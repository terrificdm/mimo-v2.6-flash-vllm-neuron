#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 1: reasoning parser tests R1-R8 (stdlib only; temperature 0).

R1 thinking on, non-stream      R2 thinking on, stream (concat == R1, reasoning before content)
R3 thinking off, non-stream     R4 thinking off, stream (no reasoning deltas, concat == R3)
R5 thinking truncated (max_tokens 32): finish 'length', everything in reasoning
R6 multi-turn: R1 assistant turn sent back with 'reasoning' and with 'reasoning_content', then "再乘以 2 呢？" -> 782
R7 4 concurrent: R1-style x2 + R3-style x2, each meets its own criteria
R8 /v1/completions raw prompt (parser does not apply; behaviour recorded)
The response field is judged as 'reasoning' (vLLM 0.24, docs/design-notes.md#reasoning-field); whether 'reasoning_content' is present is
recorded too. A stream/non-stream mismatch is re-run once: reproducible -> partial (batch numeric difference),
otherwise fail. --quick = R1 + R2.
Usage: test_reasoning.py --out PREFIX [--tag T] [--quick] [--cases R1,R3] [--timeout S] [--max-tokens N]
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

Q = "17×23 等于多少？"
EXPECT = "391"
FOLLOW_UP = "再乘以 2 呢？"
EXPECT2 = "782"
THINK_TAGS = ("<think>", "</think>")
OFF = {"enable_thinking": False}
ALL = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]


def _tags(*texts):
    return [t for t in THINK_TAGS for s in texts if s and t in s]


def judge_on(reasoning, content, finish, expect=EXPECT):
    """Thinking-on criteria; returns list of problems (empty = pass)."""
    p = []
    if not (reasoning or "").strip():
        p.append("reasoning empty")
    if expect not in (content or ""):
        p.append(f"content does not contain {expect}")
    tags = _tags(reasoning, content)
    if tags:
        p.append(f"literal {sorted(set(tags))} in output")
    if finish != "stop":
        p.append(f"finish_reason={finish!r} (want 'stop')")
    return p


def judge_off(reasoning, content, finish, expect=EXPECT):
    p = []
    if (reasoning or "").strip():
        p.append("reasoning not empty with enable_thinking=false")
    if expect not in (content or ""):
        p.append(f"content does not contain {expect}")
    if _tags(reasoning, content):
        p.append("literal <think>/</think> in output")
    if finish != "stop":
        p.append(f"finish_reason={finish!r} (want 'stop')")
    return p


def stream_order_ok(chunks):
    """All reasoning deltas must come before the first content delta."""
    first_content, last_reasoning, i = None, None, 0
    for d, _ in C.stream_deltas(chunks):
        if d.get("reasoning") or d.get("reasoning_content"):
            last_reasoning = i
        if d.get("content"):
            first_content = i if first_content is None else first_content
        i += 1
    return first_content is None or last_reasoning is None or last_reasoning < first_content


def nonstream_fields(rec):
    m = C.message(rec)
    return {"reasoning": C.reasoning_of(m), "content": m.get("content"), "finish": C.choice0(rec).get("finish_reason"),
            "has_reasoning_key": "reasoning" in m, "has_reasoning_content_key": "reasoning_content" in m}


def stream_fields(rec):
    ch = rec["chunks"]
    keys = set()
    for d, _ in C.stream_deltas(ch):
        keys.update(k for k in ("reasoning", "reasoning_content") if d.get(k))
    return {"reasoning": C.stream_concat(ch, "reasoning"), "content": C.stream_concat(ch, "content"),
            "finish": C.stream_finish(ch), "done": rec.get("done"), "delta_reasoning_keys": sorted(keys),
            "order_ok": stream_order_ok(ch), "n_chunks": len(ch)}


class Runner:
    def __init__(self, suite, timeout, max_tokens):
        self.s, self.timeout, self.mt = suite, timeout, max_tokens
        self.cache = {}

    def body(self, thinking, max_tokens=None, messages=Q):
        kw = {"max_tokens": max_tokens or self.mt}
        if not thinking:
            kw["chat_template_kwargs"] = OFF
        return C.chat_body(messages, **kw)

    def ns(self, key, thinking):
        if key not in self.cache:
            self.cache[key] = C.post("/v1/chat/completions", self.body(thinking), self.timeout)
        return self.cache[key]

    def _nonstream_case(self, cid, desc, key, thinking):
        rec = self.ns(key, thinking)
        f = nonstream_fields(rec)
        if rec["http_status"] != 200:
            return self.s.add(cid, desc, "fail", f"HTTP {rec['http_status']}", [rec], fields=f)
        p = (judge_on if thinking else judge_off)(f["reasoning"], f["content"], f["finish"])
        note = f"has 'reasoning' key={f['has_reasoning_key']}, has 'reasoning_content' key={f['has_reasoning_content_key']}"
        return self.s.add(cid, desc, "fail" if p else "pass", "; ".join(p + [note]), [rec], fields=f)

    def _stream_case(self, cid, desc, key, thinking):
        ref = nonstream_fields(self.ns(key, thinking))
        rec = C.post_stream("/v1/chat/completions", self.body(thinking), self.timeout)
        if rec["http_status"] != 200:
            return self.s.add(cid, desc, "fail", f"HTTP {rec['http_status']} {rec.get('error')}", [rec])
        f = stream_fields(rec)
        p = (judge_on if thinking else judge_off)(f["reasoning"], f["content"], f["finish"])
        if not f["done"]:
            p.append("stream did not end with [DONE]")
        if not f["order_ok"]:
            p.append("a reasoning delta arrived after the first content delta")
        if not thinking and f["delta_reasoning_keys"]:
            p.append(f"reasoning deltas present with thinking off: {f['delta_reasoning_keys']}")
        same = (f["reasoning"] or "") == (ref["reasoning"] or "") and (f["content"] or "") == (ref["content"] or "")
        records, verdict = [rec], "fail" if p else "pass"
        if not same:
            rec2 = C.post_stream("/v1/chat/completions", self.body(thinking), self.timeout)
            f2 = stream_fields(rec2) if rec2["http_status"] == 200 else {}
            records.append(rec2)
            repro = f2.get("reasoning") == f["reasoning"] and f2.get("content") == f["content"]
            if repro and not p:
                verdict = "partial"
                p.append("stream concat != non-stream, but the stream result is reproducible "
                         "(batch numeric difference)")
            else:
                verdict = "fail"
                p.append(f"stream concat != non-stream (stream re-run reproducible={repro})")
        reason = "; ".join(p) or f"concat == non-stream; {f['n_chunks']} chunks; delta keys {f['delta_reasoning_keys']}"
        return self.s.add(cid, desc, verdict, reason, records, fields=f, ref=ref)

    def R1(self):
        return self._nonstream_case("R1", "thinking on, non-stream", "on", True)

    def R2(self):
        return self._stream_case("R2", "thinking on, stream", "on", True)

    def R3(self):
        return self._nonstream_case("R3", "thinking off, non-stream", "off", False)

    def R4(self):
        return self._stream_case("R4", "thinking off, stream", "off", False)

    def R5(self):
        rec = C.post("/v1/chat/completions", self.body(True, max_tokens=32), self.timeout)
        f = nonstream_fields(rec)
        p = [] if rec["http_status"] == 200 else [f"HTTP {rec['http_status']}"]
        if not p:
            if f["finish"] != "length":
                p.append(f"finish_reason={f['finish']!r} (want 'length')")
            if not (f["reasoning"] or "").strip():
                p.append("reasoning empty")
            if (f["content"] or "").strip():
                p.append(f"content not empty: {f['content'][:80]!r}")
        return self.s.add("R5", "thinking truncated at max_tokens 32", "fail" if p else "pass",
                          "; ".join(p) or "all text in reasoning, content empty, finish length", [rec], fields=f)

    def R6(self):
        first = nonstream_fields(self.ns("on", True))
        recs, ok = [], []
        for fld in ("reasoning", "reasoning_content"):
            msgs = [{"role": "user", "content": Q},
                    {"role": "assistant", "content": first["content"] or "", fld: first["reasoning"] or ""},
                    {"role": "user", "content": FOLLOW_UP}]
            rec = C.post("/v1/chat/completions", self.body(True, messages=msgs), self.timeout)
            f = nonstream_fields(rec)
            good = rec["http_status"] == 200 and EXPECT2 in (f["content"] or "")
            rec["variant"], rec["judged"] = fld, {"pass": good, "content": f["content"], "finish": f["finish"]}
            recs.append(rec)
            ok.append(good)
        verdict = "pass" if all(ok) else ("partial" if any(ok) else "fail")
        reason = ", ".join(f"{v}: {'contains' if g else 'NO'} {EXPECT2} (HTTP {r['http_status']})"
                           for v, g, r in zip(("reasoning", "reasoning_content"), ok, recs))
        return self.s.add("R6", "multi-turn with previous reasoning sent back", verdict, reason, recs)

    def R7(self):
        modes = [True, False, True, False]
        with ThreadPoolExecutor(4) as ex:
            futs = [ex.submit(C.post, "/v1/chat/completions", self.body(m), self.timeout) for m in modes]
            recs = [fu.result() for fu in futs]
        probs = []
        for i, (m, rec) in enumerate(zip(modes, recs)):
            f = nonstream_fields(rec)
            p = [f"HTTP {rec['http_status']}"] if rec["http_status"] != 200 else \
                (judge_on if m else judge_off)(f["reasoning"], f["content"], f["finish"])
            rec["judged"] = {"thinking": m, "problems": p}
            probs += [f"#{i} ({'on' if m else 'off'}): {x}" for x in p]
        return self.s.add("R7", "4 concurrent, thinking on x2 + off x2", "fail" if probs else "pass",
                          "; ".join(probs) or "all 4 meet their criteria", recs)

    def R8(self):
        body = {"model": C.MODEL, "prompt": "问题：17×23=?\n回答：", "temperature": 0, "max_tokens": 64}
        rec = C.post("/v1/completions", body, self.timeout)
        ch = C.choice0(rec)
        text = ch.get("text")
        info = f"text={(text or '')[:120]!r}; choice keys={sorted(ch)}; contains 391={EXPECT in (text or '')}"
        ok = rec["http_status"] == 200 and bool(text)
        return self.s.add("R8", "/v1/completions raw prompt (parser not applied)", "pass" if ok else "fail",
                          (info if ok else f"HTTP {rec['http_status']}; " + info), [rec])


def main(argv=None):
    ap = C.base_argparser(__doc__)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args(argv)
    cases = ["R1", "R2"] if args.quick else ALL
    if args.cases:
        cases = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
        bad = [c for c in cases if c not in ALL]
        if bad:
            print(f"unknown cases {bad}; valid {ALL}", file=sys.stderr)
            return 2
    C.require_server()
    suite = C.Suite("reasoning", args.out, args.tag, meta={"cases_requested": cases, "max_tokens": args.max_tokens})
    r = Runner(suite, args.timeout, args.max_tokens)
    for c in cases:
        getattr(r, c)()
    print(f"reasoning: {suite.counts()}")
    return suite.exit_code()


if __name__ == "__main__":
    sys.exit(main())
