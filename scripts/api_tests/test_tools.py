#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 2: tool calling tests T1-T10 (stdlib only, jsonschema used when installed; temperature 0).

Tools: get_weather(city: string, unit: enum[celsius, fahrenheit]), get_time(timezone: string),
calculator(expression: string).
T1 single call (auto, non-stream)        T2 parallel calls (>=2; English / "please call in parallel" retries;
                                            3x one call -> partial, model behaviour)
T2b offline parser check (offline_parser_check.py, CPU)
T3 multi-turn with tool result -> final answer (contains 23 and 晴)
T4 streaming tool calls (T1 and T2 with stream; per-index id/name once; concatenated arguments == non-stream)
T5 tool_choice none                       T6 named function get_time (schema-valid)
T7 tool_choice required ("你好")           T8 thinking on + tools
T9 temperature 1.0 / top_p 0.95 x3 (>=2/3 valid get_weather)
T10 concurrent T1 + T5 + T6 + T7
T1-T7, T9, T10 use enable_thinking=false. --quick = T1 + T4 (T1 streamed only).
Usage: test_tools.py --out PREFIX [--tag T] [--quick] [--cases T1,T6] [--timeout S] [--skip-offline]
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

try:
    import jsonschema  # noqa: F401
except Exception:  # noqa: BLE001
    jsonschema = None


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


TOOLS = [
    _fn("get_weather", "查询某个城市当前的天气 / Get the current weather of a city",
        {"city": {"type": "string", "description": "城市名，例如 北京"},
         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}}, ["city"]),
    _fn("get_time", "查询某个时区当前的时间 / Get the current time in a timezone",
        {"timezone": {"type": "string", "description": "IANA 时区名，例如 Asia/Tokyo"}}, ["timezone"]),
    _fn("calculator", "计算一个数学表达式 / Evaluate a math expression",
        {"expression": {"type": "string", "description": "数学表达式，例如 17*23"}}, ["expression"]),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
OFF = {"enable_thinking": False}
Q1 = "北京现在天气怎么样？"
Q2 = ["同时告诉我北京和上海的天气，以及东京现在几点。",
      "Tell me the current weather in Beijing and in Shanghai, and the current time in Tokyo.",
      "请并行调用多个工具，一次性完成：查询北京的天气、查询上海的天气、查询东京现在的时间。"]
Q6 = "东京现在几点？"
Q7 = "你好"
TOOL_RESULT = {"temp": 23, "condition": "晴"}
ALL = ["T1", "T2", "T2b", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]


# ------------------------------------------------------------------ validation helpers
def validate_args(name, args):
    """Return a list of problems for arguments `args` (already-parsed object) of tool `name`."""
    if name not in SCHEMAS:
        return [f"unknown tool {name!r}"]
    if not isinstance(args, dict):
        return [f"{name}: arguments are not an object: {args!r}"]
    sch = SCHEMAS[name]
    if jsonschema is not None:
        try:
            jsonschema.validate(args, sch)
            return []
        except jsonschema.ValidationError as e:
            return [f"{name}: schema: {e.message}"]
    p = [f"{name}: missing {k}" for k in sch["required"] if k not in args]
    for k, v in args.items():
        ps = sch["properties"].get(k)
        if ps is None:
            continue
        if ps.get("type") == "string" and not isinstance(v, str):
            p.append(f"{name}.{k} not a string")
        if "enum" in ps and v not in ps["enum"]:
            p.append(f"{name}.{k}={v!r} not in {ps['enum']}")
    return p


def parse_calls(tool_calls):
    """[(id, name, args_obj_or_None, raw_args, problems)]"""
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        probs = []
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = None
            probs.append(f"arguments not JSON: {raw!r}")
        if args is not None:
            probs += validate_args(fn.get("name"), args)
        out.append((tc.get("id"), fn.get("name"), args, raw, probs))
    return out


def has_tag(*texts):
    return any(s and "<tool_call>" in s for s in texts)


def judge_weather_call(msg, finish, want_finish=True):
    """T1 criteria; returns problems."""
    p = []
    if want_finish and finish != "tool_calls":
        p.append(f"finish_reason={finish!r} (want 'tool_calls')")
    calls = parse_calls(msg.get("tool_calls"))
    if len(calls) != 1:
        p.append(f"{len(calls)} tool calls (want exactly 1)")
    for _id, name, args, _raw, probs in calls[:1]:
        p += probs
        if name != "get_weather":
            p.append(f"name={name!r} (want get_weather)")
        elif isinstance(args, dict):
            city = str(args.get("city", ""))
            if "北京" not in city and "beijing" not in city.lower():
                p.append(f"city={city!r} (want 北京/Beijing)")
    if has_tag(msg.get("content")):
        p.append("content contains literal <tool_call>")
    return p


def judge_none(msg):
    p = []
    if msg.get("tool_calls"):
        p.append(f"{len(msg['tool_calls'])} tool_calls with tool_choice none")
    if not (msg.get("content") or "").strip():
        p.append("content empty")
    return p


def judge_named(msg, name="get_time"):
    calls = parse_calls(msg.get("tool_calls"))
    p = [] if len(calls) == 1 else [f"{len(calls)} tool calls (want exactly 1)"]
    for _id, n, _a, _raw, probs in calls[:1]:
        p += probs
        if n != name:
            p.append(f"name={n!r} (want {name})")
    return p


def judge_required(msg):
    calls = parse_calls(msg.get("tool_calls"))
    p = [] if calls else ["no tool call with tool_choice required"]
    for _id, n, _a, _raw, probs in calls:
        p += probs
    return p


def accumulate_stream_calls(chunks):
    """Accumulate delta.tool_calls by index. Returns (calls dict index->{id,name,arguments}, problems)."""
    calls, p = {}, []
    for d, _ in C.stream_deltas(chunks):
        for tc in d.get("tool_calls") or []:
            i = tc.get("index")
            if i is None:
                p.append("tool_call delta without index")
                continue
            cur = calls.setdefault(i, {"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                if cur["id"] is not None:
                    p.append(f"index {i}: id sent twice")
                cur["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                if cur["name"] is not None:
                    p.append(f"index {i}: name sent twice")
                cur["name"] = fn["name"]
            if fn.get("arguments"):
                cur["arguments"] += fn["arguments"]
    return calls, p


def compare_stream(chunks, ref_msg):
    """Stream vs non-stream: same number of calls, same names, json-equal arguments, final finish tool_calls."""
    calls, p = accumulate_stream_calls(chunks)
    ref = parse_calls(ref_msg.get("tool_calls"))
    if len(calls) != len(ref):
        p.append(f"stream has {len(calls)} calls, non-stream {len(ref)}")
    for i, (_id, name, args, _raw, _pr) in enumerate(ref):
        c = calls.get(i)
        if c is None:
            continue
        if c["id"] is None or c["name"] is None:
            p.append(f"index {i}: id or name missing in stream")
        try:
            sargs = json.loads(c["arguments"])
        except ValueError:
            p.append(f"index {i}: streamed arguments not JSON: {c['arguments']!r}")
            continue
        if c["name"] != name or sargs != args:
            p.append(f"index {i}: stream {c['name']} {sargs} != non-stream {name} {args}")
    fin = C.stream_finish(chunks)
    if fin != "tool_calls":
        p.append(f"last finish_reason={fin!r} (want 'tool_calls')")
    return calls, p


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, suite, timeout, skip_offline):
        self.s, self.timeout, self.skip_offline = suite, timeout, skip_offline
        self.cache = {}

    def body(self, q, thinking=False, tool_choice=None, max_tokens=None, **kw):
        b = C.chat_body(q, tools=TOOLS, max_tokens=max_tokens or (2048 if thinking else 512), **kw)
        if not thinking:
            b["chat_template_kwargs"] = OFF
        if tool_choice is not None:
            b["tool_choice"] = tool_choice
        return b

    def call(self, key, body):
        if key not in self.cache:
            self.cache[key] = C.post("/v1/chat/completions", body, self.timeout)
        return self.cache[key]

    def t1(self):
        return self.call("t1", self.body(Q1))

    def _http_fail(self, cid, desc, rec):
        return self.s.add(cid, desc, "fail", f"HTTP {rec['http_status']}: {str(rec['response'])[:300]}", [rec])

    @staticmethod
    def _calls_summary(msg):
        return [(n, a if a is not None else raw) for _i, n, a, raw, _p in parse_calls(msg.get("tool_calls"))]

    def T1(self):
        rec = self.t1()
        if rec["http_status"] != 200:
            return self._http_fail("T1", "single tool call (auto)", rec)
        m = C.message(rec)
        p = judge_weather_call(m, C.choice0(rec).get("finish_reason"))
        return self.s.add("T1", "single tool call (auto)", "fail" if p else "pass",
                          "; ".join(p) or f"calls {self._calls_summary(m)}", [rec])

    def T2(self):
        recs, best = [], None
        for i, q in enumerate(Q2):
            rec = self.call(f"t2_{i}", self.body(q))
            recs.append(rec)
            if rec["http_status"] != 200:
                continue
            calls = parse_calls(C.message(rec).get("tool_calls"))
            bad = [x for c in calls for x in c[4]]
            if len(calls) >= 2 and not bad:
                best = (i, calls)
                break
        if best is not None:
            i, calls = best
            return self.s.add("T2", "parallel tool calls", "pass",
                              f"prompt #{i}: {len(calls)} valid calls {[(c[1], c[2]) for c in calls]}", recs)
        singles = [r for r in recs if r["http_status"] == 200 and
                   len(parse_calls(C.message(r).get("tool_calls"))) == 1 and
                   not any(x for c in parse_calls(C.message(r).get("tool_calls")) for x in c[4])]
        if len(singles) == len(Q2):
            return self.s.add("T2", "parallel tool calls", "partial",
                              "model gave exactly 1 valid call for all 3 prompts (model behaviour; parser parallel "
                              "capability is checked by T2b)", recs)
        reasons = []
        for i, r in enumerate(recs):
            calls = parse_calls(C.message(r).get("tool_calls")) if r["http_status"] == 200 else []
            reasons.append(f"#{i}: HTTP {r['http_status']}, {len(calls)} calls, problems "
                           f"{[x for c in calls for x in c[4]]}")
        return self.s.add("T2", "parallel tool calls", "fail", "; ".join(reasons), recs)

    def T2b(self):
        if self.skip_offline:
            return self.s.add("T2b", "offline parser check", "info", "skipped (--skip-offline)")
        here = os.path.dirname(os.path.abspath(__file__))
        out = (self.s.out + "_T2b") if self.s.out else None
        cmd = [sys.executable, os.path.join(here, "offline_parser_check.py")] + (["--out", out] if out else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            rc, tail = p.returncode, (p.stdout + p.stderr)[-1500:]
        except Exception as e:  # noqa: BLE001
            rc, tail = -1, repr(e)
        verdict = "pass" if rc == 0 else "fail"
        return self.s.add("T2b", "offline parser check (2 <tool_call> blocks)", verdict, f"rc={rc}",
                          [{"cmd": cmd, "rc": rc, "output_tail": tail}])

    def T3(self):
        r1 = self.t1()
        m1 = C.message(r1)
        calls = m1.get("tool_calls") or []
        if r1["http_status"] != 200 or not calls:
            return self.s.add("T3", "multi-turn with tool result", "fail", "T1 gave no tool call to answer", [r1])
        tc = calls[0]
        msgs = [{"role": "user", "content": Q1},
                {"role": "assistant", "content": m1.get("content") or "", "tool_calls": [
                    {"id": tc.get("id"), "type": "function", "function": tc.get("function")}]},
                {"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(TOOL_RESULT, ensure_ascii=False)}]
        rec = C.post("/v1/chat/completions", self.body(msgs), self.timeout)
        if rec["http_status"] != 200:
            return self._http_fail("T3", "multi-turn with tool result", rec)
        m = C.message(rec)
        p = []
        if m.get("tool_calls"):
            p.append(f"called tools again: {self._calls_summary(m)}")
        c = m.get("content") or ""
        if "23" not in c or "晴" not in c:
            p.append(f"content lacks 23 / 晴: {c[:200]!r}")
        return self.s.add("T3", "multi-turn with tool result", "fail" if p else "pass",
                          "; ".join(p) or f"final answer: {c[:160]!r}", [rec])

    def _stream_one(self, label, key, body):
        ref = self.call(key, body)
        rec = C.post_stream("/v1/chat/completions", body, self.timeout)
        if rec["http_status"] != 200 or ref["http_status"] != 200:
            return "fail", [f"{label}: HTTP stream {rec['http_status']} / non-stream {ref['http_status']}"], [rec]
        calls, p = compare_stream(rec["chunks"], C.message(ref))
        if not rec.get("done"):
            p.append("stream did not end with [DONE]")
        if not p:
            return "pass", [f"{label}: {len(calls)} calls match non-stream"], [rec]
        rec2 = C.post_stream("/v1/chat/completions", body, self.timeout)
        calls2, _ = accumulate_stream_calls(rec2.get("chunks") or [])
        repro = rec2["http_status"] == 200 and calls2 == calls
        structural = [x for x in p if "twice" in x or "missing" in x or "without index" in x or "not JSON" in x
                      or "finish_reason" in x or "[DONE]" in x]
        if repro and not structural:
            return "partial", [f"{label}: " + "; ".join(p) + " (stream result reproducible: batch numeric "
                                                            "difference)"], [rec, rec2]
        return "fail", [f"{label}: " + "; ".join(p) + f" (re-run reproducible={repro})"], [rec, rec2]

    def T4(self, quick=False):
        items = [("T1", "t1", self.body(Q1))]
        if not quick:
            items.append(("T2", "t2_0", self.body(Q2[0])))
        verdicts, reasons, recs = [], [], []
        for label, key, body in items:
            v, r, rs = self._stream_one(label, key, body)
            verdicts.append(v)
            reasons += r
            recs += rs
        verdict = "fail" if "fail" in verdicts else ("partial" if "partial" in verdicts else "pass")
        return self.s.add("T4", "streaming tool calls" + (" (T1 only)" if quick else ""), verdict,
                          "; ".join(reasons), recs)

    def T5(self):
        rec = self.call("t5", self.body(Q1, tool_choice="none"))
        if rec["http_status"] != 200:
            return self._http_fail("T5", "tool_choice none", rec)
        m = C.message(rec)
        p = judge_none(m)
        note = f"content={(m.get('content') or '')[:120]!r}; literal <tool_call> in content={has_tag(m.get('content'))}"
        return self.s.add("T5", "tool_choice none", "fail" if p else "pass", "; ".join(p + [note]), [rec])

    def T6(self):
        rec = self.call("t6", self.body(Q6, tool_choice={"type": "function", "function": {"name": "get_time"}}))
        if rec["http_status"] != 200:
            return self._http_fail("T6", "named function get_time", rec)
        m = C.message(rec)
        p = judge_named(m)
        return self.s.add("T6", "named function get_time", "fail" if p else "pass",
                          "; ".join(p) or f"calls {self._calls_summary(m)}; jsonschema={'yes' if jsonschema else 'no'}",
                          [rec])

    def T7(self):
        rec = self.call("t7", self.body(Q7, tool_choice="required"))
        if rec["http_status"] != 200:
            return self._http_fail("T7", "tool_choice required", rec)
        m = C.message(rec)
        p = judge_required(m)
        return self.s.add("T7", "tool_choice required", "fail" if p else "pass",
                          "; ".join(p) or f"calls {self._calls_summary(m)}", [rec])

    def T8(self):
        rec = C.post("/v1/chat/completions", self.body(Q1, thinking=True), self.timeout)
        if rec["http_status"] != 200:
            return self._http_fail("T8", "thinking on + tools", rec)
        m = C.message(rec)
        r = C.reasoning_of(m)
        p = judge_weather_call(m, C.choice0(rec).get("finish_reason"))
        if has_tag(r):
            p.append("reasoning contains literal <tool_call>")
        note = f"reasoning {'non-empty' if (r or '').strip() else 'empty'} ({len(r or '')} chars)"
        return self.s.add("T8", "thinking on + tools", "fail" if p else "pass", "; ".join(p + [note]), [rec])

    def T9(self):
        recs, good = [], 0
        for _ in range(3):
            rec = C.post("/v1/chat/completions", self.body(Q1, temperature=1.0, top_p=0.95), self.timeout)
            ok = rec["http_status"] == 200 and not judge_weather_call(C.message(rec), C.choice0(rec).get("finish_reason"))
            rec["judged_ok"] = ok
            good += ok
            recs.append(rec)
        return self.s.add("T9", "temperature 1.0 / top_p 0.95 x3", "pass" if good >= 2 else "fail",
                          f"{good}/3 valid get_weather calls", recs)

    def T10(self):
        jobs = [("T1", self.body(Q1), lambda m, f: judge_weather_call(m, f)),
                ("T5", self.body(Q1, tool_choice="none"), lambda m, f: judge_none(m)),
                ("T6", self.body(Q6, tool_choice={"type": "function", "function": {"name": "get_time"}}),
                 lambda m, f: judge_named(m)),
                ("T7", self.body(Q7, tool_choice="required"), lambda m, f: judge_required(m))]
        with ThreadPoolExecutor(len(jobs)) as ex:
            futs = [ex.submit(C.post, "/v1/chat/completions", b, self.timeout) for _, b, _ in jobs]
            recs = [f.result() for f in futs]
        probs = []
        for (label, _b, judge), rec in zip(jobs, recs):
            p = [f"HTTP {rec['http_status']}"] if rec["http_status"] != 200 else \
                judge(C.message(rec), C.choice0(rec).get("finish_reason"))
            rec["judged"] = {"as": label, "problems": p}
            probs += [f"{label}: {x}" for x in p]
        return self.s.add("T10", "concurrent T1 + T5 + T6 + T7", "fail" if probs else "pass",
                          "; ".join(probs) or "all 4 meet their criteria", recs)


def main(argv=None):
    ap = C.base_argparser(__doc__)
    ap.add_argument("--skip-offline", action="store_true", help="skip T2b (needs vllm + tokenizer)")
    args = ap.parse_args(argv)
    cases = ["T1", "T4"] if args.quick else ALL
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
        bad = [c for c in cases if c not in ALL]
        if bad:
            print(f"unknown cases {bad}; valid {ALL}", file=sys.stderr)
            return 2
    C.require_server()
    suite = C.Suite("tools", args.out, args.tag, meta={"cases_requested": cases, "quick": args.quick,
                                                       "jsonschema": jsonschema is not None})
    r = Runner(suite, args.timeout, args.skip_offline)
    for c in cases:
        if c == "T4":
            r.T4(quick=args.quick)
        else:
            getattr(r, c)()
    print(f"tools: {suite.counts()}")
    return suite.exit_code()


if __name__ == "__main__":
    sys.exit(main())
