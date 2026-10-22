#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 3: sampling / OpenAI API parameter tests (stdlib only, jsonschema used when installed).

--mode A|B is recorded in the result (A = on-device sampling, B = CPU sampling). Every case gets
a support conclusion (field `support`): supported / partial / unsupported / unverified; the Suite verdict is
pass / partial / fail / info respectively. Chat cases use enable_thinking=false.

Cases (ids are the keys param_matrix.py merges on):
  temperature_top_p   5x T=1.0/top_p=0.95 all contain 北京, not all identical; T=1.0/top_p=0.001 == greedy;
                      + logprobs/top_logprobs 20 nucleus check (only steps whose top-20 mass >= 0.95 are counted)
  top_k               top_k=1 (T=1.0) == greedy; top_k=5 + top_logprobs 5: every sampled token in that step's top-5
  seed                T=1.0: same seed twice identical, a different seed differs
  stop                stop [", 5"] on counting: no ", 5", finish stop, stop_reason ", 5"
  stop_token_ids      token id of the first "5" (from a greedy run with return_tokens_as_token_ids): stops before it
  max_tokens          max_tokens 7: completion_tokens == 7, finish length
  logprobs_chat       logprobs + top_logprobs 5: one entry per token, 5 sorted values <= 0, greedy token == top-1,
                      at T=1.0 the sampled token's logprob equals its value in the top list
  logprobs_completions  /v1/completions logprobs 5, same checks
  n_chat / n_completions          n=3, T=1.0: 3 choices, index 0-2, not all identical
  stream_usage_chat / stream_usage_completions  stream_options.include_usage: only the last chunk has usage and it
                      equals the non-stream usage
  presence_penalty / frequency_penalty  1.5 vs none on "apple x40" (greedy): output differs and apple count <= half;
                      if greedy is unchanged: same seed T=1.0 without penalty x2 (identical) vs penalty 2.0 (differs)
  json_object         response_format json_object: content json.loads to an object
  json_schema         response_format json_schema strict: json.loads + schema valid
  generation_config   --gencfg vllm|auto (default A->vllm, B->auto): (1) serve log line "Default vLLM sampling
                      parameters have been overridden" absent (vllm) / present (auto) after the last serve start;
                      (2) chat without max_tokens, ignore_eos: > 2048 tokens (vllm) / == 2048 (auto);
                      (3) chat without max_tokens ends with finish_reason stop
--quick = temperature_top_p + stream_usage_chat + stream_usage_completions.
--lp-greedy-only skips every logprobs request with temperature > 0 (nucleus check, top-5 membership, T=1.0 logprobs
consistency; recorded as skipped in the reason). Reason: in mode A such a request killed the EngineCore on api-4096
(seen in a server crash log: vllm/v1/sample/sampler.py:237 apply_temperature
"Expected all tensors to be on the same device ... neuron:N and cpu"; vllm_neuron neuron_model_runner.py:8101 moves
the gathered logits to CPU but sampling_metadata stays on the neuron device).
After every case GET /health is checked; if the engine is gone the run stops (meta engine_dead_after), exit 2.
Exit: 0 when every CORE case that ran is 'supported' (the acceptance list); 1 otherwise;
2 server unreachable / bad args.
Usage: test_params.py --mode A|B --out PREFIX [--tag T] [--quick] [--cases a,b] [--timeout S] [--gencfg vllm|auto]
       [--serve-log PATH] [--lp-greedy-only]
"""

import json
import math
import os
import re
import sys
import traceback
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

try:
    import jsonschema
except Exception:  # noqa: BLE001
    jsonschema = None

CHAT, COMP = "/v1/chat/completions", "/v1/completions"
OFF = {"enable_thinking": False}
SUPPORTS = ("supported", "partial", "unsupported", "unverified")
VERDICT = {"supported": "pass", "partial": "partial", "unsupported": "fail", "unverified": "info"}
GENCFG_LOG_LINE = "Default vLLM sampling parameters have been overridden"
SERVE_START_MARK = "===== [serve] start"
GENCFG_MAX_NEW = 2048  # reference/hf/generation_config.json max_new_tokens

Q_CAP = "中国的首都是哪里？用一句话回答，并简单介绍它。"
Q_OPEN = "Describe autumn in two sentences."
Q_POEM = "写一首关于秋天的四行短诗。"
Q_COUNT = "从 1 数到 20，用英文逗号加一个空格分隔，只输出数字。"
Q_INTRO = "介绍一下长城。"
Q_APPLE = "把 apple 这个词重复写 40 遍，用空格分隔，不要输出别的内容。"
Q_JSON = "用 JSON 给出北京的名称和人口，只输出 JSON。"
Q_SCHEMA = "介绍北京：给出名称、人口和至少两个地标，按要求的 JSON 格式输出。"
P_COMP = "The capital of France is"
P_AUTUMN = "写一句关于秋天的话："
Q_LONG_STORY = "讲一个很长的童话故事。"
CITY_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "population": {"type": "integer"},
    "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
    "required": ["name", "population", "landmarks"], "additionalProperties": False}

ALL = ["temperature_top_p", "top_k", "seed", "stop", "stop_token_ids", "max_tokens", "logprobs_chat",
       "logprobs_completions", "n_chat", "n_completions", "stream_usage_chat", "stream_usage_completions",
       "presence_penalty", "frequency_penalty", "json_object", "json_schema", "generation_config"]
QUICK = ["temperature_top_p", "stream_usage_chat", "stream_usage_completions"]
CORE = ["temperature_top_p", "top_k", "stop", "max_tokens", "logprobs_chat", "logprobs_completions",
        "stream_usage_chat", "stream_usage_completions", "json_object", "json_schema", "generation_config"]
PARAM = {
    "temperature_top_p": "temperature=1.0 + top_p=0.95", "top_k": "top_k", "seed": "seed", "stop": "stop（字符串）",
    "stop_token_ids": "stop_token_ids", "max_tokens": "max_tokens", "logprobs_chat": "logprobs / top_logprobs（chat）",
    "logprobs_completions": "logprobs（completions）", "n_chat": "n>1（chat）", "n_completions": "n>1（completions）",
    "stream_usage_chat": "stream + stream_options.include_usage（chat）",
    "stream_usage_completions": "stream + stream_options.include_usage（completions）",
    "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
    "json_object": "response_format: json_object", "json_schema": "response_format: json_schema",
    "generation_config": "--generation-config"}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def chat_lp_entries(rec):
    """Chat logprobs -> [{token, logprob, top_logprobs: [{token, logprob}] sorted desc}]."""
    lp = C.choice0(rec).get("logprobs") or {}
    out = []
    for e in lp.get("content") or []:
        tops = sorted(e.get("top_logprobs") or [], key=lambda t: -t["logprob"])
        out.append({"token": e.get("token"), "logprob": e.get("logprob"), "top_logprobs": tops})
    return out


def comp_lp_entries(choice):
    """Completions logprobs {tokens, token_logprobs, top_logprobs: [dict]} -> same shape as chat_lp_entries."""
    lp = (choice or {}).get("logprobs") or {}
    toks, vals, tops = lp.get("tokens") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or []
    out = []
    for i, t in enumerate(toks):
        d = tops[i] if i < len(tops) and tops[i] else {}
        out.append({"token": t, "logprob": vals[i] if i < len(vals) else None,
                    "top_logprobs": sorted(({"token": k, "logprob": v} for k, v in d.items()),
                                           key=lambda x: -x["logprob"])})
    return out


def find_in_tops(entry, tops, tol=1e-2):
    """Index of the sampled token in the top list (token string and logprob match; token-only match as fallback)."""
    tok, lp = entry.get("token"), entry.get("logprob")
    for j, t in enumerate(tops):
        if t.get("token") == tok and lp is not None and abs(t["logprob"] - lp) <= tol:
            return j
    for j, t in enumerate(tops):
        if t.get("token") == tok:
            return j
    return None


def nucleus_check(entries, top_p=0.95, tol=0.01):
    """Count steps where the sampled token lies outside the top_p nucleus computed from top_logprobs (raw
    distribution). Only steps whose listed mass >= top_p are 'covered'. A token at rank j is inside when the mass of
    the tokens ranked before it is < top_p + tol."""
    res = {"steps": len(entries), "covered": 0, "out_of_nucleus": 0, "examples": []}
    for i, e in enumerate(entries):
        tops = e["top_logprobs"]
        probs = [math.exp(t["logprob"]) for t in tops]
        if sum(probs) < top_p:
            continue
        res["covered"] += 1
        j = find_in_tops(e, tops)
        before = sum(probs[:j]) if j is not None else None
        if j is None or before >= top_p + tol:
            res["out_of_nucleus"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append({"step": i, "token": e["token"], "logprob": e["logprob"], "rank": j,
                                        "mass_before": None if before is None else round(before, 4)})
    return res


def topk_check(entries, k, tol=1e-2):
    """Steps whose sampled token is not among the step's top-k (ties within tol of the k-th value allowed)."""
    bad = []
    for i, e in enumerate(entries):
        tops = e["top_logprobs"][:k]
        if not tops:
            bad.append({"step": i, "token": e["token"], "why": "no top_logprobs"})
            continue
        j = find_in_tops(e, tops)
        if j is None and (e["logprob"] is None or e["logprob"] < tops[-1]["logprob"] - tol):
            bad.append({"step": i, "token": e["token"], "logprob": e["logprob"], "kth": tops[-1]["logprob"]})
    return bad


def logprob_shape_problems(entries, k, n_tokens, exact_k=True):
    p = []
    if n_tokens is not None and len(entries) != n_tokens:
        p.append(f"{len(entries)} logprob entries for {n_tokens} completion tokens")
    short = 0
    for i, e in enumerate(entries):
        tops = e["top_logprobs"]
        if (len(tops) != k) if exact_k else (len(tops) < k):
            short += 1
        vals = [t["logprob"] for t in tops]
        if any(v > 1e-6 for v in vals):
            p.append(f"step {i}: positive logprob {max(vals)}")
            break
    # completions top_logprobs is a dict keyed by token text: duplicate strings may collapse (allow <=5% of steps)
    if short and (exact_k or short > 0.05 * max(1, len(entries))):
        p.append(f"{short}/{len(entries)} steps without {k} top_logprobs")
    return p


def greedy_top1_problems(entries, tol=1e-2):
    bad = [i for i, e in enumerate(entries)
           if e["top_logprobs"] and (e["logprob"] is None or e["logprob"] < e["top_logprobs"][0]["logprob"] - tol)]
    return [f"greedy token below top-1 at steps {bad[:10]}"] if bad else []


def sampled_consistency_problems(entries, tol=1e-2):
    """At T>0: the sampled token's logprob must equal its value in the top list (when listed) and be <= top-1."""
    bad = []
    for i, e in enumerate(entries):
        tops = e["top_logprobs"]
        if not tops or e["logprob"] is None:
            continue
        j = find_in_tops(e, tops, tol=float("inf"))
        if j is not None and abs(tops[j]["logprob"] - e["logprob"]) > tol:
            bad.append(i)
        elif e["logprob"] > tops[0]["logprob"] + tol:
            bad.append(i)
    return [f"sampled-token logprob inconsistent with top list at steps {bad[:10]}"] if bad else []


def count_word(text, word="apple"):
    return len(re.findall(re.escape(word), text or "", re.I))


def gencfg_log_check(path, mark=SERVE_START_MARK, line=GENCFG_LOG_LINE):
    """Return (found, start_line_no, hit_line_no) for the override line after the last serve start; None if the
    log is not readable."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    start = max((i for i, ln in enumerate(lines) if mark in ln), default=-1)
    for i in range(start + 1, len(lines)):
        if line in lines[i]:
            return True, start + 1, i + 1
    return False, start + 1, None


def engine_alive(base=None, timeout=10):
    try:
        with urllib.request.urlopen((base or C.base_url()) + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def usage_stream_problems(chunks, ref_usage):
    p = []
    with_usage = [i for i, c in enumerate(chunks) if c.get("usage")]
    if not with_usage:
        return ["no chunk carries usage"]
    if with_usage != [len(chunks) - 1]:
        p.append(f"usage on chunks {with_usage[:5]} of {len(chunks)} (want only the last)")
    u = chunks[with_usage[-1]]["usage"]
    for k in ("prompt_tokens", "completion_tokens"):
        if (ref_usage or {}).get(k) != u.get(k):
            p.append(f"stream {k}={u.get(k)} != non-stream {(ref_usage or {}).get(k)}")
    return p


def schema_problems(obj, schema=CITY_SCHEMA):
    if jsonschema is not None:
        try:
            jsonschema.validate(obj, schema)
            return []
        except jsonschema.ValidationError as e:
            return [f"schema: {e.message}"]
    if not isinstance(obj, dict):
        return ["not an object"]
    p = [f"missing {k}" for k in schema["required"] if k not in obj]
    p += [f"extra key {k}" for k in obj if k not in schema["properties"]]
    if "name" in obj and not isinstance(obj["name"], str):
        p.append("name not a string")
    if "population" in obj and (not isinstance(obj["population"], int) or isinstance(obj["population"], bool)):
        p.append("population not an integer")
    lm = obj.get("landmarks")
    if "landmarks" in obj and (not isinstance(lm, list) or len(lm) < 2 or not all(isinstance(x, str) for x in lm)):
        p.append("landmarks not an array of >=2 strings")
    return p


# ------------------------------------------------------------------ runner
class HttpProblem(Exception):
    def __init__(self, support, reason):
        super().__init__(reason)
        self.support, self.reason = support, reason


def content_of(rec, i=0):
    try:
        ch = rec["response"]["choices"][i]
    except (KeyError, IndexError, TypeError):
        return ""
    return (ch.get("message") or {}).get("content") or ch.get("text") or ""


def usage_of(rec):
    return (rec.get("response") or {}).get("usage") or {}


class Runner:
    def __init__(self, suite, mode, timeout, gencfg, serve_log, lp_greedy_only=False):
        self.s, self.mode, self.t, self.gencfg, self.serve_log = suite, mode, timeout, gencfg, serve_log
        self.lp_greedy_only = lp_greedy_only
        self.recs, self.cache = [], {}

    # -- request helpers: every record is kept for the current case
    def chat(self, q, **kw):
        kw.setdefault("chat_template_kwargs", OFF)
        return C.chat_body(q, **kw)

    @staticmethod
    def comp(prompt, **kw):
        return dict({"model": C.MODEL, "prompt": prompt, "temperature": 0}, **kw)

    def _check(self, rec):
        st = rec.get("http_status")
        if st == 200:
            return rec
        msg = str(rec.get("response") or rec.get("error"))[:400]
        if st is None or st < 0:
            raise HttpProblem("unverified", f"request error / timeout: {msg}")
        raise HttpProblem("unsupported", f"HTTP {st}: {msg}")

    def post(self, body, ep=CHAT, key=None, timeout=None):
        if key is not None and key in self.cache:
            rec = self.cache[key]
        else:
            rec = C.post(ep, body, timeout or self.t)
            if key is not None:
                self.cache[key] = rec
        self.recs.append(rec)
        return self._check(rec)

    def stream(self, body, ep=CHAT):
        rec = C.post_stream(ep, body, self.t)
        self.recs.append(rec)
        if rec.get("http_status") == 200 and not rec.get("error"):
            return rec
        st = rec.get("http_status")
        msg = str(rec.get("error"))[:400]
        raise HttpProblem("unverified" if (st is None or st < 0) else "unsupported", f"HTTP {st}: {msg}")

    def run(self, cid):
        self.recs = []
        try:
            support, reason, extra = getattr(self, "c_" + cid)()
        except HttpProblem as e:
            support, reason, extra = e.support, e.reason, {}
        except Exception as e:  # noqa: BLE001
            support, reason = "unverified", f"exception in test code: {e!r}"
            extra = {"traceback": traceback.format_exc()[-2000:]}
        return self.s.add(cid, PARAM[cid], VERDICT[support], reason, self.recs, support=support, mode=self.mode,
                          param=PARAM[cid], core=cid in CORE, **extra)

    # -- cases: each returns (support, reason, extra-dict)
    def c_temperature_top_p(self):
        texts = [content_of(self.post(self.chat(Q_CAP, temperature=1.0, top_p=0.95, max_tokens=64)))
                 for _ in range(5)]
        p = []
        miss = [i for i, t in enumerate(texts) if "北京" not in t]
        if miss:
            p.append(f"answers {miss} lack 北京")
        distinct = len(set(texts))
        if distinct == 1:
            p.append("5 answers identical (no sampling observed)")
        g = content_of(self.post(self.chat(Q_OPEN, max_tokens=64), key="greedy_open"))
        tiny = content_of(self.post(self.chat(Q_OPEN, temperature=1.0, top_p=0.001, max_tokens=64)))
        if tiny != g:
            p.append(f"top_p=0.001 output differs from greedy: {tiny[:80]!r} vs {g[:80]!r}")
        nuc = {"steps": 0, "covered": 0, "out_of_nucleus": 0, "examples": []}
        nuc_err = "skipped (--lp-greedy-only)" if self.lp_greedy_only else None
        for q in ((Q_CAP, Q_OPEN) if not self.lp_greedy_only else ()):
            try:
                rec = self.post(self.chat(q, temperature=1.0, top_p=0.95, max_tokens=96, logprobs=True,
                                          top_logprobs=20))
            except HttpProblem as e:
                nuc_err = e.reason
                break
            r = nucleus_check(chat_lp_entries(rec))
            for k in ("steps", "covered", "out_of_nucleus"):
                nuc[k] += r[k]
            nuc["examples"] += r["examples"]
        note = []
        if nuc_err and self.lp_greedy_only:
            note.append("nucleus check skipped (--lp-greedy-only)")
        elif nuc_err:
            p.append(f"nucleus check not possible (logprobs request: {nuc_err})")
        elif nuc["covered"] == 0:
            p.append("nucleus check: no covered steps")
        elif nuc["out_of_nucleus"]:
            p.append(f"{nuc['out_of_nucleus']}/{nuc['covered']} covered steps sampled outside the 0.95 nucleus")
        reason = f"{distinct}/5 distinct answers; top_p=0.001 == greedy: {tiny == g}; nucleus " \
                 f"{nuc['out_of_nucleus']} out of {nuc['covered']} covered steps ({nuc['steps']} total)"
        return ("partial" if p else "supported"), "; ".join(p + note + [reason]), {"nucleus": nuc, "texts": texts}

    def c_top_k(self):
        g = content_of(self.post(self.chat(Q_OPEN, max_tokens=64), key="greedy_open"))
        k1 = content_of(self.post(self.chat(Q_OPEN, temperature=1.0, top_k=1, max_tokens=64)))
        p = []
        if k1 != g:
            p.append(f"top_k=1 output differs from greedy: {k1[:80]!r} vs {g[:80]!r}")
        if self.lp_greedy_only:
            return ("partial" if p else "supported"), "; ".join(p) or \
                "top_k=1 == greedy; top-5 membership skipped (--lp-greedy-only)", {}
        rec5 = self.post(self.chat(Q_OPEN, temperature=1.0, top_k=5, max_tokens=64, logprobs=True, top_logprobs=5))
        ents = chat_lp_entries(rec5)
        bad = topk_check(ents, 5)
        if not ents:
            p.append("top_k=5 response has no logprobs (membership not checkable)")
        elif bad:
            p.append(f"{len(bad)}/{len(ents)} steps sampled outside top-5: {bad[:3]}")
        support = "supported" if not p else ("unsupported" if k1 != g and bad else "partial")
        return support, "; ".join(p) or f"top_k=1 == greedy; top-5 membership ok over {len(ents)} steps", {}

    def c_seed(self):
        body = lambda s: self.chat(Q_POEM, temperature=1.0, max_tokens=48, seed=s)  # noqa: E731
        a, b, c = (content_of(self.post(body(s))) for s in (1234, 1234, 5678))
        info = {"same_seed_equal": a == b, "other_seed_differs": c != a}
        if a == b and c != a:
            return "supported", "same seed identical, different seed differs", info
        if a != b:
            return "unsupported", "same seed gave different outputs (seed ignored)", info
        return "unverified", "all three outputs identical (no randomness observed)", info

    def c_stop(self):
        rec = self.post(self.chat(Q_COUNT, max_tokens=128, stop=[", 5"]))
        ch, txt = C.choice0(rec), content_of(rec)
        p = []
        if ", 5" in txt:
            p.append("content contains ', 5'")
        if ch.get("finish_reason") != "stop":
            p.append(f"finish_reason={ch.get('finish_reason')!r}")
        if ch.get("stop_reason") != ", 5":
            p.append(f"stop_reason={ch.get('stop_reason')!r}")
        if not txt.strip():
            p.append("content empty")
        sup = "supported" if not p else ("unsupported" if ", 5" in txt else "partial")
        return sup, "; ".join(p) or f"content {txt!r}", {}

    def c_stop_token_ids(self):
        base = self.post(self.chat(Q_COUNT, max_tokens=64, logprobs=True), key="count_lp")
        ids = self.post(self.chat(Q_COUNT, max_tokens=64, logprobs=True, return_tokens_as_token_ids=True))
        strs = [e["token"] for e in chat_lp_entries(base)]
        tids = [e["token"] for e in chat_lp_entries(ids)]
        idx = next((i for i, s in enumerate(strs) if "5" in (s or "")), None)
        if idx is None or idx >= len(tids) or not str(tids[idx]).startswith("token_id:"):
            return "unverified", f"could not find the token id of '5' (strings {strs[:12]}, ids {tids[:12]})", {}
        tid = int(str(tids[idx]).split(":", 1)[1])
        rec = self.post(self.chat(Q_COUNT, max_tokens=64, stop_token_ids=[tid]))
        ch, txt = C.choice0(rec), content_of(rec)
        prefix = "".join(strs[:idx]).strip()
        p = []
        if "5" in txt:
            p.append(f"content contains '5': {txt!r}")
        if ch.get("finish_reason") != "stop":
            p.append(f"finish_reason={ch.get('finish_reason')!r}")
        if ch.get("stop_reason") != tid:
            p.append(f"stop_reason={ch.get('stop_reason')!r} (want {tid})")
        if txt.strip() != prefix:
            p.append(f"content {txt.strip()!r} != greedy prefix {prefix!r}")
        sup = "supported" if not p else ("unsupported" if "5" in txt else "partial")
        return sup, "; ".join(p) or f"stopped before token {tid}; content {txt!r}", {"stop_token_id": tid}

    def c_max_tokens(self):
        rec = self.post(self.chat(Q_INTRO, max_tokens=7))
        ct, fr = usage_of(rec).get("completion_tokens"), C.choice0(rec).get("finish_reason")
        ok = ct == 7 and fr == "length"
        return ("supported" if ok else "unsupported"), f"completion_tokens={ct} finish_reason={fr!r}", {}

    def _lp_judge(self, greedy_rec, greedy_ents, temp_ents, k, exact_k):
        p = logprob_shape_problems(greedy_ents, k, usage_of(greedy_rec).get("completion_tokens"), exact_k)
        if not greedy_ents:
            return "unsupported", "no logprobs returned"
        p += greedy_top1_problems(greedy_ents)
        if temp_ents is None:
            p.append("T=1.0 logprobs not tested (--lp-greedy-only: such requests crash the engine in mode A)")
        else:
            p += sampled_consistency_problems(temp_ents)
        return ("partial" if p else "supported"), "; ".join(p) or \
            f"{len(greedy_ents)} greedy entries ok; {len(temp_ents)} T=1.0 entries consistent"

    def c_logprobs_chat(self):
        g = self.post(self.chat(Q_OPEN, max_tokens=32, logprobs=True, top_logprobs=5))
        te = None
        if not self.lp_greedy_only:
            te = chat_lp_entries(self.post(self.chat(Q_OPEN, temperature=1.0, max_tokens=32, logprobs=True,
                                                     top_logprobs=5)))
        ge = chat_lp_entries(g)
        sup, reason = self._lp_judge(g, ge, te, 5, True)
        joined = "".join(e["token"] or "" for e in ge)
        return sup, reason, {"tokens_join_equals_content": joined.strip() == content_of(g).strip()}

    def c_logprobs_completions(self):
        g = self.post(self.comp(P_COMP, max_tokens=16, logprobs=5), COMP)
        te = None
        if not self.lp_greedy_only:
            te = comp_lp_entries(C.choice0(self.post(self.comp(P_COMP, max_tokens=16, logprobs=5, temperature=1.0),
                                                     COMP)))
        ge = comp_lp_entries(C.choice0(g))
        sup, reason = self._lp_judge(g, ge, te, 5, False)
        joined = "".join(e["token"] or "" for e in ge)
        return sup, reason, {"tokens_join_equals_text": joined == content_of(g)}

    def _n_judge(self, rec):
        chs = (rec.get("response") or {}).get("choices") or []
        idx = sorted(c.get("index") for c in chs)
        texts = [content_of(rec, i) for i in range(len(chs))]
        p = []
        if len(chs) != 3 or idx != [0, 1, 2]:
            p.append(f"{len(chs)} choices with index {idx}")
        if len(set(texts)) < 2:
            p.append("all choices identical")
        if len(chs) == 1:
            return "unsupported", "; ".join(p), {"texts": texts}
        return ("partial" if p else "supported"), "; ".join(p) or f"{len(set(texts))}/3 distinct", {"texts": texts}

    def c_n_chat(self):
        return self._n_judge(self.post(self.chat(Q_POEM, temperature=1.0, max_tokens=48, n=3)))

    def c_n_completions(self):
        return self._n_judge(self.post(self.comp(P_AUTUMN, temperature=1.0, max_tokens=32, n=3), COMP))

    def _usage_case(self, body, ep):
        ref = self.post(body, ep)
        rec = self.stream(dict(body, stream_options={"include_usage": True}), ep)
        p = usage_stream_problems(rec["chunks"], usage_of(ref))
        if not rec.get("done"):
            p.append("stream did not end with [DONE]")
        sup = "supported" if not p else ("unsupported" if p == ["no chunk carries usage"] else "partial")
        u = rec["chunks"][-1].get("usage") if rec["chunks"] else None
        return sup, "; ".join(p) or f"last chunk usage {u} == non-stream", {}

    def c_stream_usage_chat(self):
        return self._usage_case(self.chat(Q_CAP, max_tokens=32), CHAT)

    def c_stream_usage_completions(self):
        return self._usage_case(self.comp(P_COMP, max_tokens=16), COMP)

    def _penalty(self, name):
        base = content_of(self.post(self.chat(Q_APPLE, max_tokens=160), key="apple_base"))
        pen = content_of(self.post(self.chat(Q_APPLE, max_tokens=160, **{name: 1.5})))
        nb, npn = count_word(base), count_word(pen)
        info = {"apple_count_base": nb, "apple_count_penalized": npn}
        if nb < 6:
            return "unverified", f"baseline repeated apple only {nb} times", info
        if pen == base:
            seeded = self._seeded_penalty_check(name)
            info["seeded_check"] = seeded
            if seeded["applied"]:
                return "supported", (f"greedy apple x{nb} unchanged with {name}=1.5 (a presence penalty is "
                                     f"subtracted once per distinct output token, too small to break the repeat "
                                     f"instruction); seeded T=1.0 check: same seed without penalty identical, "
                                     f"with {name}=2.0 differs -> penalty applied"), info
            return "unsupported", (f"output identical to no-penalty output (silently ignored; apple x{nb}); "
                                   f"seeded check: {seeded['reason']}"), info
        if npn <= nb / 2:
            return "supported", f"apple {nb} -> {npn}", info
        return "partial", f"output changed but apple {nb} -> {npn} (> half)", info

    def _seeded_penalty_check(self, name):
        """Same seed, T=1.0: no penalty twice (must be identical) vs penalty 2.0 (differs if it is applied)."""
        body = lambda **kw: self.chat(Q_POEM, temperature=1.0, max_tokens=96, seed=4321, **kw)  # noqa: E731
        a, b = (content_of(self.post(body())) for _ in range(2))
        c = content_of(self.post(body(**{name: 2.0})))
        if a != b:
            return {"applied": False, "reason": "same seed without penalty differs (seed not honoured, cannot tell)"}
        if c == a:
            return {"applied": False, "reason": f"same seed with {name}=2.0 identical to no penalty"}
        return {"applied": True, "reason": f"same seed with {name}=2.0 differs from no penalty"}

    def c_presence_penalty(self):
        return self._penalty("presence_penalty")

    def c_frequency_penalty(self):
        return self._penalty("frequency_penalty")

    def c_json_object(self):
        txt = content_of(self.post(self.chat(Q_JSON, max_tokens=256, response_format={"type": "json_object"})))
        try:
            obj = json.loads(txt)
        except ValueError:
            return "unsupported", f"content is not JSON: {txt[:200]!r}", {}
        if not isinstance(obj, dict):
            return "partial", f"JSON but not an object: {txt[:200]!r}", {}
        return "supported", f"JSON object keys {list(obj)[:6]}", {}

    def c_json_schema(self):
        rf = {"type": "json_schema", "json_schema": {"name": "city", "schema": CITY_SCHEMA, "strict": True}}
        txt = content_of(self.post(self.chat(Q_SCHEMA, max_tokens=256, response_format=rf)))
        try:
            obj = json.loads(txt)
        except ValueError:
            return "unsupported", f"content is not JSON: {txt[:200]!r}", {}
        p = schema_problems(obj)
        return ("partial" if p else "supported"), "; ".join(p) or \
            f"schema valid (jsonschema={'yes' if jsonschema else 'no'}): {txt[:160]!r}", {}

    def c_generation_config(self):
        want_auto = self.gencfg == "auto"
        p, info = [], {"gencfg": self.gencfg, "serve_log": self.serve_log}
        lc = gencfg_log_check(self.serve_log)
        if lc is None:
            p.append(f"serve log {self.serve_log} not readable (log check unverified)")
        else:
            found, start, hit = lc
            info.update(log_override_line_found=found, log_serve_start_line=start, log_hit_line=hit)
            if found != want_auto:
                p.append(f"log line '{GENCFG_LOG_LINE}' found={found} (want {want_auto}) after line {start}")
        # No-max_tokens check goes through chat: the completions API turns a missing or null max_tokens
        # into 16 (OpenAI legacy default; vllm/entrypoints/openai/completion/protocol.py:60 and the
        # normalize_null_max_tokens validator at :351-354), so it never reaches the generation config there.
        rec = self.post(self.chat(Q_LONG_STORY, ignore_eos=True), CHAT, timeout=max(self.t, 900))
        ct, fr = usage_of(rec).get("completion_tokens"), C.choice0(rec).get("finish_reason")
        info.update(no_max_tokens_completion_tokens=ct, no_max_tokens_finish=fr)
        if want_auto and ct != GENCFG_MAX_NEW:
            p.append(f"auto: completion_tokens={ct} (want {GENCFG_MAX_NEW})")
        if not want_auto and not (ct or 0) > GENCFG_MAX_NEW:
            p.append(f"vllm: completion_tokens={ct} (want > {GENCFG_MAX_NEW})")
        chat = self.post(self.chat(Q_INTRO))
        cfr = C.choice0(chat).get("finish_reason")
        info["chat_finish_reason"] = cfr
        if cfr != "stop" or not content_of(chat).strip():
            p.append(f"chat without max_tokens: finish_reason={cfr!r}, content empty={not content_of(chat).strip()}")
        sup = "supported" if not p else ("unverified" if lc is None and len(p) == 1 else "partial")
        return sup, "; ".join(p) or f"{self.gencfg}: log check ok, ignore_eos -> {ct} tokens, chat finish stop", info


def main(argv=None):
    ap = C.base_argparser(__doc__)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--gencfg", choices=["vllm", "auto"], default=None,
                    help="server's --generation-config (default: A -> vllm, B -> auto)")
    ap.add_argument("--serve-log", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "logs/serve.log"))
    ap.add_argument("--lp-greedy-only", action="store_true",
                    help="skip logprobs requests with temperature > 0 (they crash the engine in mode A)")
    args = ap.parse_args(argv)
    gencfg = args.gencfg or ("vllm" if args.mode == "A" else "auto")
    cases = QUICK if args.quick else ALL
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
        bad = [c for c in cases if c not in ALL]
        if bad:
            print(f"unknown cases {bad}; valid {ALL}", file=sys.stderr)
            return 2
    C.require_server()
    suite = C.Suite("params", args.out, args.tag, meta={"mode": args.mode, "gencfg": gencfg, "cases_requested": cases,
                                                        "quick": args.quick, "jsonschema": jsonschema is not None,
                                                        "lp_greedy_only": args.lp_greedy_only})
    r = Runner(suite, args.mode, args.timeout, gencfg, args.serve_log, args.lp_greedy_only)
    for c in cases:
        r.run(c)
        if not engine_alive():
            suite.meta["engine_dead_after"] = c
            suite.meta["not_run"] = cases[cases.index(c) + 1:]
            suite.write()
            print(f"params[{args.mode}]: engine not healthy after case {c}; stopping (not run: "
                  f"{suite.meta['not_run']})", file=sys.stderr)
            return 2
    not_ok = [c["id"] for c in suite.cases if c["core"] and c["support"] != "supported"]
    suite.meta["core_not_supported"] = not_ok
    suite.write()
    print(f"params[{args.mode}]: {suite.counts()}; core not supported: {not_ok or 'none'}")
    return 1 if not_ok else 0


if __name__ == "__main__":
    sys.exit(main())
