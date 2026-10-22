# SPDX-License-Identifier: Apache-2.0
"""scripts/api_tests common.py (SSE parsing, Suite), smoke_gate.py, run_suite.sh argument errors,
and test_reasoning.py end to end against a fake OpenAI-style server (stdlib only, no model)."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "scripts", "api_tests")
sys.path.insert(0, API)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_api_{name}", os.path.join(API, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("common")
G = _load("smoke_gate")
R = _load("test_reasoning")


# ---------------------------------------------------------------- SSE parsing
def test_parse_sse_basic_and_done():
    lines = [b'data: {"a": 1}\n', b"\n", b": keep-alive comment\n", b"\n", b'data: {"a": 2}\n', b"\n",
             b"data: [DONE]\n", b"\n", b'data: {"after": "done"}\n', b"\n"]
    payloads, done = C.parse_sse(lines)
    assert payloads == [{"a": 1}, {"a": 2}]
    assert done is True


def test_parse_sse_multiline_data_and_crlf():
    lines = ['data: {"x":\r\n', 'data:  "y"}\r\n', "\r\n", "event: ping\n", "data: not-json\n", "\n"]
    payloads, done = C.parse_sse(lines)
    assert payloads == [{"x": "y"}, {"_raw": "not-json"}]
    assert done is False


def test_parse_sse_last_event_without_blank_line():
    payloads, done = C.parse_sse(['data: {"k": 3}'])
    assert payloads == [{"k": 3}] and not done


def _chunks(parts):
    out = []
    for fld, txt, fin in parts:
        delta = {fld: txt} if fld else {}
        out.append({"choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
    return out


def test_stream_concat_and_order():
    ch = _chunks([("role", "assistant", None), ("reasoning", "think ", None), ("reasoning_content", "more", None),
                  ("content", "391", None), ("content", "!", None), (None, None, "stop")])
    assert C.stream_concat(ch, "reasoning") == "think more"
    assert C.stream_concat(ch, "content") == "391!"
    assert C.stream_finish(ch) == "stop"
    assert R.stream_order_ok(ch)
    bad = _chunks([("content", "a", None), ("reasoning", "late", None)])
    assert not R.stream_order_ok(bad)


def test_suite_writes_json_md_and_exit_code(tmp_path):
    s = C.Suite("x", str(tmp_path / "res" / "x_t"), "t")
    s.add("A", "desc a", "pass", "ok")
    assert s.exit_code() == 0
    s.add("B", "desc|b", "fail", "bad\nline")
    assert s.exit_code() == 1
    doc = json.load(open(tmp_path / "res" / "x_t.json"))
    assert doc["counts"]["fail"] == 1 and [c["id"] for c in doc["cases"]] == ["A", "B"]
    md = open(tmp_path / "res" / "x_t.md").read()
    assert "| B | fail | desc\\|b | bad line |" in md


def test_judges():
    assert R.judge_on("let me think", "答案是 391。", "stop") == []
    assert R.judge_on("", "391", "stop") == ["reasoning empty"]
    assert any("literal" in p for p in R.judge_on("x</think>", "391", "stop"))
    assert any("finish_reason" in p for p in R.judge_on("x", "391", "length"))
    assert R.judge_off(None, "391", "stop") == []
    assert any("reasoning not empty" in p for p in R.judge_off("x", "391", "stop"))


# ---------------------------------------------------------------- smoke_gate
def _smoke_entry(endpoint, q, expect, answer, status=200):
    if endpoint == "/v1/completions":
        resp = {"choices": [{"text": answer}]}
    else:
        resp = {"choices": [{"message": {"content": answer}}]}
    return {"endpoint": endpoint, "question": q, "expect": expect, "http_status": status, "response": resp}


def _write_smoke(d, ans391=("17×23=391", "391"), ans20=("是 20", "20"), status=200):
    single = [_smoke_entry("/v1/completions", "中国", None, "北京", status),
              _smoke_entry("/v1/completions", "17×23=?", "391", ans391[0]),
              _smoke_entry("/v1/chat/completions", "一半", "20", ans20[0])]
    conc = [_smoke_entry("/v1/chat/completions", "长城", None, "长城很长"),
            _smoke_entry("/v1/chat/completions", "17×23 等于多少？", "391", ans391[1]),
            _smoke_entry("/v1/chat/completions", "一半", "20", ans20[1])]
    os.makedirs(d, exist_ok=True)
    json.dump(single, open(os.path.join(d, "single.json"), "w"))
    json.dump(conc, open(os.path.join(d, "concurrent4.json"), "w"))


def test_smoke_gate_pass(tmp_path):
    d = str(tmp_path / "smoke")
    _write_smoke(d)
    assert G.main([d, "--out", str(tmp_path / "gate")]) == 0
    assert json.load(open(tmp_path / "gate.json"))["pass"] is True


@pytest.mark.parametrize("kw", [{"ans391": ("17×23=390", "391")}, {"ans20": ("是 20", "十")}, {"status": 500}])
def test_smoke_gate_fail(tmp_path, kw):
    d = str(tmp_path / "smoke")
    _write_smoke(d, **kw)
    assert G.main([d]) == 1


def test_smoke_gate_missing_expected_entries(tmp_path):
    d = str(tmp_path / "smoke")
    os.makedirs(d)
    json.dump([_smoke_entry("/v1/completions", "17×23=?", "391", "391")], open(os.path.join(d, "single.json"), "w"))
    json.dump([], open(os.path.join(d, "concurrent4.json"), "w"))
    assert G.main([d]) == 1


def test_smoke_gate_missing_files(tmp_path):
    assert G.main([str(tmp_path / "nope")]) == 2


# ---------------------------------------------------------------- run_suite.sh argument errors
RUN_SUITE = os.path.join(API, "run_suite.sh")


@pytest.mark.parametrize("args", [[], ["bogus"], ["reasoning", "--tag"], ["reasoning", "--tag", ""],
                                  ["params", "--mode", "C"], ["reasoning", "--tag", "../x"]])
def test_run_suite_bad_args_exit_2(args, tmp_path):
    env = dict(os.environ, RUN_SUITE_NO_VENV="1", MIMO_RESULTS=str(tmp_path))
    p = subprocess.run(["bash", RUN_SUITE] + args, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2, p.stderr


def test_run_suite_reasoning_server_down_exit_2(tmp_path):
    env = dict(os.environ, RUN_SUITE_NO_VENV="1", MIMO_RESULTS=str(tmp_path), MIMO_API_BASE="http://127.0.0.1:9",
               PORT="9")
    p = subprocess.run(["bash", RUN_SUITE, "reasoning", "--tag", "t", "--quick"], env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr


# ---------------------------------------------------------------- test_reasoning.py against a fake server
REASON, ANSWER = "17 乘以 23：17×20=340，17×3=51，合计 391。", "17×23 = **391**。"


def _fake_reply(body):
    msgs = body.get("messages") or []
    off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    last = msgs[-1]["content"] if msgs else ""
    if "prompt" in body:
        return {"text": " 391", "finish_reason": "stop"}
    if body.get("max_tokens") == 32 and not off:
        return {"reasoning": "17 乘以 23", "content": None, "finish_reason": "length"}
    if last == R.FOLLOW_UP:
        return {"reasoning": "391×2", "content": "782", "finish_reason": "stop"}
    return {"reasoning": None if off else REASON, "content": ANSWER, "finish_reason": "stop"}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._json({"data": [{"id": C.MODEL}]})

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rep = _fake_reply(body)
        if "prompt" in body:
            return self._json({"choices": [{"index": 0, "text": rep["text"], "finish_reason": "stop"}]})
        if not body.get("stream"):
            return self._json({"choices": [{"index": 0, "finish_reason": rep["finish_reason"], "message": {
                "role": "assistant", "content": rep["content"], "reasoning": rep["reasoning"]}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        parts = [{"role": "assistant"}]
        for fld in ("reasoning", "content"):
            t = rep[fld] or ""
            parts += [{fld: t[i:i + 5]} for i in range(0, len(t), 5)]
        for i, d in enumerate(parts):
            fin = rep["finish_reason"] if i == len(parts) - 1 else None
            ev = {"choices": [{"index": 0, "delta": d, "finish_reason": fin}]}
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def fake_server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("MIMO_API_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


def test_reasoning_all_cases_pass_on_fake_server(fake_server, tmp_path):
    out = str(tmp_path / "reasoning_fake")
    assert R.main(["--out", out, "--tag", "fake"]) == 0
    doc = json.load(open(out + ".json"))
    assert [c["id"] for c in doc["cases"]] == R.ALL
    assert all(c["verdict"] == "pass" for c in doc["cases"]), [(c["id"], c["reason"]) for c in doc["cases"]]


def test_reasoning_quick_and_unknown_case(fake_server, tmp_path):
    out = str(tmp_path / "q")
    assert R.main(["--out", out, "--quick"]) == 0
    assert [c["id"] for c in json.load(open(out + ".json"))["cases"]] == ["R1", "R2"]
    assert R.main(["--out", out, "--cases", "R9"]) == 2


# ---------------------------------------------------------------- test_tools.py helpers and fake server (tool-calling suite)
T = _load("test_tools")


def test_tools_validate_and_parse():
    assert T.validate_args("get_weather", {"city": "北京", "unit": "celsius"}) == []
    assert T.validate_args("get_weather", {"unit": "kelvin"})
    assert T.validate_args("nope", {}) == ["unknown tool 'nope'"]
    calls = T.parse_calls([{"id": "a", "function": {"name": "get_time", "arguments": '{"timezone": "Asia/Tokyo"}'}},
                           {"id": "b", "function": {"name": "get_time", "arguments": "{bad"}}])
    assert calls[0][4] == [] and calls[1][4]


def test_tools_stream_accumulate_detects_duplicates():
    ch = [{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "get_time",
                                                                                                      "arguments": ""}}]}}]},
          {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"timezone": '}}]}}]},
          {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"arguments": '"UTC"}'}}]}}]}]
    calls, p = T.accumulate_stream_calls(ch)
    assert calls[0]["arguments"] == '{"timezone": "UTC"}'
    assert p == ["index 0: id sent twice"]


def _tool_call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args,
                                                                                                    ensure_ascii=False)}}


def _fake_tools_reply(body):
    msgs = body["messages"]
    last = msgs[-1]
    tc = body.get("tool_choice", "auto")
    if last["role"] == "tool":
        return {"content": "北京现在晴，23 度。", "tool_calls": [], "finish_reason": "stop"}
    if tc == "none":
        return {"content": "我无法实时查询天气。", "tool_calls": [], "finish_reason": "stop"}
    if isinstance(tc, dict):
        return {"content": None, "tool_calls": [_tool_call(0, "get_time", {"timezone": "Asia/Tokyo"})],
                "finish_reason": "tool_calls"}
    q = last["content"]
    if tc == "required":
        return {"content": None, "tool_calls": [_tool_call(0, "calculator", {"expression": "1+1"})],
                "finish_reason": "tool_calls"}
    if "上海" in q or "Shanghai" in q:
        return {"content": None, "finish_reason": "tool_calls", "tool_calls": [
            _tool_call(0, "get_weather", {"city": "北京"}), _tool_call(1, "get_weather", {"city": "上海"}),
            _tool_call(2, "get_time", {"timezone": "Asia/Tokyo"})]}
    return {"content": None, "tool_calls": [_tool_call(0, "get_weather", {"city": "北京", "unit": "celsius"})],
            "finish_reason": "tool_calls", "reasoning": None if body.get("chat_template_kwargs") else "需要查天气"}


class _ToolsHandler(_Handler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rep = _fake_tools_reply(body)
        if not body.get("stream"):
            return self._json({"choices": [{"index": 0, "finish_reason": rep["finish_reason"], "message": {
                "role": "assistant", "content": rep["content"], "reasoning": rep.get("reasoning"),
                "tool_calls": rep["tool_calls"]}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        deltas = [{"role": "assistant"}]
        for i, tc in enumerate(rep["tool_calls"]):
            a = tc["function"]["arguments"]
            deltas.append({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                           "function": {"name": tc["function"]["name"], "arguments": ""}}]})
            deltas += [{"tool_calls": [{"index": i, "function": {"arguments": a[k:k + 4]}}]} for k in range(0, len(a), 4)]
        if rep["content"]:
            deltas.append({"content": rep["content"]})
        for k, d in enumerate(deltas):
            fin = rep["finish_reason"] if k == len(deltas) - 1 else None
            self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": d, "finish_reason": fin}]})
                             .encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def fake_tools_server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ToolsHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MIMO_API_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


def test_tools_all_cases_on_fake_server(fake_tools_server, tmp_path):
    out = str(tmp_path / "tools_fake")
    assert T.main(["--out", out, "--skip-offline"]) == 0
    doc = json.load(open(out + ".json"))
    assert [c["id"] for c in doc["cases"]] == T.ALL
    bad = [(c["id"], c["verdict"], c["reason"]) for c in doc["cases"] if c["verdict"] not in ("pass", "info")]
    assert not bad, bad


def test_tools_quick_on_fake_server(fake_tools_server, tmp_path):
    out = str(tmp_path / "tq")
    assert T.main(["--out", out, "--quick", "--skip-offline"]) == 0
    doc = json.load(open(out + ".json"))
    assert [c["id"] for c in doc["cases"]] == ["T1", "T4"]
    assert doc["cases"][1]["desc"].endswith("(T1 only)")


# ---------------------------------------------------------------- test_params.py helpers + fake server, param_matrix (parameter suite)
P = _load("test_params")
M = _load("param_matrix")


def _ent(tok, lp, tops):
    return {"token": tok, "logprob": lp, "top_logprobs": [{"token": t, "logprob": v} for t, v in tops]}


def test_params_nucleus_check():
    import math
    ln = math.log
    inside = _ent("b", ln(0.3), [("a", ln(0.6)), ("b", ln(0.3)), ("c", ln(0.08))])     # mass before b = 0.6
    outside = _ent("c", ln(0.08), [("a", ln(0.6)), ("b", ln(0.36)), ("c", ln(0.03))])   # mass before c = 0.96
    uncovered = _ent("z", ln(0.1), [("a", ln(0.5)), ("z", ln(0.1))])                     # mass 0.6 < 0.95
    r = P.nucleus_check([inside, outside, uncovered])
    assert (r["steps"], r["covered"], r["out_of_nucleus"]) == (3, 2, 1)
    assert r["examples"][0]["step"] == 1
    missing = _ent("q", ln(0.01), [("a", ln(0.97)), ("b", ln(0.02))])
    assert P.nucleus_check([missing])["out_of_nucleus"] == 1


def test_params_topk_and_logprob_checks():
    tops = [("a", -0.1), ("b", -1.0), ("c", -2.0)]
    assert P.topk_check([_ent("b", -1.0, tops)], 2) == []
    assert len(P.topk_check([_ent("c", -2.0, tops)], 2)) == 1
    assert P.topk_check([_ent("x", -1.0, tops)], 2) == []          # tie with the k-th value is allowed
    assert P.greedy_top1_problems([_ent("a", -0.1, tops)]) == []
    assert P.greedy_top1_problems([_ent("b", -1.0, tops)])
    assert P.sampled_consistency_problems([_ent("b", -1.0, tops)]) == []
    assert P.sampled_consistency_problems([_ent("b", -0.5, tops)])
    assert P.logprob_shape_problems([_ent("a", -0.1, tops)], 3, 1) == []
    assert P.logprob_shape_problems([_ent("a", -0.1, tops)], 5, 1)
    assert P.logprob_shape_problems([_ent("a", -0.1, tops)], 3, 2)
    assert P.logprob_shape_problems([_ent("a", 0.5, [("a", 0.5)])], 1, 1)


def test_params_comp_entries_usage_and_misc(tmp_path):
    ch = {"logprobs": {"tokens": ["a", "b"], "token_logprobs": [-0.1, -0.2],
                       "top_logprobs": [{"x": -3.0, "a": -0.1}, {"b": -0.2}]}}
    e = P.comp_lp_entries(ch)
    assert [x["token"] for x in e] == ["a", "b"] and e[0]["top_logprobs"][0] == {"token": "a", "logprob": -0.1}
    assert P.count_word("Apple apple APPLE banana") == 3
    chunks = [{"choices": [{"index": 0, "delta": {"content": "x"}}], "usage": None},
              {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}]
    assert P.usage_stream_problems(chunks, {"prompt_tokens": 5, "completion_tokens": 2}) == []
    assert P.usage_stream_problems(chunks, {"prompt_tokens": 5, "completion_tokens": 3})
    assert P.usage_stream_problems(chunks[:1], {}) == ["no chunk carries usage"]
    assert P.schema_problems({"name": "北京", "population": 1, "landmarks": ["a", "b"]}) == []
    assert P.schema_problems({"name": "北京", "population": "1", "landmarks": ["a"]})
    log = tmp_path / "serve.log"
    log.write_text(f"{P.SERVE_START_MARK} old\nWARNING {P.GENCFG_LOG_LINE} by x\n{P.SERVE_START_MARK} new\nok\n")
    assert P.gencfg_log_check(str(log)) == (False, 3, None)
    log.write_text(f"{P.SERVE_START_MARK} a\nWARNING {P.GENCFG_LOG_LINE}\n")
    assert P.gencfg_log_check(str(log)) == (True, 1, 2)
    assert P.gencfg_log_check(str(tmp_path / "missing.log")) is None


_FAKE_PRESENCE_GREEDY_NOOP = False  # True: presence_penalty leaves the greedy apple output unchanged (like MiMo)


def _fake_params_reply(path, body):
    import random
    chat = path.endswith("/chat/completions")
    q = body["messages"][-1]["content"] if chat else body["prompt"]
    temp, seed = body.get("temperature", 1.0), body.get("seed")
    rf = (body.get("response_format") or {}).get("type")
    if rf == "json_schema":
        text = json.dumps({"name": "北京", "population": 21893095, "landmarks": ["故宫", "长城"]}, ensure_ascii=False)
    elif rf == "json_object":
        text = '{"name": "北京", "population": 21893095}'
    elif "apple" in q:
        pen = body.get("frequency_penalty") or (body.get("presence_penalty") and not _FAKE_PRESENCE_GREEDY_NOOP)
        text = "apple banana cherry" if pen else " ".join(["apple"] * 40)
    elif "数到 20" in q:
        text = ", ".join(str(i) for i in range(1, 21))
    elif "首都" in q:
        text = "北京是中国的首都。"
    else:
        text = "Autumn leaves fall."
    n = body.get("n", 1)
    outs = []
    for i in range(n):
        t = text
        if temp and body.get("top_k") != 1 and (body.get("top_p") or 1) > 0.01 and not rf and "数到" not in q \
                and "apple" not in q:
            rnd = random.Random(seed) if seed is not None else random.Random()
            t += str(rnd.randint(0, 10 ** 9)) + ("P" if body.get("presence_penalty") else "")
        toks, fin, sr = list(t), "stop", None
        for s in body.get("stop") or []:
            if s in "".join(toks):
                toks, sr = list("".join(toks)[:"".join(toks).index(s)]), s
        for j, c in enumerate(toks):
            if ord(c) in (body.get("stop_token_ids") or []):
                toks, sr = toks[:j], ord(c)
                break
        mt = body.get("max_tokens")
        if mt is not None and len(toks) > mt:
            toks, fin = toks[:mt], "length"
        ct = len(toks)
        if body.get("ignore_eos") and mt is None:
            ct, fin = 4000, "length"
        outs.append((toks, fin, sr, ct))
    return outs


class _ParamsHandler(_Handler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        chat = self.path.endswith("/chat/completions")
        outs = _fake_params_reply(self.path, body)
        k = body.get("top_logprobs") or 0 if chat else (body.get("logprobs") or 0)
        ids = body.get("return_tokens_as_token_ids")
        choices, usage = [], {"prompt_tokens": 9, "completion_tokens": 0}
        for i, (toks, fin, sr, ct) in enumerate(outs):
            usage["completion_tokens"] += ct
            text = "".join(toks)
            tk = [f"token_id:{ord(c)}" if ids else c for c in toks]
            ch = {"index": i, "finish_reason": fin, "stop_reason": sr}
            if chat:
                ch["message"] = {"role": "assistant", "content": text}
                if body.get("logprobs"):
                    ch["logprobs"] = {"content": [{"token": t, "logprob": -0.01, "top_logprobs": [
                        {"token": t, "logprob": -0.01}] + [{"token": f"alt{j}", "logprob": -5.0 - j}
                                                           for j in range(k - 1)]} for t in tk]}
            else:
                ch["text"] = text
                if body.get("logprobs"):
                    ch["logprobs"] = {"tokens": tk, "token_logprobs": [-0.01] * len(tk), "top_logprobs": [
                        dict({t: -0.01}, **{f"alt{j}": -5.0 - j for j in range(k - 1)}) for t in tk]}
            choices.append(ch)
        if not body.get("stream"):
            return self._json({"choices": choices, "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        toks = outs[0][0]
        for j, c in enumerate(toks):
            d = {"index": 0, "finish_reason": outs[0][1] if j == len(toks) - 1 else None}
            d.update({"delta": {"content": c}} if chat else {"text": c})
            self.wfile.write(b"data: " + json.dumps({"choices": [d], "usage": None}).encode() + b"\n\n")
        if (body.get("stream_options") or {}).get("include_usage"):
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def fake_params_server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ParamsHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MIMO_API_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


def test_params_all_cases_on_fake_server(fake_params_server, tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(f"{P.SERVE_START_MARK} x\nINFO ready\n")
    out = str(tmp_path / "params_A_fake")
    assert P.main(["--mode", "A", "--out", out, "--serve-log", str(log)]) == 0
    doc = json.load(open(out + ".json"))
    assert [c["id"] for c in doc["cases"]] == P.ALL
    assert doc["mode"] == "A" and doc["gencfg"] == "vllm" and doc["core_not_supported"] == []
    bad = [(c["id"], c["support"], c["reason"]) for c in doc["cases"] if c["support"] != "supported"]
    assert not bad, bad


def test_params_presence_penalty_seeded_fallback(fake_params_server, tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_FAKE_PRESENCE_GREEDY_NOOP", True)
    out = str(tmp_path / "pp")
    assert P.main(["--mode", "B", "--out", out, "--cases", "presence_penalty"]) == 0
    c = json.load(open(out + ".json"))["cases"][0]
    assert c["support"] == "supported" and "penalty applied" in c["reason"]
    assert c["seeded_check"]["applied"] is True


def test_params_quick_auto_and_bad_case(fake_params_server, tmp_path):
    out = str(tmp_path / "pq")
    assert P.main(["--mode", "B", "--out", out, "--quick", "--serve-log", str(tmp_path / "none.log")]) == 0
    doc = json.load(open(out + ".json"))
    assert [c["id"] for c in doc["cases"]] == P.QUICK and doc["gencfg"] == "auto"
    # auto expects the override log line and 2048 tokens; the fake server gives neither -> core case not supported
    assert P.main(["--mode", "B", "--out", out, "--cases", "generation_config",
                   "--serve-log", str(tmp_path / "none.log")]) == 1
    assert P.main(["--mode", "A", "--out", out, "--cases", "nope"]) == 2


def _pdoc(mode, supports):
    return {"tag": mode.lower(), "cases": [{"id": cid, "param": f"p_{cid}", "support": s, "reason": f"r{mode}{cid}"}
                                           for cid, s in supports]}


def test_param_matrix_overall_rules():
    assert M.overall({"A": "supported", "B": "supported"}) == ("supported", [])
    assert M.overall({"A": "supported", "B": None}) == ("supported", [])
    assert M.overall({"A": "unsupported", "B": "supported"}) == ("partial", ["B"])
    assert M.overall({"A": "partial", "B": "unsupported"}) == ("partial", [])
    assert M.overall({"A": "unsupported", "B": "unverified"}) == ("unsupported", [])
    assert M.overall({"A": "unverified", "B": None}) == ("unverified", [])


def test_param_matrix_merge_and_files(tmp_path):
    a = _pdoc("A", [("seed", "unsupported"), ("top_k", "supported")])
    a["cases"][0]["evidence"] = "sampler.py:56-164"
    b = _pdoc("B", [("top_k", "supported"), ("seed", "supported"), ("min_p", "supported")])
    rows = M.merge({"A": a, "B": b}, {"seed": ["neuron_model_runner.py:1986"]})
    assert [r["id"] for r in rows] == ["seed", "top_k", "min_p"]
    seed = rows[0]
    assert (seed["A"], seed["B"], seed["overall"], seed["only_in"]) == ("unsupported", "supported", "partial", ["B"])
    assert seed["evidence"] == ["A: rAseed", "A: sampler.py:56-164", "B: rBseed", "neuron_model_runner.py:1986"]
    assert rows[2]["A"] == "not_run" and rows[2]["overall"] == "supported"
    pa, pb, pn = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "n.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    pn.write_text(json.dumps({"seed": "x|y"}))
    out = str(tmp_path / "m" / "param_matrix")
    assert M.main(["--a", str(pa), "--b", str(pb), "--notes", str(pn), "--out", out]) == 0
    doc = json.load(open(out + ".json"))
    assert len(doc["rows"]) == 3 and doc["modes"]["A"]["tag"] == "a"
    md = open(out + ".md").read()
    assert "| p_seed | 不支持 | 支持 | 部分支持（只在模式 B 生效） |" in md and "x\\|y" in md
    assert M.main(["--a", str(pa), "--out", out]) == 0
    assert json.load(open(out + ".json"))["rows"][0]["B"] == "not_run"
    assert M.main(["--a", str(tmp_path / "missing.json"), "--out", out]) == 2


def test_params_lp_greedy_only(fake_params_server, tmp_path):
    out = str(tmp_path / "pg")
    assert P.main(["--mode", "A", "--out", out, "--lp-greedy-only", "--cases",
                   "temperature_top_p,top_k,logprobs_chat,logprobs_completions"]) == 1
    doc = json.load(open(out + ".json"))
    sup = {c["id"]: c["support"] for c in doc["cases"]}
    assert sup == {"temperature_top_p": "supported", "top_k": "supported", "logprobs_chat": "partial",
                   "logprobs_completions": "partial"}, [(c["id"], c["reason"]) for c in doc["cases"]]
    assert "skipped" in doc["cases"][0]["reason"]
    reqs = [r["request"] for c in doc["cases"] for r in c["records"]]
    assert not [r for r in reqs if r.get("logprobs") and r.get("temperature", 1.0) > 0]


class _DyingHandler(_ParamsHandler):
    posts = 0

    def do_GET(self):
        if self.path == "/health" and _DyingHandler.posts > 0:
            self.send_response(503)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        _DyingHandler.posts += 1
        super().do_POST()


def test_params_stops_when_engine_dies(monkeypatch, tmp_path):
    _DyingHandler.posts = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _DyingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MIMO_API_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        out = str(tmp_path / "pd")
        assert P.main(["--mode", "A", "--out", out, "--cases", "max_tokens,stop"]) == 2
        doc = json.load(open(out + ".json"))
        assert doc["engine_dead_after"] == "max_tokens" and doc["not_run"] == ["stop"]
        assert [c["id"] for c in doc["cases"]] == ["max_tokens"]
    finally:
        srv.shutdown()
