#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the API tests (stdlib only).

- post(path, body)           -> {"endpoint","request","http_status","elapsed_s","response"}
- post_stream(path, body)    -> {"endpoint","request","http_status","elapsed_s","chunks","done","error"}
- parse_sse(lines)           -> (list of decoded data payloads, saw_done)  (SSE: 'data:' lines, blank line ends an
                                event, consecutive data lines are joined with '\\n', '[DONE]' ends the stream)
- Suite                      collects per-case records (request, response / chunks, verdict, reason) and writes
                             <out>.json and <out>.md
Verdicts: pass / partial / fail / info. Exit codes (Suite.exit_code): 0 no fail, 1 some fail, 2 server unreachable.
Environment: MIMO_API_BASE (default http://127.0.0.1:$PORT, PORT default 8000), MIMO_API_MODEL (default mimo-v2.6-flash).
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("MIMO_API_MODEL", "mimo-v2.6-flash")
DEFAULT_TIMEOUT = 600
VERDICTS = ("pass", "partial", "fail", "info")


def base_url():
    return os.environ.get("MIMO_API_BASE") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"


def _request(path, body, base=None):
    return urllib.request.Request((base or base_url()) + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})


def post(path, body, timeout=DEFAULT_TIMEOUT, base=None):
    t = time.time()
    try:
        with urllib.request.urlopen(_request(path, body, base), timeout=timeout) as r:
            status, resp = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, resp = e.code, {"error": e.read().decode(errors="replace")}
    except Exception as e:  # noqa: BLE001
        status, resp = -1, {"error": repr(e)}
    return {"endpoint": path, "request": body, "http_status": status, "elapsed_s": round(time.time() - t, 2),
            "response": resp}


def parse_sse(lines):
    """Decode SSE lines (str or bytes, with or without trailing newline). Returns (payloads, saw_done); each payload
    is the parsed JSON object, or {"_raw": text} when it is not JSON. Comment lines (':') and other fields are
    ignored."""
    payloads, buf, done = [], [], False

    def flush():
        nonlocal done
        if not buf:
            return
        data = "\n".join(buf)
        buf.clear()
        if data.strip() == "[DONE]":
            done = True
            return
        try:
            payloads.append(json.loads(data))
        except ValueError:
            payloads.append({"_raw": data})

    for ln in lines:
        if isinstance(ln, bytes):
            ln = ln.decode("utf-8", errors="replace")
        ln = ln.rstrip("\r\n")
        if ln == "":
            flush()
            if done:
                break
            continue
        if ln.startswith(":"):
            continue
        if ln.startswith("data:"):
            v = ln[5:]
            buf.append(v[1:] if v.startswith(" ") else v)
    flush()
    return payloads, done


def post_stream(path, body, timeout=DEFAULT_TIMEOUT, base=None):
    body = dict(body, stream=True)
    t = time.time()
    rec = {"endpoint": path, "request": body, "http_status": None, "chunks": [], "done": False, "error": None}
    try:
        with urllib.request.urlopen(_request(path, body, base), timeout=timeout) as r:
            rec["http_status"] = r.status
            rec["chunks"], rec["done"] = parse_sse(iter(r.readline, b""))
    except urllib.error.HTTPError as e:
        rec["http_status"], rec["error"] = e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        rec["http_status"] = rec["http_status"] or -1
        rec["error"] = repr(e)
    rec["elapsed_s"] = round(time.time() - t, 2)
    return rec


def server_ready(base=None, timeout=10):
    try:
        with urllib.request.urlopen((base or base_url()) + "/v1/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def chat_body(messages, **kw):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return dict({"model": MODEL, "messages": messages, "temperature": 0}, **kw)


def choice0(rec):
    try:
        return rec["response"]["choices"][0]
    except (KeyError, IndexError, TypeError):
        return {}


def message(rec):
    return choice0(rec).get("message") or {}


def reasoning_of(msg):
    """vLLM 0.24 names the field 'reasoning' (docs/design-notes.md#reasoning-field); fall back to 'reasoning_content' if only that exists."""
    if msg.get("reasoning") is not None:
        return msg.get("reasoning")
    return msg.get("reasoning_content")


def stream_deltas(chunks, index=0):
    """Yield (delta, finish_reason) for choice `index` in order."""
    for c in chunks:
        for ch in c.get("choices") or []:
            if ch.get("index", 0) == index:
                yield ch.get("delta") or {}, ch.get("finish_reason")


def stream_concat(chunks, field, index=0):
    """Concatenate delta[field] over the stream (for 'reasoning' also accepts 'reasoning_content')."""
    out = []
    for d, _ in stream_deltas(chunks, index):
        v = d.get(field)
        if v is None and field == "reasoning":
            v = d.get("reasoning_content")
        if v:
            out.append(v)
    return "".join(out)


def stream_finish(chunks, index=0):
    fr = None
    for _, f in stream_deltas(chunks, index):
        if f is not None:
            fr = f
    return fr


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Suite:
    def __init__(self, name, out, tag=None, meta=None):
        self.name, self.out, self.tag = name, out, tag
        self.meta = dict(meta or {}, suite=name, tag=tag, started_utc=now_utc(), base_url=base_url(), model=MODEL)
        self.cases = []

    def add(self, case_id, desc, verdict, reason, records=None, **extra):
        assert verdict in VERDICTS, verdict
        c = {"id": case_id, "desc": desc, "verdict": verdict, "reason": reason, "records": records or []}
        c.update(extra)
        self.cases.append(c)
        print(f"[{self.name}] {case_id}: {verdict.upper()} - {reason}", flush=True)
        self.write()
        return c

    def counts(self):
        return {v: sum(1 for c in self.cases if c["verdict"] == v) for v in VERDICTS}

    def exit_code(self):
        return 1 if any(c["verdict"] == "fail" for c in self.cases) else 0

    def write(self):
        if not self.out:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.out)) or ".", exist_ok=True)
        doc = dict(self.meta, finished_utc=now_utc(), counts=self.counts(), cases=self.cases)
        tmp = self.out + ".json.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.out + ".json")
        lines = [f"# {self.name} ({self.tag or '-'})", "",
                 f"- base_url: {self.meta['base_url']}  model: {MODEL}",
                 f"- started: {self.meta['started_utc']}  finished: {doc['finished_utc']}",
                 "- counts: " + ", ".join(f"{k} {v}" for k, v in doc["counts"].items()), "",
                 "| case | verdict | desc | reason |", "|---|---|---|---|"]
        for c in self.cases:
            lines.append(f"| {c['id']} | {c['verdict']} | {_md(c['desc'])} | {_md(c['reason'])} |")
        with open(self.out + ".md", "w") as f:
            f.write("\n".join(lines) + "\n")


def _md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")[:400]


def base_argparser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--out", default=None, help="output path prefix (writes <out>.json and <out>.md)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default=None, help="comma-separated case ids to run (default: all)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return ap


def require_server():
    if not server_ready():
        print(f"server not reachable at {base_url()}/v1/models", file=sys.stderr)
        sys.exit(2)
