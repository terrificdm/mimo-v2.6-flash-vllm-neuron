#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefix caching check (docs/limitations.md): same long prompt twice, then a follow-up turn on top of it.

For each saved needle prompt (needle.py *_prompts.json): request 1 (cold), request 2 (identical, should hit the
prefix cache), request 3 (multi-turn: prompt + request-1 answer + a new user question). All greedy, thinking off,
streamed; TTFT is the time to the first streamed chunk with content. Prints and saves the TTFTs, the
``prompt_tokens_details.cached_tokens`` usage field when the server reports it, and the answers.
``--compare <earlier out.json>`` (e.g. a run with prefix caching off) checks that the answers are identical.

  python scripts/api_tests/prefix_cache_test.py --prompts $R/longctx/needle_ctx32k_prompts.json --ids N2@30000@50 \
      --max-tokens 64 --out $R/prefix/pc_on.json [--compare $R/prefix/pc_off.json]
Exit codes: 0 ok, 1 request failed or answers differ, 2 bad arguments / server not reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FOLLOW_UP = "Repeat the special magic number you found, then write one short sentence about where it was."


def stream_one(messages, max_tokens, timeout):
    body = C.chat_body(messages, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                       chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(C.base_url() + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"http_status": None, "ttft_s": None, "elapsed_s": None, "usage": None, "text": "", "error": None}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["http_status"] = r.status
            for ln in iter(r.readline, b""):
                ln = ln.decode("utf-8", errors="replace").strip()
                if not ln.startswith("data:"):
                    continue
                data = ln[5:].strip()
                if data == "[DONE]":
                    break
                c = json.loads(data)
                if c.get("usage"):
                    rec["usage"] = c["usage"]
                for ch in c.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = round(time.time() - t0, 3)
                        rec["text"] += piece
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def cached_tokens(rec):
    return ((rec.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ids", default="", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--compare", default=None, help="earlier output; answers must be identical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not C.server_ready():
        print("prefix_cache_test: server not reachable", file=sys.stderr)
        return 2
    doc = json.load(open(a.prompts))
    want = {x for x in a.ids.split(",") if x}
    prompts = [p for p in doc["prompts"] if not want or p["id"] in want]
    if not prompts:
        ap.error("no prompt matches --ids")
    ref = {c["id"]: c for c in json.load(open(a.compare))["cases"]} if a.compare else {}
    cases, rc = [], 0
    for p in prompts:
        r1 = stream_one(p["messages"], a.max_tokens, a.timeout)
        r2 = stream_one(p["messages"], a.max_tokens, a.timeout)
        turn2 = list(p["messages"]) + [{"role": "assistant", "content": r1["text"]},
                                       {"role": "user", "content": FOLLOW_UP}]
        r3 = stream_one(turn2, a.max_tokens, a.timeout)
        recs = {"cold": r1, "repeat": r2, "follow_up": r3}
        ok = all(r["http_status"] == 200 and not r["error"] and r["ttft_s"] is not None for r in recs.values())
        same = r1["text"] == r2["text"]
        case = {"id": p["id"], "prompt_tokens": (r1.get("usage") or {}).get("prompt_tokens"),
                "ttft_s": {k: r["ttft_s"] for k, r in recs.items()},
                "cached_tokens": {k: cached_tokens(r) for k, r in recs.items()},
                "repeat_identical": same, "records": recs}
        if p["id"] in ref:
            case["matches_compare"] = {k: recs[k]["text"] == ref[p["id"]]["records"][k]["text"] for k in recs}
            same = same and all(case["matches_compare"].values())
        cases.append(case)
        print(f"prefix_cache_test: {p['id']} prompt_tokens={case['prompt_tokens']} ttft={case['ttft_s']} "
              f"cached={case['cached_tokens']} repeat_identical={case['repeat_identical']} "
              f"matches_compare={case.get('matches_compare')}", flush=True)
        if not ok or not same:
            rc = 1
    with open(a.out, "w") as f:
        json.dump({"utc": C.now_utc(), "prompts_file": a.prompts, "max_tokens": a.max_tokens, "compare": a.compare,
                   "cases": cases}, f, indent=1, ensure_ascii=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
