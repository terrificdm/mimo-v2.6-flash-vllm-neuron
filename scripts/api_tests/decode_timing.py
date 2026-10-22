#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-token decode time at a long prompt (t_dec, used to extrapolate long-context timings).

Reuses saved needle prompts (needle.py *_prompts.json), sends each one alone (B=1) as a streamed chat request with
thinking off, temperature 0, ``ignore_eos`` and a fixed ``max_tokens``, and records TTFT, total time and
t_dec = (elapsed - ttft) / (completion_tokens - 1). Output has needle.py's case/record layout so
tools/longctx_extrapolate.py --needle can read it.

  run in the vLLM venv: python scripts/api_tests/decode_timing.py --prompts $R2/longctx/needle_ctx32k_prompts.json \
      --ids N2@30000@50 --max-tokens 128 --out $R2/longctx/tdec_ctx32k.json
Exit codes: 0 ok, 1 a request failed, 2 bad arguments / server not reachable.
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


def one(messages, max_tokens, timeout):
    body = C.chat_body(messages, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                       ignore_eos=True, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(C.base_url() + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    rec = {"http_status": None, "ttft_s": None, "usage": None, "error": None, "n_chunks": 0}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["http_status"] = r.status
            chunks, _ = C.parse_sse(iter(r.readline, b""))
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        rec["elapsed_s"] = round(time.time() - t0, 3)
        return rec
    rec["elapsed_s"] = round(time.time() - t0, 3)
    # parse_sse reads the whole stream, so the first-token time is not seen here; main() takes the TTFT
    # from a separate 1-token request on the same prompt.
    for c in chunks:
        if isinstance(c, dict) and c.get("usage"):
            rec["usage"] = c["usage"]
    rec["n_chunks"] = len(chunks)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ids", default="", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.max_tokens < 2:
        ap.error("--max-tokens must be >= 2")
    if not C.server_ready():
        print("decode_timing: server not reachable", file=sys.stderr)
        return 2
    doc = json.load(open(a.prompts))
    want = {x for x in a.ids.split(",") if x}
    prompts = [p for p in doc["prompts"] if not want or p["id"] in want]
    if not prompts:
        ap.error("no prompt matches --ids")
    cases, rc = [], 0
    for p in prompts:
        r1 = one(p["messages"], 1, a.timeout)      # prefill + 1 token = TTFT
        rn = one(p["messages"], a.max_tokens, a.timeout)
        n = (rn.get("usage") or {}).get("completion_tokens")
        ok = r1["http_status"] == 200 and rn["http_status"] == 200 and not r1["error"] and not rn["error"] and n
        t_dec = round((rn["elapsed_s"] - r1["elapsed_s"]) / (n - 1), 4) if ok and n > 1 else None
        rec = dict(rn, ttft_s=r1["elapsed_s"])   # longctx_extrapolate: (elapsed - ttft) / (n - 1)
        cases.append({"id": p["id"], "kind": "tdec", "prompt_tokens": (rn.get("usage") or {}).get("prompt_tokens"),
                      "ttft_s": r1["elapsed_s"], "t_dec_s": t_dec, "records": [rec]})
        print(f"decode_timing: {p['id']} prompt_tokens={cases[-1]['prompt_tokens']} ttft(1 tok)={r1['elapsed_s']}s "
              f"elapsed({n} tok)={rn['elapsed_s']}s t_dec={t_dec}s", flush=True)
        if not ok:
            rc = 1
    with open(a.out, "w") as f:
        json.dump({"utc": C.now_utc(), "prompts_file": a.prompts, "max_tokens": a.max_tokens, "cases": cases}, f,
                  indent=1, ensure_ascii=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
