#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTFT and vision cost of a long video / large image on a running server.

For each item (file:// URL under the server's --allowed-local-media-path), thinking off, greedy, max_tokens 1
(so the request time ~ TTFT):
  cold    first request with the item (API-side media decode + HF processor + ViT encode + prefill)
  repeat  the same request again (encoder / processor caches may be reused; media decode still runs)
  text    a text-only request with about the same prompt_tokens (prefill only, the baseline)
vision_overhead_s = cold - text  (everything the item adds on top of prefilling the same number of tokens).

Files: $MIMO_MM_ASSETS/large/ (README Step 7 generates them) and the test video digits_1234.mp4.

  python3 scripts/api_tests/mm_long_timing.py --out $MIMO_WORK/results/mm_long_timing
Exit codes: 0 ok, 1 a request failed, 2 server not reachable.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FILLER = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
A = os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
ITEMS = [
    ("video_720p_3min", "video_url", os.path.join(A, "large/long_720p_180s.mp4")),
    ("image_2048x1365", "image_url", os.path.join(A, "large/photo_2048x1365.jpg")),
    ("video_digits", "video_url", os.path.join(A, "digits_1234.mp4")),
]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _req(content, timeout):
    r = C.post("/v1/chat/completions", C.chat_body(content, max_tokens=1, **OFF), timeout=timeout)
    usage = (r.get("response") or {}).get("usage") or {}
    return {"http_status": r["http_status"], "elapsed_s": r["elapsed_s"], "prompt_tokens": usage.get("prompt_tokens"),
            "error": (r.get("response") or {}).get("error")}


def _text_like(n_tokens, timeout):
    """Text-only request with about n_tokens prompt tokens: probe once, rescale, measure."""
    k = max(1, n_tokens // 16)
    probe = _req(FILLER * k, timeout)
    if probe["http_status"] != 200 or not probe["prompt_tokens"]:
        return probe
    k = max(1, round(k * n_tokens / probe["prompt_tokens"]))
    return _req("Count words. " + FILLER * k, timeout)


def main(argv=None) -> int:
    ap = C.base_argparser(__doc__.split("\n")[0])
    a = ap.parse_args(argv)
    C.require_server()
    res, ok = {"time": C.now_utc(), "base": C.base_url(), "items": {}}, True
    for name, kind, path in ITEMS:
        if a.cases and name not in a.cases.split(","):
            continue
        part = {"type": kind, kind: {"url": "file://" + os.path.realpath(path)}}   # resolved, like mm_qa.py
        content = [{"role": "user", "content": [part, {"type": "text", "text": "Describe it."}]}]
        cold = _req(content, a.timeout)
        rep = _req(content, a.timeout)
        text = _text_like(cold["prompt_tokens"] or 1000, a.timeout) if cold["http_status"] == 200 else None
        rec = {"path": path, "cold": cold, "repeat": rep, "text": text}
        if text and text["http_status"] == 200:
            rec["vision_overhead_s"] = round(cold["elapsed_s"] - text["elapsed_s"], 2)
        rec["ok"] = all(x and x["http_status"] == 200 for x in (cold, rep, text))
        ok &= rec["ok"]
        res["items"][name] = rec
        print(name, json.dumps(rec, ensure_ascii=False), flush=True)
    res["ok"] = ok
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out + ".json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
