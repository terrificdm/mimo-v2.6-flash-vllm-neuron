#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Image request latency (docs/design-notes.md#vision-dp-group): vision-encoder cost per request on a running server.

Sends ``--n`` requests per image size, each with a fresh random-noise PNG (so no encoder / prefix cache can be
reused), ``max_tokens=1``, greedy, thinking off, and a text-only request with the same question as a baseline.
The difference (image - text) is mostly the vision tower plus the extra prompt tokens; compare two server
configs (e.g. MIMO_VIT_DP_GROUP=0 vs world) with the same arguments.

  python scripts/api_tests/mm_encode_timing.py --sizes 448,896 --n 5 --out $R/mm/T47_vit_timing_default.json
Exit codes: 0 ok, 1 a request failed, 2 server not reachable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

QUESTION = "What is in the image? Answer in one word."


def png_data_url(size: int, seed: int) -> str:
    from PIL import Image

    rnd = random.Random(seed)
    img = Image.frombytes("RGB", (size, size), bytes(rnd.getrandbits(8) for _ in range(size * size * 3)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def one(content):
    body = C.chat_body([{"role": "user", "content": content}], max_tokens=1,
                       chat_template_kwargs={"enable_thinking": False})
    t = time.perf_counter()
    rec = C.post("/v1/chat/completions", body, timeout=600)
    dt = time.perf_counter() - t
    usage = rec["response"].get("usage", {}) if isinstance(rec["response"], dict) else {}
    return {"http_status": rec["http_status"], "latency_s": round(dt, 4),
            "prompt_tokens": usage.get("prompt_tokens"), "error": rec["response"].get("error")
            if rec["http_status"] != 200 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="448,896", help="square image sizes in pixels, comma-separated")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not C.server_ready():
        print("mm_encode_timing: server not reachable", file=sys.stderr)
        return 2
    one(QUESTION)  # warm-up
    res = {"started_utc": C.now_utc(), "vit_dp_group": os.environ.get("MIMO_VIT_DP_GROUP"), "n": a.n, "sizes": {}}
    text = [one(QUESTION) for _ in range(a.n)]
    res["text_only"] = {"runs": text, "median_s": statistics.median(r["latency_s"] for r in text)}
    bad = any(r["http_status"] != 200 for r in text)
    for i, size in enumerate(int(s) for s in a.sizes.split(",")):
        runs = []
        for j in range(a.n + 1):  # first run of each size is a warm-up (bucket / graph load)
            url = png_data_url(size, 1000 * i + j)
            r = one([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": QUESTION}])
            if j:
                runs.append(r)
            bad |= r["http_status"] != 200
        med = statistics.median(r["latency_s"] for r in runs)
        res["sizes"][str(size)] = {"runs": runs, "median_s": med,
                                   "minus_text_s": round(med - res["text_only"]["median_s"], 4)}
        print(f"mm_encode_timing: {size}px prompt_tokens={runs[0]['prompt_tokens']} median {med:.3f} s "
              f"(text-only {res['text_only']['median_s']:.3f} s)")
    res["finished_utc"] = C.now_utc()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
