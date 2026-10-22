#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks of a running server started with MIMO_MM_VISION_BUCKETS (e.g. 4096,65536; docs/design-notes.md#vision-request-blocks).

Cases (files under the server's --allowed-local-media-path, default $MIMO_MM_ASSETS/large; README Step 7 generates them):
  text      text-only baseline latency (max_tokens 1)
  video1    3-minute 720p video, max_tokens 1 (first request: encoder + prefill time ~ TTFT), prompt tokens
  video     same video, short description (thinking off)
  image     2048 x 1365 image (resized to <= max_pixels by the processor)
  images2   two images in one request (needs > 1 encoder block per request)
  reject    video + two images: 16 + 1 + 1 blocks > 16 -> expect HTTP 400 (API-side block check), server stays up
  after     text request after the rejection (server still serving)

  python3 scripts/api_tests/vision_limits_check.py --out $MIMO_WORK/results/vision_limits_check
Exit codes: 0 all expectations met, 1 some failed, 2 server not reachable.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _content(*parts, text):
    return [{"role": "user", "content": list(parts) + [{"type": "text", "text": text}]}]


def main(argv=None) -> int:
    ap = C.base_argparser(__doc__.split("\n")[0])
    ap.add_argument("--dir", default=os.path.join(os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets")), "large"))
    ap.add_argument("--video", default="long_720p_180s.mp4")
    ap.add_argument("--image", default="photo_2048x1365.jpg")
    ap.add_argument("--image2", default="photo_small.jpg")
    a = ap.parse_args(argv)
    C.require_server()
    url = lambda n: "file://" + os.path.join(os.path.realpath(a.dir), n)  # noqa: E731  (resolved, like mm_qa.py)
    vid = {"type": "video_url", "video_url": {"url": url(a.video)}}
    im1 = {"type": "image_url", "image_url": {"url": url(a.image)}}
    im2 = {"type": "image_url", "image_url": {"url": url(a.image2)}}
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    cases = [
        ("text", C.chat_body("Say OK.", max_tokens=1, **off), 200),
        ("video1", C.chat_body(_content(vid, text="Describe the video."), max_tokens=1, **off), 200),
        ("video", C.chat_body(_content(vid, text="Describe this video in two sentences."), max_tokens=96, **off), 200),
        ("image", C.chat_body(_content(im1, text="Describe the image in one sentence."), max_tokens=64, **off), 200),
        ("images2", C.chat_body(_content(im1, im2, text="What animal is in each image? Answer briefly."),
                                max_tokens=48, **off), 200),
        ("reject", C.chat_body(_content(vid, im1, im2, text="Describe."), max_tokens=8, **off), 400),
        ("after", C.chat_body("Say OK.", max_tokens=4, **off), 200),
    ]
    want = set(a.cases.split(",")) if a.cases else None
    res, ok = {"cases": {}}, True
    for cid, body, expect in cases:
        if want and cid not in want:
            continue
        r = C.post("/v1/chat/completions", body, timeout=a.timeout)
        msg = C.message(r)
        rec = {"http_status": r["http_status"], "expect": expect, "elapsed_s": r["elapsed_s"],
               "usage": (r["response"] or {}).get("usage"), "content": (msg.get("content") or "")[:600],
               "finish_reason": C.choice0(r).get("finish_reason")}
        if r["http_status"] != 200:
            rec["error"] = str((r["response"] or {}).get("error"))[:800]
        rec["ok"] = r["http_status"] == expect
        ok = ok and rec["ok"]
        res["cases"][cid] = rec
        print(json.dumps({cid: rec}, ensure_ascii=False), flush=True)
    res.update(ok=ok, time=C.now_utc(), base=C.base_url())
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out + ".json", "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
