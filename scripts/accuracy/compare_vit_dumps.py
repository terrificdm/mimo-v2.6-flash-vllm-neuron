#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the device ViT rows dumped by a running server (MIMO_VIT_DUMP_DIR) with the HF ViT goldens.

Goldens come from ``gen_vit_reference.py`` (test assets) or ``gen_vit_reference_files.py`` (any image / video file).
Start the multimodal service with ``MIMO_VIT_DUMP_DIR=<dir>`` in its env prefix, send the image / video requests
(``mm_qa.py`` or ``ask.py``), then run this script. Items are matched to dumps by the sum of their pixel values and
the grid (same as ``run_mm_validation.py --capture-vision``); criteria (docs/design-notes.md#vit-accuracy):
cos(device, HF BF16) >= 0.999, else fallback three-way + cos(device, HF FP32) >= cos(HF BF16, HF FP32) - 0.002.

Usage: compare_vit_dumps.py --dumps DIR [--goldens $MIMO_WORK/goldens/mm/vit_goldens.pt]
                            [--items digits_1234.mp4,red_square.png] --out result.json
Exit codes: 0 all compared items pass, 1 some fail, 3 config error (no matching dump).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common as C  # noqa: E402
from run_mm_validation import vit_compare  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--goldens", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "goldens/mm/vit_goldens.pt"))
    ap.add_argument("--items", default=None, help="comma-separated golden item names (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    v = torch.load(a.goldens, weights_only=False)["items"]
    names = [s.strip() for s in a.items.split(",")] if a.items else list(v)
    g = {"mm": {n: {"pixel_values": v[n]["pixel_values"], "grid_thw": v[n]["grid_thw"].reshape(1, 3).tolist(),
                    "vit_fp32": v[n]["hf_fp32"], "vit_bf16": v[n]["hf_bf16"]} for n in names}}
    res = vit_compare(a.dumps, g, names)
    res.update(dumps=a.dumps, goldens=a.goldens, kinds={n: v[n]["kind"] for n in names})
    C.write_json(a.out, C.to_py(res, max_len=256))
    C.log(f"RESULT vit dumps: passed={res['passed']} items={names}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    C.run_main(main)
