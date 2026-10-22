#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF MiMo ViT goldens on the instance CPU (docs/design-notes.md#mm-reference-inputs).

Instantiates only HF ``MiMoVisionTransformer`` (patched HF remote code) with the ``visual.*`` weights of the original
checkpoint (BF16 on disk; the 309B text model is not loaded) and runs it in FP32 and BF16 on the inputs the vLLM
server actually feeds the model (vLLM MiMo-Omni processor output after the BF16 cast; 4 test images + the 8-frame
test video, grid_t = 4).

Saved (torch.save, --out): {"meta": {...}, "items": {name: {"kind", "grid_thw", "pixel_values" (BF16),
"hf_fp32" [tokens, 4096] fp32, "hf_bf16" [tokens, 4096] BF16, and for --layer-items also "layers_in_fp32",
"layers_out_fp32", "layers_out_bf16" (28 x [patches, 1280], stored BF16, in the order each layer ran)}}}.
A JSON summary (shapes, sha256 of the file, cos(BF16, FP32) per item, timings) goes to --summary.

  env OMP_NUM_THREADS=64 PYTHONPATH=<repo>/plugin python scripts/accuracy/gen_vit_reference.py \
      --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --assets $MIMO_WORK/mm_assets \
      --out $MIMO_WORK/goldens/mm/vit_goldens.pt
Exit codes: 0 ok, 1 failure, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import vit_ref  # noqa: E402


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
    ap.add_argument("--assets", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None, help="JSON summary path (default: <out>.summary.json)")
    ap.add_argument("--layer-items", default="red_square.png",
                    help="comma-separated items whose per-layer inputs/outputs are saved")
    args = ap.parse_args()
    if not os.path.isdir(args.src) or not os.path.isdir(args.assets):
        print("bad --src / --assets", file=sys.stderr)
        return 2
    summary_path = args.summary or args.out + ".summary.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    items = vit_ref.served_inputs(args.src, args.assets)
    t_proc = time.time() - t0
    ckpt = vit_ref.VisualCkpt(args.src)
    vits = {torch.float32: vit_ref.hf_vit(args.src, torch.float32, ckpt),
            torch.bfloat16: vit_ref.hf_vit(args.src, torch.bfloat16, ckpt)}
    layer_items = set(x for x in args.layer_items.split(",") if x)
    out = {"meta": {"src": args.src, "assets": args.assets, "vision_config": vit_ref.vision_config(args.src),
                    "input": "vLLM MiMoV2OmniMultiModalProcessor output after the BF16 cast",
                    "merger_biases": "zero (checkpoint has none)", "torch": torch.__version__},
           "items": {}}
    summ = {"items": {}, "t_processor_s": round(t_proc, 1)}
    for it in items:
        name, pv, grid = it["name"], it["pixel_values"], it["grid_thw"]
        rec = {"kind": it["kind"], "grid_thw": grid.clone(), "pixel_values": pv.to(torch.bfloat16).clone()}
        rec_s = {"kind": it["kind"], "grid_thw": grid.tolist(), "patches": int(pv.shape[0])}
        record = name in layer_items
        for dt, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            t1 = time.time()
            y, ins, outs = vit_ref.hf_run(vits[dt], pv.to(torch.bfloat16), grid, record=record)
            rec_s[f"t_{tag}_s"] = round(time.time() - t1, 1)
            rec[f"hf_{tag}"] = y if dt == torch.float32 else y.to(torch.bfloat16)
            if record:
                rec[f"layers_out_{tag}"] = [o.to(torch.bfloat16) for o in outs]
                if dt == torch.float32:
                    rec["layers_in_fp32"] = [i.to(torch.bfloat16) for i in ins]
            if not torch.isfinite(y).all():
                print(f"{name} {tag}: non-finite output", file=sys.stderr)
                return 1
        rec_s["tokens"] = int(rec["hf_fp32"].shape[0])
        rec_s["cos_bf16_vs_fp32"] = _cos(rec["hf_bf16"], rec["hf_fp32"])
        rec_s["norm_fp32"] = float(rec["hf_fp32"].norm())
        rec_s["layers_saved"] = record
        out["items"][name] = rec
        summ["items"][name] = rec_s
        print(json.dumps({name: rec_s}), flush=True)
    tmp = args.out + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    summ.update({"out": args.out, "bytes": os.path.getsize(args.out), "sha256": h.hexdigest(),
                 "t_total_s": round(time.time() - t0, 1)})
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "items"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
