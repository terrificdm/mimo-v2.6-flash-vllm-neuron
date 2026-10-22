#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF MiMo ViT goldens for arbitrary image / video files (instance CPU; docs/design-notes.md#mm-reference-inputs).

Same records as ``gen_vit_reference.py`` (``{"meta", "items": {name: {"kind", "grid_thw", "pixel_values" (BF16),
"hf_fp32", "hf_bf16"}}}``, so ``compare_vit_dumps.py --goldens <out> --items <name>`` works), but for the given files
and with the processor of the *served* model directory (its ``processor_config``: image / video_max_pixels) plus the
served ``--mm-processor-kwargs`` (default ``{"max_pixels": 1048576}``, scripts/serve.sh). Videos are decoded like the
API server (VideoMediaIO defaults: 32 frames). ViT weights come from ``--src`` (original checkpoint, ``visual.*``).

  env OMP_NUM_THREADS=64 PYTHONPATH=<repo>/plugin python scripts/accuracy/gen_vit_reference_files.py \
      --proc-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8-mm-audio \
      --files $MIMO_MM_ASSETS/large/long_720p_180s.mp4,$MIMO_MM_ASSETS/large/photo_2048x1365.jpg \
      --out $MIMO_WORK/goldens/mm/vit_goldens_large.pt
Exit codes: 0 ok, 1 failure, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import mm_prep  # noqa: E402
import vit_ref  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _cos(a, b):   # float64: in float32 the 57.7M-element 3-minute video gives cos > 1
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
    ap.add_argument("--proc-dir", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-fp8-mm-audio"))
    ap.add_argument("--mm-processor-kwargs", default='{"max_pixels": 1048576}')
    ap.add_argument("--files", required=True, help="comma-separated image / video paths")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = [f for f in a.files.split(",") if f]
    if not files or not all(os.path.isfile(f) for f in files) or not os.path.isdir(a.src):
        print("bad --files / --src", file=sys.stderr)
        return 2
    mm_kwargs = json.loads(a.mm_processor_kwargs)
    torch.manual_seed(0)
    t0 = time.time()
    proc = mm_prep.vllm_processor(a.proc_dir)
    ckpt = vit_ref.VisualCkpt(a.src)
    vits = {torch.float32: vit_ref.hf_vit(a.src, torch.float32, ckpt),
            torch.bfloat16: vit_ref.hf_vit(a.src, torch.bfloat16, ckpt)}
    out = {"meta": {"src": a.src, "proc_dir": a.proc_dir, "mm_processor_kwargs": mm_kwargs,
                    "input": "vLLM MiMoV2OmniMultiModalProcessor output after the BF16 cast",
                    "torch": torch.__version__}, "items": {}}
    summ = {"items": {}}
    for path in files:
        name = os.path.basename(path)
        t1 = time.time()
        if path.lower().endswith(VIDEO_EXT):
            frames, meta = mm_prep.load_video(path)
            r = mm_prep.run_vllm(proc, mm_prep.VID_PROMPT, {"video": [(frames, meta)]}, mm_kwargs)
            kind, pv = "video", mm_prep.item_field(r, "video", 0, "pixel_values_videos")
            grid = mm_prep.item_field(r, "video", 0, "video_grid_thw")
            embeds = int(r["mm_placeholders"]["video"][0].get_num_embeds())
        else:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            r = mm_prep.run_vllm(proc, mm_prep.IMG_PROMPT, {"image": [img]}, mm_kwargs)
            kind, pv = "image", mm_prep.item_field(r, "image", 0, "pixel_values")
            grid = mm_prep.item_field(r, "image", 0, "image_grid_thw")
            embeds = int(r["mm_placeholders"]["image"][0].get_num_embeds())
        grid = grid.reshape(1, 3).long()
        rec = {"kind": kind, "grid_thw": grid.clone(), "pixel_values": pv.to(torch.bfloat16).clone()}
        rs = {"kind": kind, "grid_thw": grid.tolist(), "patches": int(pv.shape[0]), "embeds": embeds,
              "t_processor_s": round(time.time() - t1, 1)}
        for dt, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            t2 = time.time()
            y, _, _ = vit_ref.hf_run(vits[dt], pv.to(torch.bfloat16), grid)
            rs[f"t_{tag}_s"] = round(time.time() - t2, 1)
            if not torch.isfinite(y).all():
                print(f"{name} {tag}: non-finite output", file=sys.stderr)
                return 1
            rec[f"hf_{tag}"] = y if dt == torch.float32 else y.to(torch.bfloat16)
        rs["tokens"] = int(rec["hf_fp32"].shape[0])
        rs["cos_bf16_vs_fp32"] = _cos(rec["hf_bf16"], rec["hf_fp32"])
        out["items"][name] = rec
        summ["items"][name] = rs
        print(json.dumps({name: rs}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, a.out)
    summ.update(out=a.out, bytes=os.path.getsize(a.out), t_total_s=round(time.time() - t0, 1))
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "items"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
