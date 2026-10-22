#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF reference for the audio towers (docs/design-notes.md#audio-criteria; used by the device check). Instance CPU.

  start_bg.sh p2_audio_ref env OMP_NUM_THREADS=32 bash scripts/venv_exec.sh python scripts/accuracy/gen_audio_reference.py \
      --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --assets $MIMO_WORK/mm_assets/audio \
      --out $MIMO_WORK/goldens/audio/audio_goldens.pt --summary $MIMO_WORK/results/audio/audio_ref_summary.json

For every WAV asset: the mel vLLM's processor serves (input), and HF FP32 / BF16 pre-VQ features ``[T2, 1024]``,
RVQ codes ``[T2, 20]`` and output embeddings ``[G, 4096]``. The summary records shapes, the BF16-vs-FP32 code agreement
and embedding cosine (the HF noise floor the plugin is judged against, docs/design-notes.md#audio-criteria).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_ref  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args(argv)
    mels = audio_ref.served_mels(a.src, a.assets)
    M, se, enc, tok = audio_ref.hf_towers(a.src)
    data, summ = {}, {}
    for name, (mel, tl) in mels.items():
        r32 = audio_ref.hf_run(M, se, enc, tok, mel, torch.float32)
        r16 = audio_ref.hf_run(M, se, enc, tok, mel, torch.bfloat16)
        data[name] = {"mel": mel, "token_len": tl, "fp32": r32, "bf16": r16}
        summ[name] = {"mel_frames": int(mel.shape[0]), "token_len": tl, "codes_shape": list(r32["codes"].shape),
                      "embeds_shape": list(r32["embeds"].shape),
                      "bf16_codes_eq_fp32": float((r16["codes"] == r32["codes"]).float().mean()),
                      "bf16_embeds_cos_fp32": float(torch.nn.functional.cosine_similarity(
                          r16["embeds"].float().flatten(), r32["embeds"].float().flatten(), dim=0))}
        print(name, summ[name], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(data, a.out)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.summary))), exist_ok=True)
    with open(os.path.expanduser(a.summary), "w") as f:
        json.dump({"out": a.out, "items": summ}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
