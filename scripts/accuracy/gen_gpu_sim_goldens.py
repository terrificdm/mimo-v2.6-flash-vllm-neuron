#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""“GPU 算法模拟”的 L3 goldens（docs/design-notes.md#fp8-alignment）. Instance CPU.

  start_bg.sh p2_goldens_gpusim env OMP_NUM_THREADS=96 bash scripts/venv_exec.sh python \
      scripts/accuracy/gen_gpu_sim_goldens.py --model-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16-L3 \
      --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --layers 0,1,5 --out $MIMO_WORK/goldens/L3_gpusim \
      --prompts scripts/accuracy/prompts.json --record-routing

Same procedure and output format as ``gen_goldens.py`` (FP32 autoregressive 16 steps + BF16 teacher-forced, optional
routing.pt), but the HF model computes like the official GPU deployment (vLLM 0.24, H100):

  * q/k/v and the dense MLP (layer 0): W8A8 with the original checkpoint weights -- E4M3FN codes x 128x128 F32
    ``weight_scale_inv`` dequantised exactly in FP32 (not the BF16-rounded copy of the -bf16 directory) -- and the
    input of each such linear quantised per token per 128 channels exactly like ``fp8_utils.per_token_group_quant_fp8``
    (``plugin/.../fp8_linear.quant_act_group``: absmax * (1/448), clamp, E4M3FN round-half-even); the product is
    accumulated in FP32 and cast to the activation dtype (GPU: FP32 epilogue, BF16 output);
  * experts: the MXFP4 values exactly (the -bf16 directory holds them exactly), BF16 activations (H100 Marlin
    W4A16); o_proj, embeddings, lm_head, norms, router, sinks: unchanged BF16 / FP32 (as on the GPU).
Not modelled: summation order of the GPU kernels; vLLM's K/V re-quantisation for TP != 4 (``_requantize_fp8``);
the B200 MXFP8 expert activations. Exit codes as gen_goldens.py.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "plugin"), os.path.join(ROOT, "scripts", "accuracy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_goldens as G  # noqa: E402

GROUP = 128


class W8A8Linear(nn.Module):
    """GPU block-FP8 linear: dynamic per-token-group activation FP8 x exact FP8 weight, FP32 accumulation."""

    def __init__(self, w_fp32: torch.Tensor, fp8_max: float = 448.0):
        super().__init__()
        self.register_buffer("w", w_fp32.to(torch.float32).contiguous(), persistent=False)
        self.fp8_max = fp8_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from mimo_v2_neuron.model.mimo_v2.fp8_linear import quant_act_group

        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        t, k = x2.shape
        codes, s = quant_act_group(x2, self.fp8_max)
        xdq = (codes.reshape(t, k // GROUP, GROUP) * s.unsqueeze(-1)).reshape(t, k)   # exact FP32 values
        return torch.matmul(xdq, self.w.t()).to(x.dtype).reshape(*shp[:-1], -1)


def patch_model(model, src_dir: str, layers: list[int]) -> int:
    """Replace q/k/v (every layer) and the dense MLP projections (non-MoE layers) by W8A8Linear."""
    import json

    from tools.convert_checkpoint import Src, attn_dims
    from tools.dequant import dequant_fp8_block, split_fused_qkv

    src = Src(src_dir)
    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    n = 0
    for new, orig in enumerate(layers):
        L = model.model.layers[new]
        sp = f"model.layers.{orig}."
        d = attn_dims(cfg, orig)
        w = src.get(sp + "self_attn.qkv_proj.weight")
        q, k, v = split_fused_qkv(w, src.get(sp + "self_attn.qkv_proj.weight_scale_inv"), d["nh"], d["nkv"],
                                  d["hd"], d["vd"])
        for name, t in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            old = getattr(L.self_attn, name)
            if tuple(old.weight.shape) != tuple(t.shape) or old.bias is not None:
                G.fail(f"layer {new} {name}: shape {tuple(old.weight.shape)} vs {tuple(t.shape)} / bias")
            setattr(L.self_attn, name, W8A8Linear(t))
            n += 1
        if not cfg["moe_layer_freq"][orig]:
            for name in ("gate_proj", "up_proj", "down_proj"):
                key = sp + f"mlp.{name}.weight"
                t = dequant_fp8_block(src.get(key), src.get(key + "_scale_inv"))
                old = getattr(L.mlp, name)
                if tuple(old.weight.shape) != tuple(t.shape):
                    G.fail(f"layer {new} mlp.{name}: shape {tuple(old.weight.shape)} vs {tuple(t.shape)}")
                setattr(L.mlp, name, W8A8Linear(t))
                n += 1
    return n


def main(argv=None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--src", required=True, help="original checkpoint (FP8 / MXFP4)")
    ap.add_argument("--layers", required=True, help="original layer ids of the reduced model, e.g. 0,1,5")
    a, rest = ap.parse_known_args(argv)
    layers = [int(x) for x in a.layers.split(",")]
    orig_load = G.load_model

    def load_model(model_dir, dtype):
        m = orig_load(model_dir, dtype)
        n = patch_model(m, a.src, layers)
        G.log(f"gpu-sim: {n} linear layers replaced by W8A8 (dtype {dtype})")
        return m

    G.load_model = load_model
    return G.main(rest)


if __name__ == "__main__":
    sys.exit(main())
