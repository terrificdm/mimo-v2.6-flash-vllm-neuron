#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny random MiMo-V2 checkpoint (split layout, BF16 + FP32 routing bias) for CPU e2e tests.

Tiny config (tiny-model drill): hidden 512, 8 Q heads, GA kv 2 / SWA kv 4, head 192 / v 128, rope 64,
16 experts top-4, expert I 128, dense I 1024, 3 layers (dense+GA, MoE+SWA+sink, MoE+GA),
window 128, vocab 152576 (real tokenizer files are copied from --like).
``--with-vision`` (multimodal support) adds a tiny MiMo ViT (hidden 256, 4 Q / 2 KV heads of 64, intermediate 512,
4 blocks: full / row-order window / column-order window / full, sinks, out_hidden 512) saved as ``visual.*``
(merger / ln_q biases zeroed and not saved, like the real checkpoint) and sets
``architectures = ["MiMoV2OmniForCausalLM"]``.

Weights are drawn with a fixed seed. Matrices use std 1/sqrt(fan_in) and the embedding std 1 so that
every sub-layer contributes O(1) to the residual stream (the HF default init of 0.02 would make the
layers almost invisible and the test insensitive to layer bugs).

Usage: make_tiny_checkpoint.py --out $MIMO_WORK/tiny/tiny3
         [--like $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16-L3] [--seed 0] [--with-vision]
Exit codes: 0 ok, 2 bad args.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch

TINY = {
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "head_dim": 192,
    "v_head_dim": 128,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 128,
    "intermediate_size": 1024,
    "num_hidden_layers": 3,
    "hybrid_layer_pattern": [0, 1, 0],
    "moe_layer_freq": [0, 1, 1],
    "sliding_window": 128,
    "sliding_window_size": 128,
    "attention_projection_layout": "split",
    "vision_config": {},
    "audio_config": {},
}

TINY_VISION = {
    "depth": 4, "fullatt_block_indexes": [0, 3], "hidden_act": "silu", "hidden_size": 256, "in_chans": 3,
    "intermediate_size": 512, "num_heads": 4, "num_key_value_heads": 2, "out_hidden_size": 512, "patch_size": 16,
    "spatial_merge_size": 2, "spatial_patch_size": 16, "temporal_patch_size": 2, "tokens_per_second": 2,
    "use_sink": True, "visual_token_window_size": 64, "vit_window_attn_types": [-1, 0, 1, -1], "window_size": 128,
}
MERGER_BIASES = ("visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias", "visual.merger.ln_q.bias")


def init_(name: str, p: torch.Tensor, g: torch.Generator) -> None:
    with torch.no_grad():
        if name.startswith("visual."):
            if name in MERGER_BIASES:
                p.zero_()
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p[0].numel()))
            elif name.endswith(".sinks"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:   # RMSNorm / LayerNorm weights
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
        elif name.endswith("e_score_correction_bias"):
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.2)
        elif name.endswith("attention_sink_bias"):
            p.copy_(torch.randn(p.shape, generator=g))
        elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
        elif name.endswith("embed_tokens.weight"):
            p.copy_(torch.randn(p.shape, generator=g))
        elif p.dim() == 2:
            p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
        else:
            raise ValueError(f"no init rule for {name} {tuple(p.shape)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--like", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16-L3"),
                    help="converted checkpoint dir to copy config/tokenizer/remote code from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-vision", action="store_true", help="add a tiny ViT (MiMoV2OmniForCausalLM)")
    a = ap.parse_args(argv)
    if not os.path.isfile(os.path.join(a.like, "config.json")):
        print(f"ERROR: --like {a.like} has no config.json", file=sys.stderr)
        return 2

    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(a.like):
        if fn.endswith(".safetensors") or fn.endswith(".tmp") or fn in ("config.json", "model.safetensors.index.json"):
            continue
        src = os.path.join(a.like, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(a.out, fn))
    with open(os.path.join(a.like, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    cfg.update(TINY)
    if a.with_vision:
        cfg["vision_config"] = dict(TINY_VISION)
        cfg["architectures"] = ["MiMoV2OmniForCausalLM"]
        # cap images at 1024 patches (= the tiny e2e vision block), like make_mm_checkpoint.py does at 4096
        cfg["processor_config"] = {**(cfg.get("processor_config") or {}), "image_max_pixels": 262144,
                                   "video_max_pixels": 262144}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    from transformers import AutoConfig, AutoModelForCausalLM

    hf_cfg = AutoConfig.from_pretrained(a.out, trust_remote_code=True)
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True, dtype=torch.float32)
    g = torch.Generator().manual_seed(a.seed)
    tensors = {}
    for name, p in sorted(model.named_parameters()):
        init_(name, p, g)
        if name in MERGER_BIASES:
            continue
        t = p.detach().clone()
        tensors[name] = t.float().contiguous() if name.endswith("e_score_correction_bias") \
            else t.to(torch.bfloat16).contiguous()

    from safetensors.torch import save_file

    shard = "model.safetensors"
    save_file(tensors, os.path.join(a.out, shard), metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: shard for k in sorted(tensors)}}
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1)
    n_params = sum(t.numel() for t in tensors.values())
    bias = [k for k in tensors if k.endswith("e_score_correction_bias")]
    print(f"tiny checkpoint -> {a.out}: {len(tensors)} tensors, {n_params / 1e6:.1f}M params, "
          f"{total / 1e6:.1f} MB; FP32 routing bias: {bias} {[str(tensors[k].dtype) for k in bias]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
