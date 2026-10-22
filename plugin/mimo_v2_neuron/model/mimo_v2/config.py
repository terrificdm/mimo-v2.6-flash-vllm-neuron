# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 (MiMo-V2.6-Flash) configuration for the vllm-neuron native model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from ... import switches

DECODE_MOE_IMPLS = ("tkg", "cte", "torch")
KEY_BLOCK_ENVS = ("MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK")
DEFAULT_DECODE_KEY_BLOCK = 32768


def default_prefill_key_block(max_model_len: int | None) -> int:
    """Prior-key block size of segmented prefill by ``max_model_len`` (docs/design-notes.md#segmented-prefill)."""
    if max_model_len is None or max_model_len <= 16384:
        return 4096
    if max_model_len <= 32768:
        return 8192
    if max_model_len <= 131072:
        return 16384
    return 32768


def key_blocks_from_env(max_model_len: int | None) -> tuple[int, int]:
    """(prefill_key_block, decode_key_block) from MIMO_ATTN_{PREFILL,DECODE}_KEY_BLOCK or the defaults.

    Only the integer parse happens here; range / block-size checks are in ``factory._validate_config``.
    """
    vals = []
    for name, default in zip(KEY_BLOCK_ENVS, (default_prefill_key_block(max_model_len), DEFAULT_DECODE_KEY_BLOCK)):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            vals.append(default)
            continue
        try:
            vals.append(int(raw))
        except ValueError:
            raise ValueError(f"MiMoV2 Neuron: {name} must be an integer, got {raw!r}") from None
    return vals[0], vals[1]


@dataclass
class MiMoV2Config:
    # ── text model architecture ───────────────────────────────────────────
    vocab_size: int = 152576
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    # global (full) attention layers
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    rope_theta: float = 1e7
    # sliding-window attention layers
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 1e4
    sliding_window: int = 128
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    partial_rotary_factor: float = 0.334
    attention_value_scale: float | None = 0.707
    attention_projection_layout: str = "split"
    hybrid_layer_pattern: list = field(default_factory=list)   # 0 = GA, 1 = SWA
    # feed-forward
    intermediate_size: int = 16384
    moe_layer_freq: list = field(default_factory=list)          # 0 = dense, 1 = MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float | None = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    n_group: int = 1
    topk_group: int = 1
    hidden_act: str = "silu"
    layernorm_epsilon: float = 1e-6
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 1048576
    torch_dtype: torch.dtype = torch.bfloat16
    # ── derived / runtime ────────────────────────────────────────────────
    has_quantization_config: bool = False
    decode_moe_impl: str = "tkg"
    prefill_key_block: int = 4096        # segmented prefill: prior keys per block (MIMO_ATTN_PREFILL_KEY_BLOCK)
    decode_key_block: int = DEFAULT_DECODE_KEY_BLOCK   # decode: keys per block (MIMO_ATTN_DECODE_KEY_BLOCK)
    neuron_config: object | None = None
    # FP8 checkpoint (``neuron_fp8_config`` written by tools/convert_checkpoint.py --fp8; docs/design-notes.md#fp8-alignment)
    fp8_weights: bool = False
    fp8_linear_group: int = 128
    fp8_expert_group: int = 32
    fp8_act: dict | None = None          # fp8_linear.act_settings() when fp8_weights
    # FP8 experts in decode (docs/design-notes.md#fp8-alignment): "expand" = BF16 expansion in the graph + BF16 moe_tkg (default, validated);
    # "row" = NKI moe_tkg ROW mode reads the FP8 codes from HBM, BF16 hidden, per-output-channel dequant after
    # the matmul (MIMO_FP8_EXPERT_IMPL)
    fp8_expert_impl: str = "expand"
    # FP8 "row" experts in prefill (docs/design-notes.md#moe-hybrid): "shard_on_i" = nkilib moe_cte shard_on_i (static loop over
    # all N blocks, default); "hybrid" = shard_on_i_hybrid with num_static_block=0 (runtime loop over the non-empty
    # blocks only; same per-block code) (MIMO_MOE_CTE_IMPL)
    moe_cte_impl: str = "shard_on_i"
    def __post_init__(self):
        if not self.hybrid_layer_pattern:
            self.hybrid_layer_pattern = [0] * self.num_hidden_layers
        if not self.moe_layer_freq:
            self.moe_layer_freq = [1] * self.num_hidden_layers
        if self.routed_scaling_factor is None:
            self.routed_scaling_factor = 1.0

    # per-layer helpers
    def is_swa(self, layer_idx: int) -> bool:
        return self.hybrid_layer_pattern[layer_idx] == 1

    def is_moe(self, layer_idx: int) -> bool:
        return bool(self.moe_layer_freq[layer_idx])

    def attn_dims(self, layer_idx: int) -> dict:
        if self.is_swa(layer_idx):
            return dict(num_heads=self.swa_num_attention_heads, num_kv_heads=self.swa_num_key_value_heads,
                        head_dim=self.swa_head_dim, v_head_dim=self.swa_v_head_dim,
                        rope_theta=self.swa_rope_theta, sliding_window=self.sliding_window,
                        has_sink=self.add_swa_attention_sink_bias)
        return dict(num_heads=self.num_attention_heads, num_kv_heads=self.num_key_value_heads,
                    head_dim=self.head_dim, v_head_dim=self.v_head_dim,
                    rope_theta=self.rope_theta, sliding_window=None,
                    has_sink=self.add_full_attention_sink_bias)

    def rope_dim(self, layer_idx: int) -> int:
        return int(self.attn_dims(layer_idx)["head_dim"] * self.partial_rotary_factor)

    @classmethod
    def from_configs(cls, hf_config, neuron_config, max_model_len: int | None = None) -> "MiMoV2Config":
        if isinstance(hf_config, dict):
            d = dict(hf_config)
        else:
            d = hf_config.to_dict()
        qc = d.get("quantization_config")
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in names and k not in ("torch_dtype", "neuron_config",
                                                                     "prefill_key_block", "decode_key_block")}
        dt = d.get("torch_dtype", d.get("dtype", "bfloat16"))
        if isinstance(dt, str):
            dt = getattr(torch, dt)
        kw["torch_dtype"] = dt if isinstance(dt, torch.dtype) else torch.bfloat16
        kw["has_quantization_config"] = bool(qc)
        kw["decode_moe_impl"] = switches.get("MIMO_DECODE_MOE_IMPL")
        kw["prefill_key_block"], kw["decode_key_block"] = key_blocks_from_env(max_model_len)
        kw["neuron_config"] = neuron_config
        fc = d.get("neuron_fp8_config")
        if fc:
            from .fp8_linear import act_settings

            if fc.get("format") != "neuron_fp8":
                raise ValueError(f"MiMoV2 Neuron: unsupported neuron_fp8_config format {fc.get('format')!r}")
            kw["fp8_weights"] = True
            kw["fp8_linear_group"] = int(fc.get("linear_group_size", 128))
            kw["fp8_expert_group"] = int(fc.get("expert_group_size", 32))
            kw["fp8_act"] = act_settings()
            impl = switches.get("MIMO_FP8_EXPERT_IMPL")   # default row on the device
            if impl not in ("expand", "row"):
                raise ValueError(f"MIMO_FP8_EXPERT_IMPL must be expand or row, got {impl!r}")
            kw["fp8_expert_impl"] = impl
            cte = switches.get("MIMO_MOE_CTE_IMPL")   # default hybrid on the device (row experts only)
            if cte not in ("shard_on_i", "hybrid"):
                raise ValueError(f"MIMO_MOE_CTE_IMPL must be shard_on_i or hybrid, got {cte!r}")
            kw["moe_cte_impl"] = cte
        return cls(**kw)
