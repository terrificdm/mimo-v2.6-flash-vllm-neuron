# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vllm-project/vllm-neuron (Apache-2.0): vllm_neuron/model/gpt_oss/model_bf16.py
#   (model layout, GptOssForCausalLM.forward); modified for MiMo-V2. See NOTICE.
"""MiMo-V2 (MiMo-V2.6-Flash) for vllm-neuron, adapted from ``vllm_neuron/model/gpt_oss/model_bf16.py``.

Layout conventions follow GPT-OSS: during prefill the inter-layer hidden state is
sequence-parallel ``[T/tp, H]`` (attention all-gathers on entry and reduce-scatters on
exit); during decode it is replicated ``[B, H]`` and attention all-reduces its O-proj
partial sums.

Differences from GPT-OSS handled here:
  * QK head_dim 192 != V head_dim 128 -> PyTorch attention (``attention_torch``), plain matmul QKV;
  * partial RoPE (first 64 dims, neox rotate_half), GA theta 1e7 / SWA theta 1e4;
  * sink only on SWA layers; V scaled by 0.707 before the cache;
  * one KV-cache head_size (192): V is zero-padded to 192 on write and sliced on read;
  * sigmoid + noaux_tc routing (``routing.py``, FP32) feeding ``NF.moe_cte`` (prefill) and, for
    decode, ``NF.moe_tkg`` / ``NF.moe_cte`` / ``_torch_local_experts`` (``MIMO_DECODE_MOE_IMPL``);
    SiLU experts without bias or clamp; dense SwiGLU FFN on layer 0;
  * no hidden padding, no YaRN, no Eagle3 / DP variants.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from ... import switches
from .attention_torch import decode_attention, prefill_attention, segmented_prefill_attention
from .config import MiMoV2Config
from .routing import noaux_tc_route
from .weight_loaders import (
    expert_checkpoint_keys,
    expert_down_fp8_loader,
    expert_down_loader,
    expert_fp8_checkpoint_keys,
    expert_gate_up_fp8_loader,
    expert_gate_up_loader,
    fp32_vector_loader,
    kv_heads_for_rank,
    o_proj_weight_loader,
    qkv_fp8_loader,
    qkv_weight_loader,
    rowgroup_shard_loader,
    sink_loader,
)
from .fp8_linear import fp8_linear

logger = logging.getLogger(__name__)

# MIMO_DEBUG_CAPTURE=1: inline vllm-neuron tensor captures inside attention (accuracy debugging with
# neuron_config.tensor_capture, scripts/accuracy/diag_capture.py). Off by default -> no graph change.
_DEBUG_CAPTURE = os.environ.get("MIMO_DEBUG_CAPTURE") == "1"


def _dbg(name: str, t: torch.Tensor) -> None:
    if _DEBUG_CAPTURE:
        from vllm_neuron.accuracy import capture_tensor

        capture_tensor(name, t)


def _set_loader(param, loader):
    from vllm_neuron.utils.weight_loader import set_weight_loader

    set_weight_loader(param, loader)


def _all_reduce(group, x):
    res = group.all_reduce(x)
    return x if res is None else res


# =============================================================================
# RMSNorm (HF MiMoV2RMSNorm: normalise in fp32, cast back, then multiply by weight)
# =============================================================================
class MiMoV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


# =============================================================================
# Partial RoPE (default rope, no scaling): rotate the first ``rope_dim`` dims, neox rotate_half
# =============================================================================
class MiMoV2RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, rope_dim: int):
        super().__init__()
        if rope_dim % 2:
            raise ValueError(f"MiMoV2 Neuron: rope_dim must be even, got {rope_dim}")
        self.rope_dim = rope_dim
        # No buffer: the runner builds the model on the meta device and only loads parameters, so a
        # buffer would stay on meta (``model.to(device)`` then fails). Computed per call like llama3.
        self.theta = theta

    def _compute_inv_freq(self, device) -> torch.Tensor:
        exps = torch.arange(0, self.rope_dim, 2, dtype=torch.int64, device=device).float() / self.rope_dim
        return 1.0 / (self.theta ** exps)

    @property
    def inv_freq(self) -> torch.Tensor:
        return self._compute_inv_freq(torch.device("cpu"))

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions [T] -> (cos, sin) each [T, rope_dim] in ``dtype`` (HF casts to x.dtype)."""
        inv_freq = self._compute_inv_freq(positions.device)
        freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, nh, D]; cos/sin [T, rope_dim] -> rotate x[..., :rope_dim], keep the rest."""
    rd = cos.shape[-1]
    x_rope, x_pass = x[..., :rd], x[..., rd:]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    x_rope = x_rope * c + _rotate_half(x_rope) * s
    return torch.cat([x_rope, x_pass], dim=-1)


# =============================================================================
# Attention
# =============================================================================
class MiMoV2Attention(nn.Module):
    """GA / SWA attention with TP head sharding.

    Per rank: ``num_heads // tp`` Q heads (contiguous), KV heads from ``kv_heads_for_rank``
    (TP=64: 1 Q head and 1 KV head per rank for both layer types).

    Parameters (loaded from the split BF16 checkpoint):
      qkv_proj_weight [H, nq*192 + nkv*192 + nkv*128]  (q | k | v columns)
      o_proj_weight   [nq*128, H]                       (row slice of o_proj input dim)
      sinks           [nq] fp32                         (SWA layers only)
    """

    def __init__(self, config: MiMoV2Config, layer_idx: int, tp_group=None):
        super().__init__()
        if tp_group is None:
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.dtype = config.torch_dtype

        d = config.attn_dims(layer_idx)
        tp = self.world_size
        self.num_heads = d["num_heads"]
        self.num_kv_heads = d["num_kv_heads"]
        self.head_dim = d["head_dim"]
        self.v_head_dim = d["v_head_dim"]
        self.sliding_window = d["sliding_window"]
        self.rope_dim = config.rope_dim(layer_idx)
        self.scaling = self.head_dim ** -0.5
        self.v_scale = config.attention_value_scale
        self.cache_head_dim = max(self.head_dim, self.v_head_dim)
        # docs/design-notes.md#v-no-pad: MIMO_KV_VNOPAD=1 stores V at its own width (128) instead of zero-padded to 192; the runner
        # still allocates one (2, blocks, Hkv, bs, head_size) tensor per layer, so get_kv_spec declares
        # head_size = (192 + 128) / 2 and bind_kv_cache re-views that storage as K [.., 192] + V [.., 128]
        # (default 1 on the device; an odd head_dim + v_head_dim falls back to the padded layout)
        self.kv_vnopad = switches.get("MIMO_KV_VNOPAD") == "1"
        if self.kv_vnopad and (self.head_dim + self.v_head_dim) % 2:
            if switches.is_set("MIMO_KV_VNOPAD"):
                raise ValueError("MiMoV2 Neuron: MIMO_KV_VNOPAD needs head_dim + v_head_dim even")
            logger.warning("MiMoV2 Neuron: head_dim %d + v_head_dim %d is odd; default MIMO_KV_VNOPAD=1 not used",
                           self.head_dim, self.v_head_dim)
            self.kv_vnopad = False
        # MIMO_KV_WRITE=nki writes K/V rows with an in-place NKI scatter (nki_kv.kv_write)
        self.kv_write_nki = switches.get("MIMO_KV_WRITE") == "nki"   # default nki on the device
        self.prefill_key_block = config.prefill_key_block
        self.decode_key_block = config.decode_key_block

        self.num_heads_per_rank = self.num_heads // tp
        self.kv_heads = kv_heads_for_rank(self.rank, tp, self.num_kv_heads)
        self.num_kv_heads_per_rank = len(self.kv_heads)

        self.q_size = self.num_heads_per_rank * self.head_dim
        self.k_size = self.num_kv_heads_per_rank * self.head_dim
        self.v_size = self.num_kv_heads_per_rank * self.v_head_dim
        hidden = config.hidden_size

        # FP8 checkpoint: q/k/v stay FP8 (codes [H, N] + F32 row-group scale [H // 128, N]); o_proj stays BF16
        # like the GPU deployment (it is in ignored_layers). BF16 checkpoint: unchanged.
        self.fp8 = config.fp8_weights
        self.fp8_act = config.fp8_act
        n_qkv = self.q_size + self.k_size + self.v_size
        if self.fp8:
            self.qkv_proj_weight = nn.Parameter(torch.empty(hidden, n_qkv, dtype=torch.float8_e4m3fn),
                                                requires_grad=False)
            self.qkv_proj_scale = nn.Parameter(torch.empty(hidden // config.fp8_linear_group, n_qkv,
                                                           dtype=torch.float32), requires_grad=False)
        else:
            self.qkv_proj_weight = nn.Parameter(torch.empty(hidden, n_qkv, dtype=self.dtype), requires_grad=False)
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.num_heads_per_rank * self.v_head_dim, hidden, dtype=self.dtype), requires_grad=False)
        if d["has_sink"]:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads_per_rank, dtype=torch.float32),
                                      requires_grad=False)
        else:
            self.sinks = None

        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        tp = self.world_size
        if self.fp8:
            for prm in (self.qkv_proj_weight, self.qkv_proj_scale):
                _set_loader(prm, qkv_fp8_loader(self.num_heads, self.num_kv_heads, self.head_dim, self.v_head_dim, tp))
        else:
            _set_loader(self.qkv_proj_weight, qkv_weight_loader(
                self.num_heads, self.num_kv_heads, self.head_dim, self.v_head_dim, tp))
        _set_loader(self.o_proj_weight, o_proj_weight_loader(self.num_heads, self.v_head_dim, tp))
        if self.sinks is not None:
            _set_loader(self.sinks, sink_loader(self.num_heads, tp))

    @staticmethod
    def checkpoint_keys(layer_idx: int, has_sink: bool, fp8: bool = False) -> dict:
        """Parameter name (relative to the attention module) -> checkpoint key(s)."""
        p = f"model.layers.{layer_idx}.self_attn."
        m = {
            "qkv_proj_weight": [p + "q_proj.weight", p + "k_proj.weight", p + "v_proj.weight"],
            "o_proj_weight": p + "o_proj.weight",
        }
        if fp8:
            m["qkv_proj_scale"] = [p + "q_proj.weight_scale", p + "k_proj.weight_scale", p + "v_proj.weight_scale"]
        if has_sink:
            m["sinks"] = p + "attention_sink_bias"
        return m

    # ── helpers ──────────────────────────────────────────────────────────
    def _project(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """hidden [T, H] -> q [T, nq, 192], k [T, nkv, 192] (RoPE applied), v [T, nkv, 128] (scaled)."""
        t = hidden.shape[0]
        if self.fp8:
            qkv = fp8_linear(hidden.to(self.dtype), self.qkv_proj_weight, self.qkv_proj_scale, self.fp8_act,
                             out_dtype=self.dtype)
        else:
            qkv = torch.matmul(hidden.to(self.dtype), self.qkv_proj_weight)
        _dbg(f"l{self.layer_idx}.qkv_raw", qkv)
        # torch.tensor_split (as GPT-OSS), not torch.split: on Neuron, split_with_sizes on the last dim
        # returned wrong data (rows > 0 garbage; device bug, reproduced by tools/repro_split.py). CPU eager is fine.
        q, k, v = torch.tensor_split(qkv, [self.q_size, self.q_size + self.k_size], dim=-1)
        q = q.reshape(t, self.num_heads_per_rank, self.head_dim)
        k = k.reshape(t, self.num_kv_heads_per_rank, self.head_dim)
        v = v.reshape(t, self.num_kv_heads_per_rank, self.v_head_dim)
        if self.v_scale is not None:
            v = v * self.v_scale
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        return q, k, v

    def _write_cache(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, block_size: int):
        """k [T, nkv, 192], v [T, nkv, 128] -> paged caches [num_blocks, nkv, bs, 192] at slot_mapping.

        Padding slots from the runner are slot 0 (null block); negative slots are redirected to 0
        defensively.
        """
        nkh = self.num_kv_heads_per_rank
        slots = torch.where(slot_mapping < 0, torch.zeros_like(slot_mapping), slot_mapping).to(torch.long)
        kd, vd = self.k_cache.shape[-1], self.v_cache.shape[-1]   # 192 / 192, or 192 / 128 (MIMO_KV_VNOPAD)
        if vd > self.v_head_dim:
            v = torch.nn.functional.pad(v, (0, vd - self.v_head_dim))
        if kd > self.head_dim:
            k = torch.nn.functional.pad(k, (0, kd - self.head_dim))
        k_flat = k.transpose(0, 1).reshape(-1, kd).to(self.k_cache.dtype)
        v_flat = v.transpose(0, 1).reshape(-1, vd).to(self.v_cache.dtype)
        n = slots.shape[0]
        blk = (slots // block_size).repeat(nkh)
        off = (slots % block_size).repeat(nkh)
        head = torch.arange(nkh, dtype=torch.long, device=slots.device).repeat_interleave(n)
        if self.kv_write_nki:
            # docs/design-notes.md#kv-write: in-place scatter DMA (NKI kernel, cache aliased in/out) instead of a functional index_put_
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            from .nki_kv import kv_write
            rows = ((blk * nkh + head) * block_size + off).to(torch.int32).to(torch.uint32).reshape(-1, 1)
            wrap_nki(kv_write)[2](k_cache=self.k_cache, v_cache=self.v_cache, k_rows=k_flat.contiguous(),
                                  v_rows=v_flat.contiguous(), idx=rows)
            return
        self.k_cache.index_put_((blk, head, off), k_flat)
        self.v_cache.index_put_((blk, head, off), v_flat)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor], attn_metadata: dict) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        if meta["max_query_len"] <= meta["decode_token_threshold"]:
            out = self.forward_decode(hidden_states, positions, position_embeddings, meta)
            if self.world_size > 1:
                out = _all_reduce(self.tp_group, out)
            return out
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = self.forward_prefill(hidden_states, positions, position_embeddings, meta)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    def forward_prefill(self, hidden: torch.Tensor, positions: torch.Tensor,
                        position_embeddings: tuple[torch.Tensor, torch.Tensor], meta: dict) -> torch.Tensor:
        """Single-request prefill. hidden [T, H] (full sequence or one segment) -> O-proj partial [T, H].

        Segmented prefill (``meta["kv_segment_size"]`` != 0): T is one segment; earlier tokens are read
        from the paged cache via ``segmented_prefill_attention``.
        """
        seg = meta.get("kv_segment_size")
        if seg:
            # MIMO_PREFILL_BUCKETS: a prefill bucket may be smaller than the segment size (T < seg)
            if hidden.shape[0] > seg:
                raise ValueError(f"MiMoV2 Neuron: segmented prefill expects T <= kv_segment_size "
                                 f"(T={hidden.shape[0]}, kv_segment_size={seg})")
            return self._forward_prefill_segment(hidden, position_embeddings, meta)
        cos, sin = position_embeddings
        tag = f"l{self.layer_idx}.prefill"
        _dbg(f"{tag}.attn_in", hidden)
        _dbg(f"{tag}.cos", cos)
        q, k, v = self._project(hidden, cos, sin)
        _dbg(f"{tag}.q", q)
        _dbg(f"{tag}.k", k)
        _dbg(f"{tag}.v", v)
        self._write_cache(k, v, meta["slot_mapping"], meta["block_size"])
        attn = prefill_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), self.scaling,
                                 sliding_window=self.sliding_window, sink=self.sinks)   # [T, nq*128]
        _dbg(f"{tag}.attn", attn)
        out = torch.matmul(attn, self.o_proj_weight)
        _dbg(f"{tag}.oproj_partial", out)
        return out

    def _forward_prefill_segment(self, hidden: torch.Tensor, position_embeddings, meta: dict) -> torch.Tensor:
        cos, sin = position_embeddings
        q, k, v = self._project(hidden, cos, sin)
        self._write_cache(k, v, meta["slot_mapping"], meta["block_size"])
        attn = segmented_prefill_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), self.k_cache, self.v_cache,
            meta["block_table_tensor"], meta["cached_seq_len"], self.scaling, self.v_head_dim,
            meta["block_size"], self.prefill_key_block, sliding_window=self.sliding_window,
            sink=self.sinks)                                                              # [T, nq*128]
        return torch.matmul(attn, self.o_proj_weight)

    def forward_decode(self, hidden: torch.Tensor, positions: torch.Tensor,
                       position_embeddings: tuple[torch.Tensor, torch.Tensor], meta: dict) -> torch.Tensor:
        """hidden [B, H] (one token per sequence) -> O-proj partial [B, H]."""
        cos, sin = position_embeddings
        q, k, v = self._project(hidden, cos, sin)
        self._write_cache(k, v, meta["slot_mapping"], meta["block_size"])
        offset = meta.get("swa_kv_pos_offset") if self.sliding_window is not None else None
        attn = decode_attention(q, self.k_cache, self.v_cache, meta["block_table_tensor"], positions,
                                self.scaling, self.v_head_dim, sliding_window=self.sliding_window,
                                sink=self.sinks, kv_pos_offset=offset,
                                key_block=self.decode_key_block)                         # [B, nq*128]
        return torch.matmul(attn, self.o_proj_weight)


# =============================================================================
# Dense FFN (layer 0): SwiGLU, intermediate sharded by TP (llama3 MLP pattern)
# =============================================================================
class MiMoV2DenseMLP(nn.Module):
    """``down(silu(h @ gate) * (h @ up))``; prefill all-gather -> ``NF.mlp`` -> reduce-scatter,
    decode ``NF.mlp`` -> all-reduce.

    Parameters: gate_proj_weight / up_proj_weight ``[H, I/tp]``, down_proj_weight ``[I/tp, H]``.
    """

    def __init__(self, config: MiMoV2Config, layer_idx: int, tp_group=None):
        super().__init__()
        from vllm_neuron.utils.weight_loader import sharding_weight_loader

        if tp_group is None:
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.layer_idx = layer_idx
        tp = self.world_size
        if config.intermediate_size % tp:
            raise ValueError(f"MiMoV2 Neuron: intermediate_size ({config.intermediate_size}) must be "
                             f"divisible by tensor_parallel_size ({tp})")
        self.intermediate_size_per_rank = config.intermediate_size // tp
        h, i, dt = config.hidden_size, self.intermediate_size_per_rank, config.torch_dtype
        self.fp8 = config.fp8_weights
        self.fp8_act = config.fp8_act
        if self.fp8:   # gate / up / down stay FP8 (block-scaled in the checkpoint), computed W8A8 (fp8_linear)
            g = config.fp8_linear_group
            f8 = torch.float8_e4m3fn
            self.gate_proj_weight = nn.Parameter(torch.empty(h, i, dtype=f8), requires_grad=False)
            self.up_proj_weight = nn.Parameter(torch.empty(h, i, dtype=f8), requires_grad=False)
            self.down_proj_weight = nn.Parameter(torch.empty(i, h, dtype=f8), requires_grad=False)
            self.gate_proj_scale = nn.Parameter(torch.empty(h // g, i, dtype=torch.float32), requires_grad=False)
            self.up_proj_scale = nn.Parameter(torch.empty(h // g, i, dtype=torch.float32), requires_grad=False)
            self.down_proj_scale = nn.Parameter(torch.empty(i // g, h, dtype=torch.float32), requires_grad=False)
            for prm in (self.gate_proj_weight, self.up_proj_weight):
                _set_loader(prm, rowgroup_shard_loader(True, i, tp, g))
            for prm in (self.gate_proj_scale, self.up_proj_scale):
                _set_loader(prm, rowgroup_shard_loader(True, i, tp, g, is_scale=True))
            _set_loader(self.down_proj_weight, rowgroup_shard_loader(False, i, tp, g))
            _set_loader(self.down_proj_scale, rowgroup_shard_loader(False, i, tp, g, is_scale=True))
            return
        self.gate_proj_weight = nn.Parameter(torch.empty(h, i, dtype=dt), requires_grad=False)
        self.up_proj_weight = nn.Parameter(torch.empty(h, i, dtype=dt), requires_grad=False)
        self.down_proj_weight = nn.Parameter(torch.empty(i, h, dtype=dt), requires_grad=False)
        gate_up_loader = sharding_weight_loader(shard_dim=1, shard_size=i, num_shards=tp, is_storage_transposed=True)
        down_loader = sharding_weight_loader(shard_dim=0, shard_size=i, num_shards=tp, is_storage_transposed=True)
        _set_loader(self.gate_proj_weight, gate_up_loader)
        _set_loader(self.up_proj_weight, gate_up_loader)
        _set_loader(self.down_proj_weight, down_loader)

    @staticmethod
    def checkpoint_keys(layer_idx: int, fp8: bool = False) -> dict:
        p = f"model.layers.{layer_idx}.mlp."
        m = {"gate_proj_weight": p + "gate_proj.weight", "up_proj_weight": p + "up_proj.weight",
             "down_proj_weight": p + "down_proj.weight"}
        if fp8:
            m.update({n.replace("_weight", "_scale"): k + "_scale" for n, k in list(m.items())})
        return m

    def forward_local(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [T, H] (already normalised, full sequence / batch) -> partial output [T, H]."""
        import vllm_neuron.functional as NF

        if self.fp8:   # GPU: merged gate_up W8A8, SiluAndMul in BF16, down W8A8 (input re-quantised)
            dt = hidden.dtype
            g = fp8_linear(hidden, self.gate_proj_weight, self.gate_proj_scale, self.fp8_act, out_dtype=dt)
            u = fp8_linear(hidden, self.up_proj_weight, self.up_proj_scale, self.fp8_act, out_dtype=dt)
            a = torch.nn.functional.silu(g) * u
            return fp8_linear(a, self.down_proj_weight, self.down_proj_scale, self.fp8_act, out_dtype=dt)
        return NF.mlp(hidden, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, is_decode: bool,
                rank: torch.Tensor | None = None) -> torch.Tensor:
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = self.forward_local(hidden_states)
        if self.world_size > 1:
            out = _all_reduce(self.tp_group, out) if is_decode else self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# MoE (layers 1..47): 256 routed experts, top-8, sigmoid + noaux_tc routing, SiLU experts
# =============================================================================
def _torch_local_experts(hidden: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                         local_affinities: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch local experts (CPU path, ``moe_tkg`` oracle, last-resort device fallback).

    hidden [T, H]; gate_up [E_L, H, 2, I]; down [E_L, I, H]; local_affinities [T, E_L] fp32 (0 for
    experts not selected by the token). Matches HF ``MiMoV2MoE.moe``: BF16 expert MLP, output times the
    FP32 routing weight, FP32 accumulation, cast back to ``hidden.dtype`` (POST_SCALE semantics).
    """
    out = torch.zeros(hidden.shape[0], down.shape[-1], dtype=torch.float32, device=hidden.device)
    h = hidden.to(gate_up.dtype)
    for e in range(gate_up.shape[0]):
        g = torch.matmul(h, gate_up[e, :, 0, :])
        u = torch.matmul(h, gate_up[e, :, 1, :])
        y = torch.matmul(torch.nn.functional.silu(g) * u, down[e])
        out = out + y.to(torch.float32) * local_affinities[:, e:e + 1].to(torch.float32)
    return out.to(hidden.dtype)


def _ep_settings(tp_group) -> dict:
    """EP degree / rank / EP-TP sub-group from the Neuron parallel state (GPT-OSS ``GptOssExperts``)."""
    from vllm.config import get_current_vllm_config

    vc = get_current_vllm_config()
    if vc.parallel_config.enable_expert_parallel:
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
            get_neuron_ep_tp_group,
        )

        return {"degree": get_neuron_ep_degree(), "rank": get_neuron_ep_rank(), "tp_group": get_neuron_ep_tp_group()}
    return {"degree": 1, "rank": 0, "tp_group": tp_group}


class MiMoV2MoE(nn.Module):
    """Routed-expert FFN with EP (linear placement: EP rank k owns experts [k*E_L, (k+1)*E_L)) and the
    intermediate dim sharded over the EP-TP sub-group (``tp_degree``; 1 under full EP).

    Parameters:
      router_weight            [E, H] BF16 (replicated)
      e_score_correction_bias  [E] FP32 (replicated; must stay FP32 as on GPU)
      gate_up_proj_weight      [E_L, H, 2, I/tp_degree] BF16 ([..., 0, :] gate, [..., 1, :] up)
      down_proj_weight         [E_L, I/tp_degree, H] BF16
    """

    BLOCK_SIZE = 256

    def __init__(self, config: MiMoV2Config, layer_idx: int, tp_group=None, ep: dict | None = None):
        super().__init__()
        if tp_group is None:
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.layer_idx = layer_idx
        if ep is None:
            ep = _ep_settings(tp_group)
        self.ep_degree = ep["degree"]
        self.ep_rank = ep["rank"]
        self.ep_tp_group = ep["tp_group"]
        self.tp_degree = self.ep_tp_group.world_size
        self.ep_tp_rank = self.ep_tp_group.rank_in_group

        e_total = config.n_routed_experts
        if e_total % self.ep_degree:
            raise ValueError(f"MiMoV2 Neuron: n_routed_experts ({e_total}) must be divisible by the EP degree "
                             f"({self.ep_degree})")
        if config.moe_intermediate_size % self.tp_degree:
            raise ValueError(f"MiMoV2 Neuron: moe_intermediate_size ({config.moe_intermediate_size}) must be "
                             f"divisible by the MoE TP degree ({self.tp_degree})")
        self.num_experts = e_total
        self.num_local_experts = e_total // self.ep_degree
        self.local_experts = range(self.ep_rank * self.num_local_experts, (self.ep_rank + 1) * self.num_local_experts)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree
        self.decode_moe_impl = config.decode_moe_impl
        # Under full EP inside the TP group the runner's ``rank`` tensor equals the EP rank; using it keeps
        # the traced graph identical on every rank (the runner passes rank as a tensor for that reason).
        self._rank_is_ep_rank = (self.ep_degree == self.world_size and self.tp_degree == 1
                                 and self.ep_rank == self.rank)

        dt = config.torch_dtype
        e_l, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size_per_rank
        self.router_weight = nn.Parameter(torch.empty(e_total, h, dtype=dt), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(e_total, dtype=torch.float32), requires_grad=False)
        self.gate_up_proj_weight = nn.Parameter(torch.empty(e_l, h, 2, i, dtype=dt), requires_grad=False)
        self.down_proj_weight = nn.Parameter(torch.empty(e_l, i, h, dtype=dt), requires_grad=False)

        i_start = self.ep_tp_rank * i
        _set_loader(self.e_score_correction_bias,
                    fp32_vector_loader(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"))
        # FP8 checkpoint (docs/design-notes.md#fp8-alignment): experts stay FP8 in HBM (MXFP4 -> FP8 codes, merged losslessly to one scale per
        # output channel at load) and are expanded to BF16 inside the graph right before the BF16 expert kernels
        # (exact: E2M1 value x 2^k), i.e. the GPU H100 path W4A16 numerically. Router weight / bias keep BF16 / FP32.
        self.fp8 = config.fp8_weights
        self.fp8_expert_impl = getattr(config, "fp8_expert_impl", "expand")
        self.moe_cte_impl = getattr(config, "moe_cte_impl", "shard_on_i")
        if self.fp8:
            g = config.fp8_expert_group
            f8 = torch.float8_e4m3fn
            self.gate_up_proj_weight = nn.Parameter(torch.empty(e_l, h, 2, i, dtype=f8), requires_grad=False)
            self.down_proj_weight = nn.Parameter(torch.empty(e_l, i, h, dtype=f8), requires_grad=False)
            self.gate_up_proj_scale = nn.Parameter(torch.empty(e_l, 2, i, dtype=torch.float32), requires_grad=False)
            self.down_proj_scale = nn.Parameter(torch.empty(e_l, h, dtype=torch.float32), requires_grad=False)
            _set_loader(self.gate_up_proj_weight, expert_gate_up_fp8_loader(e_l, i_start, i, g, "codes"))
            _set_loader(self.gate_up_proj_scale, expert_gate_up_fp8_loader(e_l, i_start, i, g, "scale"))
            _set_loader(self.down_proj_weight, expert_down_fp8_loader(e_l, i_start, i, g, "codes"))
            _set_loader(self.down_proj_scale, expert_down_fp8_loader(e_l, i_start, i, g, "scale"))
        else:
            _set_loader(self.gate_up_proj_weight, expert_gate_up_loader(e_l, i_start, i))
            _set_loader(self.down_proj_weight, expert_down_loader(e_l, i_start, i))

    def checkpoint_keys(self) -> dict:
        p = f"model.layers.{self.layer_idx}.mlp."
        m = {"router_weight": p + "gate.weight", "e_score_correction_bias": p + "gate.e_score_correction_bias"}
        if self.fp8:
            k = expert_fp8_checkpoint_keys(self.layer_idx, self.local_experts)
            k["gate_up_proj_scale"], k["down_proj_scale"] = k["gate_up_proj_weight"], k["down_proj_weight"]
            m.update(k)
        else:
            m.update(expert_checkpoint_keys(self.layer_idx, self.local_experts))
        return m

    def expert_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [E_L, H, 2, I], down [E_L, I, H]) in BF16 for the expert kernels; FP8 weights are expanded
        here (codes x per-channel power-of-two scale, exact in BF16)."""
        if not self.fp8:
            return self.gate_up_proj_weight, self.down_proj_weight
        bf = torch.bfloat16
        gu = self.gate_up_proj_weight.to(bf) * self.gate_up_proj_scale.to(bf).unsqueeze(1)
        dn = self.down_proj_weight.to(bf) * self.down_proj_scale.to(bf).unsqueeze(1)
        return gu, dn

    # ── helpers ──────────────────────────────────────────────────────────
    def route(self, hidden: torch.Tensor):
        """hidden [T, H] -> (dense affinities [T, E] fp32, topk_idx [T, k])."""
        dense, idx, _ = noaux_tc_route(hidden, self.router_weight, self.e_score_correction_bias, self.top_k,
                                       norm_topk_prob=self.norm_topk_prob,
                                       routed_scaling_factor=self.routed_scaling_factor)
        if _DEBUG_CAPTURE:
            choice = torch.sigmoid(torch.matmul(hidden.to(torch.float32), self.router_weight.to(torch.float32).t()))
            _dbg(f"l{self.layer_idx}.t{hidden.shape[0]}.moe_choice", choice + self.e_score_correction_bias)
            _dbg(f"l{self.layer_idx}.t{hidden.shape[0]}.moe_idx", idx.to(torch.int32))
        return dense, idx

    def _ep_rank_tensor(self, rank: torch.Tensor | None, device) -> torch.Tensor:
        if rank is not None and self._rank_is_ep_rank:
            return rank.reshape(1, 1).to(torch.int32)
        return torch.tensor([[self.ep_rank]], dtype=torch.int32, device=device)

    def local_affinities(self, dense: torch.Tensor, rank: torch.Tensor | None = None) -> torch.Tensor:
        """[T, E] -> [T, E_L] affinities of this rank's experts."""
        if self.ep_degree == 1:
            return dense
        import vllm_neuron.functional as NF

        e_l = self.num_local_experts
        base = self._ep_rank_tensor(rank, dense.device).reshape(1) * e_l
        idx = base + torch.arange(e_l, dtype=torch.int32, device=dense.device)
        return NF.get_local_expert_affinities(dense, idx)

    def _blockwise_cte(self, hidden: torch.Tensor, local_aff: torch.Tensor, padding_mask, rank) -> torch.Tensor:
        """Blockwise ``NF.moe_cte`` (GPT-OSS ``forward_prefill`` call, SiLU, no bias / clamp)."""
        import nki.language as nl
        import vllm_neuron.functional as NF
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        (aff_masked, token_position_to_id, block_to_expert, conditions) = NF.build_blockwise_mapping(
            expert_affinities=local_aff,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.BLOCK_SIZE,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
            rank=rank,
        )
        quant = {}
        implementation = MoECTEImplementation.shard_on_block
        if self.fp8 and self.fp8_expert_impl == "row" and can_run_kernel(hidden):
            # docs/design-notes.md#fp8-alignment: FP8 codes into moe_cte (weight-only FP8: BF16 hidden, per-output-channel dequant after the
            # matmul); scales in the kernel layout [E_L, 1, 2*I] (gate | up) / [E_L, 1, H]. shard_on_block only
            # has a placeholder for FP8 scales (nkilib bwmm_shard_on_block.py: gup_scale = None), shard_on_i
            # applies them, so the FP8 path uses shard_on_i.
            implementation = MoECTEImplementation.shard_on_i
            gate_up, down = self.gate_up_proj_weight, self.down_proj_weight
            e_l, _, two, i = gate_up.shape
            quant = dict(gate_up_proj_scale=self.gate_up_proj_scale.reshape(e_l, 1, two * i),
                         down_proj_scale=self.down_proj_scale.unsqueeze(1))
            if self.moe_cte_impl == "hybrid":
                # docs/design-notes.md#moe-hybrid: shard_on_i loops statically over all N blocks (N = ceil((T*min(k, E_L) -
                # E_L + 1) / B) + E_L - 1 = 67 for T=4096 under EP=64), but a rank only gets ~T*k/EP routed tokens
                # (a handful of non-empty blocks), so ~90 % of the block computations ran on padding. The hybrid
                # variant runs the same per-block code (compute_one_block) in a runtime loop whose trip count is
                # sum(conditions) = the number of non-empty blocks (they come first: blocks are laid out expert by
                # expert, padding blocks last). num_static_block=0: no static part, so the trip count is never
                # negative. The kernel wants conditions [N+1] with a trailing 0.
                implementation = MoECTEImplementation.shard_on_i_hybrid
                quant["num_static_block"] = 0
                conditions = torch.cat([conditions.reshape(-1).to(torch.int32),
                                        torch.zeros(1, dtype=torch.int32, device=conditions.device)])
        else:
            gate_up, down = self.expert_weights()
        return NF.moe_cte(
            implementation=implementation,
            conditions=conditions,
            hidden_states=hidden,
            expert_affinities_masked=aff_masked,
            gate_up_proj_weight=gate_up,
            down_proj_weight=down,
            gate_and_up_proj_bias=None,
            down_proj_bias=None,
            activation_function=ActFnType.SiLU,
            block_size=self.BLOCK_SIZE,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            **quant,
        )

    def experts_prefill(self, hidden: torch.Tensor, dense: torch.Tensor, positions: torch.Tensor | None,
                        rank: torch.Tensor | None = None) -> torch.Tensor:
        """Full-sequence hidden [T, H] + dense affinities [T, E] -> this rank's partial output [T, H]."""
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            padding_mask = torch.arange(positions.shape[0], device=positions.device) <= last_real_idx
        return self._blockwise_cte(hidden, self.local_affinities(dense, rank), padding_mask, rank)

    def experts_decode(self, hidden: torch.Tensor, dense: torch.Tensor, topk_idx: torch.Tensor,
                       rank: torch.Tensor | None = None) -> torch.Tensor:
        """hidden [B, H] (replicated) -> this rank's partial output [B, H] via ``decode_moe_impl``."""
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        impl = self.decode_moe_impl
        if impl == "tkg" and can_run_kernel(hidden):
            import vllm_neuron.functional as NF
            from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

            if self.fp8 and self.fp8_expert_impl == "row":
                # docs/design-notes.md#fp8-alignment: FP8 codes straight from HBM into the kernel (ROW quantisation: BF16 hidden x FP8 weight
                # on the PE, FP32 PSUM, then x per-output-channel scale) -- no BF16 copy of the experts.
                # NF.moe_tkg refuses float8 weights in its Python dtype check (only bf16 / MX uint16 / uint32),
                # while nkilib moe_tkg supports FP8 ROW; so call the same wrapped kernel directly (same HOP and
                # grid [2] as NF.moe_tkg, installed files unchanged).
                from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
                from vllm_neuron.functional.moe.moe_tkg_wrapper import moe_tkg_wrapper

                return wrap_nki(moe_tkg_wrapper)[2](
                    hidden_input=hidden,
                    expert_gate_up_weights=self.gate_up_proj_weight,
                    expert_down_weights=self.down_proj_weight,
                    expert_affinities=dense,
                    expert_index=topk_idx.to(torch.int32),
                    is_all_expert=True,
                    rank_id=self._ep_rank_tensor(rank, hidden.device),
                    expert_gate_up_weights_scale=self.gate_up_proj_scale,
                    expert_down_weights_scale=self.down_proj_scale,
                    activation_fn=ActFnType.SiLU,
                    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                )
            gate_up, down = self.expert_weights()
            return NF.moe_tkg(
                hidden_input=hidden,
                expert_gate_up_weights=gate_up,
                expert_down_weights=down,
                expert_affinities=dense,
                expert_index=topk_idx.to(torch.int32),
                is_all_expert=True,
                rank_id=self._ep_rank_tensor(rank, hidden.device),
                activation_fn=ActFnType.SiLU,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            )
        if impl == "cte":
            # hidden is already replicated in the TP group during decode -> no all-gather; positions in a
            # decode batch are not monotonic, so no positions-based padding mask (padding rows only add
            # their own, discarded, outputs).
            return self._blockwise_cte(hidden, self.local_affinities(dense, rank), None, rank)
        gate_up, down = self.expert_weights()
        return _torch_local_experts(hidden, gate_up, down, self.local_affinities(dense, rank))

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, is_decode: bool,
                rank: torch.Tensor | None = None) -> torch.Tensor:
        """hidden_states already normalised by post_attention_layernorm.

        Prefill: SP shard [T/tp, H] -> route locally -> all-gather affinities and hidden -> experts ->
        reduce-scatter back to SP. Decode: replicated [B, H] -> route -> experts -> all-reduce.
        """
        dense, idx = self.route(hidden_states)
        if is_decode:
            out = self.experts_decode(hidden_states, dense, idx, rank)
            return _all_reduce(self.tp_group, out) if self.world_size > 1 else out
        if self.world_size > 1:
            dense = self.tp_group.all_gather(dense, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = self.experts_prefill(hidden_states, dense, positions, rank)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# Decoder layer / backbone / LM head
# =============================================================================
def _is_decode(attn_metadata: dict, layer_idx: int = 0) -> bool:
    meta = attn_metadata[f"layers.{layer_idx}.self_attn"]
    return meta["max_query_len"] <= meta["decode_token_threshold"]


class MiMoV2DecoderLayer(nn.Module):
    """residual + attn(input_layernorm(h)); residual + mlp(post_attention_layernorm(h)) (HF order)."""

    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = config.is_swa(layer_idx)
        self.is_moe = config.is_moe(layer_idx)
        eps, dt = config.layernorm_epsilon, config.torch_dtype
        self.input_layernorm = MiMoV2RMSNorm(config.hidden_size, eps, dt)
        self.post_attention_layernorm = MiMoV2RMSNorm(config.hidden_size, eps, dt)
        self.self_attn = MiMoV2Attention(config, layer_idx)
        self.mlp = MiMoV2MoE(config, layer_idx) if self.is_moe else MiMoV2DenseMLP(config, layer_idx)

    def checkpoint_keys(self) -> dict:
        """Parameter name (relative to ``model.layers.{i}``) -> checkpoint key(s)."""
        p = f"model.layers.{self.layer_idx}."
        m = {"input_layernorm.weight": p + "input_layernorm.weight",
             "post_attention_layernorm.weight": p + "post_attention_layernorm.weight"}
        attn = MiMoV2Attention.checkpoint_keys(self.layer_idx, self.self_attn.sinks is not None, self.self_attn.fp8)
        m.update({"self_attn." + k: v for k, v in attn.items()})
        mlp = self.mlp.checkpoint_keys() if self.is_moe else \
            MiMoV2DenseMLP.checkpoint_keys(self.layer_idx, self.mlp.fp8)
        m.update({"mlp." + k: v for k, v in mlp.items()})
        return m

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata, rank=None):
        is_decode = _is_decode(attn_metadata, self.layer_idx)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, position_embeddings, attn_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, positions=positions, is_decode=is_decode, rank=rank)
        return residual + hidden_states


class MiMoV2Model(nn.Module):
    """Embedding (vocab-sharded, SP scatter in prefill) -> layers -> final norm -> SP all-gather."""

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        from vllm.distributed.parallel_state import get_tp_group
        from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.embed_tokens = VocabDimShardedEmbedding(vocab_size=config.vocab_size, embed_dim=config.hidden_size,
                                                     dtype=config.torch_dtype, tp_group=self.tp_group.device_group)
        self.layers = nn.ModuleList([MiMoV2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MiMoV2RMSNorm(config.hidden_size, config.layernorm_epsilon, config.torch_dtype)
        # one RoPE table per layer type (GA theta 1e7, SWA theta 1e4); both rotate the first 64 dims
        self.rotary_ga = None
        self.rotary_swa = None
        for i in range(config.num_hidden_layers):
            d = config.attn_dims(i)
            if config.is_swa(i) and self.rotary_swa is None:
                self.rotary_swa = MiMoV2RotaryEmbedding(d["rope_theta"], config.rope_dim(i))
            if not config.is_swa(i) and self.rotary_ga is None:
                self.rotary_ga = MiMoV2RotaryEmbedding(d["rope_theta"], config.rope_dim(i))

    def forward(self, input_ids, positions, attn_metadata, rank=None, inputs_embeds=None, is_token_ids=None,
                vision_embedding_blocks=None, vision_positions=None):
        import vllm_neuron.functional as NF

        is_prefill = not _is_decode(attn_metadata)
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        if is_prefill and self.world_size > 1 and inputs_embeds is not None and is_token_ids is not None:
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start:start + local_len]
            is_token_ids = is_token_ids[start:start + local_len]
        hidden_states = NF.merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids)
        if is_prefill and vision_embedding_blocks is not None and vision_positions is not None:
            # multimodal (MiMoV2OmniForCausalLM): scatter the encoder-cache rows into this
            # SP rank's slice of the embeddings; text-only calls never take this branch
            from .model_mm import merge_vision_embeddings

            hidden_states = merge_vision_embeddings(hidden_states, vision_embedding_blocks, vision_positions,
                                                    rank if rank is not None else self.rank)

        dt = hidden_states.dtype
        pe_ga = self.rotary_ga(positions, dt) if self.rotary_ga is not None else None
        pe_swa = self.rotary_swa(positions, dt) if self.rotary_swa is not None else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, pe_swa if layer.is_swa else pe_ga, attn_metadata,
                                  rank=rank)
        hidden_states = self.norm(hidden_states)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


def _describe_metadata(attn_metadata: dict, positions: torch.Tensor, layer_names) -> dict:
    """Python-side summary of the runner metadata (MIMO_DEBUG_METADATA=1; design V3/V4)."""

    def desc(v):
        if isinstance(v, torch.Tensor):
            d = {"shape": tuple(v.shape), "dtype": str(v.dtype)}
            try:
                if v.numel() <= 64:
                    d["values"] = v.detach().cpu().tolist()
                elif v.dim() == 1:   # long 1-D tensors (prefill positions): ends + whether they step by 1
                    c = v.detach().cpu()
                    d["head"], d["tail"] = c[:4].tolist(), c[-4:].tolist()
                    d["step1_runs"] = int((c[1:] - c[:-1] != 1).sum()) + 1
            except Exception:  # fake / lazy tensors during tracing
                pass
            return d
        return v

    out = {"positions": desc(positions)}
    for name in layer_names:
        if name in attn_metadata:
            out[name] = {k: desc(v) for k, v in attn_metadata[name].items()}
    return out


class MiMoV2ForCausalLMNeuron(nn.Module):
    """MiMo-V2 language model + column-parallel LM head + optional on-device sampling.

    ``forward`` follows ``GptOssForCausalLM.forward`` (model_bf16.py) without Eagle3 / DP branches.
    """

    _DEBUG_MAX_PRINTS = 6   # MIMO_DEBUG_METADATA=1: metadata of the first N prefill and N decode calls

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        from vllm.distributed.parallel_state import get_tp_group
        import vllm_neuron.nn as neuron_nn

        self.config = config
        self.model = MiMoV2Model(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc is not None else None
        debug_logits_enabled = nc is not None and getattr(nc, "debug_logits_dir", None) is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits_enabled
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config, tp_group=self.tp_group.device_group)
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler

            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
        self._debug_metadata = os.environ.get("MIMO_DEBUG_METADATA") == "1"
        self._debug_printed = {"prefill": 0, "decode": 0}   # prints per phase (first calls only)
        impls = {"tkg": "NF.moe_tkg (falls back to PyTorch experts when NKI kernels cannot run)",
                 "cte": "NF.moe_cte (blockwise)", "torch": "PyTorch local experts"}
        logger.info("MiMoV2 Neuron: decode MoE implementation = %s -> %s", config.decode_moe_impl,
                    impls.get(config.decode_moe_impl, "?"))

    @classmethod
    def from_configs(cls, hf_config, neuron_config) -> "MiMoV2ForCausalLMNeuron":
        return cls(MiMoV2Config.from_configs(hf_config, neuron_config))

    @torch.no_grad()
    def forward(self, input_ids, positions, inputs_embeds=None, is_token_ids=None, attn_metadata=None,
                sampling_positions=None, sampling_params=None, spec_decode_metadata=None, logit_mask=None,
                rank=None, vision_embedding_blocks=None, vision_positions=None, **kwargs):
        positions = positions.to(torch.int32)
        is_prefill = not _is_decode(attn_metadata)
        phase = "prefill" if is_prefill else "decode"
        if self._debug_metadata and self._debug_printed[phase] < self._DEBUG_MAX_PRINTS:
            self._debug_printed[phase] += 1
            names = ["layers.0.self_attn"] + [f"layers.{i}.self_attn" for i in range(self.config.num_hidden_layers)
                                              if self.config.is_swa(i)][:1]
            logger.warning("MIMO_DEBUG_METADATA %s %s", phase, _describe_metadata(attn_metadata, positions, names))
        t = input_ids.shape[0]
        if is_prefill and ((t <= self.world_size) or (t % self.world_size != 0)):
            raise ValueError(f"Prompt Length ({t}) must be > world_size ({self.world_size}) for SP.")

        hidden_states = self.model(input_ids, positions, attn_metadata, rank=rank, inputs_embeds=inputs_embeds,
                                   is_token_ids=is_token_ids, vision_embedding_blocks=vision_embedding_blocks,
                                   vision_positions=vision_positions)
        hidden_states = torch.index_select(hidden_states, dim=0, index=sampling_positions)
        logits = self.lm_head(hidden_states.to(self.config.torch_dtype))

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        if self.on_device_sampling_config is None:
            return logits
        sampled_tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled_tokens, gathered_logits

    # ── KV cache ─────────────────────────────────────────────────────────
    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

        layers = []
        for i, layer in enumerate(self.model.layers):
            a = layer.self_attn
            layers.append(LayerSpec(name=f"layers.{i}.self_attn", num_kv_heads=a.num_kv_heads_per_rank,
                                    head_size=((a.head_dim + a.v_head_dim) // 2 if a.kv_vnopad
                                               else a.cache_head_dim), dtype=a.dtype,
                                    sliding_window_size=a.sliding_window, chunk_size=None))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise Exception(f"KV cache for layer {name} not initialized")
            a = layer.self_attn
            if a.kv_vnopad:
                a.k_cache, a.v_cache = _split_kv_storage(kv_caches[name][0], kv_caches[name][1], a.head_dim,
                                                         a.v_head_dim)
            else:
                a.k_cache = kv_caches[name][0]
                a.v_cache = kv_caches[name][1]

    # ── weights ──────────────────────────────────────────────────────────
    def weight_mappings(self) -> dict:
        """Parameter name -> checkpoint key(s) for every parameter of this rank."""
        m = {"model.embed_tokens.weight": "model.embed_tokens.weight", "model.norm.weight": "model.norm.weight",
             "lm_head.weight": "lm_head.weight"}
        for i, layer in enumerate(self.model.layers):
            for k, v in layer.checkpoint_keys().items():
                m[f"model.layers.{i}.{k}"] = v
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        mappings = self.weight_mappings()
        params = dict(self.named_parameters())
        unmapped = sorted(set(params) - set(mappings))
        if unmapped:
            raise KeyError(f"MiMoV2 Neuron: parameters without a checkpoint mapping: {unmapped[:8]}")
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        names = set(checkpoint.get_tensor_names())
        missing = [(p, k) for p, ks in mappings.items() for k in (ks if isinstance(ks, list) else [ks])
                   if k not in names]
        if missing:
            raise KeyError(f"MiMoV2 Neuron: {len(missing)} checkpoint key(s) missing in {checkpoint_path}, "
                           f"e.g. parameter {missing[0][0]} <- {missing[0][1]}")
        logger.info("MiMoV2 Neuron: loading %d parameters (tp_rank=%d/%d)", len(params), self.rank, self.world_size)
        state_dict = checkpoint.load_sharded_pipelined(self.rank, self.world_size, self, mappings, device).state_dict
        for name, param in params.items():
            t = state_dict.get(name)
            if t is None:
                raise KeyError(f"MiMoV2 Neuron: parameter {name} was not loaded (checkpoint key {mappings[name]})")
            if tuple(t.shape) != tuple(param.shape):
                raise ValueError(f"MiMoV2 Neuron: shape mismatch for {name} <- {mappings[name]}: "
                                 f"checkpoint {tuple(t.shape)} vs parameter {tuple(param.shape)}")
            if name.endswith("e_score_correction_bias") and t.dtype != torch.float32:
                raise ValueError(f"MiMoV2 Neuron: {name} must be FP32 after loading, got {t.dtype}")
        self.load_state_dict(state_dict, strict=False, assign=True)


def _split_kv_storage(k0: torch.Tensor, v0: torch.Tensor, k_dim: int, v_dim: int):
    """MIMO_KV_VNOPAD (docs/design-notes.md#v-no-pad): the runner's per-layer cache is one contiguous (2, nb, H, bs, (k_dim + v_dim) / 2)
    tensor handed over as its two halves ``k0`` / ``v0``. Re-view the same storage as
    K [nb, H, bs, k_dim] followed by V [nb, H, bs, v_dim] (same bytes, no copy)."""
    nb, h, bs, half = k0.shape
    if 2 * half != k_dim + v_dim or tuple(v0.shape) != tuple(k0.shape):
        raise ValueError(f"MiMoV2 Neuron: unexpected KV cache shapes {tuple(k0.shape)} / {tuple(v0.shape)}")
    per = nb * h * bs * half
    if not (k0.is_contiguous() and v0.is_contiguous() and v0.storage_offset() == k0.storage_offset() + per):
        raise ValueError("MiMoV2 Neuron: KV cache halves are not one contiguous tensor; cannot use MIMO_KV_VNOPAD")
    flat = k0.as_strided((2 * per,), (1,), k0.storage_offset())
    nk = nb * h * bs * k_dim
    return flat[:nk].view(nb, h, bs, k_dim), flat[nk:].view(nb, h, bs, v_dim)
