# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vllm-project/vllm-neuron (Apache-2.0): vllm_neuron/model/qwen3_vl/vision_encoder_bf16.py
#   (block-packed layout, vision TP / DP groups, encoder-cache scatter); modified. The maths follows the
#   XiaomiMiMo/MiMo-V2.6-Flash-MOPD remote code (modeling_mimo_v2.py, revision b12f557;
#   Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team, Apache-2.0 file header). See NOTICE.
"""MiMo-V2 vision transformer (ViT, 681M, 28 layers) for vllm-neuron (docs/design-notes.md#vision-bucket).

Structure follows ``vllm_neuron/model/qwen3_vl/vision_encoder_bf16.py`` (block-packed layout
``[num_blocks, block_size, ...]``, separate vision TP / DP groups, merger output scatter-written into the
encoder cache), but the maths follows HF ``MiMoVisionTransformer`` (``reference/hf/modeling_mimo_v2.py``):

  * RMSNorm (eps 1e-6) and a SwiGLU MLP (gate / up / down all with bias), intermediate 4608;
  * GQA attention: 32 Q heads, 8 KV heads, head_dim 64, fused qkv ``[Q 2048 | K 512 | V 512]`` with bias;
  * 2D RoPE only (16 row + 16 column frequencies, ``cat(emb, emb)`` -> 64), no learned position embedding;
  * attention is confined to a *time group* (each ``grid_t`` slice = 2 frames, HF ``cu_seqlens``);
    layers 0 / 9 / 18 / 27 are full attention inside the group, the other 24 a band window ``|i - j| <= 64``
    with a per-head sink added (additive bias) to the score of the group's first key;
  * ``vit_window_attn_types == 1`` layers run in "column order": before a run of type-1 layers the tokens are
    permuted (2x2 merge units transposed per time group, HF ``get_window_index_1d(col=True)``) and the
    column-order RoPE table is used; the permutation is undone when the run ends;
  * merger: LayerNorm (weight only) -> Linear(5120, 5120) -> GELU(erf) -> Linear(5120, 4096), no biases.

Everything that depends on the image sizes is computed on the CPU (:func:`block_inputs`, :func:`pack_items`)
and passed in as static-shape tensors: two RoPE tables (row / column order), the column-order gather index and
its inverse, and per-row ``[min, max)`` key bounds for full-attention and window layers. Because the column
permutation only moves tokens inside their time group, the group boundaries -- and therefore both bound sets and
the sink key (= the group start, ``full_min``) -- are the same in row and column order.

Padding rows (block tail) get ``[p, p + 1)`` bounds (attend to themselves only, no NaN) and an identity
permutation; their outputs are dropped by :func:`unpack_merged`.

Attention is plain PyTorch with an FP32 softmax (the NKI flash-attention kernel has no sink bias; speed-only).
``MIMO_VIT_SINKS=0`` drops the sinks (vLLM's GPU implementation ignores them; docs/design-notes.md#vit-sinks) for comparisons.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .weight_loaders import kv_heads_for_rank, o_proj_weight_loader, sink_loader, vision_qkv_loader


# =============================================================================
# Config
# =============================================================================
@dataclass
class MiMoVisionConfig:
    hidden_size: int = 1280
    intermediate_size: int = 4608
    out_hidden_size: int = 4096
    depth: int = 28
    num_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_sink: bool = True
    window: int = 64
    fullatt_block_indexes: list = field(default_factory=lambda: [0, 9, 18, 27])
    vit_window_attn_types: list = field(default_factory=list)

    @classmethod
    def from_hf(cls, vc) -> "MiMoVisionConfig":
        """``vc``: HF ``vision_config`` (dict or attribute object), same defaults as HF ``MiMoVisionTransformer``."""
        g = (lambda k, d=None: vc.get(k, d)) if isinstance(vc, dict) else (lambda k, d=None: getattr(vc, k, d))
        depth = g("depth")
        return cls(
            hidden_size=g("hidden_size"), intermediate_size=g("intermediate_size"),
            out_hidden_size=g("out_hidden_size"), depth=depth, num_heads=g("num_heads"),
            num_key_value_heads=g("num_key_value_heads", g("num_heads")), head_dim=g("qk_channels", 64),
            patch_size=g("patch_size"), temporal_patch_size=g("temporal_patch_size"),
            in_channels=g("in_channels") or g("in_chans", 3), spatial_merge_size=g("spatial_merge_size", 2),
            rms_norm_eps=g("rms_norm_eps", 1e-6), use_sink=bool(g("use_sink", False)),
            window=g("visual_token_window_size", -1), fullatt_block_indexes=list(g("fullatt_block_indexes", [])),
            vit_window_attn_types=list(g("vit_window_attn_types", None) or [-1] * depth),
        )

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size ** 2


# =============================================================================
# CPU preprocessing (exactly the HF functions, per block)
# =============================================================================
def _inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    dim = head_dim // 2  # HF MiMoVisionRotaryEmbedding(head_dim // 2)
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


def rot_pos_ids(t: int, h: int, w: int, merge: int) -> torch.Tensor:
    """HF ``rot_pos_emb`` position ids of one item: ``[t*h*w, 2]`` (row, col) in merge-unit order."""
    hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
    hpos = hpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
    wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
    wpos = wpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
    return torch.stack([hpos, wpos], dim=-1).repeat(t, 1)


def col_merge_index(t: int, h: int, w: int, merge: int) -> torch.Tensor:
    """HF ``get_window_index_1d(col=True)`` of one item, in merge units: ``[t * h/m * w/m]``."""
    lh, lw = h // merge, w // merge
    return torch.arange(t * lh * lw).reshape(t, lh, lw).transpose(1, 2).reshape(-1)


def block_inputs(grids, block_size: int, cfg: MiMoVisionConfig) -> dict:
    """Static-shape per-block inputs for the segments ``grids`` (list of ``(t, h, w)``) packed back to back.

    Returns ``cos_row / sin_row / cos_col / sin_col [S, head_dim]`` fp32, ``col_index / row_index [S]`` int64
    (``x_col = x[col_index]``, ``x_row = x_col[row_index]``) and ``full_min / full_max / win_min / win_max [S, 1]``
    int32 (keys ``[min, max)`` visible to each row; ``full_min`` is also the sink key).
    """
    m, mu = cfg.spatial_merge_size, cfg.merge_unit
    lengths = [t * h * w for t, h, w in grids]
    used = sum(lengths)
    if used > block_size:
        raise ValueError(f"MiMo ViT: {used} patches do not fit in vision block of {block_size}")
    for t, h, w in grids:
        if h % m or w % m:
            raise ValueError(f"MiMo ViT: grid {(t, h, w)} not divisible by spatial_merge_size {m}")
    s = block_size
    inv = _inv_freq(cfg.head_dim, cfg.rope_theta)
    max_hw = max([max(h, w) for _, h, w in grids] + [1])
    table = torch.outer(torch.arange(max_hw, dtype=inv.dtype), inv)  # HF rotary_pos_emb(max_grid_size)
    emb = torch.zeros(s, cfg.head_dim, dtype=torch.float32)
    col = torch.arange(s, dtype=torch.int64)
    p = torch.arange(s, dtype=torch.int64)
    gmin, gmax = p.clone(), p + 1  # padding rows: themselves only
    off = 0
    for (t, h, w), n in zip(grids, lengths):
        if n == 0:
            continue
        ids = rot_pos_ids(t, h, w, m)
        rot = table[ids].flatten(1)
        emb[off:off + n] = torch.cat((rot, rot), dim=-1)
        mi = col_merge_index(t, h, w, m)
        col[off:off + n] = (mi.unsqueeze(1) * mu + torch.arange(mu)).flatten() + off
        g = h * w
        for k in range(t):
            a, b = off + k * g, off + (k + 1) * g
            gmin[a:b], gmax[a:b] = a, b
        off += n
    row = torch.argsort(col)
    emb_col = emb[col]
    if cfg.window > 0:
        wmin = torch.maximum(gmin, p - cfg.window)
        wmax = torch.minimum(gmax, p + cfg.window + 1)
    else:
        wmin, wmax = gmin, gmax
    i32 = lambda x: x.to(torch.int32).unsqueeze(-1)  # noqa: E731
    return {
        "cos_row": emb.cos(), "sin_row": emb.sin(), "cos_col": emb_col.cos(), "sin_col": emb_col.sin(),
        "col_index": col, "row_index": row,
        "full_min": i32(gmin), "full_max": i32(gmax), "win_min": i32(wmin), "win_max": i32(wmax),
    }


INPUT_KEYS = ("cos_row", "sin_row", "cos_col", "sin_col", "col_index", "row_index",
              "full_min", "full_max", "win_min", "win_max")


def pack_items(pixel_values: torch.Tensor, grid_thw, block_size: int, cfg: MiMoVisionConfig,
               num_blocks: int | None = None):
    """Pack items (images / videos, rows of ``grid_thw``) into vision blocks, one item per block run.

    An item larger than a block is split at time-group boundaries (exact: attention never crosses a group).
    Returns ``(inputs, layout)``: ``inputs`` has ``pixel_values [N, S, patch_dim]`` plus :data:`INPUT_KEYS`
    stacked to ``[N, S, ...]``; ``layout[i]`` lists ``(block, start_patch, num_patches)`` of item ``i``.
    Unused trailing blocks (up to ``num_blocks``) are all padding.
    """
    grid_rows = [tuple(int(v) for v in r) for r in (grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw)]
    blocks, layout = [], []
    row0 = 0
    for t, h, w in grid_rows:
        g = h * w
        if g > block_size:
            raise ValueError(f"MiMo ViT: one time group ({h}x{w} = {g} patches) exceeds vision block {block_size}")
        per = block_size // g
        spans = []
        k = 0
        while k < t:
            n = min(per, t - k)
            spans.append((len(blocks), 0, n * g))
            blocks.append(([(n, h, w)], row0 + k * g, n * g))
            k += n
        layout.append(spans)
        row0 += t * g
    if row0 != pixel_values.shape[0]:
        raise ValueError(f"MiMo ViT: grid_thw covers {row0} patches, pixel_values has {pixel_values.shape[0]}")
    n_blk = len(blocks) if num_blocks is None else num_blocks
    if n_blk < len(blocks):
        raise ValueError(f"MiMo ViT: need {len(blocks)} vision blocks, bucket has {n_blk}")
    pix = torch.zeros(n_blk, block_size, pixel_values.shape[1], dtype=pixel_values.dtype)
    per_block = []
    for b in range(n_blk):
        if b < len(blocks):
            grids, r, n = blocks[b]
            pix[b, :n] = pixel_values[r:r + n]
        else:
            grids = []
        per_block.append(block_inputs(grids, block_size, cfg))
    inputs = {"pixel_values": pix}
    for k in INPUT_KEYS:
        inputs[k] = torch.stack([d[k] for d in per_block], dim=0)
    return inputs, layout


def unpack_merged(merged: torch.Tensor, layout, merge_unit: int) -> list:
    """``merged [N, S / merge_unit, D]`` -> per-item ``[num_tokens, D]`` (HF merger output order)."""
    out = []
    for spans in layout:
        parts = [merged[b, s // merge_unit:(s + n) // merge_unit] for b, s, n in spans]
        out.append(torch.cat(parts, dim=0))
    return out


# =============================================================================
# Device modules
# =============================================================================
@torch.compiler.allow_in_graph
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, = HF ``nn.GELU()``; traceable on Neuron (libtorch_neuronx_lite patches F.gelu)."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``x @ w + b`` with ``w`` stored ``[in, out]``; bias folded into the matmul (one BF16 rounding, like HF
    ``nn.Linear``; ``x @ w + b`` rounds twice and measurably grows the 28-layer error, docs/design-notes.md#vit-accuracy)."""
    return F.linear(x, w.t(), b)


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU in FP32, rounded once (as ``F.silu`` on BF16), written with plain ops so it traces on Neuron."""
    xf = x.to(torch.float32)
    return (xf * torch.sigmoid(xf)).to(x.dtype)


def _all_reduce(group, x):
    res = group.all_reduce(x)
    return x if res is None else res


def _row_parallel(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, group) -> torch.Tensor:
    """Row-parallel ``x @ w (+ b)``: FP32 partial sums, FP32 all-reduce, bias, then one rounding to ``x.dtype``.

    BF16 partials would be rounded per rank, after the sum and after the bias; over 28 layers that measurably grows
    the error vs HF (docs/design-notes.md#vit-accuracy). The vision tower is small, so the FP32 matmul / all-reduce cost is acceptable.
    """
    out = torch.matmul(x.to(torch.float32), w.to(torch.float32))
    if group.world_size > 1:
        out = _all_reduce(group, out)
    if b is not None:
        out = out + b.to(torch.float32)
    return out.to(x.dtype)


def _set_loader(param, loader):
    from vllm_neuron.utils.weight_loader import set_weight_loader

    set_weight_loader(param, loader)


def _loader(transform):
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    return SafetensorsWeightLoader(transform=transform)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x [N, S, D]``, ``index [N, S]`` -> ``x[n, index[n, s]]``."""
    return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _p(*shape, dtype):
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class MiMoVisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.to(torch.float32)).to(x.dtype)


class MiMoVisionAttention(nn.Module):
    """Per rank: ``num_heads / tp`` Q heads and their KV head(s) (:func:`kv_heads_for_rank`)."""

    def __init__(self, cfg: MiMoVisionConfig, layer_idx: int, tp_group, dtype: torch.dtype):
        super().__init__()
        self.tp_group = tp_group
        tp = tp_group.world_size
        nh, nkv, hd = cfg.num_heads, cfg.num_key_value_heads, cfg.head_dim
        if nh % tp:
            raise ValueError(f"MiMo ViT: num_heads {nh} not divisible by vision tp {tp}")
        self.nq = nh // tp
        self.nkv = len(kv_heads_for_rank(0, tp, nkv))
        self.head_dim = hd
        self.scale = hd ** -0.5
        self.full = layer_idx in cfg.fullatt_block_indexes
        use_sinks = cfg.use_sink and not self.full and os.environ.get("MIMO_VIT_SINKS", "1") != "0"
        h = cfg.hidden_size
        width = (self.nq + 2 * self.nkv) * hd
        self.qkv_weight = _p(h, width, dtype=dtype)
        self.qkv_bias = _p(width, dtype=dtype)
        self.proj_weight = _p(self.nq * hd, h, dtype=dtype)
        self.proj_bias = _p(h, dtype=dtype)
        self.sinks = _p(self.nq, dtype=torch.float32) if use_sinks else None

        _set_loader(self.qkv_weight, vision_qkv_loader(nh, nkv, hd, tp, bias=False))
        _set_loader(self.qkv_bias, vision_qkv_loader(nh, nkv, hd, tp, bias=True))
        _set_loader(self.proj_weight, o_proj_weight_loader(nh, hd, tp))
        if self.sinks is not None:
            _set_loader(self.sinks, sink_loader(nh, tp))

    def forward(self, x, cos, sin, bmin, bmax, sink_idx):
        """x ``[N, S, H]``; cos/sin ``[N, S, hd]`` fp32; bmin/bmax/sink_idx ``[N, S, 1]`` int32."""
        n, s, _ = x.shape
        hd, nq, nkv = self.head_dim, self.nq, self.nkv
        qkv = _linear(x, self.qkv_weight, self.qkv_bias)
        q = qkv[..., :nq * hd].reshape(n, s, nq, hd)
        k = qkv[..., nq * hd:(nq + nkv) * hd].reshape(n, s, nkv, hd)
        v = qkv[..., (nq + nkv) * hd:].reshape(n, s, nkv, hd)
        c, si = cos.unsqueeze(2).to(torch.float32), sin.unsqueeze(2).to(torch.float32)
        qf, kf = q.to(torch.float32), k.to(torch.float32)
        q = (qf * c + _rotate_half(qf) * si).to(x.dtype)  # HF rotates in fp32 and casts back
        k = (kf * c + _rotate_half(kf) * si).to(x.dtype)
        q = q.permute(0, 2, 1, 3)  # [N, nq, S, hd]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        g = nq // nkv
        if g > 1:
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)) * self.scale
        j = torch.arange(s, device=x.device, dtype=torch.int32).view(1, 1, s)
        visible = (j >= bmin) & (j < bmax)  # [N, S, S]
        if self.sinks is not None:
            onehot = (j == sink_idx).to(torch.float32)  # [N, S, S]
            scores = scores + onehot.unsqueeze(1) * self.sinks.view(1, nq, 1, 1)
        scores = scores.masked_fill(~visible.unsqueeze(1), torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.to(torch.float32)).to(x.dtype)  # [N, nq, S, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, s, nq * hd)
        return _row_parallel(out, self.proj_weight, self.proj_bias, self.tp_group)


class MiMoVisionMLP(nn.Module):
    def __init__(self, cfg: MiMoVisionConfig, tp_group, dtype: torch.dtype):
        super().__init__()
        from vllm_neuron.utils.weight_loader import sharding_weight_loader

        self.tp_group = tp_group
        tp = tp_group.world_size
        if cfg.intermediate_size % tp:
            raise ValueError(f"MiMo ViT: intermediate_size {cfg.intermediate_size} not divisible by tp {tp}")
        i, h = cfg.intermediate_size // tp, cfg.hidden_size
        self.gate_weight, self.up_weight = _p(h, i, dtype=dtype), _p(h, i, dtype=dtype)
        self.gate_bias, self.up_bias = _p(i, dtype=dtype), _p(i, dtype=dtype)
        self.down_weight, self.down_bias = _p(i, h, dtype=dtype), _p(h, dtype=dtype)
        col_w = sharding_weight_loader(shard_dim=1, shard_size=i, num_shards=tp, is_storage_transposed=True)
        col_b = sharding_weight_loader(shard_dim=0, shard_size=i, num_shards=tp)
        for prm in (self.gate_weight, self.up_weight):
            _set_loader(prm, col_w)
        for prm in (self.gate_bias, self.up_bias):
            _set_loader(prm, col_b)
        _set_loader(self.down_weight,
                    sharding_weight_loader(shard_dim=0, shard_size=i, num_shards=tp, is_storage_transposed=True))

    def forward(self, x):
        gate = _linear(x, self.gate_weight, self.gate_bias)
        up = _linear(x, self.up_weight, self.up_bias)
        return _row_parallel(_silu(gate) * up, self.down_weight, self.down_bias, self.tp_group)


class MiMoVisionBlock(nn.Module):
    def __init__(self, cfg: MiMoVisionConfig, layer_idx: int, tp_group, dtype: torch.dtype):
        super().__init__()
        self.norm1 = MiMoVisionRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.norm2 = MiMoVisionRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.attn = MiMoVisionAttention(cfg, layer_idx, tp_group, dtype)
        self.mlp = MiMoVisionMLP(cfg, tp_group, dtype)

    def forward(self, x, cos, sin, bmin, bmax, sink_idx):
        x = x + self.attn(self.norm1(x), cos, sin, bmin, bmax, sink_idx)
        return x + self.mlp(self.norm2(x))


class MiMoVisionMerger(nn.Module):
    """``ln_q`` LayerNorm (weight only) -> fc1 (column-parallel) -> GELU -> fc2 (row-parallel), no biases."""

    def __init__(self, cfg: MiMoVisionConfig, tp_group, dtype: torch.dtype):
        super().__init__()
        from vllm_neuron.utils.weight_loader import sharding_weight_loader

        self.tp_group = tp_group
        tp = tp_group.world_size
        self.merged = cfg.hidden_size * cfg.merge_unit
        if self.merged % tp:
            raise ValueError(f"MiMo ViT: merger width {self.merged} not divisible by tp {tp}")
        i = self.merged // tp
        self.ln_weight = _p(cfg.hidden_size, dtype=dtype)
        self.fc1_weight = _p(self.merged, i, dtype=dtype)
        self.fc2_weight = _p(i, cfg.out_hidden_size, dtype=dtype)
        _set_loader(self.fc1_weight,
                    sharding_weight_loader(shard_dim=1, shard_size=i, num_shards=tp, is_storage_transposed=True))
        _set_loader(self.fc2_weight,
                    sharding_weight_loader(shard_dim=0, shard_size=i, num_shards=tp, is_storage_transposed=True))
        self.eps = 1e-6

    def forward(self, x):
        """x ``[N, S, H]`` -> ``[N, S / merge_unit, out_hidden]``."""
        n, s, h = x.shape
        y = F.layer_norm(x.to(torch.float32), (h,), self.ln_weight.to(torch.float32), None, self.eps).to(x.dtype)
        y = y.reshape(n, s * h // self.merged, self.merged)
        return _row_parallel(gelu(y @ self.fc1_weight), self.fc2_weight, None, self.tp_group)


class MiMoVisionModel(nn.Module):
    """Patch embed -> 28 blocks -> merger, block-packed; optional DP over blocks and encoder-cache write."""

    def __init__(self, cfg: MiMoVisionConfig, dtype: torch.dtype = torch.bfloat16, tp_group=None, dp_group=None,
                 use_dp_group: bool = True, world_group=None):
        super().__init__()
        if tp_group is None:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_tp_group

            tp_group = get_neuron_vision_tp_group()
        if dp_group is None and use_dp_group and world_group is None:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_dp_group

            dp_group = get_neuron_vision_dp_group()
        self.cfg = cfg
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.dp_size = dp_group.world_size if dp_group is not None else 1
        # docs/design-notes.md#vision-dp-group: DP over blocks without the column group: vision TP group g = world rank // vision tp (contiguous
        # ranks) encodes block slice g, then one all-gather over the whole world (a group the device supports)
        # and every vision tp-th slot is kept.
        self.world_group = world_group
        if world_group is not None:
            self.dp_size = world_group.world_size // tp_group.world_size
        self.patch_weight = _p(cfg.patch_dim, cfg.hidden_size, dtype=dtype)
        _set_loader(self.patch_weight, _loader(lambda slices, rank: slices[0][:].reshape(cfg.hidden_size, -1).t()))
        self.blocks = nn.ModuleList([MiMoVisionBlock(cfg, i, tp_group, dtype) for i in range(cfg.depth)])
        self.merger = MiMoVisionMerger(cfg, tp_group, dtype)
        self.dtype = dtype

    def encode(self, pixel_values, cos_row, sin_row, cos_col, sin_col, col_index, row_index,
               full_min, full_max, win_min, win_max, capture: list | None = None):
        """Block-packed ViT forward (no DP, no cache write) -> merged ``[N, S / 4, out_hidden]``.

        ``capture``: if a list, each block's output is appended in the order that layer ran in (row or column
        order, like HF's hidden state; tests / debugging).
        """
        types = self.cfg.vit_window_attn_types
        x = pixel_values.to(self.dtype) @ self.patch_weight
        in_col = False
        for i, blk in enumerate(self.blocks):
            if types[i] == 1 and not in_col:
                x = _gather_rows(x, col_index)
                in_col = True
            elif types[i] != 1 and in_col:
                x = _gather_rows(x, row_index)
                in_col = False
            cos, sin = (cos_col, sin_col) if in_col else (cos_row, sin_row)
            bmin, bmax = (full_min, full_max) if blk.attn.full else (win_min, win_max)
            x = blk(x, cos, sin, bmin, bmax, full_min)
            if capture is not None:
                capture.append(x)
        return self.merger(x)

    def forward(self, pixel_values, cos_row, sin_row, cos_col, sin_col, col_index, row_index,
                full_min, full_max, win_min, win_max, encoder_cache_buffer=None, write_block_ids=None):
        """Same inputs as :meth:`encode` for all blocks; DP-sliced by block, gathered, then written to
        ``encoder_cache_buffer[write_block_ids]`` (returns the buffer) or returned when no buffer is given."""
        args = [pixel_values, cos_row, sin_row, cos_col, sin_col, col_index, row_index,
                full_min, full_max, win_min, win_max]
        if self.dp_size > 1:
            nb = pixel_values.shape[0]
            if nb % self.dp_size:
                raise ValueError(f"MiMo ViT: num_blocks {nb} not divisible by vision dp {self.dp_size}")
            per = nb // self.dp_size
            if self.world_group is not None:
                r = self.world_group.rank_in_group // self.tp_group.world_size
            else:
                r = self.dp_group.rank_in_group
            args = [a[r * per:(r + 1) * per] for a in args]
        merged = self.encode(*args)
        if self.dp_size > 1 and self.world_group is not None:
            g = self.world_group.all_gather(merged, dim=0)  # [world * per, ...]; slot w = group w // tp
            tp = self.tp_group.world_size
            merged = g.reshape(self.dp_size, tp, *merged.shape)[:, 0].reshape(self.dp_size * merged.shape[0],
                                                                              *merged.shape[1:])
        elif self.dp_size > 1:
            merged = self.dp_group.all_gather(merged, dim=0)
        if encoder_cache_buffer is None:
            return merged
        encoder_cache_buffer.index_put_((write_block_ids,), merged.to(encoder_cache_buffer.dtype))
        return encoder_cache_buffer

    # ── weights ─────────────────────────────────────────────────────────────
    @staticmethod
    def weight_mappings(cfg: MiMoVisionConfig, prefix: str = "visual.", sinks: bool = True) -> dict:
        """Parameter name -> checkpoint key (keys as in the original checkpoint, ``visual.*``)."""
        m = {"patch_weight": f"{prefix}patch_embed.proj.weight",
             "merger.ln_weight": f"{prefix}merger.ln_q.weight",
             "merger.fc1_weight": f"{prefix}merger.mlp.0.weight",
             "merger.fc2_weight": f"{prefix}merger.mlp.2.weight"}
        for i in range(cfg.depth):
            p, c = f"blocks.{i}.", f"{prefix}blocks.{i}."
            m.update({p + "norm1.weight": c + "norm1.weight", p + "norm2.weight": c + "norm2.weight",
                      p + "attn.qkv_weight": c + "attn.qkv.weight", p + "attn.qkv_bias": c + "attn.qkv.bias",
                      p + "attn.proj_weight": c + "attn.proj.weight", p + "attn.proj_bias": c + "attn.proj.bias",
                      p + "mlp.gate_weight": c + "mlp.gate_proj.weight", p + "mlp.gate_bias": c + "mlp.gate_proj.bias",
                      p + "mlp.up_weight": c + "mlp.up_proj.weight", p + "mlp.up_bias": c + "mlp.up_proj.bias",
                      p + "mlp.down_weight": c + "mlp.down_proj.weight",
                      p + "mlp.down_bias": c + "mlp.down_proj.bias"})
            if sinks and cfg.use_sink and i not in cfg.fullatt_block_indexes:
                m[p + "attn.sinks"] = c + "attn.sinks"
        return m

    def mappings(self, prefix: str = "visual.") -> dict:
        names = dict(self.named_parameters())
        return {k: v for k, v in self.weight_mappings(self.cfg, prefix).items() if k in names}

    @staticmethod
    def check_no_merger_bias(keys) -> None:
        """The checkpoint has no merger / ln_q biases (they are zero by construction here)."""
        bad = [k for k in keys if k.endswith(("merger.mlp.0.bias", "merger.mlp.2.bias", "merger.ln_q.bias"))]
        if bad:
            raise ValueError(f"MiMo ViT: unexpected merger bias keys in checkpoint: {bad}")


def num_vision_blocks(num_patches: int, block_size: int, dp_size: int = 1) -> int:
    """Blocks for a bucket of ``num_patches``, rounded up to a multiple of ``dp_size``."""
    return math.ceil(math.ceil(num_patches / block_size) / dp_size) * dp_size
