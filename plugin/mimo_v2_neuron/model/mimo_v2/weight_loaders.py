# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the MiMo-V2 Neuron model (BF16 checkpoint with split q/k/v).

All transforms take ``(slices, rank)`` where ``slices`` are safetensors ``PySafeSlice``
objects in the order given by the model's ``load_weights`` mappings and ``rank`` is the
TP rank. HF linear weights are stored ``[out_features, in_features]``.

KV-head ownership is defined only by :func:`kv_heads_for_rank`; both the attention
weight loader and the KV-cache spec go through it (the single place that owns these invariants).
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def kv_heads_for_rank(rank: int, tp: int, n_kv: int) -> range:
    """KV heads held by TP rank ``rank``.

    ``tp >= n_kv``: every rank holds one KV head, replicated over ``tp // n_kv`` ranks
    (head ``rank * n_kv // tp``; e.g. TP=64: GA (4 heads) ``rank // 16``, SWA (8) ``rank // 8``).
    ``tp < n_kv``: rank holds the contiguous block ``[rank * n_kv/tp, (rank+1) * n_kv/tp)``.
    """
    rank = rank % tp
    if tp >= n_kv:
        if tp % n_kv:
            raise ValueError(f"tp ({tp}) must be a multiple of n_kv ({n_kv})")
        h = rank * n_kv // tp
        return range(h, h + 1)
    if n_kv % tp:
        raise ValueError(f"n_kv ({n_kv}) must be a multiple of tp ({tp})")
    per = n_kv // tp
    return range(rank * per, (rank + 1) * per)


def kv_head_for_rank(rank: int, tp: int, n_kv: int) -> int:
    """First (for ``tp >= n_kv`` the only) KV head of ``rank``."""
    return kv_heads_for_rank(rank, tp, n_kv).start


def qkv_weight_loader(num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int,
                      tp: int) -> SafetensorsWeightLoader:
    """[q_proj, k_proj, v_proj] -> fused, transposed ``[hidden, q_local | k_local | v_local]``.

    Q heads are split contiguously across ranks; K/V heads follow :func:`kv_heads_for_rank`.
    """
    nq = num_heads // tp

    def transform(slices, rank):
        assert len(slices) == 3, "qkv loader expects [q_proj, k_proj, v_proj]"
        r = rank % tp
        kvh = kv_heads_for_rank(r, tp, num_kv_heads)
        q = slices[0][r * nq * head_dim:(r + 1) * nq * head_dim, :]
        k = slices[1][kvh.start * head_dim:kvh.stop * head_dim, :]
        v = slices[2][kvh.start * v_head_dim:kvh.stop * v_head_dim, :]
        return torch.cat([q, k, v], dim=0).t()

    return SafetensorsWeightLoader(transform=transform)


def vision_qkv_loader(num_heads: int, num_kv_heads: int, head_dim: int, tp: int,
                      bias: bool = False) -> SafetensorsWeightLoader:
    """ViT fused qkv ``[(nq + 2 nkv) * hd, H]`` (rows ``Q all | K all | V all``) -> rank's
    ``[H, q_local | k_local | v_local]`` (weight, transposed) or ``[q_local | k_local | v_local]`` (bias).

    Q heads are split contiguously; K/V heads follow :func:`kv_heads_for_rank` (TP=8: one KV head per rank).
    """
    nq, q_rows, k_rows = num_heads // tp, num_heads * head_dim, num_kv_heads * head_dim

    def transform(slices, rank):
        assert len(slices) == 1, "vision qkv loader expects the fused qkv tensor"
        r = rank % tp
        kvh = kv_heads_for_rank(r, tp, num_kv_heads)
        s = slices[0]
        spans = [(r * nq * head_dim, (r + 1) * nq * head_dim),
                 (q_rows + kvh.start * head_dim, q_rows + kvh.stop * head_dim),
                 (q_rows + k_rows + kvh.start * head_dim, q_rows + k_rows + kvh.stop * head_dim)]
        fused = torch.cat([s[a:b] if bias else s[a:b, :] for a, b in spans], dim=0)
        return fused if bias else fused.t()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(num_heads: int, v_head_dim: int, tp: int) -> SafetensorsWeightLoader:
    """o_proj ``[hidden, num_heads * v_head_dim]`` -> rank's input slice, transposed ``[nq*v_head_dim, hidden]``."""
    shard = (num_heads // tp) * v_head_dim

    def transform(slices, rank):
        assert len(slices) == 1
        r = rank % tp
        return slices[0][:, r * shard:(r + 1) * shard].t()

    return SafetensorsWeightLoader(transform=transform)


def sink_loader(num_heads: int, tp: int) -> SafetensorsWeightLoader:
    """attention_sink_bias ``[num_heads]`` -> rank's heads, FP32 (softmax runs in FP32)."""
    nq = num_heads // tp

    def transform(slices, rank):
        assert len(slices) == 1
        r = rank % tp
        return slices[0][r * nq:(r + 1) * nq].to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


# ── MoE ──────────────────────────────────────────────────────────────────────
def expert_checkpoint_keys(layer_idx: int, experts) -> dict:
    """Checkpoint keys of the given (local) experts, in the order the expert loaders expect."""
    p = f"model.layers.{layer_idx}.mlp.experts."
    gate_up = []
    for e in experts:
        gate_up += [f"{p}{e}.gate_proj.weight", f"{p}{e}.up_proj.weight"]
    return {"gate_up_proj_weight": gate_up,
            "down_proj_weight": [f"{p}{e}.down_proj.weight" for e in experts]}


def expert_gate_up_loader(num_local_experts: int, i_start: int, i_size: int) -> SafetensorsWeightLoader:
    """[gate_e0, up_e0, gate_e1, up_e1, ...] (each HF ``[I, H]``) -> ``[E_L, H, 2, i_size]``.

    ``[..., 0, :]`` is gate, ``[..., 1, :]`` is up (the ``[E, H, 2, I]`` layout of
    ``NF.moe_cte`` / ``NF.moe_tkg``); the intermediate dim is sliced to this rank's
    ``[i_start, i_start + i_size)`` (whole range under full EP). The mapping already lists
    only this rank's experts, so ``rank`` is not used here.
    """

    def transform(slices, rank):
        assert len(slices) == 2 * num_local_experts, (len(slices), num_local_experts)
        out = []
        for e in range(num_local_experts):
            g = slices[2 * e][i_start:i_start + i_size, :].t()
            u = slices[2 * e + 1][i_start:i_start + i_size, :].t()
            out.append(torch.stack([g, u], dim=1))
        return torch.stack(out, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def expert_down_loader(num_local_experts: int, i_start: int, i_size: int) -> SafetensorsWeightLoader:
    """[down_e0, down_e1, ...] (each HF ``[H, I]``) -> ``[E_L, i_size, H]``."""

    def transform(slices, rank):
        assert len(slices) == num_local_experts, (len(slices), num_local_experts)
        return torch.stack([s[:, i_start:i_start + i_size].t() for s in slices], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# ── FP8 (``-fp8`` directory, tools/convert_checkpoint.py --fp8; docs/design-notes.md#fp8-alignment) ─────────────────────────────────────────
# Each FP8 tensor ``X.weight`` (float8_e4m3fn [R, C]) has ``X.weight_scale`` (F32 [R, C // G]); dequant[r, c] =
# code[r, c] * weight_scale[r, c // G]. Codes are kept as FP8 parameters ``[K, N]`` with an F32 scale ``[K // G, N]``.
def _cat_rows(parts: list) -> torch.Tensor:
    if parts[0].dtype == torch.float8_e4m3fn:   # concatenate the bit patterns (no dtype conversion)
        return torch.cat([p.contiguous().view(torch.uint8) for p in parts], dim=0).view(torch.float8_e4m3fn)
    return torch.cat(parts, dim=0)


def qkv_fp8_loader(num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int,
                   tp: int) -> SafetensorsWeightLoader:
    """[q, k, v] codes (or their row-group scales) -> ``[K, q_local | k_local | v_local]`` (codes FP8, scale F32
    ``[K // 128, N]``). Same head split as :func:`qkv_weight_loader`; scales are per row, so rows slice alike."""
    nq = num_heads // tp

    def transform(slices, rank):
        assert len(slices) == 3, "qkv fp8 loader expects [q_proj, k_proj, v_proj] (weight or weight_scale)"
        r = rank % tp
        kvh = kv_heads_for_rank(r, tp, num_kv_heads)
        q = slices[0][r * nq * head_dim:(r + 1) * nq * head_dim, :]
        k = slices[1][kvh.start * head_dim:kvh.stop * head_dim, :]
        v = slices[2][kvh.start * v_head_dim:kvh.stop * v_head_dim, :]
        out = _cat_rows([q, k, v])
        return out.t().contiguous() if out.dtype != torch.float8_e4m3fn else \
            out.view(torch.uint8).t().contiguous().view(torch.float8_e4m3fn)

    return SafetensorsWeightLoader(transform=transform)


def rowgroup_shard_loader(shard_rows: bool, shard_size: int, tp: int, group: int = 128,
                          is_scale: bool = False) -> SafetensorsWeightLoader:
    """Dense MLP FP8 codes / scales, TP-sharded and transposed to ``[K, N]`` / ``[K // group, N]``.

    ``shard_rows``: shard the HF output dim (gate / up: rows of ``[I, H]``); else the HF input dim (down: columns
    of ``[H, I]``, scale columns sharded by ``shard_size // group``)."""
    if not shard_rows and shard_size % group:
        raise ValueError(f"MiMoV2 Neuron: FP8 input-dim shard {shard_size} must be a multiple of {group}")

    def transform(slices, rank):
        assert len(slices) == 1
        r = rank % tp
        s = slices[0]
        if shard_rows:
            x = s[r * shard_size:(r + 1) * shard_size, :]
        else:
            n = shard_size // group if is_scale else shard_size
            x = s[:, r * n:(r + 1) * n]
        if x.dtype == torch.float8_e4m3fn:
            return x.contiguous().view(torch.uint8).t().contiguous().view(torch.float8_e4m3fn)
        return x.to(torch.float32).t().contiguous()

    return SafetensorsWeightLoader(transform=transform)


def expert_fp8_checkpoint_keys(layer_idx: int, experts) -> dict:
    """Like :func:`expert_checkpoint_keys`, each weight followed by its ``weight_scale``."""
    p = f"model.layers.{layer_idx}.mlp.experts."
    gate_up, down = [], []
    for e in experts:
        for n in ("gate_proj", "up_proj"):
            gate_up += [f"{p}{e}.{n}.weight", f"{p}{e}.{n}.weight_scale"]
        down += [f"{p}{e}.down_proj.weight", f"{p}{e}.down_proj.weight_scale"]
    return {"gate_up_proj_weight": gate_up, "down_proj_weight": down}


def merge_row_scale(codes: torch.Tensor, scale: torch.Tensor, group: int, headroom: float = 32.0):
    """Per-group FP8 expert codes -> one scale per row (output channel), losslessly (docs/design-notes.md#fp8-alignment).

    ``codes`` [R, C] float8 (E2M1 values), ``scale`` [R, C // group] F32 powers of two. Row scale =
    ``max(scale) / headroom``; codes are multiplied by ``scale / row_scale`` (a power of two <= headroom: E2M1 6 x 32
    = 192 <= 240). Raises when any code would change (underflow or > 240), so a lossy merge never loads silently.
    Returns (codes' [R, C] float8_e4m3fn, row_scale [R] F32)."""
    r, c = codes.shape
    sc = scale.to(torch.float32)
    row = sc.amax(dim=1, keepdim=True) / headroom
    row = torch.where(row > 0, row, torch.ones_like(row))
    v = (codes.to(torch.float32).reshape(r, c // group, group) * (sc / row).unsqueeze(-1)).reshape(r, c)
    q = v.to(torch.float8_e4m3fn)
    if bool((v.abs() > 240).any()) or not torch.equal(q.to(torch.float32), v):
        raise ValueError("MiMoV2 Neuron: FP8 expert per-row scale merge would change codes (not lossless)")
    return q, row.squeeze(1)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def expert_gate_up_fp8_loader(num_local_experts: int, i_start: int, i_size: int, group: int = 32,
                              part: str = "codes") -> SafetensorsWeightLoader:
    """FP8 experts kept FP8 with one scale per output channel (NKI ``moe_tkg`` ROW layout).

    ``part="codes"`` -> ``[E_L, H, 2, i_size]`` float8 (same layout as :func:`expert_gate_up_loader`);
    ``part="scale"`` -> ``[E_L, 2, i_size]`` F32. Slices: per expert [gate.weight, gate.weight_scale, up.weight,
    up.weight_scale]."""

    def transform(slices, rank):
        assert len(slices) == 4 * num_local_experts, (len(slices), num_local_experts)
        codes, scales = [], []
        for e in range(num_local_experts):
            pair_c, pair_s = [], []
            for j in (0, 2):
                c, s = merge_row_scale(slices[4 * e + j][i_start:i_start + i_size, :],
                                       slices[4 * e + j + 1][i_start:i_start + i_size, :], group)
                pair_c.append(_u8(c).t())            # [H, i]
                pair_s.append(s)
            codes.append(torch.stack(pair_c, dim=1))  # [H, 2, i]
            scales.append(torch.stack(pair_s, dim=0))  # [2, i]
        if part == "scale":
            return torch.stack(scales, dim=0)
        return torch.stack(codes, dim=0).contiguous().view(torch.float8_e4m3fn)

    return SafetensorsWeightLoader(transform=transform)


def expert_down_fp8_loader(num_local_experts: int, i_start: int, i_size: int, group: int = 32,
                           part: str = "codes") -> SafetensorsWeightLoader:
    """``part="codes"`` -> ``[E_L, i_size, H]`` float8; ``part="scale"`` -> ``[E_L, H]`` F32 (per output channel).
    Slices: per expert [down.weight, down.weight_scale]."""
    if i_start % group or i_size % group:
        raise ValueError(f"MiMoV2 Neuron: FP8 expert intermediate shard ({i_start}, {i_size}) must align to {group}")

    def transform(slices, rank):
        assert len(slices) == 2 * num_local_experts, (len(slices), num_local_experts)
        g0, gn = i_start // group, i_size // group
        codes, scales = [], []
        for e in range(num_local_experts):
            c, s = merge_row_scale(slices[2 * e][:, i_start:i_start + i_size], slices[2 * e + 1][:, g0:g0 + gn], group)
            codes.append(_u8(c).t())                  # [i, H]
            scales.append(s)                          # [H]
        if part == "scale":
            return torch.stack(scales, dim=0)
        return torch.stack(codes, dim=0).contiguous().view(torch.float8_e4m3fn)

    return SafetensorsWeightLoader(transform=transform)


def fp32_vector_loader(key: str) -> SafetensorsWeightLoader:
    """Load a replicated tensor that must be stored as F32 in the checkpoint (routing bias)."""

    def transform(slices, rank):
        assert len(slices) == 1
        dt = slices[0].get_dtype()
        if dt != "F32":
            raise ValueError(f"MiMoV2 Neuron: checkpoint tensor {key} must be F32 (got {dt}); "
                             "re-run tools/convert_checkpoint.py, which keeps the routing bias in FP32")
        return slices[0][:].to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)
