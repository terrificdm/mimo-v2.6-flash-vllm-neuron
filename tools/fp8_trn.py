# SPDX-License-Identifier: Apache-2.0
"""FP8 weight conversion for Trainium2 (trn2) -- helpers shared by
tools/convert_checkpoint.py (``--fp8``), tools/check_fp8_converted.py and
tests/test_fp8_convert.py.  Format notes: docs/gpu-alignment.md section 3.

Output format ("neuron_fp8", per-row group scales):
  * Every FP8 weight is ``[R, C]`` ``float8_e4m3fn`` whose codes all satisfy
    ``|code| <= 240`` (trn2 FP8 is E4M3 with max 240; for |x| <= 240 the bit
    patterns of E4M3 and E4M3FN are identical, docs/design-notes.md#fp8-alignment), plus an F32
    ``weight_scale`` of shape ``[R, C // G]``:
        dequant[r, c] = float(code[r, c]) * weight_scale[r, c // G]
  * Linear layers (q/k/v split from the fused qkv, dense MLP): G = 128.  The
    original 128x128 block scale is copied to every row of its block, so the
    split q/k/v (whose GA K/V rows do not start on 128-row boundaries) and any
    TP row shard keep exact scales.
  * MoE experts (MXFP4: E2M1 + E8M0 per 32 elements): G = 32.  The FP8 code is
    the E2M1 value itself (0, +-0.5 ... +-6, all exact in E4M3) and the scale is
    2^(e - 127) -- bit-exact, no rescaling needed.
  * E4M3FN -> trn2 E4M3: in each (row, 128-column) group whose max |code| > 240
    the codes are halved (round-to-nearest-even) and the scale doubled.  Only
    codes in the subnormal range with an odd last mantissa bit change (or
    underflow to zero); ``rescale_to_trn`` counts them.
"""

from __future__ import annotations

import torch

TRN_FP8_MAX = 240.0
LINEAR_GROUP = 128
EXPERT_GROUP = 32
FP8_BLOCK = 128
QKV_TP_BLOCKS = 4

_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def empty_stats() -> dict:
    return {"elements": 0, "groups": 0, "groups_rescaled": 0, "changed": 0, "underflow_to_zero": 0,
            "max_abs_dequant_err": 0.0}


def add_stats(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0.0), v) if k.startswith("max_") else out.get(k, 0) + v
    return out


def dequant_rowgroup(w: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """``[R, C]`` FP8 codes x ``[R, C // group]`` scale -> float32 ``[R, C]``."""
    r, c = w.shape
    if c % group or tuple(scale.shape) != (r, c // group):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not fit weight {tuple(w.shape)} / group {group}")
    return (w.float().view(r, c // group, group) * scale.float().unsqueeze(-1)).view(r, c)


def block_scale_to_rows(scale_inv: torch.Tensor, rows: int, block: int = FP8_BLOCK) -> torch.Tensor:
    """``[ceil(R/128), ceil(C/128)]`` block scale -> per-row ``[R, ceil(C/128)]`` (exact copy)."""
    return scale_inv.float().repeat_interleave(block, 0)[:rows].contiguous()


def rescale_to_trn(w: torch.Tensor, scale: torch.Tensor, group: int = LINEAR_GROUP):
    """E4M3FN codes (max 448) -> trn2-safe codes (max 240), per (row, group).

    Returns ``(codes, scale, stats)``; ``codes`` stay ``float8_e4m3fn``.
    """
    if w.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected float8_e4m3fn, got {w.dtype}")
    r, c = w.shape
    if c % group:
        raise ValueError(f"columns {c} not divisible by group {group}")
    wf = w.float().view(r, c // group, group)
    if not torch.isfinite(wf).all():
        raise ValueError("FP8 weight contains NaN")
    need = wf.abs().amax(-1) > TRN_FP8_MAX                                    # [R, G]
    half = (wf / 2).to(torch.float8_e4m3fn).float()
    new = torch.where(need.unsqueeze(-1), half, wf)
    changed = need.unsqueeze(-1) & (half * 2 != wf)
    s = scale.float()
    s_new = torch.where(need, s * 2, s)
    err = ((new * s_new.unsqueeze(-1)) - (wf * s.unsqueeze(-1))).abs().max().item()
    stats = {"elements": w.numel(), "groups": need.numel(), "groups_rescaled": int(need.sum()),
             "changed": int(changed.sum()), "underflow_to_zero": int((changed & (half == 0)).sum()),
             "max_abs_dequant_err": float(err)}
    out = new.view(r, c).to(torch.float8_e4m3fn)
    assert out.float().abs().max().item() <= TRN_FP8_MAX
    return out, s_new.contiguous(), stats


def fp8_linear(w: torch.Tensor, scale_inv: torch.Tensor):
    """Original FP8 block-scaled linear -> (codes, per-row-group scale, stats)."""
    return rescale_to_trn(w, block_scale_to_rows(scale_inv, w.shape[0]), LINEAR_GROUP)


def split_fused_qkv_fp8(w: torch.Tensor, scale_inv: torch.Tensor, num_heads: int, num_kv_heads: int,
                        head_dim: int, v_head_dim: int, tp_blocks: int = QKV_TP_BLOCKS,
                        block: int = FP8_BLOCK):
    """TP-pre-split fused FP8 qkv -> ((q, sq), (k, sk), (v, sv)) FP8 codes + per-row scales.

    Same row order as ``tools.dequant.split_fused_qkv``; no requantization (codes copied,
    each row keeps the scale of the 128x128 tile it came from).
    """
    qr = num_heads // tp_blocks * head_dim
    kr = num_kv_heads // tp_blocks * head_dim
    vr = num_kv_heads // tp_blocks * v_head_dim
    rows = qr + kr + vr
    srows = (rows + block - 1) // block
    if w.shape[0] != rows * tp_blocks or scale_inv.shape[0] != srows * tp_blocks:
        raise ValueError(f"qkv shapes {tuple(w.shape)} / {tuple(scale_inv.shape)} do not match "
                         f"{tp_blocks} x ({rows} rows, {srows} scale rows)")
    parts = {"q": ([], []), "k": ([], []), "v": ([], [])}
    for b in range(tp_blocks):
        wb = w[b * rows:(b + 1) * rows]
        sb = block_scale_to_rows(scale_inv[b * srows:(b + 1) * srows], rows, block)
        for name, lo, hi in (("q", 0, qr), ("k", qr, qr + kr), ("v", qr + kr, rows)):
            parts[name][0].append(wb[lo:hi])
            parts[name][1].append(sb[lo:hi])
    return tuple((torch.cat(parts[n][0]), torch.cat(parts[n][1])) for n in ("q", "k", "v"))


def mxfp4_to_fp8(blocks: torch.Tensor, scales: torch.Tensor, group: int = EXPERT_GROUP):
    """MXFP4 (``[R, C/2]`` u8 nibbles, ``[R, C/32]`` u8 E8M0) -> (FP8 codes ``[R, C]``, F32 scale ``[R, C/32]``)."""
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"expected uint8 inputs, got {blocks.dtype}/{scales.dtype}")
    r, half = blocks.shape
    c = half * 2
    if tuple(scales.shape) != (r, c // group):
        raise ValueError(f"mxfp4 scale shape {tuple(scales.shape)} != {(r, c // group)}")
    if (scales == 255).any():
        raise ValueError("E8M0 scale 255 (NaN) present")
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    vals = torch.stack([lut[(blocks & 0x0F).long()], lut[(blocks >> 4).long()]], dim=-1).reshape(r, c)
    codes = vals.to(torch.float8_e4m3fn)
    scale = torch.pow(2.0, scales.to(torch.int32).float() - 127.0)
    return codes, scale.contiguous()


def coarse_scale_stats(codes: torch.Tensor, scales_u8: torch.Tensor, merge: int | None) -> dict:
    """How many expert elements would change if the 32-element E8M0 scales were merged into
    one power-of-two scale per ``merge`` groups (``None`` = whole row, i.e. per-output-channel
    scale as used by the NKI MoE "row" FP8 kernels).  Diagnostic only."""
    r, c = codes.shape
    g = c // EXPERT_GROUP
    m = g if merge is None else merge
    vals = codes.float().view(r, g, EXPERT_GROUP)
    e = scales_u8.to(torch.int32).view(r, g)
    nonzero = vals.abs().amax(-1) > 0
    e_eff = torch.where(nonzero, e, torch.full_like(e, -1000))
    emax = e_eff.view(r, g // m, m).amax(-1, keepdim=True).expand(r, g // m, m).reshape(r, g)
    shift = (e - emax + 5).clamp(min=-60).float()                          # 6 * 2^5 = 192 <= 240
    q = vals * torch.pow(2.0, shift).unsqueeze(-1)
    rt = q.to(torch.float8_e4m3fn).float()
    changed = (rt != q) & nonzero.unsqueeze(-1)
    return {"elements": codes.numel(), "changed": int(changed.sum()),
            "underflow_to_zero": int((changed & (rt == 0)).sum()),
            "rows_lossless": int((~changed.view(r, -1).any(-1)).sum()), "rows": r}
