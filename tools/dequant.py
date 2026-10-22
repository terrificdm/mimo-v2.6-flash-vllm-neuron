# SPDX-License-Identifier: Apache-2.0
"""Dequantization helpers for the MiMo-V2.6-Flash-MOPD checkpoint.

Shared by tools/convert_checkpoint.py and tests/test_dequant.py.

Checkpoint formats (see docs/gpu-alignment.md section 1):
  * FP8 E4M3 weights with ``weight_scale_inv`` F32 block scales (128x128 blocks).
  * Fused ``qkv_proj`` stored pre-split for TP=4: the rows are 4 contiguous
    blocks ``[Q_r | K_r | V_r]`` (r = 0..3), each with its own scale rows
    (ceil(rows_per_block / 128) scale rows per block).  Q_r = 16 heads x 192,
    K_r = (num_kv / 4) heads x 192, V_r = (num_kv / 4) heads x 128.
  * Routed experts in MXFP4: U8 bytes with two FP4 E2M1 values each (low nibble
    = even element, high nibble = odd element, OCP MX layout) and U8 E8M0
    exponents (one per 32 elements), value = fp4 * 2^(scale - 127).
"""

from __future__ import annotations

import torch

FP8_BLOCK = 128
MXFP4_GROUP = 32
QKV_TP_BLOCKS = 4

_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = FP8_BLOCK) -> torch.Tensor:
    """FP8 block dequantization -> float32.

    ``w`` is [R, C] (float8_e4m3fn), ``scale_inv`` is [ceil(R/block), ceil(C/block)].
    Each scale entry multiplies one ``block x block`` tile; the expanded scale is
    truncated to the weight shape.
    """
    if w.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(f"expected 2-D tensors, got {tuple(w.shape)} / {tuple(scale_inv.shape)}")
    r, c = w.shape
    need = ((r + block - 1) // block, (c + block - 1) // block)
    if tuple(scale_inv.shape) != need:
        raise ValueError(f"scale shape {tuple(scale_inv.shape)} does not match weight {tuple(w.shape)} "
                         f"(expected {need})")
    s = scale_inv.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:r, :c]
    return w.float() * s


def split_fused_qkv(w: torch.Tensor, scale_inv: torch.Tensor, num_heads: int, num_kv_heads: int,
                    head_dim: int, v_head_dim: int, tp_blocks: int = QKV_TP_BLOCKS,
                    block: int = FP8_BLOCK):
    """De-interleave the TP-pre-split fused FP8 qkv_proj -> (q, k, v) float32.

    Returns q [num_heads*head_dim, H], k [num_kv*head_dim, H], v [num_kv*v_head_dim, H]
    in standard (unsharded) head order.
    """
    if num_heads % tp_blocks or num_kv_heads % tp_blocks:
        raise ValueError(f"heads {num_heads}/{num_kv_heads} not divisible by tp_blocks={tp_blocks}")
    qr = num_heads // tp_blocks * head_dim
    kr = num_kv_heads // tp_blocks * head_dim
    vr = num_kv_heads // tp_blocks * v_head_dim
    rows = qr + kr + vr
    if w.shape[0] != rows * tp_blocks:
        raise ValueError(f"qkv rows {w.shape[0]} != {tp_blocks} x {rows}")
    srows = (rows + block - 1) // block
    if scale_inv.shape[0] != srows * tp_blocks:
        raise ValueError(f"qkv scale rows {scale_inv.shape[0]} != {tp_blocks} x {srows}")
    qs, ks, vs = [], [], []
    for b in range(tp_blocks):
        d = dequant_fp8_block(w[b * rows:(b + 1) * rows], scale_inv[b * srows:(b + 1) * srows], block)
        qs.append(d[:qr])
        ks.append(d[qr:qr + kr])
        vs.append(d[qr + kr:])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                  group: int = MXFP4_GROUP) -> torch.Tensor:
    """MXFP4 dequantization -> bfloat16.

    ``blocks`` is [R, C/2] uint8 (two FP4 values per byte), ``scales`` is
    [R, C/group] uint8 E8M0 exponents.  Returns [R, C] bfloat16.
    """
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"expected uint8 inputs, got {blocks.dtype}/{scales.dtype}")
    r, half = blocks.shape
    c = half * 2
    if tuple(scales.shape) != (r, c // group):
        raise ValueError(f"mxfp4 scale shape {tuple(scales.shape)} != {(r, c // group)}")
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    lo = (blocks & 0x0F).long()
    hi = (blocks >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(r, c)          # even, odd
    exp = scales.to(torch.int32) - 127
    vals = vals.reshape(r, c // group, group) * torch.pow(2.0, exp.float()).unsqueeze(-1)
    return vals.reshape(r, c).to(torch.bfloat16)
