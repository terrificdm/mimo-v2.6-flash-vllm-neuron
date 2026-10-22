# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scaled linear layers aligned with the GPU deployment (docs/design-notes.md#fp8-alignment).

The official GPU deployment (vLLM ``Fp8LinearMethod``, ``block_quant=True``, ``activation_scheme=dynamic``) runs
q/k/v and the dense MLP as W8A8:

  * activations: per token, per 128 input channels, ``scale = amax / 448``, ``code = clamp(x / scale, +-448)``
    cast to E4M3FN with round-to-nearest-even (``fp8_utils.per_token_group_quant_fp8``);
  * weights: E4M3FN codes with one F32 scale per 128x128 block;
  * ``out = sum_kb (x_code[:, kb] @ w_code[kb, :]) * x_scale[:, kb] * w_scale[kb, :]`` with FP32 accumulation.

On trn2 the weights come from the ``-fp8`` directory (``tools/convert_checkpoint.py --fp8``): E4M3 codes
(|code| <= 240, same bit patterns as E4M3FN in that range) and a per-row-group scale ``[R, C // 128]``
(``weight_scale``). The parameters stay FP8 on the device (``*_weight`` float8_e4m3fn ``[K, N]``) with an F32
scale ``[K // 128, N]`` (``*_scale``).

Computation (all in the graph, no NKI kernel):

  * activation quantisation is emulated in FP32 arithmetic (quantisation step ``2^(max(e,-6)-3)`` built from
    comparisons with powers of two, ``torch.round`` = round-half-even), giving the GPU's E4M3FN codes for
    ``fp8_max=448`` (default, ``MIMO_FP8_ACT_MAX``; CPU bit-exact, device: 0.07 % of codes one step off because
    the device's FP32 division rounds differently, device probe). ``240`` uses the trn2 E4M3 range instead (same
    rounding grid, different scale). A plain ``.to(float8)`` round trip is not used: the compiler folds
    BF16->FP8->BF16 in one graph;
  * every FP8 code has at most 4 significant bits, so codes are exact in BF16 and every product of two codes
    is exact; the per-128-block partial sums are computed by one batched matmul (``[KB, T, 128] @ [KB, 128, N]``)
    and scaled per block, which is the GPU's block GEMM up to summation order. ``MIMO_FP8_MM=bf16`` (default)
    feeds the matmul BF16 operands (exact FP8 values, FP32 accumulation; device probe: 1.2e-7 relative to the
    float64 GPU block GEMM); ``fp32`` uses FP32 operands, which returns NaN on the device (probe ``mm_block_fp32``;
    CPU only). FP8 weights are converted to BF16 first (exact).

No native FP8 x FP8 GEMM is reachable from PyTorch in this SDK (probe: ``torch.matmul`` on float8 returns float8,
``torch._scaled_mm`` aborts); the BF16 PE path with FP8-valued operands gives the same numbers.

``MIMO_FP8_ACT=0`` switches activation quantisation off (FP8 storage, BF16 activations) for comparisons.
"""

from __future__ import annotations

import logging
import os

import torch

GROUP = 128
E4M3_MANTISSA_BITS = 3
E4M3_MIN_NORMAL_EXP = -6     # E4M3 and E4M3FN share the exponent bias (7) and the subnormal step 2^-9


def act_settings() -> dict:
    """Runtime switches read once when the model is built (env prefix of the serve / validation command)."""
    mx = float(os.environ.get("MIMO_FP8_ACT_MAX", "448"))
    if mx not in (448.0, 240.0):
        raise ValueError(f"MIMO_FP8_ACT_MAX must be 448 (GPU E4M3FN) or 240 (trn2 E4M3), got {mx}")
    mm = os.environ.get("MIMO_FP8_MM", "bf16")
    if mm not in ("fp32", "bf16"):
        raise ValueError(f"MIMO_FP8_MM must be fp32 or bf16, got {mm!r}")
    from ... import switches

    impl = switches.get("MIMO_FP8_IMPL")   # default nki on the device, emul in CPU mode
    if impl not in ("emul", "nki"):
        raise ValueError(f"MIMO_FP8_IMPL must be emul or nki, got {impl!r}")
    if impl == "nki" and mx != 448.0:
        if switches.is_set("MIMO_FP8_IMPL"):
            raise ValueError("MIMO_FP8_IMPL=nki implements the GPU scale (MIMO_FP8_ACT_MAX=448) only")
        logging.getLogger(__name__).warning("MiMoV2 Neuron: MIMO_FP8_ACT_MAX=%s: default MIMO_FP8_IMPL=nki needs "
                                            "448, using emul", mx)
        impl = "emul"
    return {"quant_act": os.environ.get("MIMO_FP8_ACT", "1") != "0", "fp8_max": mx, "mm": mm, "impl": impl}


def round_to_e4m3(v: torch.Tensor) -> torch.Tensor:
    """FP32 ``v`` (|v| <= 448) -> nearest E4M3 value (round half to even), as FP32. Exact emulation of
    ``v.to(torch.float8_e4m3fn).float()`` for |v| <= 448.

    The quantisation step ``2^(max(e, -6) - 3)`` is built from comparisons with powers of two and exact
    doublings (no bit-pattern view: ``Tensor.view(int32)`` / bitwise ops hit a torch-xla device assert on
    Neuron, device probe), so every value in the graph is exact and only ``torch.round`` (half to even) rounds."""
    a = v.abs()
    step = torch.full_like(v, 2.0 ** (E4M3_MIN_NORMAL_EXP - E4M3_MANTISSA_BITS))
    two = _c(2.0, v)
    for e in range(E4M3_MIN_NORMAL_EXP + 1, 9):           # exponents -5 .. 8 (448 = 1.75 * 2^8)
        step = torch.where(a >= _c(2.0 ** e, v), step * two, step)
    return torch.round(v / step) * step


def _c(x: float, like: torch.Tensor) -> torch.Tensor:
    """FP32 scalar constant on ``like``'s device. Python floats became f64 constants on Neuron (NCC_ESPP004);
    ``torch.tensor`` is not traceable by the runner's fake-tensor graph capture, the ``torch.full`` factory is."""
    return torch.full((), x, dtype=torch.float32, device=like.device)


def quant_act_group(x: torch.Tensor, fp8_max: float = 448.0, group: int = GROUP):
    """x [T, K] -> (codes [T, K] FP32 holding E4M3 values, scale [T, K // group] FP32).

    Same maths as vLLM ``per_token_group_quant_fp8`` (``fp8_utils.py:143-146``: ``absmax = max(amax, 1e-10)``,
    ``scale = absmax * (1 / fp8_max)`` -- multiply by the reciprocal, like the Triton kernel --, clamp, RNE; FP32
    scales, i.e. not the Blackwell UE8M0 power-of-two variant)."""
    t, k = x.shape
    xg = x.to(torch.float32).reshape(t, k // group, group)
    amax = torch.maximum(xg.abs().amax(dim=-1, keepdim=True), _c(1e-10, xg))
    scale = amax * _c(1.0 / fp8_max, xg)
    v = torch.minimum(torch.maximum(xg / scale, _c(-fp8_max, xg)), _c(fp8_max, xg))
    return round_to_e4m3(v).reshape(t, k), scale.reshape(t, k // group)


def block_matmul(x_codes: torch.Tensor, x_scale: torch.Tensor, w_codes: torch.Tensor, w_scale: torch.Tensor,
                 mm: str = "fp32", group: int = GROUP) -> torch.Tensor:
    """sum_kb (x_codes[:, kb] @ w_codes[kb]) * x_scale[:, kb, None] * w_scale[kb] -> [T, N] FP32.

    x_codes [T, K] (exact FP8 values, any float dtype), x_scale [T, KB], w_codes [K, N] (float8 or float),
    w_scale [KB, N]."""
    t, k = x_codes.shape
    n = w_codes.shape[1]
    kb = k // group
    dt = torch.float32 if mm == "fp32" else torch.bfloat16
    if w_codes.dtype == torch.float8_e4m3fn:
        w_codes = w_codes.to(torch.bfloat16)     # exact (every E4M3 value is a BF16 value)
    xb = x_codes.to(dt).reshape(t, kb, group).transpose(0, 1)          # [KB, T, G]
    wb = w_codes.to(dt).reshape(kb, group, n)                          # [KB, G, N]
    part = torch.matmul(xb, wb).to(torch.float32)                      # [KB, T, N]
    part = part * x_scale.transpose(0, 1).unsqueeze(-1) * w_scale.unsqueeze(1)
    return part.sum(dim=0)


def fp8_linear(x: torch.Tensor, w_codes: torch.Tensor, w_scale: torch.Tensor, settings: dict,
               out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """x [T, K] @ dequant(w) [K, N] the GPU W8A8 way (or with BF16 activations when quant_act is off)."""
    if settings["quant_act"] and settings.get("impl") == "nki" and out_dtype == torch.bfloat16:
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if can_run_kernel(x):
            # native FP8 x FP8 on the Tensor Engine (docs/design-notes.md#fp8-alignment) (nki_fp8.w8a8_block_linear), LNC=2 grid when N splits
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            from .nki_fp8 import w8a8_block_linear

            grid = 2 if w_codes.shape[1] % 2 == 0 else 1
            return wrap_nki(w8a8_block_linear)[grid](x=x.to(torch.bfloat16).contiguous(), w=w_codes, ws=w_scale)
    if settings["quant_act"]:
        codes, xs = quant_act_group(x, settings["fp8_max"])
        return block_matmul(codes, xs, w_codes, w_scale, settings["mm"]).to(out_dtype)
    t, k = x.shape
    ones = torch.ones(t, k // GROUP, dtype=torch.float32, device=x.device)
    return block_matmul(x, ones, w_codes, w_scale, settings["mm"]).to(out_dtype)


def dequant_rowgroup(codes: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """codes [R, C] (float8), scale [R, C // group] F32 -> FP32 ``codes * scale`` (exact for MXFP4 experts)."""
    r, c = codes.shape
    return (codes.to(torch.float32).reshape(r, c // group, group) * scale.to(torch.float32).unsqueeze(-1)).reshape(r, c)
