# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the FP8 W8A8 linear path (docs/design-notes.md#fp8-alignment): ``fp8_linear``, the FP8 weight loaders, and the FP8
attention / dense MLP / expert loading on the real ``-fp8-L3`` checkpoint against the ``-bf16-L3`` one.

GPU reference semantics (vLLM 0.24 ``fp8_utils.per_token_group_quant_fp8`` + block GEMM) are written out here with
``torch.float8_e4m3fn`` casts and float64 sums, independent of the emulation in ``fp8_linear``.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from mimo_v2_neuron.model.mimo_v2 import fp8_linear as F

FP8_DIR = os.environ.get("MIMO_FP8_L3_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-fp8-L3"))
BF16_DIR = os.environ.get("MIMO_L3_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16-L3"))
needs_ckpt = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(d, "model.safetensors.index.json")) for d in (FP8_DIR, BF16_DIR)),
    reason="fp8-L3 / bf16-L3 checkpoints not available")
SETTINGS = {"quant_act": True, "fp8_max": 448.0, "mm": "fp32"}


def gpu_quant(x, fp8_max=448.0, group=128):
    """vLLM per_token_group_quant_fp8 (torch-native form): codes as float8_e4m3fn, scale F32."""
    t, k = x.shape
    xg = x.float().reshape(t, k // group, group)
    s = torch.clamp(xg.abs().amax(-1, keepdim=True), min=1e-10) * torch.tensor(1.0 / fp8_max)  # fp8_utils.py:144
    q = torch.clamp(xg / s, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q.reshape(t, k), s.reshape(t, k // group)


def gpu_block_gemm(xq, xs, wq, ws, group=128):
    """float64 sum_kb (xq @ wq)_kb * xs * ws; xq [T, K], wq [K, N], ws [K // 128, N]."""
    t, k = xq.shape
    kb = k // group
    xb = xq.double().reshape(t, kb, group).transpose(0, 1)
    wb = wq.double().reshape(kb, group, -1)
    return (torch.matmul(xb, wb) * xs.double().t().unsqueeze(-1) * ws.double().unsqueeze(1)).sum(0)


# ── pure maths ──────────────────────────────────────────────────────────────
def test_round_to_e4m3_matches_cast():
    g = torch.Generator().manual_seed(0)
    v = torch.cat([torch.randn(200000, generator=g) * s for s in (1e-3, 0.05, 1.0, 30.0, 150.0)])
    v = torch.clamp(v, -448, 448)
    edges = torch.tensor([0.0, 448.0, -448.0, 240.0, 2 ** -6, 2 ** -9, 2 ** -10, 1.5 * 2 ** -10, 3 * 2 ** -10,
                          1.0625, 1.1875, 15.5, 17.0, 432.0, 440.0, 447.0])
    v = torch.cat([v, edges, -edges])
    ref = v.to(torch.float8_e4m3fn).float()
    assert torch.equal(F.round_to_e4m3(v), ref)


@pytest.mark.parametrize("fp8_max", [448.0, 240.0])
def test_quant_act_group_matches_gpu(fp8_max):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(64, 1024, generator=g) * torch.logspace(-3, 2, 1024)).to(torch.bfloat16)
    x[3] = 0                                   # all-zero group -> eps scale, codes 0
    codes, s = F.quant_act_group(x, fp8_max)
    rq, rs = gpu_quant(x, fp8_max)
    assert torch.equal(s, rs)
    assert torch.equal(codes, rq.float())
    assert codes.abs().max() <= fp8_max


@pytest.mark.parametrize("mm", ["fp32", "bf16"])
def test_block_matmul_matches_gpu_gemm(mm):
    g = torch.Generator().manual_seed(2)
    t, k, n = 32, 512, 96
    xq, xs = gpu_quant(torch.randn(t, k, generator=g))
    wq = torch.clamp(torch.randn(k, n, generator=g) * 60, -240, 240).to(torch.float8_e4m3fn)
    ws = torch.rand(k // 128, n, generator=g) * 1e-3
    ref = gpu_block_gemm(xq, xs, wq, ws)
    got = F.block_matmul(xq.float(), xs, wq, ws, mm).double()
    rel = float((got - ref).norm() / ref.norm())
    assert rel < (1e-6 if mm == "fp32" else 5e-3), rel


def test_fp8_linear_without_act_quant_is_dequant_matmul():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, 256, generator=g).to(torch.bfloat16)
    wq = torch.clamp(torch.randn(256, 64, generator=g) * 60, -240, 240).to(torch.float8_e4m3fn)
    ws = torch.rand(2, 64, generator=g) * 1e-3
    w = (wq.double().reshape(2, 128, 64) * ws.double().unsqueeze(1)).reshape(256, 64)
    ref = x.double() @ w
    got = F.fp8_linear(x, wq, ws, {"quant_act": False, "fp8_max": 448.0, "mm": "fp32"}, torch.float32).double()
    assert float((got - ref).norm() / ref.norm()) < 1e-6


def test_act_settings_env(monkeypatch):
    monkeypatch.delenv("MIMO_FP8_ACT", raising=False)
    monkeypatch.delenv("MIMO_FP8_ACT_MAX", raising=False)
    monkeypatch.delenv("MIMO_FP8_MM", raising=False)
    monkeypatch.delenv("MIMO_FP8_IMPL", raising=False)
    assert F.act_settings() == {"quant_act": True, "fp8_max": 448.0, "mm": "bf16", "impl": "emul"}
    monkeypatch.setenv("MIMO_FP8_ACT_MAX", "300")
    with pytest.raises(ValueError):
        F.act_settings()


# ── real L3 checkpoints ─────────────────────────────────────────────────────
class _Ck:
    def __init__(self, d):
        from safetensors import safe_open

        self.d = d
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self.files, self.open = {}, safe_open

    def slice(self, key):
        fn = self.wm[key]
        if fn not in self.files:
            self.files[fn] = self.open(os.path.join(self.d, fn), "pt")
        return self.files[fn].get_slice(key)

    def cfg(self):
        with open(os.path.join(self.d, "config.json")) as f:
            return json.load(f)


_CK = {}


def ck(which):
    if which not in _CK:
        _CK[which] = _Ck(FP8_DIR if which == "fp8" else BF16_DIR)
    return _CK[which]


def _cfg(which):
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config

    return MiMoV2Config.from_configs(ck(which).cfg(), None)


def _load(module, mapping, rank, which):
    from vllm_neuron.utils.weight_loader import get_weight_loader

    for pname, keys in mapping.items():
        keys = keys if isinstance(keys, list) else [keys]
        param = getattr(module, pname)
        t = get_weight_loader(param).load([ck(which).slice(k) for k in keys], rank)
        assert tuple(t.shape) == tuple(param.shape), (pname, t.shape, param.shape)
        assert t.dtype == param.dtype, (pname, t.dtype, param.dtype)
        param.data = t


def _grp(tp, rank):
    return SimpleNamespace(world_size=tp, rank_in_group=rank)


@needs_ckpt
def test_config_detects_fp8():
    c, b = _cfg("fp8"), _cfg("bf16")
    assert c.fp8_weights and c.fp8_linear_group == 128 and c.fp8_expert_group == 32 and c.fp8_act["quant_act"]
    assert not b.fp8_weights and b.fp8_act is None


@needs_ckpt
@pytest.mark.parametrize("layer,rank", [(0, 0), (0, 37), (1, 5), (1, 63)])
def test_attention_fp8_weights_match_bf16(layer, rank):
    """Dequantised FP8 q/k/v of one TP=64 rank == the BF16 checkpoint except the rescaled subnormal codes."""
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2Attention

    tp = 64
    a8 = MiMoV2Attention(_cfg("fp8"), layer, tp_group=_grp(tp, rank))
    ab = MiMoV2Attention(_cfg("bf16"), layer, tp_group=_grp(tp, rank))
    _load(a8, MiMoV2Attention.checkpoint_keys(layer, a8.sinks is not None, True), rank, "fp8")
    _load(ab, MiMoV2Attention.checkpoint_keys(layer, ab.sinks is not None), rank, "bf16")
    assert a8.qkv_proj_weight.dtype == torch.float8_e4m3fn and a8.o_proj_weight.dtype == torch.bfloat16
    k, n = a8.qkv_proj_weight.shape
    w = (a8.qkv_proj_weight.float().reshape(k // 128, 128, n) * a8.qkv_proj_scale.unsqueeze(1)).reshape(k, n)
    wb = ab.qkv_proj_weight.float()
    wr = w.to(torch.bfloat16).float()                  # the BF16 directory holds BF16-rounded dequant values
    assert float((wr != wb).float().mean()) < 1e-3      # only the rescaled subnormal codes (6.3e-5 in the conversion check)
    assert bool(((wr - wb).abs() <= wb.abs() * 2 ** -7 + 1e-5).all())   # at most one BF16 ulp
    assert torch.equal(a8.o_proj_weight, ab.o_proj_weight)


@needs_ckpt
def test_dense_mlp_fp8_matches_gpu_reference():
    """TP=64 sum of FP8 dense MLP partials == GPU W8A8 reference (float64), and close to the BF16 module."""
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2DenseMLP

    tp = 64
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(8, 4096, generator=g) * 0.5).to(torch.bfloat16)
    acc8 = torch.zeros(8, 4096, dtype=torch.float64)
    accb = torch.zeros(8, 4096, dtype=torch.float64)
    ref = torch.zeros(8, 4096, dtype=torch.float64)
    xq, xs = gpu_quant(x)
    for r in range(0, tp, 8):          # 8 of the 64 ranks keeps the test fast; same comparison per rank
        d8 = MiMoV2DenseMLP(_cfg("fp8"), 0, tp_group=_grp(tp, r))
        d8.fp8_act = SETTINGS
        db = MiMoV2DenseMLP(_cfg("bf16"), 0, tp_group=_grp(tp, r))
        _load(d8, MiMoV2DenseMLP.checkpoint_keys(0, True), r, "fp8")
        _load(db, MiMoV2DenseMLP.checkpoint_keys(0), r, "bf16")
        with torch.no_grad():
            acc8 += d8.forward_local(x).double()
            accb += torch.matmul(torch.nn.functional.silu(x @ db.gate_proj_weight) * (x @ db.up_proj_weight),
                                 db.down_proj_weight).double()
            gate = gpu_block_gemm(xq, xs, d8.gate_proj_weight, d8.gate_proj_scale).to(torch.bfloat16)
            up = gpu_block_gemm(xq, xs, d8.up_proj_weight, d8.up_proj_scale).to(torch.bfloat16)
            a = torch.nn.functional.silu(gate) * up
            aq, as_ = gpu_quant(a)
            ref += gpu_block_gemm(aq, as_, d8.down_proj_weight, d8.down_proj_scale).to(torch.bfloat16).double()
    rel_gpu = float((acc8 - ref).norm() / ref.norm())
    rel_bf16 = float((acc8 - accb).norm() / accb.norm())
    print(f"dense MLP FP8 vs GPU-ref rel {rel_gpu:.3e}; vs BF16 rel {rel_bf16:.3e}")
    assert rel_gpu < 2e-2          # only BF16 output rounding of g / u / a differs (summation order)
    assert rel_bf16 < 0.15         # activation quantisation error (W8A8 vs W16A16)


@needs_ckpt
@pytest.mark.parametrize("layer", [1, 2])
def test_experts_fp8_resident_equal_bf16(layer):
    """FP8 experts stay FP8 (one scale per output channel); expanded in the graph they are bit-identical to the
    BF16 directory (MXFP4 -> FP8 lossless; per-row scale merge lossless, docs/design-notes.md#fp8-alignment)."""
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE

    ep = {"degree": 64, "rank": 9, "tp_group": _grp(1, 0)}
    m8 = MiMoV2MoE(_cfg("fp8"), layer, tp_group=_grp(64, 9), ep=ep)
    mb = MiMoV2MoE(_cfg("bf16"), layer, tp_group=_grp(64, 9), ep=ep)
    _load(m8, m8.checkpoint_keys(), 9, "fp8")
    _load(mb, mb.checkpoint_keys(), 9, "bf16")
    assert m8.gate_up_proj_weight.dtype == torch.float8_e4m3fn and m8.down_proj_weight.dtype == torch.float8_e4m3fn
    assert tuple(m8.gate_up_proj_scale.shape) == (4, 2, 2048) and tuple(m8.down_proj_scale.shape) == (4, 4096)
    assert m8.router_weight.dtype == torch.bfloat16 and m8.e_score_correction_bias.dtype == torch.float32
    gu, dn = m8.expert_weights()
    assert torch.equal(gu, mb.gate_up_proj_weight) and torch.equal(dn, mb.down_proj_weight)
    for n in ("router_weight", "e_score_correction_bias"):
        assert torch.equal(getattr(m8, n), getattr(mb, n)), n


def test_merge_row_scale_lossless_or_raises():
    from mimo_v2_neuron.model.mimo_v2.weight_loaders import merge_row_scale

    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    g = torch.Generator().manual_seed(6)
    codes = (e2m1[torch.randint(0, 8, (4, 128), generator=g)] * torch.where(
        torch.rand(4, 128, generator=g) > 0.5, 1.0, -1.0)).to(torch.float8_e4m3fn)
    scale = 2.0 ** torch.randint(-12, -8, (4, 4), generator=g).float()
    c, s = merge_row_scale(codes, scale, 32)
    ref = (codes.float().reshape(4, 4, 32) * scale.unsqueeze(-1)).reshape(4, 128)
    assert torch.equal(c.float() * s.unsqueeze(1), ref)
    bad = scale.clone()
    bad[0, 0] = 2.0 ** -30                       # 2^-18 below the row max: 0.5 x 2^-18 x 32 underflows E4M3
    with pytest.raises(ValueError):
        merge_row_scale(codes.float().abs().clamp(min=0.5).to(torch.float8_e4m3fn), bad, 32)
