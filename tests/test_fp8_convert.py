# SPDX-License-Identifier: Apache-2.0
"""trn2 FP8 weight conversion tests (tools/fp8_trn.py, convert_checkpoint.py --fp8; CPU)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from tools import fp8_trn
from tools.dequant import dequant_fp8_block, dequant_mxfp4, split_fused_qkv

SRC = os.environ.get("MIMO_SRC", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_SRC = os.path.isfile(os.path.join(SRC, "model.safetensors.index.json"))


def _rand_fp8(rows, cols, gen, scale=100.0):
    return (torch.randn(rows, cols, generator=gen) * scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def test_mxfp4_to_fp8_bit_exact_all_codes_and_scales():
    # every nibble value x a wide range of E8M0 exponents (incl. 0 -> 2^-127 and 254)
    r = 255                                                     # row i uses exponent i for all groups
    blocks = torch.arange(256, dtype=torch.int32).repeat(r, 1).to(torch.uint8)       # [255, 256] -> C = 512
    scales = torch.arange(r, dtype=torch.int32).unsqueeze(1).repeat(1, 16).to(torch.uint8)
    codes, s = fp8_trn.mxfp4_to_fp8(blocks, scales)
    assert codes.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    assert codes.float().abs().max().item() <= 6
    assert torch.equal(fp8_trn.dequant_rowgroup(codes, s, 32), dequant_mxfp4(blocks, scales).float())


def test_mxfp4_random_bit_exact():
    gen = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (128, 512), dtype=torch.uint8, generator=gen)
    scales = torch.randint(100, 135, (128, 32), dtype=torch.uint8, generator=gen)
    codes, s = fp8_trn.mxfp4_to_fp8(blocks, scales)
    assert torch.equal(fp8_trn.dequant_rowgroup(codes, s, 32), dequant_mxfp4(blocks, scales).float())


def test_mxfp4_nan_scale_rejected():
    with pytest.raises(ValueError):
        fp8_trn.mxfp4_to_fp8(torch.zeros(1, 16, dtype=torch.uint8), torch.full((1, 1), 255, dtype=torch.uint8))


def test_rescale_counts_only_subnormal_odd_codes():
    # group 0: has 448 -> rescaled; codes 2^-9 (min subnormal, underflows / rounds to even) and
    # 3*2^-9 (odd mantissa) change, normal codes do not.  group 1: max 240 -> untouched.
    g0 = [448.0, 2.0 ** -9, 3 * 2.0 ** -9, 1.5, -6.0, 0.0, 2.0 ** -6, 0.25] * 16
    g1 = [240.0, 2.0 ** -9] + [1.0] * 126
    w = torch.tensor([g0 + g1], dtype=torch.float32).to(torch.float8_e4m3fn)
    s = torch.tensor([[0.5, 3.0]])
    codes, s2, st = fp8_trn.rescale_to_trn(w, s, 128)
    assert codes.float().abs().max().item() <= 240
    assert s2.tolist() == [[1.0, 3.0]]
    assert st["groups"] == 2 and st["groups_rescaled"] == 1
    assert st["changed"] == 32 and st["underflow_to_zero"] == 16
    deq_new = fp8_trn.dequant_rowgroup(codes, s2, 128)
    deq_old = fp8_trn.dequant_rowgroup(w, s, 128)
    assert int((deq_new != deq_old).sum()) == st["changed"]
    assert torch.equal(deq_new[0, 128:], deq_old[0, 128:])


def test_fp8_linear_matches_block_dequant_except_counted():
    gen = torch.Generator().manual_seed(1)
    w = _rand_fp8(300, 256, gen)
    sinv = torch.rand(3, 2, generator=gen) + 0.1
    codes, s, st = fp8_trn.fp8_linear(w, sinv)
    assert s.shape == (300, 2)
    ref = dequant_fp8_block(w, sinv)
    got = fp8_trn.dequant_rowgroup(codes, s, 128)
    assert int((got != ref).sum()) == st["changed"]
    assert st["max_abs_dequant_err"] <= 2.0 ** -9 * sinv.max().item() * 2


@pytest.mark.parametrize("nh,nkv", [(64, 4), (64, 8)])
def test_split_qkv_fp8_matches_bf16_split(nh, nkv):
    gen = torch.Generator().manual_seed(2)
    hd, vd, H = 192, 128, 256
    rows = (nh // 4) * hd + (nkv // 4) * hd + (nkv // 4) * vd
    srows = (rows + 127) // 128
    w = _rand_fp8(rows * 4, H, gen)
    sinv = torch.rand(srows * 4, H // 128, generator=gen) + 0.1
    refs = split_fused_qkv(w, sinv, nh, nkv, hd, vd)
    parts = fp8_trn.split_fused_qkv_fp8(w, sinv, nh, nkv, hd, vd)
    for (codes, s), ref in zip(parts, refs):
        assert s.shape == (codes.shape[0], H // 128)
        assert torch.equal(fp8_trn.dequant_rowgroup(codes, s, 128), ref)   # before rescale: exact copy
        c2, s2, st = fp8_trn.rescale_to_trn(codes, s)
        assert int((fp8_trn.dequant_rowgroup(c2, s2, 128) != ref).sum()) == st["changed"]


def test_coarse_scale_stats_detects_spread():
    # row 0: exponents 127 and 127-13 -> lossless per row; row 1: 127 and 127-20 with code 0.5 -> lossy
    codes = torch.full((2, 64), 0.5).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 114], [127, 107]], dtype=torch.uint8)
    st = fp8_trn.coarse_scale_stats(codes, scales, None)
    assert st["rows_lossless"] == 1 and st["changed"] == 32


@pytest.mark.skipif(not HAVE_SRC, reason="original checkpoint not present")
def test_check_tool_in_memory_layer0(tmp_path):
    # dense layer only (~15 s); experts of all layers are covered by tools/check_fp8_converted.py (results/conversion/)
    out = tmp_path / "chk.json"
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_fp8_converted.py"), "--src", SRC,
                        "--layers", "0", "--workers", "1", "--out", str(out)],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    r = json.load(open(out))
    assert r["pass"] and r["layers_checked"] == 1
    assert 0 < r["linear_e4m3fn_to_trn_e4m3"]["changed_fraction"] < 0.01


@pytest.mark.skipif(not HAVE_SRC, reason="original checkpoint not present")
def test_real_expert_bit_exact():
    from safetensors import safe_open
    wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    for proj in ("gate_proj", "down_proj"):
        k = f"model.layers.1.mlp.experts.7.{proj}.weight"
        with safe_open(os.path.join(SRC, wm[k]), "pt") as f:
            w, s = f.get_tensor(k), f.get_tensor(k + "_scale")
        codes, sc = fp8_trn.mxfp4_to_fp8(w, s)
        assert torch.equal(fp8_trn.dequant_rowgroup(codes, sc, 32), dequant_mxfp4(w, s).float())


def test_convert_fp8_flag_in_help():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "convert_checkpoint.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "--fp8" in p.stdout
