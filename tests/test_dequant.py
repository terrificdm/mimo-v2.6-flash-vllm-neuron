# SPDX-License-Identifier: Apache-2.0
"""Checkpoint dequantization tests (run on the instance CPU).

Oracles:
  * vllm_neuron.model.gpt_oss.weight_loaders_bf16._dequantize_mxfp4_to_bf16 (MXFP4)
  * vllm.model_executor.models.mimo_v2._shard_fp8_qkv_proj (fused FP8 qkv, TP chunking)
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from tools.dequant import dequant_fp8_block, dequant_mxfp4, split_fused_qkv

SRC = os.environ.get("MIMO_SRC", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_SRC = os.path.isfile(os.path.join(SRC, "model.safetensors.index.json"))


def _mxfp4_oracle():
    from vllm_neuron.model.gpt_oss.weight_loaders_bf16 import _dequantize_mxfp4_to_bf16
    return _dequantize_mxfp4_to_bf16


def _qkv_oracle():
    from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj
    return _shard_fp8_qkv_proj


def _rand_fp8(rows, cols, gen):
    return (torch.randn(rows, cols, generator=gen) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)


def _src_get(key):
    from safetensors import safe_open
    wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(SRC, wm[key]), "pt") as f:
        return f.get_tensor(key)


# ----------------------------------------------------------------------------- MXFP4
def test_mxfp4_random_matches_oracle():
    gen = torch.Generator().manual_seed(0)
    R, C = 256, 1024
    blocks = torch.randint(0, 256, (R, C // 2), dtype=torch.uint8, generator=gen)
    scales = torch.randint(110, 140, (R, C // 32), dtype=torch.uint8, generator=gen)
    ours = dequant_mxfp4(blocks, scales)
    ref = _mxfp4_oracle()(blocks.view(R, C // 32, 16), scales).reshape(R, C)
    assert ours.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16
    assert torch.equal(ours, ref)


def test_mxfp4_nibble_order():
    # byte 0x21: low nibble 1 -> 0.5 (even element), high nibble 2 -> 1.0 (odd element)
    blocks = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    out = dequant_mxfp4(blocks, scales).float()
    assert out[0, 0].item() == 0.5 and out[0, 1].item() == 1.0


@pytest.mark.skipif(not HAVE_SRC, reason="original checkpoint not present")
@pytest.mark.parametrize("layer,expert", [(1, 0), (5, 255)])
def test_mxfp4_real_expert_matches_oracle(layer, expert):
    for proj in ("gate_proj", "up_proj", "down_proj"):
        k = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        w, s = _src_get(k), _src_get(k + "_scale")
        R, half = w.shape
        ours = dequant_mxfp4(w, s)
        ref = _mxfp4_oracle()(w.view(R, half // 16, 16), s).reshape(R, half * 2)
        assert torch.equal(ours, ref), k
        assert torch.isfinite(ours.float()).all()


# ----------------------------------------------------------------------------- FP8 / QKV
def test_fp8_block_dequant_partial_tiles():
    gen = torch.Generator().manual_seed(1)
    w = _rand_fp8(200, 300, gen)
    s = torch.rand(2, 3, generator=gen) + 0.5
    d = dequant_fp8_block(w, s)
    ref2 = w.float() * torch.tensor([[s[i // 128, j // 128] for j in range(300)] for i in range(200)])
    assert torch.equal(d, ref2)


def _fake_qkv(nh, nkv, hd, vd, H, gen):
    rows = (nh // 4) * hd + (nkv // 4) * hd + (nkv // 4) * vd
    srows = (rows + 127) // 128
    w = _rand_fp8(rows * 4, H, gen)
    s = torch.rand(srows * 4, (H + 127) // 128, generator=gen) + 0.1
    return w, s, rows, srows


@pytest.mark.parametrize("nh,nkv", [(64, 4), (64, 8)])
def test_split_qkv_matches_independent_reference(nh, nkv):
    gen = torch.Generator().manual_seed(2)
    hd, vd, H = 192, 128, 256
    w, s, rows, srows = _fake_qkv(nh, nkv, hd, vd, H, gen)
    q, k, v = split_fused_qkv(w, s, nh, nkv, hd, vd)
    assert q.shape == (nh * hd, H) and k.shape == (nkv * hd, H) and v.shape == (nkv * vd, H)
    qr, kr = nh // 4 * hd, nkv // 4 * hd
    for r in range(4):
        blk = w[r * rows:(r + 1) * rows].float()
        sc = s[r * srows:(r + 1) * srows]
        full = torch.stack([blk[i] * sc[i // 128].repeat_interleave(128)[:H] for i in range(rows)])
        assert torch.equal(q[r * qr:(r + 1) * qr], full[:qr])
        assert torch.equal(k[r * kr:(r + 1) * kr], full[qr:qr + kr])
        assert torch.equal(v[r * (nkv // 4 * vd):(r + 1) * (nkv // 4 * vd)], full[qr + kr:])


def test_split_qkv_ga_matches_vllm_oracle():
    """GA layout (64 q / 4 kv heads): vLLM's TP=4 shard of the fused qkv equals our per-rank q|k|v."""
    gen = torch.Generator().manual_seed(3)
    nh, nkv, hd, vd, H = 64, 4, 192, 128, 512
    w, s, rows, _ = _fake_qkv(nh, nkv, hd, vd, H, gen)
    q, k, v = split_fused_qkv(w, s, nh, nkv, hd, vd)
    shard = _qkv_oracle()
    for r in range(4):
        w_r, s_r = shard(w, s, num_heads=nh, num_kv_heads=nkv, head_dim=hd, v_head_dim=vd, tp_rank=r, tp_size=4)
        d = dequant_fp8_block(w_r, s_r)
        qr = nh // 4 * hd
        assert torch.equal(d[:qr], q[r * qr:(r + 1) * qr])
        assert torch.equal(d[qr:qr + hd], k[r * hd:(r + 1) * hd])
        assert torch.equal(d[qr + hd:], v[r * vd:(r + 1) * vd])


@pytest.mark.skipif(not HAVE_SRC, reason="original checkpoint not present")
@pytest.mark.parametrize("layer,nkv", [(1, 8), (5, 4)])
def test_real_qkv_scale_tiling_and_oracle(layer, nkv):
    """Real checkpoint: every 128x128 FP8 tile *within each TP block* has amax 448
    (proves scales are tiled per TP block), and GA layers match the vLLM oracle."""
    nh, hd, vd = 64, 192, 128
    key = f"model.layers.{layer}.self_attn.qkv_proj.weight"
    w, s = _src_get(key), _src_get(key + "_scale_inv")
    rows = w.shape[0] // 4
    srows = s.shape[0] // 4
    assert srows == (rows + 127) // 128
    amax_ok = total = 0
    for r in range(4):
        blk = w[r * rows:(r + 1) * rows].float().abs()
        for i in range(srows):
            t = blk[i * 128:(i + 1) * 128]
            m = t.reshape(t.shape[0], -1, 128).amax(dim=(0, 2))
            amax_ok += int((m == 448).sum())
            total += m.numel()
    assert amax_ok / total > 0.99, f"only {amax_ok}/{total} tiles have amax 448"
    q, k, v = split_fused_qkv(w, s, nh, nkv, hd, vd)
    assert torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()
    if nkv == 4:
        shard = _qkv_oracle()
        for r in range(4):
            w_r, s_r = shard(w, s, num_heads=nh, num_kv_heads=nkv, head_dim=hd, v_head_dim=vd, tp_rank=r, tp_size=4)
            d = dequant_fp8_block(w_r, s_r)
            assert torch.equal(d[:16 * hd], q[r * 16 * hd:(r + 1) * 16 * hd])
    # Diagnostic (layout evidence, printed with -s): per-head rope-part vs nope-part row norms.
    kn = k.norm(dim=1).reshape(nkv, hd)
    print(f"layer {layer} K head row-norm rope/nope:",
          [(round(x[:64].mean().item(), 3), round(x[64:].mean().item(), 3)) for x in kn])


# ----------------------------------------------------------------------------- converter CLI
@pytest.mark.parametrize("args", [
    ["--src", "/nonexistent", "--dst", "/tmp/x"],
    ["--src", SRC, "--dst", SRC],
    ["--src", SRC, "--dst", "/tmp/mimo_bad", "--layers", "0,99"],
    ["--src", SRC, "--dst", "/tmp/mimo_bad", "--layers", "a,b"],
    ["--src", SRC, "--dst", "/tmp/mimo_bad", "--workers", "0"],
])
def test_convert_bad_args_exit_2(args):
    if not HAVE_SRC and "/nonexistent" not in args:
        pytest.skip("original checkpoint not present")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "convert_checkpoint.py")] + args,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stderr
