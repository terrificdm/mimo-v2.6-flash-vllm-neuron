# SPDX-License-Identifier: Apache-2.0
"""Long-context attention (layer 1): segmented prefill and key-blocked decode vs the
single-shot functions of the base configuration (``prefill_attention`` / unblocked ``decode_attention``), pure PyTorch.

Segmented prefill is simulated the way vllm-neuron runs it: one request, fixed segment size S (last
segment padded), the segment's K/V written to the paged cache first (padding tokens -> null block 0),
``cached_seq_len`` = tokens before the segment, block table row with -1 for missing blocks.
Thresholds: FP32 inputs max |err| <= 1e-5; BF16 inputs relative L2 <= 1e-2.
"""

import json
import os
import types

import pytest
import torch

from mimo_v2_neuron.model.mimo_v2 import attention_torch as at
from mimo_v2_neuron.model.mimo_v2 import factory
from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config, default_prefill_key_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DQK, DV = 192, 128
SCALE = DQK ** -0.5


def _rand(shape, gen, dtype, mul=1.0):
    return (torch.randn(shape, generator=gen) * mul).to(dtype)


def _make_seq(n, hq, hkv, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return (_rand((hq, n, DQK), g, dtype), _rand((hkv, n, DQK), g, dtype), _rand((hkv, n, DV), g, dtype),
            _rand((hq,), g, torch.float32))


def _segmented(q, k, v, seg, bs, mml, key_block, window, sink, dtype, seed, free_swa=False):
    """Run ``segmented_prefill_attention`` segment by segment; returns [n, Hq*DV] for the real tokens
    and the list of per-segment outputs (incl. padding rows)."""
    hq, n, _ = q.shape
    hkv = k.shape[0]
    width = -(-mml // bs)
    num_blocks = width + 8
    g = torch.Generator().manual_seed(seed)
    shape = (num_blocks, hkv, bs, DQK)
    k_cache = _rand(shape, g, dtype, 50.0)        # garbage: any unmasked stale read shows up
    v_cache = _rand(shape, g, dtype, 50.0)
    blocks = (torch.randperm(num_blocks - 1, generator=g) + 1)[:width]   # never block 0 (null block)
    table = torch.full((1, width), -1, dtype=torch.int32)
    outs, segs = [], []
    nseg = -(-n // seg)
    for s_i in range(nseg):
        c = s_i * seg
        real = min(seg, n - c)
        # allocate the blocks this segment touches
        nblk = -(-(c + real) // bs)
        table[0, :nblk] = blocks[:nblk].to(torch.int32)
        if free_swa and window is not None:
            # sliding-window manager frees blocks wholly outside every query's window -> null (-1)
            first_needed = max(0, c - window + 1) // bs
            table[0, :first_needed] = -1
        pad = seg - real
        qs = torch.cat([q[:, c:c + real], torch.zeros(hq, pad, DQK, dtype=dtype)], dim=1)
        ks = torch.cat([k[:, c:c + real], _rand((hkv, pad, DQK), g, dtype)], dim=1)
        vs = torch.cat([v[:, c:c + real], _rand((hkv, pad, DV), g, dtype)], dim=1)
        # write the segment (padding -> slot 0..pad of null block 0 is not used: runner pads with slot 0)
        i = torch.arange(real)
        slots = blocks[(c + i) // bs] * bs + (c + i) % bs
        slots = torch.cat([slots, torch.zeros(pad, dtype=torch.long)])
        vpad = torch.nn.functional.pad(vs, (0, DQK - DV))
        for h in range(hkv):
            k_cache[slots // bs, h, slots % bs] = ks[h]
            v_cache[slots // bs, h, slots % bs] = vpad[h]
        out = at.segmented_prefill_attention(qs, ks, vs, k_cache, v_cache, table.clone(),
                                             torch.tensor([[c]], dtype=torch.int32), SCALE, DV, bs, key_block,
                                             sliding_window=window, sink=sink)
        assert out.shape == (seg, hq * DV) and out.dtype == dtype
        segs.append(out)
        outs.append(out[:real])
    return torch.cat(outs), segs


def _check(act, ref, dtype):
    assert torch.isfinite(act).all()
    if dtype == torch.float32:
        err = (act - ref).abs().max().item()
        assert err <= 1e-5, err
    else:
        rel = ((act.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 1e-2, rel


CASES = [
    # name,        hq, hkv, window, sink, bs, key_block
    ("ga_kb512_bs32", 4, 2, None, False, 32, 512),
    ("ga_kb512_bs64", 1, 1, None, False, 64, 512),
    ("ga_kb4096_bs32", 2, 1, None, False, 32, 4096),     # one prior block, larger than the prefix
    ("swa_sink_bs32", 4, 2, 128, True, 32, 512),
    ("swa_sink_bs64", 1, 1, 128, True, 64, 512),
    ("swa_nosink_w600", 2, 2, 600, False, 32, 512),      # window > segment
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("name,hq,hkv,window,sink,bs,kb", CASES, ids=[c[0] for c in CASES])
def test_segmented_prefill_matches_single_shot(name, hq, hkv, window, sink, bs, kb, dtype):
    n, seg, mml = 2300, 512, 2304          # 5 segments, last padded; 2304 is not a multiple of 512
    q, k, v, snk = _make_seq(n, hq, hkv, dtype, seed=sum(map(ord, name)))
    snk = snk if sink else None
    ref = at.prefill_attention(q, k, v, SCALE, sliding_window=window, sink=snk)
    act, segs = _segmented(q, k, v, seg, bs, mml, kb, window, snk, dtype, seed=7)
    _check(act, ref, dtype)
    for s in segs:                          # padding rows stay finite
        assert torch.isfinite(s).all()


def test_last_prior_slice_exceeds_table_width():
    # width 72 blocks (bs 32), 16 blocks per key block -> 5 slices, the last one padded with -1
    bt = torch.arange(72, dtype=torch.int32).view(1, 72)
    c = at._table_chunk(bt, 64, 16)
    assert c.shape == (1, 16) and c[0, :8].tolist() == list(range(64, 72)) and (c[0, 8:] == -1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_swa_with_freed_blocks(dtype):
    """Blocks wholly outside the window replaced by null (-1), as vLLM's sliding-window manager does."""
    n, seg, mml = 2300, 512, 2304
    q, k, v, snk = _make_seq(n, 2, 1, dtype, seed=11)
    ref = at.prefill_attention(q, k, v, SCALE, sliding_window=128, sink=snk)
    act, _ = _segmented(q, k, v, seg, 32, mml, 512, 128, snk, dtype, seed=3, free_swa=True)
    _check(act, ref, dtype)


def test_first_segment_equals_prefill_attention():
    """cached_seq_len = 0 and a single (unpadded) segment reproduces the single-shot function."""
    q, k, v, snk = _make_seq(512, 2, 1, torch.float32, seed=5)
    for window, sink in ((None, None), (128, snk)):
        ref = at.prefill_attention(q, k, v, SCALE, sliding_window=window, sink=sink)
        act, _ = _segmented(q, k, v, 512, 32, 4096, 4096, window, sink, torch.float32, seed=1)
        _check(act, ref, torch.float32)


def test_key_block_not_multiple_of_block_size():
    q, k, v, _ = _make_seq(64, 1, 1, torch.float32, seed=1)
    cache = torch.zeros(4, 1, 64, DQK)
    with pytest.raises(ValueError, match="multiple of the KV block size"):
        at.segmented_prefill_attention(q, k, v, cache, cache, torch.zeros(1, 4, dtype=torch.int32),
                                       torch.zeros(1, 1, dtype=torch.int32), SCALE, DV, 64, 4000)


# ── prefix-cache hit: one segment starting at any 32-token boundary ──
def _hit_segment(q, k, v, hit, seg, bs, mml, key_block, window, sink, dtype, seed):
    """Tokens [0, hit) already in the paged cache (written by the earlier request), SWA blocks wholly outside the
    window of the first query replaced by null (-1) as vLLM's sliding-window manager does on a hit; then one padded
    segment with the remaining tokens at ``cached_seq_len = hit``. Returns the outputs of the real tokens."""
    hq, n, _ = q.shape
    hkv = k.shape[0]
    width = -(-mml // bs)
    g = torch.Generator().manual_seed(seed)
    num_blocks = width + 8
    k_cache = _rand((num_blocks, hkv, bs, DQK), g, dtype, 50.0)
    v_cache = _rand((num_blocks, hkv, bs, DQK), g, dtype, 50.0)
    blocks = (torch.randperm(num_blocks - 1, generator=g) + 1)[:width]
    i = torch.arange(n)
    slots = blocks[i // bs] * bs + i % bs
    vpad = torch.nn.functional.pad(v, (0, DQK - DV))
    for h in range(hkv):
        k_cache[slots // bs, h, slots % bs] = k[h]
        v_cache[slots // bs, h, slots % bs] = vpad[h]
    table = torch.full((1, width), -1, dtype=torch.int32)
    table[0, :-(-n // bs)] = blocks[:-(-n // bs)].to(torch.int32)
    if window is not None:
        table[0, :max(0, hit - window + 1) // bs] = -1
    real = n - hit
    pad = seg - real
    qs = torch.cat([q[:, hit:], torch.zeros(hq, pad, DQK, dtype=dtype)], dim=1)
    ks = torch.cat([k[:, hit:], _rand((hkv, pad, DQK), g, dtype)], dim=1)
    vs = torch.cat([v[:, hit:], _rand((hkv, pad, DV), g, dtype)], dim=1)
    out = at.segmented_prefill_attention(qs, ks, vs, k_cache, v_cache, table, torch.tensor([[hit]], dtype=torch.int32),
                                         SCALE, DV, bs, key_block, sliding_window=window, sink=sink)
    return out[:real]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("window", [None, 128], ids=["ga", "swa_sink"])
@pytest.mark.parametrize("hit", [32, 576, 1000, 1312, 1600, 4128, 4672])
def test_prefix_hit_segment_matches_single_shot(hit, window, dtype):
    """Hit lengths off the 512 / 4096 grid (SWA window start then off the 4096 grid too, leading table entries null)."""
    n = hit + 21
    q, k, v, snk = _make_seq(n, 2, 1, dtype, seed=hit)
    snk = snk if window else None
    ref = at.prefill_attention(q, k, v, SCALE, sliding_window=window, sink=snk)[hit:]
    act = _hit_segment(q, k, v, hit, 512, 32, 8192, 512, window, snk, dtype, seed=hit + 1)
    _check(act, ref, dtype)


def test_nki_whole_tiles_even():
    csl = torch.tensor([0, 32, 511, 512, 576, 1023, 1024, 1600, 4672, 469984, 940000, 120000, 29984])
    assert at._nki_whole_tiles(csl).tolist() == [0, 0, 0, 0, 0, 0, 2, 2, 8, 916, 1834, 234, 58]


@pytest.mark.parametrize("width,bs", [(64, 32), (100, 32), (37, 16), (32768, 32)])
def test_nki_key_rows_arith_equals_gather(width, bs):
    """docs/design-notes.md#moe-hybrid: MIMO_PREFILL_KIDX=arith gives the same cache rows as the original index_select, including
    -1 (unallocated) table entries and positions past the table (width * bs not a multiple of 512)."""
    g = torch.Generator().manual_seed(width + bs)
    bt = torch.randperm(width + 50, generator=g)[:width].reshape(1, width).to(torch.int32)
    bt[0, width // 2:] = -1
    nt = -(-(width * bs) // at._NKI_KT)
    ref = at._nki_key_rows(bt, bs, nt, "gather")
    new = at._nki_key_rows(bt, bs, nt, "arith")
    assert ref.shape == new.shape == (nt * at._NKI_KT,)
    assert torch.equal(ref, new)
    with pytest.raises(ValueError):
        at._nki_key_rows(bt, bs, nt, "bogus")


@pytest.mark.parametrize("hit,ntile", [(32, 0), (576, 0), (1000, 0), (1600, 2), (4672, 8)])
def test_nki_prefill_kernel_gets_even_tiles(hit, ntile, monkeypatch):
    """MIMO_PREFILL_ATTN=nki plumbing: the kernel launch (2 programs) gets an even tile count and the masked torch
    remainder block covers the rest; with no whole tiles the result equals the torch path."""
    pytest.importorskip("libtorch_neuronx_lite.nki.nki_hop")
    import libtorch_neuronx_lite.nki.nki_hop as hop

    seen = []

    class _Launch:
        def __getitem__(self, nprog):
            def run(qt, k_cache, v_cache, kidx, ntile, scale):
                seen.append((nprog, int(ntile.reshape(-1)[0])))
                s = qt.shape[1]
                return (torch.full((nprog, s, 1), torch.finfo(torch.float32).min), torch.zeros(nprog, s, 1),
                        torch.zeros(nprog, s, DV))
            return run

    monkeypatch.setattr(hop, "wrap_nki", lambda kernel: _Launch())
    n = hit + 21
    q, k, v, snk = _make_seq(n, 1, 1, torch.float32, seed=hit)
    monkeypatch.setenv("MIMO_PREFILL_ATTN", "nki")
    act = _hit_segment(q, k, v, hit, 512, 32, 8192, 512, None, snk, torch.float32, seed=3)
    assert seen == [(2, ntile)] * len(seen) and seen
    if ntile == 0:
        ref = at.prefill_attention(q, k, v, SCALE, sink=snk)[hit:]
        _check(act, ref, torch.float32)


# ── decode ─────────────────────────────────────────────────────────────────
def _decode_setup(bs, dtype, hq, hkv, seed):
    kb = 512
    length = 3 * kb + 17
    width = -(-length // bs)
    num_blocks = width * 2 + 4
    g = torch.Generator().manual_seed(seed)
    k_cache = _rand((num_blocks, hkv, bs, DQK), g, dtype)
    v_cache = _rand((num_blocks, hkv, bs, DQK), g, dtype)
    perm = (torch.randperm(num_blocks - 1, generator=g) + 1).to(torch.int32)
    bt = torch.full((3, width), -1, dtype=torch.int32)        # row 2 = padding row
    bt[0] = perm[:width]                                       # full length
    n1 = -(-700 // bs)
    bt[1, :n1] = perm[width:width + n1]                        # 700 tokens
    positions = torch.tensor([length - 1, 699, 0])
    q = _rand((3, hq, DQK), g, dtype)
    sink = _rand((hq,), g, torch.float32)
    return q, k_cache, v_cache, bt, positions, sink, kb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("bs", [32, 64])
@pytest.mark.parametrize("window,use_sink", [(None, False), (128, True)], ids=["ga", "swa_sink"])
def test_decode_blocked_matches_unblocked(dtype, bs, window, use_sink):
    q, kc, vc, bt, pos, sink, kb = _decode_setup(bs, dtype, 4, 2, seed=bs)
    sink = sink if use_sink else None
    off = torch.zeros(3, dtype=torch.int32) if window else None
    ref = at.decode_attention(q, kc, vc, bt, pos, SCALE, DV, sliding_window=window, sink=sink, kv_pos_offset=off)
    act = at.decode_attention(q, kc, vc, bt, pos, SCALE, DV, sliding_window=window, sink=sink,
                              kv_pos_offset=off, key_block=kb)
    assert bt.shape[1] * bs > kb      # blocked path taken
    assert act.shape == ref.shape and torch.isfinite(act).all()
    _check(act[:2], ref[:2], dtype)   # row 2 (padding) only needs to be finite


def test_decode_short_table_uses_single_shot_path():
    """L <= key_block: identical result (same function body as the single-shot path)."""
    q, kc, vc, bt, pos, sink, _ = _decode_setup(32, torch.bfloat16, 2, 1, seed=2)
    ref = at.decode_attention(q, kc, vc, bt, pos, SCALE, DV)
    act = at.decode_attention(q, kc, vc, bt, pos, SCALE, DV, key_block=32768)
    assert torch.equal(ref, act)


# ── config / factory ──────────────────────────────────────────────────────
def test_default_key_blocks():
    assert [default_prefill_key_block(x) for x in (2048, 4096, 16384, 32768, 131072, 524288, 1048576, None)] \
        == [4096, 4096, 4096, 8192, 16384, 32768, 32768, 4096]


def _hf_dict():
    d = json.load(open(os.path.join(ROOT, "reference/hf/config.json")))
    d.pop("quantization_config")
    d["attention_projection_layout"] = "split"
    return d


RT = dict(tp=64, ep=64, enable_prefix_caching=False, speculative_config=None,
          max_num_batched_tokens=4096, max_model_len=32768, block_size=32)


def _validate(nc, rt=None):
    r = dict(RT, **(rt or {}))
    cfg = MiMoV2Config.from_configs(_hf_dict(), nc, max_model_len=r["max_model_len"])
    factory.MiMoV2ForCausalLM._validate_config(cfg, nc, r)
    return cfg


def test_segmented_config_accepted(monkeypatch):
    for v in ("MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK"):
        monkeypatch.delenv(v, raising=False)
    nc = types.SimpleNamespace(kv_segment_size_buckets=[4096], num_batched_tokens_buckets=[4096])
    cfg = _validate(nc)
    assert cfg.prefill_key_block == 8192 and cfg.decode_key_block == 32768
    _validate(nc, {"enable_prefix_caching": True})   # prefix caching (docs/limitations.md): allowed with segmented prefill


@pytest.mark.parametrize("nc,match", [
    (types.SimpleNamespace(kv_segment_size_buckets=[4096], num_batched_tokens_buckets=[1024, 4096]),
     "must equal kv_segment_size_buckets"),
    (types.SimpleNamespace(kv_segment_size_buckets=[1024, 4096], num_batched_tokens_buckets=[1024, 4096]),
     "single value"),
    (types.SimpleNamespace(kv_segment_size_buckets=[8192], num_batched_tokens_buckets=[8192]), "single value"),
    (None, "max_num_batched_tokens"),       # chunked prefill without segmentation
])
def test_segmented_config_rejected(nc, match):
    with pytest.raises(ValueError, match="^MiMoV2 Neuron: .*" + match):
        _validate(nc)


def test_prefill_buckets_env(monkeypatch):
    """MIMO_PREFILL_BUCKETS adds prefill buckets below the segment size (num_batched_tokens_buckets)."""
    for v in ("MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MIMO_PREFILL_BUCKETS", "4096,512")
    nc = types.SimpleNamespace(kv_segment_size_buckets=[4096], num_batched_tokens_buckets=[4096])
    _validate(nc)
    assert nc.num_batched_tokens_buckets == [512, 4096]
    for bad in ("512,2048", "300,4096", "x"):
        monkeypatch.setenv("MIMO_PREFILL_BUCKETS", bad)
        nc = types.SimpleNamespace(kv_segment_size_buckets=[4096], num_batched_tokens_buckets=[4096])
        with pytest.raises(ValueError, match="MIMO_PREFILL_BUCKETS"):
            _validate(nc)


@pytest.mark.parametrize("env,val,match", [
    ("MIMO_ATTN_PREFILL_KEY_BLOCK", "1000", "multiple of the KV block size"),
    ("MIMO_ATTN_PREFILL_KEY_BLOCK", "64", "within"),
    ("MIMO_ATTN_DECODE_KEY_BLOCK", "131072", "within"),
    ("MIMO_ATTN_DECODE_KEY_BLOCK", "abc", "must be an integer"),
])
def test_key_block_env_rejected(monkeypatch, env, val, match):
    monkeypatch.setenv(env, val)
    nc = types.SimpleNamespace(kv_segment_size_buckets=[4096], num_batched_tokens_buckets=[4096])
    with pytest.raises(ValueError, match="^MiMoV2 Neuron: .*" + match):
        _validate(nc)


def test_key_block_env_override(monkeypatch):
    monkeypatch.setenv("MIMO_ATTN_PREFILL_KEY_BLOCK", "512")
    monkeypatch.setenv("MIMO_ATTN_DECODE_KEY_BLOCK", "512")
    nc = types.SimpleNamespace(kv_segment_size_buckets=[512], num_batched_tokens_buckets=[512])
    cfg = _validate(nc, rt=dict(max_num_batched_tokens=512, max_model_len=2048))
    assert cfg.prefill_key_block == 512 and cfg.decode_key_block == 512


# ── module: MiMoV2Attention segmented branch (real L3 weights, TP=1) ──────
L3_DIR = os.environ.get("MIMO_L3_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16-L3"))
needs_l3 = pytest.mark.skipif(not os.path.isfile(os.path.join(L3_DIR, "model.safetensors.index.json")),
                              reason="L3 checkpoint not available")


@needs_l3
@pytest.mark.parametrize("layer", [0, 1], ids=["ga", "swa_sink"])
def test_module_segmented_prefill(layer, monkeypatch):
    from tests import test_modules_cpu as tm

    monkeypatch.setenv("MIMO_ATTN_PREFILL_KEY_BLOCK", "256")
    n, seg, bs = 1000, 256, tm.BS
    x = tm._layer_input(layer, n, tm.SEED + 300 + layer)
    num_blocks = 40
    blocks = (torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(9)) + 1)[:32].tolist()
    width = 32                                   # max_model_len 1024
    cfg = tm._our_cfg()
    pos = torch.arange(n)
    full, _ = tm._our_attention(layer, 1, 0, num_blocks)
    with torch.no_grad():
        ref = full.forward_prefill(x, pos, tm._rope(cfg, layer, pos),
                                   {"slot_mapping": tm._slots(blocks, n), "block_size": bs, "kv_segment_size": 0})

    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2Attention

    scfg = MiMoV2Config.from_configs(tm._cfg_dict(), None, max_model_len=1024)
    assert scfg.prefill_key_block == 256
    a = MiMoV2Attention(scfg, layer, tp_group=tm._fake_group(1, 0))
    tm._load_params(a, MiMoV2Attention.checkpoint_keys(layer, a.sinks is not None), 0)
    a.k_cache, a.v_cache = full.k_cache.clone().normal_(0, 50), full.v_cache.clone().normal_(0, 50)
    table = torch.full((1, width), -1, dtype=torch.int32)
    slots_all = tm._slots(blocks, n)
    # step == seg: one segment per bucket; step == seg // 2: a prefill bucket smaller than kv_segment_size
    # (MIMO_PREFILL_BUCKETS, T < seg)
    for step in (seg, seg // 2):
        outs = []
        with torch.no_grad():
            for c in range(0, n, step):
                real = min(step, n - c)
                table[0, :-(-(c + real) // bs)] = torch.tensor(blocks[:-(-(c + real) // bs)], dtype=torch.int32)
                p = torch.cat([torch.arange(c, c + real), torch.zeros(step - real, dtype=torch.long)])
                h = torch.cat([x[c:c + real], torch.zeros(step - real, x.shape[1], dtype=x.dtype)])
                slots = torch.cat([slots_all[c:c + real], torch.zeros(step - real, dtype=torch.long)])
                meta = {"slot_mapping": slots, "block_size": bs, "kv_segment_size": seg,
                        "block_table_tensor": table.clone(),
                        "cached_seq_len": torch.tensor([[c]], dtype=torch.int32)}
                o = a.forward_prefill(h, p, tm._rope(cfg, layer, p), meta)
                assert torch.isfinite(o).all()
                outs.append(o[:real])
            with pytest.raises(ValueError, match="segmented prefill expects T <= kv_segment_size"):
                p2 = torch.cat([p] * (2 * seg // step))
                a.forward_prefill(torch.cat([h] * (2 * seg // step)), p2, tm._rope(cfg, layer, p2), meta)
        act = torch.cat(outs)
        rel = ((act.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"[module_segmented layer{layer} step {step}] relL2 vs single-shot = {rel:.2e}")
        assert rel <= 1e-2, rel
