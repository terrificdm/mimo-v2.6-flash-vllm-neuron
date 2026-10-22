# SPDX-License-Identifier: Apache-2.0
"""Module-level CPU tests against the HF reference (VLLM_NEURON_CPU_MODE=1, no NKI_SIMULATOR).

Uses the real L3 reduced checkpoint (original layers 0 / 1 / 5 -> new 0 / 1 / 2):
  layer 0: global attention (GA, 4 KV heads, theta 1e7, no sink) + dense FFN
  layer 1: sliding-window attention (SWA, 8 KV heads, theta 1e4, sink) + MoE
Three-way comparison (``vllm_neuron.accuracy.testing.assert_close_three_way``):
  baseline = HF FP32, expected = HF BF16, actual = this implementation (BF16).
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from tests.hf_ref import load as load_hf

L3_DIR = os.environ.get("MIMO_L3_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16-L3"))
needs_l3 = pytest.mark.skipif(not os.path.isfile(os.path.join(L3_DIR, "model.safetensors.index.json")),
                              reason="L3 checkpoint not available")
BS = 32            # KV block size (Neuron default)
MAX_MODEL_LEN = 1024
SEED = 1234


# ── helpers ────────────────────────────────────────────────────────────────
def _cfg_dict():
    with open(os.path.join(L3_DIR, "config.json")) as f:
        return json.load(f)


def _our_cfg():
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config

    return MiMoV2Config.from_configs(_cfg_dict(), None)


def _hf_cfg():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(L3_DIR, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    return cfg


class _Ckpt:
    def __init__(self):
        from safetensors import safe_open

        with open(os.path.join(L3_DIR, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self._files = {}
        self._open = safe_open

    def _f(self, key):
        fn = self.wm[key]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(L3_DIR, fn), "pt")
        return self._files[fn]

    def slice(self, key):
        return self._f(key).get_slice(key)

    def tensor(self, key):
        return self._f(key).get_tensor(key)


_CKPT = None
_EMBED = None


def ckpt():
    global _CKPT
    if _CKPT is None:
        _CKPT = _Ckpt()
    return _CKPT


def _load_params(module, mapping, rank):
    from vllm_neuron.utils.weight_loader import get_weight_loader

    for pname, keys in mapping.items():
        keys = keys if isinstance(keys, list) else [keys]
        param = getattr(module, pname)
        t = get_weight_loader(param).load([ckpt().slice(k) for k in keys], rank)
        assert tuple(t.shape) == tuple(param.shape), (pname, t.shape, param.shape)
        param.data = t.to(param.dtype)


def _layer_input(layer, n_tokens, seed, norm="input_layernorm"):
    """Realistic module input: RMSNorm(embed(random token ids)) with the layer's ``norm`` weight (BF16)."""
    global _EMBED
    if _EMBED is None:
        _EMBED = ckpt().tensor("model.embed_tokens.weight")
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 151000, (n_tokens,), generator=g)
    x = _EMBED[ids]
    w = ckpt().tensor(f"model.layers.{layer}.{norm}.weight")
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (w * h.to(torch.bfloat16)).to(torch.bfloat16)


def _hf_attention(layer, dtype):
    mdl, _ = load_hf()
    cfg = _hf_cfg()
    is_swa = cfg.hybrid_layer_pattern[layer] == 1
    m = mdl.MiMoV2Attention(cfg, is_swa, layer, projection_layout="split")
    p = f"model.layers.{layer}.self_attn."
    sd = {n: ckpt().tensor(p + n) for n, _ in m.named_parameters()}
    m.load_state_dict(sd, strict=True)
    rot = mdl.MiMoV2RotaryEmbedding(cfg, is_swa)
    return m.to(dtype).eval(), rot, cfg


def _hf_mask(t, window, dtype):
    qi = torch.arange(t).unsqueeze(1)
    ki = torch.arange(t).unsqueeze(0)
    masked = ki > qi
    if window is not None:
        masked = masked | (qi - ki >= window)
    return torch.zeros(t, t, dtype=dtype).masked_fill(masked, torch.finfo(dtype).min).view(1, 1, t, t)


@torch.no_grad()
def _hf_attn_out(layer, x_bf16, dtype):
    """HF MiMoV2Attention on one full sequence -> [T, H] in fp32."""
    m, rot, cfg = _hf_attention(layer, dtype)
    t = x_bf16.shape[0]
    x = x_bf16.to(dtype).unsqueeze(0)
    pos = torch.arange(t).unsqueeze(0)
    pe = rot(x, pos)
    window = cfg.sliding_window if cfg.hybrid_layer_pattern[layer] == 1 else None
    out, _ = m(hidden_states=x, position_embeddings=pe, attention_mask=_hf_mask(t, window, dtype),
               position_ids=pos)
    return out.squeeze(0).float()


def _fake_group(tp, rank):
    return SimpleNamespace(world_size=tp, rank_in_group=rank)


def _our_attention(layer, tp, rank, num_blocks):
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2Attention

    cfg = _our_cfg()
    a = MiMoV2Attention(cfg, layer, tp_group=_fake_group(tp, rank))
    _load_params(a, MiMoV2Attention.checkpoint_keys(layer, a.sinks is not None), rank)
    g = torch.Generator().manual_seed(rank + 7)
    shape = (num_blocks, a.num_kv_heads_per_rank, BS, a.cache_head_dim)
    # garbage (not zeros) so that any unmasked read of a foreign / stale slot shows up
    a.k_cache = (torch.randn(shape, generator=g) * 50).to(torch.bfloat16)
    a.v_cache = (torch.randn(shape, generator=g) * 50).to(torch.bfloat16)
    return a, cfg


def _rope(cfg, layer, positions):
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2RotaryEmbedding

    d = cfg.attn_dims(layer)
    return MiMoV2RotaryEmbedding(d["rope_theta"], cfg.rope_dim(layer))(positions, cfg.torch_dtype)


def _slots(blocks, n):
    """Token slots 0..n-1 of a sequence whose KV lives in ``blocks`` (list of block ids)."""
    i = torch.arange(n)
    return torch.tensor(blocks)[i // BS] * BS + i % BS


def _three_way(base, exp, act, name):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    for b, e, a in zip(base, exp, act):
        assert b.shape == e.shape == a.shape, (name, b.shape, e.shape, a.shape)
        assert torch.isfinite(a).all(), name
    r = assert_close_three_way(base, exp, act, name=name)
    rel = [float((a.float() - b).norm() / b.norm()) for b, a in zip(base, act)]
    rel_hf = [float((e.float() - b).norm() / b.norm()) for b, e in zip(base, exp)]
    print(f"[{name}] sigma_ratio={r.sigma_ratio:.4f} bc={r.bc:.4f} linf_ratio={r.linf_ratio:.3f} "
          f"l2_ratio={r.l2_ratio:.3f} relL2(ours vs fp32)={max(rel):.5f} relL2(hf_bf16 vs fp32)={max(rel_hf):.5f}")
    assert r.passed, r.summary()
    return r


# ── kv_head_for_rank ───────────────────────────────────────────────────────
def test_kv_head_for_rank():
    from mimo_v2_neuron.model.mimo_v2.weight_loaders import kv_head_for_rank, kv_heads_for_rank

    for r in range(64):
        assert kv_head_for_rank(r, 64, 4) == r // 16
        assert kv_head_for_rank(r, 64, 8) == r // 8
        assert len(kv_heads_for_rank(r, 64, 8)) == 1
    assert list(kv_heads_for_rank(1, 2, 4)) == [2, 3]
    assert list(kv_heads_for_rank(3, 4, 8)) == [6, 7]
    assert list(kv_heads_for_rank(5, 8, 4)) == [2]
    with pytest.raises(ValueError):
        kv_heads_for_rank(0, 6, 4)


# ── RoPE ───────────────────────────────────────────────────────────────────
@needs_l3
@pytest.mark.parametrize("layer", [0, 1], ids=["ga_theta1e7", "swa_theta1e4"])
def test_rope_matches_hf(layer):
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2RotaryEmbedding, apply_partial_rope

    mdl, _ = load_hf()
    hcfg = _hf_cfg()
    cfg = _our_cfg()
    is_swa = hcfg.hybrid_layer_pattern[layer] == 1
    hf_rot = mdl.MiMoV2RotaryEmbedding(hcfg, is_swa)
    d = cfg.attn_dims(layer)
    ours = MiMoV2RotaryEmbedding(d["rope_theta"], cfg.rope_dim(layer))
    assert cfg.rope_dim(layer) == 64
    assert torch.equal(ours.inv_freq, hf_rot.inv_freq.float())

    g = torch.Generator().manual_seed(SEED)
    pos = torch.cat([torch.arange(512), torch.tensor([1000, 4095, 65535, 1048575])])
    t = pos.shape[0]
    x = torch.randn(t, 4, 192, generator=g).to(torch.bfloat16)   # [T, nh, D]

    def hf_apply(dtype):
        xx = x.to(dtype).transpose(0, 1).unsqueeze(0)            # [1, nh, T, D]
        cos, sin = hf_rot(xx, pos.unsqueeze(0))
        rope, nope = xx.split([64, 128], dim=-1)
        rope, _ = mdl.apply_rotary_pos_emb(rope, rope, cos, sin)
        return torch.cat([rope, nope], dim=-1).squeeze(0).transpose(0, 1).float()

    cos, sin = ours(pos, torch.bfloat16)
    act = apply_partial_rope(x, cos, sin)
    assert torch.equal(act[..., 64:], x[..., 64:])
    _three_way([hf_apply(torch.float32)], [hf_apply(torch.bfloat16)], [act], f"rope.layer{layer}")


# ── attention prefill ─────────────────────────────────────────────────────
@needs_l3
@pytest.mark.parametrize("layer", [0, 1], ids=["ga", "swa_sink"])
def test_attention_prefill(layer):
    t = 400
    x = _layer_input(layer, t, SEED + layer)
    base = _hf_attn_out(layer, x, torch.float32)
    exp = _hf_attn_out(layer, x, torch.bfloat16)

    num_blocks = 40
    blocks = torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(3))[:13] + 1
    slots = _slots(blocks.tolist(), t)
    meta = {"slot_mapping": slots, "block_size": BS, "kv_segment_size": 0,
            "max_query_len": t, "decode_token_threshold": 1}
    pos = torch.arange(t)

    a, cfg = _our_attention(layer, 1, 0, num_blocks)
    assert a.sliding_window == (128 if layer == 1 else None)
    assert (a.sinks is not None) == (layer == 1)
    pe = _rope(cfg, layer, pos)
    with torch.no_grad():
        out1 = a.forward_prefill(x, pos, pe, meta)
    _three_way([base], [exp], [out1], f"attn_prefill.layer{layer}.tp1")

    # KV cache contents: K = RoPE'd k (192), V = 0.707*v zero-padded to 192.
    with torch.no_grad():
        q, k, v = a._project(x, *pe)
    blk, off = slots // BS, slots % BS
    assert torch.equal(a.k_cache[blk, :, off, :], k)
    assert torch.equal(a.v_cache[blk, :, off, :128], v)
    assert (a.v_cache[blk, :, off, 128:] == 0).all()

    # TP=64 emulation: per-rank weights from the loaders, O-proj partials summed (== all-reduce).
    acc = torch.zeros(t, cfg.hidden_size)
    for r in range(64):
        ar, _ = _our_attention(layer, 64, r, num_blocks)
        assert ar.num_heads_per_rank == 1 and ar.num_kv_heads_per_rank == 1
        with torch.no_grad():
            acc += ar.forward_prefill(x, pos, pe, meta).float()
    _three_way([base], [exp], [acc.to(torch.bfloat16)], f"attn_prefill.layer{layer}.tp64")


# ── attention decode (paged cache, batch of 3 incl. a padding row) ────────
def _swa_trim(full_bt, ctx_lens, window):
    """Replicates NeuronModelRunner._compute_swa_num_blocks/_compute_swa_decode_tensors."""
    p_max = 128
    min_blocks = window // BS + 1
    per = p_max // BS
    n = (min_blocks + per - 1) // per * per
    n = min(n, (MAX_MODEL_LEN + BS - 1) // BS)
    start = torch.clamp(ctx_lens // BS - n + 1, min=0)
    idx = start.unsqueeze(1) + torch.arange(n).unsqueeze(0)
    return full_bt.gather(1, idx.long()), (start * BS).to(torch.int32)


@needs_l3
@pytest.mark.parametrize("layer,trim", [(0, False), (1, False), (1, True)],
                         ids=["ga", "swa_full_table", "swa_trimmed_table"])
@pytest.mark.parametrize("tp", [1, 64])
def test_attention_decode(layer, trim, tp):
    lens = [400, 150]
    xs = [_layer_input(layer, n, SEED + 100 * i + layer) for i, n in enumerate(lens)]
    base = [_hf_attn_out(layer, x, torch.float32)[-1:] for x in xs]
    exp = [_hf_attn_out(layer, x, torch.bfloat16)[-1:] for x in xs]

    num_blocks = 40
    perm = (torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(5)) + 1).tolist()
    seq_blocks = [perm[:13], perm[13:18]]           # 400 -> 13 blocks, 150 -> 5 blocks
    width = MAX_MODEL_LEN // BS
    full_bt = torch.full((3, width), -1, dtype=torch.int32)   # row 2 = padding row (null block sentinel)
    for i, b in enumerate(seq_blocks):
        full_bt[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
    cfg = _our_cfg()

    acc = torch.zeros(3, cfg.hidden_size)
    for r in range(tp):
        a, _ = _our_attention(layer, tp, r, num_blocks)
        with torch.no_grad():
            # prefill the first n-1 tokens of each sequence separately (one request per prefill step)
            for x, n, b in zip(xs, lens, seq_blocks):
                pos = torch.arange(n - 1)
                meta = {"slot_mapping": _slots(b, n - 1), "block_size": BS, "kv_segment_size": 0,
                        "max_query_len": n - 1, "decode_token_threshold": 1}
                a.forward_prefill(x[: n - 1], pos, _rope(cfg, layer, pos), meta)
            # one batched decode step: seq0 @ pos 399, seq1 @ pos 149, padding row @ pos 0 / slot 0
            hid = torch.stack([xs[0][-1], xs[1][-1], torch.zeros_like(xs[0][-1])])
            pos = torch.tensor([lens[0] - 1, lens[1] - 1, 0])
            slot = torch.tensor([_slots(seq_blocks[0], lens[0])[-1], _slots(seq_blocks[1], lens[1])[-1], 0])
            bt = full_bt
            meta = {"slot_mapping": slot, "block_size": BS, "max_query_len": 1, "decode_token_threshold": 1,
                    "block_table_tensor": bt}
            if trim:
                ctx = torch.tensor([lens[0] - 1, lens[1] - 1, 0])
                bt, off = _swa_trim(full_bt, ctx, a.sliding_window)
                assert bt.shape[1] == 8 and off.tolist() == [160, 0, 0]
                meta.update(block_table_tensor=bt, swa_kv_pos_offset=off)
            elif layer == 1:
                meta["swa_kv_pos_offset"] = torch.zeros(3, dtype=torch.int32)
            acc += a.forward_decode(hid, pos, _rope(cfg, layer, pos), meta).float()
    out = acc.to(torch.bfloat16)
    assert torch.isfinite(out[2]).all(), "padding row must stay finite"
    _three_way(base, exp, [out[0:1], out[1:2]], f"attn_decode.layer{layer}.trim{int(trim)}.tp{tp}")


# ── dense FFN (layer 0) ───────────────────────────────────────────────────
@needs_l3
@pytest.mark.parametrize("tp", [1, 64])
def test_dense_mlp(tp):
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2DenseMLP

    mdl, _ = load_hf()
    t = 96
    x = _layer_input(0, t, SEED + 11, norm="post_attention_layernorm")
    p = "model.layers.0.mlp."
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = mdl.MiMoV2MLP(_hf_cfg())
        m.load_state_dict({n: ckpt().tensor(p + n) for n, _ in m.named_parameters()}, strict=True)
        with torch.no_grad():
            outs[dt] = m.to(dt)(x.to(dt)).float()
    cfg = _our_cfg()
    acc = torch.zeros(t, cfg.hidden_size)
    for r in range(tp):
        d = MiMoV2DenseMLP(cfg, 0, tp_group=_fake_group(tp, r))
        assert d.intermediate_size_per_rank == 16384 // tp
        _load_params(d, MiMoV2DenseMLP.checkpoint_keys(0), r)
        with torch.no_grad():
            acc += d.forward_local(x).float()
    _three_way([outs[torch.float32]], [outs[torch.bfloat16]], [acc.to(torch.bfloat16)], f"dense_mlp.tp{tp}")


# ── MoE (layer 1, 256 experts top-8) ──────────────────────────────────────
_HF_MOE = {}


def _hf_moe(layer, dtype):
    """HF MiMoV2MoE with real weights; routing bias restored to FP32 also in the BF16 copy (design e.2)."""
    if (layer, dtype) not in _HF_MOE:
        mdl, _ = load_hf()
        with torch.device("meta"):
            m = mdl.MiMoV2MoE(_hf_cfg())
        p = f"model.layers.{layer}.mlp."
        m.load_state_dict({n: ckpt().tensor(p + n) for n, _ in m.named_parameters()}, strict=True, assign=True)
        m = m.to(dtype).eval()
        bias_key = p + "gate.e_score_correction_bias"
        assert ckpt().slice(bias_key).get_dtype() == "F32"
        m.gate.e_score_correction_bias.data = ckpt().tensor(bias_key).float()
        _HF_MOE[(layer, dtype)] = m
    return _HF_MOE[(layer, dtype)]


@torch.no_grad()
def _hf_moe_out(layer, x, dtype):
    m = _hf_moe(layer, dtype)
    return m(x.to(dtype).unsqueeze(0)).squeeze(0).float()


@torch.no_grad()
def _hf_topk(layer, x, dtype):
    idx, _ = _hf_moe(layer, dtype).gate(x.to(dtype).unsqueeze(0))
    return idx


def _our_moe(layer, ep_degree, ep_rank, impl="tkg", moe_tp=1):
    """MiMoV2MoE for global rank ``ep_rank * moe_tp + moe_tp_rank`` emulated with fake groups."""
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE

    cfg = _our_cfg()
    cfg.decode_moe_impl = impl
    mods = []
    for j in range(moe_tp):
        world = ep_degree * moe_tp
        r = ep_rank * moe_tp + j
        m = MiMoV2MoE(cfg, layer, tp_group=_fake_group(world, r),
                      ep={"degree": ep_degree, "rank": ep_rank, "tp_group": _fake_group(moe_tp, j)})
        _load_params(m, m.checkpoint_keys(), r)
        mods.append(m)
    return mods


@needs_l3
def test_moe_weights_and_routing():
    """Loader layout ([E_L, H, 2, I] gate|up, [E_L, I, H]), FP32 bias, routing == HF-BF16 routing."""
    m = _our_moe(1, 64, 5)[0]
    assert list(m.local_experts) == [20, 21, 22, 23]
    assert m.e_score_correction_bias.dtype == torch.float32
    assert tuple(m.gate_up_proj_weight.shape) == (4, 4096, 2, 2048)
    assert tuple(m.down_proj_weight.shape) == (4, 2048, 4096)
    p = "model.layers.1.mlp.experts.21."
    assert torch.equal(m.gate_up_proj_weight[1, :, 0, :], ckpt().tensor(p + "gate_proj.weight").t())
    assert torch.equal(m.gate_up_proj_weight[1, :, 1, :], ckpt().tensor(p + "up_proj.weight").t())
    assert torch.equal(m.down_proj_weight[1], ckpt().tensor(p + "down_proj.weight").t())
    assert torch.equal(m.e_score_correction_bias, ckpt().tensor("model.layers.1.mlp.gate.e_score_correction_bias"))
    # EP16 x TP4: intermediate slice of moe-TP rank 3
    m4 = _our_moe(1, 16, 2, moe_tp=4)[3]
    assert tuple(m4.gate_up_proj_weight.shape) == (16, 4096, 2, 512)
    assert torch.equal(m4.gate_up_proj_weight[1, :, 0, :],
                       ckpt().tensor("model.layers.1.mlp.experts.33.gate_proj.weight")[1536:2048].t())
    # routing: our FP32 routing on BF16 input == HF BF16 gate (with FP32 bias) as top-8 sets
    x = _layer_input(1, 200, SEED + 21, norm="post_attention_layernorm")
    dense, idx = m.route(x)
    hf_idx = _hf_topk(1, x, torch.bfloat16)
    same = (torch.sort(idx, -1).values == torch.sort(hf_idx, -1).values).all(-1)
    assert same.all(), f"{int((~same).sum())} tokens route differently from HF"
    assert torch.allclose(dense.sum(-1), torch.ones(200), atol=1e-5)
    assert dense.dtype == torch.float32 and int((dense > 0).sum(-1).min()) == 8


@needs_l3
@pytest.mark.parametrize("ep", [1, 64], ids=["noep_all256", "ep64"])
def test_moe_prefill(ep):
    """Prefill path (route -> get_local_expert_affinities -> build_blockwise_mapping -> NF.moe_cte CPU
    fallback, POST_SCALE) summed over ranks vs HF MiMoV2MoE; trailing padding rows must be masked."""
    t, pad = 160, 32
    x = _layer_input(1, t, SEED + 31, norm="post_attention_layernorm")
    base = _hf_moe_out(1, x, torch.float32)
    exp = _hf_moe_out(1, x, torch.bfloat16)
    xp = torch.cat([x, (torch.randn(pad, x.shape[1]) * 3).to(x.dtype)])      # padded bucket
    positions = torch.cat([torch.arange(t), torch.zeros(pad, dtype=torch.long)])
    acc = torch.zeros(t + pad, 4096)
    for r in range(ep):
        m = _our_moe(1, ep, r)[0]
        with torch.no_grad():
            dense, _ = m.route(xp)
            acc += m.experts_prefill(xp, dense, positions).float()
    out = acc.to(torch.bfloat16)
    assert (out[t:] == 0).all(), "padding rows must not receive expert output"
    _three_way([base], [exp], [out[:t]], f"moe_prefill.ep{ep}")


@needs_l3
@pytest.mark.parametrize("impl", ["tkg", "cte", "torch"])
def test_moe_decode(impl):
    """Decode path (B=4 replicated tokens) under every MIMO_DECODE_MOE_IMPL value; on CPU without the NKI
    simulator ``tkg`` takes the PyTorch path. EP=64 partial sums (== all-reduce) vs HF."""
    b = 4
    x = _layer_input(1, b, SEED + 41, norm="post_attention_layernorm")
    base = _hf_moe_out(1, x, torch.float32)
    exp = _hf_moe_out(1, x, torch.bfloat16)
    acc = torch.zeros(b, 4096)
    for r in range(64):
        m = _our_moe(1, 64, r, impl=impl)[0]
        with torch.no_grad():
            dense, idx = m.route(x)
            acc += m.experts_decode(x, dense, idx).float()
    _three_way([base], [exp], [acc.to(torch.bfloat16)], f"moe_decode.{impl}.ep64")


@needs_l3
def test_moe_decode_ep16_tp4():
    """Fallback layout EP=16 x MoE-TP=4 (intermediate sharded): PyTorch experts, partial sums."""
    b = 4
    x = _layer_input(1, b, SEED + 51, norm="post_attention_layernorm")
    base = _hf_moe_out(1, x, torch.float32)
    exp = _hf_moe_out(1, x, torch.bfloat16)
    acc = torch.zeros(b, 4096)
    for e in range(16):
        for m in _our_moe(1, 16, e, impl="torch", moe_tp=4):
            with torch.no_grad():
                dense, idx = m.route(x)
                acc += m.experts_decode(x, dense, idx).float()
    _three_way([base], [exp], [acc.to(torch.bfloat16)], "moe_decode.torch.ep16_tp4")


def test_post_scale_semantics():
    """_torch_local_experts == NF.moe_cte POST_SCALE fallback == explicit HF-style loop (random tiny)."""
    import vllm_neuron.functional as NF
    from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    from mimo_v2_neuron.model.mimo_v2.model import _torch_local_experts

    g = torch.Generator().manual_seed(7)
    t, h, i, e_l = 6, 64, 32, 4
    x = torch.randn(t, h, generator=g).to(torch.bfloat16)
    gu = (torch.randn(e_l, h, 2, i, generator=g) * 0.1).to(torch.bfloat16)
    dn = (torch.randn(e_l, i, h, generator=g) * 0.1).to(torch.bfloat16)
    aff = torch.rand(t, e_l, generator=g) * (torch.rand(t, e_l, generator=g) > 0.4)
    ours = _torch_local_experts(x, gu, dn, aff)
    ref = torch.zeros(t, h)
    for tt in range(t):
        for e in range(e_l):
            if aff[tt, e] > 0:
                y = (torch.nn.functional.silu(x[tt] @ gu[e, :, 0]) * (x[tt] @ gu[e, :, 1])) @ dn[e]
                ref[tt] += y.float() * aff[tt, e]
    assert torch.allclose(ours.float(), ref.to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)
    cte = NF.moe_cte(implementation=MoECTEImplementation.shard_on_block, hidden_states=x,
                     expert_affinities_masked=aff.reshape(-1, 1), gate_up_proj_weight=gu, down_proj_weight=dn,
                     token_position_to_id=torch.zeros(1, dtype=torch.int32),
                     block_to_expert=torch.zeros(1, dtype=torch.int32), block_size=256,
                     activation_function=ActFnType.SiLU,
                     expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)
    assert torch.allclose(cte.float(), ours.float(), atol=1e-2, rtol=1e-2)


# ── whole-model wiring: every parameter mapped, loads with the right shape / dtype ─────────
@needs_l3
def test_model_weight_mappings(monkeypatch):
    import vllm.distributed.parallel_state as ps

    from mimo_v2_neuron.model.mimo_v2 import model as M

    rank, tp = 5, 64
    monkeypatch.setattr(ps, "get_tp_group", lambda: SimpleNamespace(world_size=tp, rank_in_group=rank,
                                                                     device_group=None))
    monkeypatch.setattr(M, "_ep_settings", lambda g: {"degree": tp, "rank": rank, "tp_group": _fake_group(1, 0)})
    cfg = _our_cfg()
    with torch.device("meta"):
        model = M.MiMoV2ForCausalLMNeuron(cfg)
    mappings = model.weight_mappings()
    params = dict(model.named_parameters())
    assert set(params) == set(mappings), (set(params) ^ set(mappings))
    keys = {k for v in mappings.values() for k in (v if isinstance(v, list) else [v])}
    assert keys <= set(ckpt().wm), sorted(keys - set(ckpt().wm))[:5]
    assert mappings["model.layers.2.mlp.gate_up_proj_weight"][:2] == [
        "model.layers.2.mlp.experts.20.gate_proj.weight", "model.layers.2.mlp.experts.20.up_proj.weight"]
    from vllm_neuron.utils.weight_loader import get_weight_loader

    for name, p in params.items():
        ks = mappings[name]
        ks = ks if isinstance(ks, list) else [ks]
        t = get_weight_loader(p).load([ckpt().slice(k) for k in ks], rank)
        assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
        if name.endswith("e_score_correction_bias"):
            assert t.dtype == torch.float32 and p.dtype == torch.float32
    spec = model.get_kv_spec()
    assert [(s.num_kv_heads, s.head_size, s.sliding_window_size) for s in spec.layers] == [
        (1, 192, None), (1, 192, 128), (1, 192, None)]
    assert [type(l.mlp).__name__ for l in model.model.layers] == ["MiMoV2DenseMLP", "MiMoV2MoE", "MiMoV2MoE"]
    assert model.model.layers[1].self_attn.sinks is not None and model.model.layers[0].self_attn.sinks is None
