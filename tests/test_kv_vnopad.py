# SPDX-License-Identifier: Apache-2.0
"""V stored at 128 dims (MIMO_KV_VNOPAD=1) gives bit-identical attention outputs to the padded layout
(docs/design-notes.md#v-no-pad).

Uses the L3 checkpoint helpers of test_modules_cpu (VLLM_NEURON_CPU_MODE=1, no NKI simulator). The no-pad caches
are built exactly like the runner does: one contiguous (2, nb, H, bs, (192 + 128) / 2) tensor split into halves,
then re-viewed by ``_split_kv_storage``.
"""

import pytest
import torch

import test_modules_cpu as M


def test_split_kv_storage_views_one_buffer():
    from mimo_v2_neuron.model.mimo_v2.model import _split_kv_storage

    full = torch.arange(2 * 3 * 2 * 4 * 160, dtype=torch.float32).view(2, 3, 2, 4, 160)
    k, v = _split_kv_storage(full[0], full[1], 192, 128)
    assert tuple(k.shape) == (3, 2, 4, 192) and tuple(v.shape) == (3, 2, 4, 128)
    assert k.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
    assert float(k.reshape(-1)[0]) == 0.0 and float(v.reshape(-1)[-1]) == full.numel() - 1
    k.fill_(1.0)
    v.fill_(2.0)
    assert float(full.sum()) == k.numel() * 1.0 + v.numel() * 2.0
    with pytest.raises(ValueError):
        _split_kv_storage(full[0], full[1].clone(), 192, 128)


def _decode_out(layer, vnopad, monkeypatch):
    monkeypatch.setenv("MIMO_KV_VNOPAD", "1" if vnopad else "0")
    from mimo_v2_neuron.model.mimo_v2.model import _split_kv_storage

    lens = [400, 150]
    xs = [M._layer_input(layer, n, M.SEED + 100 * i + layer) for i, n in enumerate(lens)]
    num_blocks = 40
    perm = (torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(5)) + 1).tolist()
    seq_blocks = [perm[:13], perm[13:18]]
    bt = torch.full((3, M.MAX_MODEL_LEN // M.BS), -1, dtype=torch.int32)
    for i, b in enumerate(seq_blocks):
        bt[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
    a, cfg = M._our_attention(layer, 1, 0, num_blocks)
    assert a.kv_vnopad == vnopad
    if vnopad:
        full = torch.zeros(2, num_blocks, a.num_kv_heads_per_rank, M.BS, (a.head_dim + a.v_head_dim) // 2,
                           dtype=torch.bfloat16)
        a.k_cache, a.v_cache = _split_kv_storage(full[0], full[1], a.head_dim, a.v_head_dim)
    else:
        a.k_cache.zero_()
        a.v_cache.zero_()
    outs = []
    with torch.no_grad():
        for x, n, b in zip(xs, lens, seq_blocks):
            pos = torch.arange(n - 1)
            meta = {"slot_mapping": M._slots(b, n - 1), "block_size": M.BS, "kv_segment_size": 0,
                    "max_query_len": n - 1, "decode_token_threshold": 1}
            outs.append(a.forward_prefill(x[: n - 1], pos, M._rope(cfg, layer, pos), meta))
        hid = torch.stack([xs[0][-1], xs[1][-1], torch.zeros_like(xs[0][-1])])
        pos = torch.tensor([lens[0] - 1, lens[1] - 1, 0])
        slot = torch.tensor([M._slots(seq_blocks[0], lens[0])[-1], M._slots(seq_blocks[1], lens[1])[-1], 0])
        meta = {"slot_mapping": slot, "block_size": M.BS, "max_query_len": 1, "decode_token_threshold": 1,
                "block_table_tensor": bt}
        if layer == 1:
            meta["swa_kv_pos_offset"] = torch.zeros(3, dtype=torch.int32)
        outs.append(a.forward_decode(hid, pos, M._rope(cfg, layer, pos), meta))
    v_used = a.v_cache[..., :a.v_head_dim].clone()
    return outs, v_used


@M.needs_l3
@pytest.mark.parametrize("layer", [0, 1], ids=["ga", "swa_sink"])
def test_vnopad_bit_identical(layer, monkeypatch):
    ref, v_ref = _decode_out(layer, False, monkeypatch)
    got, v_got = _decode_out(layer, True, monkeypatch)
    assert torch.equal(v_ref, v_got)
    for r, g in zip(ref, got):
        assert torch.equal(r, g)


@M.needs_l3
def test_vnopad_kv_spec(monkeypatch):
    monkeypatch.setenv("MIMO_KV_VNOPAD", "1")
    a, _ = M._our_attention(0, 64, 0, 4)
    assert a.kv_vnopad and (a.head_dim + a.v_head_dim) // 2 == 160
