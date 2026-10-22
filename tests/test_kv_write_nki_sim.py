# SPDX-License-Identifier: Apache-2.0
"""KV write / gather kernels (docs/design-notes.md#kv-write; ``nki_kv.kv_write`` via ``_write_cache`` with MIMO_KV_WRITE=nki, ``kv_gather``
via ``gather_paged_kv`` with MIMO_KV_GATHER=nki) under the
NKI CPU simulator. Run in its own pytest process with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` (see run_tests.sh).

Reference: the ``index_put_`` path of ``_write_cache``; the cache after the write must be bit-identical.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.nkisim,
    pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1" or os.environ.get("VLLM_NEURON_CPU_MODE") != "1",
                       reason="needs VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1"),
]


class _Attn:
    """Just the fields ``MiMoV2Attention._write_cache`` reads."""

    def __init__(self, nkh, head_dim, v_head_dim, kd, vd, nb, bs, nki):
        self.num_kv_heads_per_rank = nkh
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.kv_write_nki = nki
        g = torch.Generator().manual_seed(7)
        self.k_cache = torch.randn(nb, nkh, bs, kd, generator=g).to(torch.bfloat16)
        self.v_cache = torch.randn(nb, nkh, bs, vd, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("t,nkh,vd,slots", [
    (1, 1, 128, [37]),                      # decode B=1, V not padded
    (4, 2, 192, [0, 65, 130, 200]),         # decode B=4 (slot 0 = null-block padding), V padded to 192
    (160, 1, 128, list(range(32, 192))),    # one prefill segment, > 128 rows
])
def test_kv_write_nki_matches_index_put(t, nkh, vd, slots):
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2Attention

    nb, bs = 8, 32
    g = torch.Generator().manual_seed(t)
    k = torch.randn(t, nkh, 192, generator=g).to(torch.bfloat16)
    v = torch.randn(t, nkh, 128, generator=g).to(torch.bfloat16)
    sm = torch.tensor(slots, dtype=torch.int64)
    ref = _Attn(nkh, 192, 128, 192, vd, nb, bs, nki=False)
    out = _Attn(nkh, 192, 128, 192, vd, nb, bs, nki=True)
    MiMoV2Attention._write_cache(ref, k, v, sm, bs)
    MiMoV2Attention._write_cache(out, k, v, sm, bs)
    assert torch.equal(out.k_cache, ref.k_cache)
    assert torch.equal(out.v_cache, ref.v_cache)


@pytest.mark.parametrize("b,nblk,nkh,d", [(1, 4, 1, 192), (4, 3, 2, 128), (1, 130, 1, 192)])
def test_kv_gather_nki_matches_xla_gather(b, nblk, nkh, d, monkeypatch):
    from mimo_v2_neuron.model.mimo_v2.attention_torch import gather_paged_kv

    nb, bs = 160, 32
    g = torch.Generator().manual_seed(b * 100 + nblk)
    cache = torch.randn(nb, nkh, bs, d, generator=g).to(torch.bfloat16)
    table = torch.randint(0, nb, (b, nblk), generator=g)
    table[0, -1] = -1                                                    # unused table entry
    monkeypatch.setenv("MIMO_KV_GATHER", "xla")
    ref, ref_valid = gather_paged_kv(cache, table)
    monkeypatch.setenv("MIMO_KV_GATHER", "nki")
    out, valid = gather_paged_kv(cache, table)
    assert torch.equal(out, ref)
    assert torch.equal(valid, ref_valid)
