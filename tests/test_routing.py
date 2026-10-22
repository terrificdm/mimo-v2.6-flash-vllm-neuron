# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.hf_ref import load
from mimo_v2_neuron.model.mimo_v2.routing import _topk_lastdim, noaux_tc_route


def _hf_gate(E, H, k):
    mdl, cfgm = load()
    cfg = cfgm.MiMoV2Config(hidden_size=H, n_routed_experts=E, num_experts_per_tok=k, scoring_func="sigmoid",
                            topk_method="noaux_tc", n_group=1, topk_group=1, norm_topk_prob=True,
                            routed_scaling_factor=None)
    return mdl.MiMoV2MoEGate(cfg).eval()


@pytest.mark.parametrize("E,H,k,T,dtype", [(256, 4096, 8, 37, torch.bfloat16), (16, 512, 4, 9, torch.float32)])
def test_routing_matches_hf_gate(E, H, k, T, dtype):
    g = torch.Generator().manual_seed(0)
    gate = _hf_gate(E, H, k)
    w = (torch.randn(E, H, generator=g) * 0.02).to(dtype)
    bias = (torch.randn(E, generator=g) * 0.01).float()
    with torch.no_grad():
        gate.weight.data = w.clone()
        gate.e_score_correction_bias.data = bias.clone()
    h = torch.randn(1, T, H, generator=g).to(dtype)
    with torch.no_grad():
        idx_ref, w_ref = gate(h)
    dense, idx, wt = noaux_tc_route(h.view(T, H), w, bias, k)
    assert gate.e_score_correction_bias.dtype == torch.float32
    assert dense.dtype == torch.float32 and dense.shape == (T, E)
    for t in range(T):
        assert set(idx[t].tolist()) == set(idx_ref[t].tolist())
        ref = dict(zip(idx_ref[t].tolist(), w_ref[t].tolist()))
        for e, v in zip(idx[t].tolist(), wt[t].tolist()):
            assert v == pytest.approx(ref[e], rel=1e-6, abs=1e-7)
            assert dense[t, e].item() == pytest.approx(v, rel=1e-6)
    assert torch.allclose(dense.sum(-1), torch.ones(T), atol=1e-5)
    assert int((dense > 0).sum()) == T * k


def test_bias_only_affects_selection():
    E, H, k = 16, 64, 2
    h = torch.randn(3, H)
    w = torch.randn(E, H) * 0.1
    bias = torch.zeros(E)
    bias[5] = 100.0       # force expert 5 selected
    dense, idx, wt = noaux_tc_route(h, w, bias, k)
    assert (idx == 5).any(-1).all()
    s = torch.sigmoid(h @ w.t())
    for t in range(3):
        sel = idx[t]
        assert torch.allclose(wt[t], s[t, sel] / s[t, sel].sum(), atol=1e-6)


@pytest.mark.parametrize("T,E,k", [(1, 256, 8), (37, 256, 8), (5, 384, 8), (4, 16, 4)])
def test_chunked_topk_matches_torch_topk(T, E, k):
    """Device-bug workaround (tools/repro_topk.py; in-graph topk over 256 columns is wrong on Neuron): same sets as torch.topk."""
    x = torch.rand(T, E, generator=torch.Generator().manual_seed(1))
    v, i = _topk_lastdim(x, k)
    rv, ri = torch.topk(x, k, dim=-1)
    assert [set(r) for r in i.tolist()] == [set(r) for r in ri.tolist()]
    assert torch.equal(v.sort(-1).values, rv.sort(-1).values)
