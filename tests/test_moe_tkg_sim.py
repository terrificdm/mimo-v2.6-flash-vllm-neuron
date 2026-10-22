# SPDX-License-Identifier: Apache-2.0
"""``NF.moe_tkg`` semantics under the NKI CPU simulator (design V2).

Run in its own pytest process with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` (vllm_neuron reads
NKI_SIMULATOR at import time, so it cannot be toggled inside a test). Tiny shapes only (the
simulator is slow): H=512, E_L=4, I=128, T=4, E=16, top-4.

Checks the combination used by MiMoV2MoE decode: BF16 weights ``[E_L, H, 2, I]`` / ``[E_L, I, H]``,
external dense FP32 affinities ``[T, E]``, ``is_all_expert=True``, ``rank_id=[[ep_rank]]``, SiLU,
POST_SCALE -- against ``_torch_local_experts`` and an explicit FP32 per-token reference.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.nkisim,
    pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1" or os.environ.get("VLLM_NEURON_CPU_MODE") != "1",
                       reason="needs VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1"),
]

H, E_L, I, T, E, K = 512, 4, 128, 4, 16, 4


def _data(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g).to(torch.bfloat16)
    gate_up = (torch.randn(E, H, 2, I, generator=g) * 0.05).to(torch.bfloat16)   # all E experts
    down = (torch.randn(E, I, H, generator=g) * 0.05).to(torch.bfloat16)
    router = (torch.randn(E, H, generator=g) * 0.05).to(torch.bfloat16)
    bias = torch.randn(E, generator=g) * 0.01
    return x, gate_up, down, router, bias


def _fp32_reference(x, gate_up, down, idx, w, ep_rank):
    """HF-style loop in FP32 over the selected experts that live on ``ep_rank``."""
    out = torch.zeros(T, H)
    lo, hi = ep_rank * E_L, (ep_rank + 1) * E_L
    for t in range(T):
        for j in range(K):
            e = int(idx[t, j])
            if lo <= e < hi:
                xf = x[t].float()
                y = (torch.nn.functional.silu(xf @ gate_up[e, :, 0].float()) * (xf @ gate_up[e, :, 1].float())) \
                    @ down[e].float()
                out[t] += y * w[t, j]
    return out


@pytest.mark.parametrize("ep_rank", [0, 2])
def test_moe_tkg_matches_torch(ep_rank):
    import vllm_neuron.functional as NF
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    from mimo_v2_neuron.model.mimo_v2.model import _torch_local_experts
    from mimo_v2_neuron.model.mimo_v2.routing import noaux_tc_route

    x, gate_up, down, router, bias = _data(100 + ep_rank)
    assert can_run_kernel(x), "NKI simulator not active"
    dense, idx, w = noaux_tc_route(x, router, bias, K)
    lo, hi = ep_rank * E_L, (ep_rank + 1) * E_L
    gu_l, dn_l = gate_up[lo:hi].contiguous(), down[lo:hi].contiguous()

    ref32 = _fp32_reference(x, gate_up, down, idx, w, ep_rank)
    ref_torch = _torch_local_experts(x, gu_l, dn_l, dense[:, lo:hi])
    out = NF.moe_tkg(
        hidden_input=x,
        expert_gate_up_weights=gu_l,
        expert_down_weights=dn_l,
        expert_affinities=dense,
        expert_index=idx.to(torch.int32),
        is_all_expert=True,
        rank_id=torch.tensor([[ep_rank]], dtype=torch.int32),
        activation_fn=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )
    assert tuple(out.shape) == (T, H) and torch.isfinite(out.float()).all()
    n_sel = int(((idx >= lo) & (idx < hi)).sum())
    print(f"[moe_tkg ep_rank={ep_rank}] selected (token, expert) pairs on this rank: {n_sel}; "
          f"max|tkg-torch|={float((out.float() - ref_torch.float()).abs().max()):.4e}; "
          f"max|ref|={float(ref32.abs().max()):.4e}")
    r = assert_close_three_way([ref32], [ref_torch], [out], name=f"moe_tkg.ep_rank{ep_rank}")
    print(f"[moe_tkg ep_rank={ep_rank}] sigma_ratio={r.sigma_ratio:.4f} bc={r.bc:.4f} linf_ratio={r.linf_ratio:.3f}")
    assert r.passed, r.summary()


def _fp8_experts(seed):
    """FP8 expert codes like the -fp8 checkpoint (E2M1 value x 2^k, |code| <= 192) + one power-of-two F32 scale
    per output channel: gate_up codes [E_L, H, 2, I] / scale [E_L, 2, I], down codes [E_L, I, H] / scale [E_L, H]."""
    g = torch.Generator().manual_seed(seed)
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    def codes(shape):
        v = e2m1[torch.randint(0, len(e2m1), shape, generator=g)]
        return (v * 2.0 ** torch.randint(0, 6, shape, generator=g).float()).to(torch.float8_e4m3fn)

    gu = codes((E_L, H, 2, I))
    dn = codes((E_L, I, H))
    gu_s = 2.0 ** -torch.randint(9, 13, (E_L, 2, I), generator=g).float()
    dn_s = 2.0 ** -torch.randint(9, 13, (E_L, H), generator=g).float()
    return gu, gu_s, dn, dn_s


def test_moe_tkg_row_fp8_matches_bf16_expansion():
    """docs/design-notes.md#fp8-alignment: moe_tkg ROW mode (FP8 codes + per-output-channel scale, BF16 hidden) == BF16-expanded experts."""
    import vllm_neuron.functional as NF
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    from mimo_v2_neuron.model.mimo_v2.model import _torch_local_experts
    from mimo_v2_neuron.model.mimo_v2.routing import noaux_tc_route

    x, _, _, router, bias = _data(11)
    gu, gu_s, dn, dn_s = _fp8_experts(12)
    ep_rank = 1
    dense, idx, _ = noaux_tc_route(x, router, bias, K)
    # same expansion as MiMoV2MoE.expert_weights (exact in BF16)
    gu_bf = gu.to(torch.bfloat16) * gu_s.to(torch.bfloat16).unsqueeze(1)
    dn_bf = dn.to(torch.bfloat16) * dn_s.to(torch.bfloat16).unsqueeze(1)
    assert torch.equal(gu_bf.float(), gu.float() * gu_s.unsqueeze(1))
    common = dict(hidden_input=x, expert_affinities=dense, expert_index=idx.to(torch.int32), is_all_expert=True,
                  rank_id=torch.tensor([[ep_rank]], dtype=torch.int32), activation_fn=ActFnType.SiLU,
                  expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE)
    out_bf = NF.moe_tkg(expert_gate_up_weights=gu_bf, expert_down_weights=dn_bf, **common)
    # NF.moe_tkg's Python dtype check rejects float8 weights; MiMoV2MoE calls the wrapped kernel directly
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from vllm_neuron.functional.moe.moe_tkg_wrapper import moe_tkg_wrapper
    out_row = wrap_nki(moe_tkg_wrapper)[2](expert_gate_up_weights=gu, expert_down_weights=dn,
                                           expert_gate_up_weights_scale=gu_s, expert_down_weights_scale=dn_s,
                                           **common)
    ref = _torch_local_experts(x, gu_bf, dn_bf, dense[:, ep_rank * E_L:(ep_rank + 1) * E_L])
    scale = float(ref.float().abs().max())
    d_row_bf = float((out_row.float() - out_bf.float()).abs().max())
    d_row_ref = float((out_row.float() - ref.float()).abs().max())
    print(f"[moe_tkg ROW fp8] max|row-bf16exp|={d_row_bf:.4e} max|row-torch|={d_row_ref:.4e} max|ref|={scale:.4e} "
          f"bitexact_vs_bf16exp={torch.equal(out_row, out_bf)}")
    assert tuple(out_row.shape) == (T, H) and torch.isfinite(out_row.float()).all()
    assert d_row_bf <= 0.02 * scale + 1e-3


def test_moe_module_decode_fp8_row_vs_expand():
    """MiMoV2MoE.experts_decode with FP8 experts: fp8_expert_impl='row' vs the default 'expand'."""
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE

    x, _, _, router, bias = _data(13)
    gu, gu_s, dn, dn_s = _fp8_experts(14)
    ep_degree, ep_rank = E // E_L, 2
    outs = {}
    for impl in ("expand", "row"):
        cfg = MiMoV2Config(hidden_size=H, n_routed_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
                           num_hidden_layers=2, decode_moe_impl="tkg", fp8_weights=True, fp8_expert_impl=impl)
        m = MiMoV2MoE(cfg, 1, tp_group=SimpleNamespace(world_size=ep_degree, rank_in_group=ep_rank),
                      ep={"degree": ep_degree, "rank": ep_rank,
                          "tp_group": SimpleNamespace(world_size=1, rank_in_group=0)})
        m.router_weight.data = router
        m.e_score_correction_bias.data = bias
        m.gate_up_proj_weight.data, m.gate_up_proj_scale.data = gu, gu_s
        m.down_proj_weight.data, m.down_proj_scale.data = dn, dn_s
        with torch.no_grad():
            dense, idx = m.route(x)
            outs[impl] = m.experts_decode(x, dense, idx)
    err = float((outs["row"].float() - outs["expand"].float()).abs().max())
    scale = float(outs["expand"].float().abs().max())
    print(f"[moe_module fp8 row vs expand] max|diff|={err:.4e} max|ref|={scale:.4e} "
          f"bitexact={torch.equal(outs['row'], outs['expand'])}")
    assert err <= 0.02 * scale + 1e-3


def test_moe_module_prefill_fp8_row_vs_expand():
    """MiMoV2MoE.experts_prefill (moe_cte shard_on_block) with FP8 experts: 'row' vs 'expand'."""
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE

    x, _, _, router, bias = _data(15)
    # prefill needs a longer sequence (blockwise mapping kernels tile T); 256 tokens = one 256-token block
    x = torch.randn(256, H, generator=torch.Generator().manual_seed(17)).to(torch.bfloat16)
    gu, gu_s, dn, dn_s = _fp8_experts(16)
    ep_degree, ep_rank = E // E_L, 0
    outs = {}
    for impl in ("expand", "row"):
        cfg = MiMoV2Config(hidden_size=H, n_routed_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
                           num_hidden_layers=2, decode_moe_impl="tkg", fp8_weights=True, fp8_expert_impl=impl)
        m = MiMoV2MoE(cfg, 1, tp_group=SimpleNamespace(world_size=ep_degree, rank_in_group=ep_rank),
                      ep={"degree": ep_degree, "rank": ep_rank,
                          "tp_group": SimpleNamespace(world_size=1, rank_in_group=0)})
        m.router_weight.data = router
        m.e_score_correction_bias.data = bias
        m.gate_up_proj_weight.data, m.gate_up_proj_scale.data = gu, gu_s
        m.down_proj_weight.data, m.down_proj_scale.data = dn, dn_s
        with torch.no_grad():
            dense, _ = m.route(x)
            outs[impl] = m.experts_prefill(x, dense, None, rank=torch.tensor([[ep_rank]], dtype=torch.int32))
    err = float((outs["row"].float() - outs["expand"].float()).abs().max())
    scale = float(outs["expand"].float().abs().max())
    print(f"[moe_module prefill fp8 row vs expand] max|diff|={err:.4e} max|ref|={scale:.4e} "
          f"bitexact={torch.equal(outs['row'], outs['expand'])}")
    assert scale > 0 and err <= 0.02 * scale + 1e-3


def test_moe_module_decode_tkg_path():
    """MiMoV2MoE.experts_decode with decode_moe_impl='tkg' really takes the kernel path here."""
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE, _torch_local_experts

    cfg = MiMoV2Config(hidden_size=H, n_routed_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
                       num_hidden_layers=2, decode_moe_impl="tkg")
    x, gate_up, down, router, bias = _data(7)
    ep_degree, ep_rank = E // E_L, 1
    m = MiMoV2MoE(cfg, 1, tp_group=SimpleNamespace(world_size=ep_degree, rank_in_group=ep_rank),
                  ep={"degree": ep_degree, "rank": ep_rank, "tp_group": SimpleNamespace(world_size=1, rank_in_group=0)})
    lo, hi = ep_rank * E_L, (ep_rank + 1) * E_L
    m.router_weight.data = router
    m.e_score_correction_bias.data = bias
    m.gate_up_proj_weight.data = gate_up[lo:hi].contiguous()
    m.down_proj_weight.data = down[lo:hi].contiguous()
    with torch.no_grad():
        dense, idx = m.route(x)
        out = m.experts_decode(x, dense, idx)
    ref = _torch_local_experts(x, m.gate_up_proj_weight, m.down_proj_weight, dense[:, lo:hi])
    err = float((out.float() - ref.float()).abs().max())
    scale = float(ref.float().abs().max())
    print(f"[moe_module tkg] max|tkg-torch|={err:.4e} max|ref|={scale:.4e}")
    assert err <= 0.02 * scale + 1e-3


@pytest.mark.parametrize("t,ep_rank", [(256, 0), (512, 3)])
def test_moe_module_prefill_fp8_hybrid_vs_shard_on_i(t, ep_rank):
    """docs/design-notes.md#moe-hybrid: MiMoV2MoE.experts_prefill with FP8 'row' experts, moe_cte_impl='hybrid'
    (nkilib shard_on_i_hybrid, num_static_block=0: runtime loop over the non-empty blocks only) against the validated
    'shard_on_i' (static loop over all N blocks). Same per-block code -> same values; also against the BF16 expansion."""
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config
    from mimo_v2_neuron.model.mimo_v2.model import MiMoV2MoE

    _, _, _, router, bias = _data(31 + t)
    x = torch.randn(t, H, generator=torch.Generator().manual_seed(33 + t)).to(torch.bfloat16)
    gu, gu_s, dn, dn_s = _fp8_experts(32 + t)
    ep_degree = E // E_L
    outs = {}
    for impl, cte in (("expand", "shard_on_i"), ("row", "shard_on_i"), ("row", "hybrid")):
        cfg = MiMoV2Config(hidden_size=H, n_routed_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
                           num_hidden_layers=2, decode_moe_impl="tkg", fp8_weights=True, fp8_expert_impl=impl,
                           moe_cte_impl=cte)
        m = MiMoV2MoE(cfg, 1, tp_group=SimpleNamespace(world_size=ep_degree, rank_in_group=ep_rank),
                      ep={"degree": ep_degree, "rank": ep_rank,
                          "tp_group": SimpleNamespace(world_size=1, rank_in_group=0)})
        m.router_weight.data = router
        m.e_score_correction_bias.data = bias
        m.gate_up_proj_weight.data, m.gate_up_proj_scale.data = gu, gu_s
        m.down_proj_weight.data, m.down_proj_scale.data = dn, dn_s
        with torch.no_grad():
            dense, _ = m.route(x)
            outs[impl, cte] = m.experts_prefill(x, dense, None, rank=torch.tensor([[ep_rank]], dtype=torch.int32))
    hyb, base, exp = outs["row", "hybrid"], outs["row", "shard_on_i"], outs["expand", "shard_on_i"]
    scale = float(exp.float().abs().max())
    d_base = float((hyb.float() - base.float()).abs().max())
    d_exp = float((hyb.float() - exp.float()).abs().max())
    print(f"[moe_module prefill fp8 hybrid t={t} ep_rank={ep_rank}] max|hybrid-shard_on_i|={d_base:.4e} "
          f"max|hybrid-expand|={d_exp:.4e} max|ref|={scale:.4e} bitexact_vs_shard_on_i={torch.equal(hyb, base)}")
    assert scale > 0 and torch.isfinite(hyb.float()).all()
    assert torch.equal(hyb, base)
    assert d_exp <= 0.02 * scale + 1e-3
