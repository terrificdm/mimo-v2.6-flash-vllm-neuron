# SPDX-License-Identifier: Apache-2.0
"""GA prefill prior-key flash kernel (docs/design-notes.md#segmented-prefill; ``nki_attn.ga_prior_flash`` via ``segmented_prefill_attention`` with
MIMO_PREFILL_ATTN=nki) under the NKI CPU simulator. Run in its own pytest process with
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` (see run_tests.sh).

Reference: the torch path (MIMO_PREFILL_ATTN=torch). The kernel's softmax numerator is BF16 for the PV matmul
(as FlashAttention), so the outputs agree to BF16 rounding, not bit for bit.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.nkisim,
    pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1" or os.environ.get("VLLM_NEURON_CPU_MODE") != "1",
                       reason="needs VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1"),
]


@pytest.mark.parametrize("csl,vd,sink", [
    (0, 128, True),        # first segment: no prior tile, only the masked remainder block
    (512, 128, True),      # one whole tile -> rounded down to 0 tiles, 512 keys in the remainder block
    (1280, 128, False),    # two tiles + 256-key remainder
    (1536, 192, True),     # three tiles -> two tiles + 512-key remainder, V padded to 192
    # prefix-cache hits start at any 32-token boundary (vLLM block size), not on 512 / 4096
    (32, 128, True),
    (576, 128, True),      # 1 whole tile (odd) -> remainder 576 keys
    (1000, 192, False),
    (1312, 128, True),
    (1600, 128, True),     # 3 whole tiles (odd) -> 2 tiles + 576 keys
    (4128, 128, True),     # 4096 + 32: 8 tiles + 32 keys
    (4672, 192, True),     # 4096 + 576: 9 tiles (odd) -> 8 tiles + 576 keys
])
def test_ga_prior_flash_matches_torch(csl, vd, sink, monkeypatch):
    from mimo_v2_neuron.model.mimo_v2.attention_torch import segmented_prefill_attention

    seg, bs = 256, 32
    width = max(64, -(-(csl + seg) // bs) + 8)
    nb = width + 32
    g = torch.Generator().manual_seed(csl + vd)
    q = torch.randn(1, seg, 192, generator=g).to(torch.bfloat16)
    k = torch.randn(1, seg, 192, generator=g).to(torch.bfloat16)
    v = torch.randn(1, seg, 128, generator=g).to(torch.bfloat16)
    k_cache = torch.randn(nb, 1, bs, 192, generator=g).to(torch.bfloat16)
    v_cache = torch.randn(nb, 1, bs, vd, generator=g).to(torch.bfloat16)
    table = torch.randperm(nb, generator=g)[:width].reshape(1, width).to(torch.int32)   # reused, non-monotonic ids
    table[0, (csl + seg) // bs:] = -1                                     # blocks past the segment unallocated
    csl_t = torch.tensor([[csl]], dtype=torch.int32)
    sinks = torch.randn(1, generator=g) if sink else None
    args = (q, k, v, k_cache, v_cache, table, csl_t, 192 ** -0.5, 128, bs, 1024)
    monkeypatch.setenv("MIMO_PREFILL_ATTN", "torch")
    ref = segmented_prefill_attention(*args, sliding_window=None, sink=sinks).float()
    monkeypatch.setenv("MIMO_PREFILL_ATTN", "nki")
    out = segmented_prefill_attention(*args, sliding_window=None, sink=sinks).float()
    err = (out - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"csl={csl} vd={vd} sink={sink}: max|diff|={err:.3e} max|ref|={ref.abs().max().item():.3e} cos={cos:.6f}")
    assert cos > 0.9999
    assert err < 2e-2 * ref.abs().max().item()


@pytest.mark.parametrize("csl", [1280, 4672])
def test_ga_prior_flash_kidx_arith_identical(csl, monkeypatch):
    """docs/design-notes.md#moe-hybrid: MIMO_PREFILL_KIDX=arith builds the same key rows as the index_select path, so the kernel
    output is bit-identical."""
    from mimo_v2_neuron.model.mimo_v2.attention_torch import segmented_prefill_attention

    seg, bs = 256, 32
    width = max(64, -(-(csl + seg) // bs) + 8) + 3                       # width * bs not a multiple of 512
    nb = width + 32
    g = torch.Generator().manual_seed(csl)
    q = torch.randn(1, seg, 192, generator=g).to(torch.bfloat16)
    k = torch.randn(1, seg, 192, generator=g).to(torch.bfloat16)
    v = torch.randn(1, seg, 128, generator=g).to(torch.bfloat16)
    k_cache = torch.randn(nb, 1, bs, 192, generator=g).to(torch.bfloat16)
    v_cache = torch.randn(nb, 1, bs, 128, generator=g).to(torch.bfloat16)
    table = torch.randperm(nb, generator=g)[:width].reshape(1, width).to(torch.int32)
    table[0, (csl + seg) // bs:] = -1
    sinks = torch.randn(1, generator=g)
    args = (q, k, v, k_cache, v_cache, table, torch.tensor([[csl]], dtype=torch.int32), 192 ** -0.5, 128, bs, 1024)
    monkeypatch.setenv("MIMO_PREFILL_ATTN", "nki")
    monkeypatch.setenv("MIMO_PREFILL_KIDX", "gather")
    ref = segmented_prefill_attention(*args, sliding_window=None, sink=sinks)
    monkeypatch.setenv("MIMO_PREFILL_KIDX", "arith")
    out = segmented_prefill_attention(*args, sliding_window=None, sink=sinks)
    print(f"csl={csl}: kidx arith vs gather bitexact={torch.equal(out, ref)}")
    assert torch.equal(out, ref)
