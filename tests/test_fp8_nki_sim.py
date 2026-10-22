# SPDX-License-Identifier: Apache-2.0
"""Native FP8 linear kernel (docs/design-notes.md#fp8-alignment; ``nki_fp8.w8a8_block_linear`` via ``fp8_linear`` with impl="nki") under the NKI
CPU simulator. Run in its own pytest process with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` (see run_tests.sh).

Reference: the GPU W8A8 block linear (``quant_act_group`` with 448 + block GEMM) in float64; the kernel stores the
activation codes halved as trn2 E4M3, which only differs from the GPU codes below 2^-5 (see nki_fp8.py).
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.nkisim,
    pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1" or os.environ.get("VLLM_NEURON_CPU_MODE") != "1",
                       reason="needs VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1"),
]


def _inputs(t, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(t, k, generator=g) * 2.0).to(torch.bfloat16)
    w = (torch.randn(k, n, generator=g) * 40.0).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    ws = (torch.rand(k // 128, n, generator=g) * 1e-3 + 1e-4).float()
    return x, w, ws


@pytest.mark.parametrize("t,k,n", [(4, 4096, 768), (130, 256, 1024), (3, 512, 96), (1, 256, 64)])
def test_w8a8_nki_matches_gpu_reference(t, k, n):
    from mimo_v2_neuron.model.mimo_v2 import fp8_linear as F

    x, w, ws = _inputs(t, k, n, 100 + t)
    codes, xs = F.quant_act_group(x, 448.0)
    ref = F.block_matmul(codes.double(), xs.double(), w.double(), ws.double(), mm="fp32").double()
    nki_out = F.fp8_linear(x, w, ws, {"quant_act": True, "fp8_max": 448.0, "mm": "bf16", "impl": "nki"})
    emul = F.fp8_linear(x, w, ws, {"quant_act": True, "fp8_max": 448.0, "mm": "bf16", "impl": "emul"})
    assert nki_out.dtype == torch.bfloat16 and tuple(nki_out.shape) == (t, n)
    rel = float((nki_out.double() - ref).norm() / ref.norm())
    rel_emul = float((emul.double() - ref).norm() / ref.norm())
    print(f"[w8a8 nki t={t} k={k} n={n}] rel_l2 nki={rel:.3e} emul={rel_emul:.3e}")
    # BF16 output rounding alone is ~1.6e-3; the kernel must not be worse than the emulation
    assert rel <= max(rel_emul, 2.5e-3)
