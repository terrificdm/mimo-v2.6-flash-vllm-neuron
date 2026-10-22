# SPDX-License-Identifier: Apache-2.0
"""Native FP8 NKI kernels for trn2 (docs/design-notes.md#fp8-alignment).

``w8a8_block_linear``: the GPU's FP8 block linear (vLLM ``Fp8LinearMethod`` block_quant + dynamic activations,
see ``fp8_linear.py``) with the FP8 x FP8 product on the Tensor Engine instead of the BF16 emulation.

  x [T, K] BF16, w [K, N] FP8 codes (|c| <= 240, ``-fp8`` checkpoint), ws [K / 128, N] F32  ->  [T, N] BF16

Per token and 128-channel block the activation scale is the GPU's ``max(amax, 1e-10) / 448`` (E4M3FN). trn2's FP8
is E4M3 (max 240), so the code is stored halved, ``c = e4m3(x / scale / 2)``, with scale ``2 * scale``: the same
grid as the GPU's E4M3FN codes except for |x / scale| < 2^-5 (E4M3 subnormal step 2^-9 vs the halved grid's
2^-10), which only moves values smaller than amax / 14336. Each K block is one FP8 nc_matmul (PSUM FP32), then
``x (2 * scale)[t] x ws[kb, n]`` and an FP32 sum over blocks in SBUF, i.e. the GPU block GEMM up to summation order.

Launched with ``wrap_nki(kernel)[2]`` like vllm-neuron's kernels (LNC=2): program ``i`` computes its share of the
N columns. Called from the torch.compile graph through ``libtorch_neuronx_lite.nki.nki_hop.wrap_nki``
(a stand-alone ``nki.jit`` call needs ``torch_neuronx``, which this venv lacks).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

_P = 128          # partitions / K block / T tile
_NT = 512         # PSUM free size per matmul (one FP32 bank)


@nki.jit
def w8a8_block_linear(x, w, ws):
    t_all, k = x.shape
    _, n = w.shape
    kb_n = k // _P
    nprog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    n_per = n // nprog            # caller guarantees n % nprog == 0
    n_lo = pid * n_per
    n_cnt = n_per
    out = nl.ndarray((t_all, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for t0 in range(0, t_all, _P):
        t = min(_P, t_all - t0)
        xs = nl.ndarray((t, k), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=x[t0:t0 + t, 0:k])
        # activation quantisation of the whole T tile: codes transposed to [K block (partitions), KB, t]
        qt = nl.ndarray((_P, kb_n, t), dtype=w.dtype, buffer=nl.sbuf)
        sc2 = nl.ndarray((t, kb_n), dtype=nl.float32, buffer=nl.sbuf)
        for kb in range(kb_n):
            xk = xs[0:t, kb * _P:(kb + 1) * _P]
            ax = nl.ndarray((t, _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ax, op=nl.abs, data=xk)
            amax = nl.ndarray((t, 1), dtype=nl.float32, buffer=nl.sbuf)
            # abs first, then max (tensor_reduce op=abs_max returned 0 for rows with a negative max |x| in the
            # NKI simulator)
            nisa.tensor_reduce(dst=amax, op=nl.maximum, data=ax, axis=(1,))
            sc = nl.ndarray((t, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sc, data=amax, op0=nl.maximum, operand0=1e-10, op1=nl.multiply,
                               operand1=1.0 / 448.0)
            # x / scale: the Vector Engine has no divide in tensor_scalar on trn2 (compile error "unsupported
            # operator 'divide'"), so multiply by the reciprocal (can move a code by one step where x / scale
            # lies within an ulp of a rounding boundary)
            inv = nl.ndarray((t, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv, data=sc)
            q = nl.ndarray((t, _P), dtype=w.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=q, data=xk, op0=nl.multiply, operand0=inv, op1=nl.multiply, operand1=0.5)
            # transpose through BF16 (FP8 values are exact in BF16): a PE transpose of an FP8 tile writes 16-bit
            # PSUM elements and fails to compile here (NCC_IBTN174 "FP8 transpose output must write with even
            # step sizes")
            qb = nl.ndarray((t, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qb, src=q)
            q_ps = nl.ndarray((_P, t), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=q_ps, data=qb)
            nisa.tensor_copy(dst=qt[0:_P, kb, 0:t], src=q_ps)
            nisa.tensor_scalar(dst=sc2[0:t, kb:kb + 1], data=sc, op0=nl.multiply, operand0=2.0)
        for nt0 in range(0, n_cnt, _NT):
            nt = min(_NT, n_cnt - nt0)
            c0 = n_lo + nt0
            acc = nl.ndarray((t, nt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)
            # all K blocks of this column tile in one DMA each: w as [128 (row in block), KB, nt], ws broadcast
            # over the t partitions as [t, KB, nt]
            wk = nl.ndarray((_P, kb_n, nt), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wk, src=w.ap(pattern=[[n, _P], [_P * n, kb_n], [1, nt]], offset=c0))
            wsb = nl.ndarray((t, kb_n, nt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wsb, src=ws.ap(pattern=[[0, t], [n, kb_n], [1, nt]], offset=c0))
            for kb in range(kb_n):
                ps = nl.ndarray((t, nt), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=qt[0:_P, kb, 0:t], moving=wk[0:_P, kb, 0:nt])
                part = nl.ndarray((t, nt), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=part, data1=ps, data2=wsb[0:t, kb, 0:nt], op=nl.multiply)
                nisa.scalar_tensor_tensor(dst=acc, data=part, op0=nl.multiply, operand0=sc2[0:t, kb:kb + 1],
                                          op1=nl.add, operand1=acc)
            r = nl.ndarray((t, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=r, src=acc)
            nisa.dma_copy(dst=out[t0:t0 + t, nl.ds(c0, nt)], src=r)
    return out
