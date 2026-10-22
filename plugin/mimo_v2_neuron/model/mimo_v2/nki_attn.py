# SPDX-License-Identifier: Apache-2.0
# The register_load + fori_loop pattern follows nkilib (Apache-2.0) core/attention/attention_segmented_cte.py
#   and attention_cte.py, used as a read-only reference (no code copied). See NOTICE.
"""Flash (online-softmax) attention over the prior keys of one GA prefill segment, as an NKI kernel (docs/design-notes.md#segmented-prefill).

``segmented_prefill_attention`` (torch) builds the scores one key block at a time, but the compiled graph is
unrolled over the *maximum* context (1M keys = 32 blocks of 32768 per GA layer) and the compiler keeps the score
tiles of all blocks alive: prefill scratch ~ segment x context x 4 B (docs/design-notes.md#segmented-prefill), which is why the 1M configuration runs
512-token segments and every segment pays for 1M keys (0.58 s per segment, 940K TTFT 1065 s).

This kernel covers the part that scales with the context: queries of the segment against the ``ntile * 512``
earlier keys (all visible for a GA layer, no mask needed). The loop over key tiles is a runtime loop
(``nl.fori_loop`` with the trip count read from ``ntile``), so a segment early in the prompt only touches the keys
that exist, and no score matrix leaves the chip. QK uses the 192-wide head as two contractions (128 + 64) into
one PSUM tile; V is 128 wide. Scores stay FP32 in PSUM, the softmax numerator is BF16 for the PV matmul (as
FlashAttention on GPU), running max / sum / accumulator are FP32.

  qt [192, S] BF16 (Q of the segment, transposed; RoPE applied; S % 128 == 0)
  k_cache [nb, 1, bs, dk], v_cache [nb, 1, bs, dv] (BF16; dk = 192, dv = 128 or 192, first 128 used)
  kidx [nt, 128, 4] uint32: row (block * bs + offset) of key ``t * 512 + c * 128 + p`` at [t, p, c]
  ntile [1, 1] int32: number of 512-key tiles to process, a multiple of the number of programs (the caller passes
     ``cached_seq_len // 512`` rounded down to even, ``attention_torch._nki_whole_tiles``): every program must run the
     same number of loop trips, an odd count hung the device execution on prefix-cache hits (docs/design-notes.md#prefix-cache-odd-tiles)
  -> m [nprog, S, 1] (running max of the *scaled* scores), l [nprog, S, 1], acc [nprog, S, 128], all FP32,
     unnormalised: softmax-weighted sum = acc / l relative to exp(scale * s - m)

Launched with ``wrap_nki(kernel)[2]`` (LNC=2): program ``p`` takes key tiles p, p + 2, ...; the caller merges the
two partial states with the in-segment block (``attention_torch.segmented_prefill_attention``) and adds the sink.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

_P = 128          # partitions / query tile / key sub-tile
_KT = 512         # keys per tile (one FP32 PSUM bank of scores)
_NC = _KT // _P   # key sub-tiles per tile
_NEG_INIT = -1.0e30


@nki.jit
def ga_prior_flash(qt, k_cache, v_cache, kidx, ntile, scale):
    dqk, s_len = qt.shape
    dk = k_cache.shape[3]
    dv_cache = v_cache.shape[3]
    dv = 128
    rows_k = k_cache.shape[0] * k_cache.shape[1] * k_cache.shape[2]
    rows_v = v_cache.shape[0] * v_cache.shape[1] * v_cache.shape[2]
    da = min(dqk, _P)
    db = dqk - da
    nq = s_len // _P
    nprog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    k2 = k_cache.reshape((rows_k, dk))
    v2 = v_cache.reshape((rows_v, dv_cache))

    m_out = nl.ndarray((nprog, s_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    l_out = nl.ndarray((nprog, s_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    acc_out = nl.ndarray((nprog, s_len, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # Q^T resident in SBUF for the whole loop: [128, S] + [64, S]
    qa = nl.ndarray((da, s_len), dtype=qt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=qa, src=qt[0:da, 0:s_len])
    if db:
        qb = nl.ndarray((db, s_len), dtype=qt.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qb, src=qt[da:dqk, 0:s_len])

    # running state per query tile (FP32): max of unscaled scores, sum, accumulator
    m_st = nl.ndarray((_P, nq), dtype=nl.float32, buffer=nl.sbuf)
    l_st = nl.ndarray((_P, nq), dtype=nl.float32, buffer=nl.sbuf)
    acc_st = nl.ndarray((_P, nq, dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=m_st, value=_NEG_INIT)
    nisa.memset(dst=l_st, value=0.0)
    nisa.memset(dst=acc_st, value=0.0)

    nt_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sb, src=ntile[0:1, 0:1])
    nt_reg = nisa.register_alloc()
    nisa.register_load(nt_reg, nt_sb)

    def body(i):
        # gather the tile's K / V rows (keys on partitions) and build K^T [d, 512]
        ka = nl.ndarray((da, _KT), dtype=k_cache.dtype, buffer=nl.sbuf)
        if db:
            kb = nl.ndarray((db, _KT), dtype=k_cache.dtype, buffer=nl.sbuf)
        vt = nl.ndarray((_P, _NC, dv), dtype=v_cache.dtype, buffer=nl.sbuf)
        for c in range(_NC):
            it = nl.ndarray((_P, 1), dtype=kidx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=it, src=kidx.ap(pattern=[[_NC, _P], [1, 1]], offset=c, scalar_offset=i,
                                              indirect_dim=0))
            krow = nl.ndarray((_P, dk), dtype=k_cache.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=krow, src=k2.ap(pattern=[[dk, _P], [1, dk]], offset=0, vector_offset=it,
                                              indirect_dim=0))
            nisa.dma_copy(dst=vt[0:_P, c, 0:dv], src=v2.ap(pattern=[[dv_cache, _P], [1, dv]], offset=0,
                                                           vector_offset=it, indirect_dim=0))
            tpa = nl.ndarray((da, _P), dtype=k_cache.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tpa, data=krow[0:_P, 0:da])
            nisa.tensor_copy(dst=ka[0:da, c * _P:(c + 1) * _P], src=tpa)
            if db:
                tpb = nl.ndarray((db, _P), dtype=k_cache.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tpb, data=krow[0:_P, da:dqk])
                nisa.tensor_copy(dst=kb[0:db, c * _P:(c + 1) * _P], src=tpb)

        for j in range(nq):
            q0 = j * _P
            s_ps = nl.ndarray((_P, _KT), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=s_ps, stationary=qa[0:da, q0:q0 + _P], moving=ka, accumulate=False)
            if db:
                nisa.nc_matmul(dst=s_ps, stationary=qb[0:db, q0:q0 + _P], moving=kb, accumulate=True)
            mt = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mt, op=nl.maximum, data=s_ps, axis=(1,))
            mn = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mn, data1=m_st[0:_P, j:j + 1], data2=mt, op=nl.maximum)
            nb_ = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nb_, data=mn, op0=nl.multiply, operand0=-scale)
            # alpha = exp(scale * (m_old - m_new)); p = exp(scale * s - scale * m_new), row sums in FP32
            alpha = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=alpha, op=nl.exp, data=m_st[0:_P, j:j + 1], bias=nb_, scale=scale)
            p_sb = nl.ndarray((_P, _KT), dtype=nl.bfloat16, buffer=nl.sbuf)
            rs = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=p_sb, op=nl.exp, data=s_ps, reduce_op=nl.add, reduce_res=rs, bias=nb_,
                                   scale=scale)
            nisa.tensor_copy(dst=m_st[0:_P, j:j + 1], src=mn)
            nisa.scalar_tensor_tensor(dst=l_st[0:_P, j:j + 1], data=l_st[0:_P, j:j + 1], op0=nl.multiply,
                                      operand0=alpha, op1=nl.add, operand1=rs)
            # P^T per 128-key sub-tile, then PV accumulated over the sub-tiles in PSUM
            pv = nl.ndarray((_P, dv), dtype=nl.float32, buffer=nl.psum)
            for c in range(_NC):
                ptp = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=ptp, data=p_sb[0:_P, c * _P:(c + 1) * _P])
                pts = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pts, src=ptp)
                nisa.nc_matmul(dst=pv, stationary=pts, moving=vt[0:_P, c, 0:dv], accumulate=(c > 0))
            nisa.scalar_tensor_tensor(dst=acc_st[0:_P, j, 0:dv], data=acc_st[0:_P, j, 0:dv], op0=nl.multiply,
                                      operand0=alpha, op1=nl.add, operand1=pv)

    nl.fori_loop(pid, nt_reg, body, step=nprog)

    for j in range(nq):
        q0 = j * _P
        ms = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms, data=m_st[0:_P, j:j + 1], op0=nl.multiply, operand0=scale)
        nisa.dma_copy(dst=m_out[pid, q0:q0 + _P, 0:1], src=ms)
        nisa.dma_copy(dst=l_out[pid, q0:q0 + _P, 0:1], src=l_st[0:_P, j:j + 1])
        nisa.dma_copy(dst=acc_out[pid, q0:q0 + _P, 0:dv], src=acc_st[0:_P, j, 0:dv])
    return m_out, l_out, acc_out
