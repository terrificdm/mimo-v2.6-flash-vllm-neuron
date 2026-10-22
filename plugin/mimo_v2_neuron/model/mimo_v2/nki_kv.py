# SPDX-License-Identifier: Apache-2.0
# Portions adapted from nkilib (Apache-2.0, installed with the Neuron SDK): the indirect-DMA cache write
#   pattern of _update_block_cache_vectorized. See NOTICE.
"""Paged KV cache write (in place) and block gather as NKI kernels (docs/design-notes.md#kv-write).

``kv_gather`` (MIMO_KV_GATHER=nki) replaces the XLA gather ``cache[block_table]`` in ``gather_paged_kv`` with
indirect DMA reads of whole blocks.

``_write_cache`` used ``index_put_`` on the whole paged cache. In the compiled graph that scatter is functional: the
compiler materialises copies of the full cache tensor (about four per decode graph were measured), so the cost of
one decode step grows with the KV pool size instead of with the number of new tokens. This kernel writes only the
new rows with indirect (scatter) DMA and returns the cache tensors it was given, which the NKI HOP declares as
input/output aliases, so the graph updates the cache in place (the same mechanism as vllm-neuron's in-kernel KV
update in ``attention_decode`` / ``qkv_cte``).

  k_cache [nb, nkh, bs, dk], v_cache [nb, nkh, bs, dv] (BF16; dv = dk, or 128 with MIMO_KV_VNOPAD)
  k_rows [n, dk], v_rows [n, dv], idx [n, 1] uint32 = (block * nkh + head) * bs + offset
  -> (k_cache, v_cache) with rows idx overwritten; numerically identical to the index_put_ path.

Launched with ``wrap_nki(kv_write)[2]`` (LNC=2): program 0 writes V, program 1 writes K (as vllm-neuron's
``_update_block_cache_vectorized`` does when the token count does not split evenly).
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

_P = 128


@nki.jit
def kv_write(k_cache, v_cache, k_rows, v_rows, idx):
    n = idx.shape[0]
    dk = k_cache.shape[3]
    dv = v_cache.shape[3]
    rows_k = k_cache.shape[0] * k_cache.shape[1] * k_cache.shape[2]
    rows_v = v_cache.shape[0] * v_cache.shape[1] * v_cache.shape[2]
    nprog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    do_v = nprog == 1 or pid == 0
    do_k = nprog == 1 or pid == 1
    k2 = k_cache.reshape((rows_k, dk))
    v2 = v_cache.reshape((rows_v, dv))
    for r0 in range(0, n, _P):
        t = min(_P, n - r0)
        it = nl.ndarray((t, 1), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=it, src=idx[r0:r0 + t, 0:1])
        if do_v:
            vt = nl.ndarray((t, dv), dtype=v_rows.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt, src=v_rows[r0:r0 + t, 0:dv])
            nisa.dma_copy(dst=v2.ap(pattern=[[dv, t], [1, dv]], offset=0, vector_offset=it, indirect_dim=0),
                          src=vt)
        if do_k:
            kt = nl.ndarray((t, dk), dtype=k_rows.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt, src=k_rows[r0:r0 + t, 0:dk])
            nisa.dma_copy(dst=k2.ap(pattern=[[dk, t], [1, dk]], offset=0, vector_offset=it, indirect_dim=0),
                          src=kt)
    return k_cache, v_cache


@nki.jit
def kv_gather(cache, idx):
    """cache [nb, nkh, bs, d], idx [n, 1] uint32 block ids (>= 0) -> ([n, nkh * bs * d] = cache[idx], cache)
    (indirect DMA reads of whole blocks; replaces the XLA gather ``cache[idx]``).

    The cache is also returned unchanged so the NKI HOP declares it an input/output alias: the FX aliasing
    pass then threads every read into the same chain as the ``kv_write`` writes (write -> read -> ... -> next
    write). Without it each cache version has a reader and an in-place writer side by side, and the compiler
    copied the whole cache tensor to keep the read safe (L3 device profile: ~2 full-pool copies per cache tensor
    per decode step)."""
    nb = cache.shape[0]
    row = cache.shape[1] * cache.shape[2] * cache.shape[3]
    n = idx.shape[0]
    nprog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    per = (n + nprog - 1) // nprog
    lo = pid * per
    hi = min(n, lo + per)
    c2 = cache.reshape((nb, row))
    out = nl.ndarray((n, row), dtype=cache.dtype, buffer=nl.shared_hbm)
    for r0 in range(lo, hi, _P):
        t = min(_P, hi - r0)
        it = nl.ndarray((t, 1), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=it, src=idx[r0:r0 + t, 0:1])
        buf = nl.ndarray((t, row), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf, src=c2.ap(pattern=[[row, t], [1, row]], offset=0, vector_offset=it, indirect_dim=0))
        nisa.dma_copy(dst=out[r0:r0 + t, 0:row], src=buf)
    return out, cache
