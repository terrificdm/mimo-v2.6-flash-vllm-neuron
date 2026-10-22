# SPDX-License-Identifier: Apache-2.0
"""PyTorch attention for MiMo-V2 (QK head_dim 192 != V head_dim 128, so the NKI
attention kernels -- MAX_HEAD_DIM 128, single head_dim -- are not used).

Semantics follow HF ``eager_attention_forward`` + transformers' causal / sliding-window
masks: key ``k`` is visible to query ``q`` iff ``k <= q`` and, for SWA, ``q - k < window``;
an optional per-head sink logit is appended as an extra column before the FP32 softmax
and dropped afterwards. Masked scores use ``finfo(float32).min`` (never -inf) so rows with
no visible key (padding rows) stay finite.
"""

from __future__ import annotations

import os

import torch

from ... import switches

_NEG = torch.finfo(torch.float32).min


def _softmax_with_sink(scores: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    """scores [..., Hq, Sq, Sk] fp32; sink [Hq] fp32 or None -> probs [..., Hq, Sq, Sk] fp32."""
    if sink is None:
        return torch.softmax(scores, dim=-1)
    s = sink.to(torch.float32).view(*([1] * (scores.dim() - 3)), -1, 1, 1)
    s = s.expand(*scores.shape[:-1], 1)
    probs = torch.softmax(torch.cat([scores, s], dim=-1), dim=-1)
    return probs[..., :-1]


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                      sliding_window: int | None = None, sink: torch.Tensor | None = None) -> torch.Tensor:
    """Single-sequence causal prefill attention (token index == position offset).

    q [Hq, T, Dqk], k [Hkv, T, Dqk], v [Hkv, T, Dv]; Hq % Hkv == 0.
    Returns [T, Hq * Dv] in q.dtype (heads concatenated, matching HF o_proj input order).
    """
    hq, t, _ = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    if g > 1:
        k = k.repeat_interleave(g, dim=0)
        v = v.repeat_interleave(g, dim=0)
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)) * scale  # [Hq, T, T]
    qi = torch.arange(t, device=q.device).unsqueeze(1)
    ki = torch.arange(t, device=q.device).unsqueeze(0)
    masked = ki > qi
    if sliding_window is not None:
        masked = masked | (qi - ki >= sliding_window)
    scores = scores.masked_fill(masked.unsqueeze(0), _NEG)
    probs = _softmax_with_sink(scores, sink)
    out = torch.matmul(probs, v.to(torch.float32))  # [Hq, T, Dv]
    return out.transpose(0, 1).reshape(t, -1).to(q.dtype)


def gather_paged_kv(cache: torch.Tensor, block_table: torch.Tensor):
    """cache [num_blocks, Hkv, bs, D]; block_table [B, nblk] (-1 = no block).

    Returns (kv [B, Hkv, nblk*bs, D], block_valid [B, nblk*bs] bool).
    """
    b, nblk = block_table.shape
    bs = cache.shape[2]
    valid = block_table >= 0
    idx = torch.clamp(block_table, min=0).to(torch.long)
    if os.environ.get("MIMO_KV_GATHER", "xla") == "nki":
        # indirect-DMA block reads (nki_kv.kv_gather) instead of the XLA gather
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .nki_kv import kv_gather
        # the returned cache alias is dropped here: the FX aliasing pass rewires later cache users to it
        rows, _ = wrap_nki(kv_gather)[2](cache=cache, idx=idx.reshape(-1, 1).to(torch.int32).to(torch.uint32))
        kv = rows.reshape(b, nblk, *cache.shape[1:])
    else:
        kv = cache[idx]                               # [B, nblk, Hkv, bs, D]
    kv = kv.permute(0, 2, 1, 3, 4).reshape(b, cache.shape[1], nblk * bs, cache.shape[3])
    valid = valid.unsqueeze(-1).expand(b, nblk, bs).reshape(b, nblk * bs)
    return kv, valid


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_table: torch.Tensor, positions: torch.Tensor, scale: float, v_head_dim: int,
                     sliding_window: int | None = None, sink: torch.Tensor | None = None,
                     kv_pos_offset: torch.Tensor | None = None, key_block: int | None = None) -> torch.Tensor:
    """One query token per sequence, keys read from the paged cache (current token already written).

    q [B, Hq, Dqk]; caches [num_blocks, Hkv, bs, D] (V zero-padded to D, first ``v_head_dim`` used);
    block_table [B, nblk]; positions [B] absolute; kv_pos_offset [B] absolute position of the first
    slot of ``block_table`` (runner's ``swa_kv_pos_offset`` for trimmed SWA tables) or None (= 0).
    ``key_block``: when the table covers more than ``key_block`` tokens, keys are processed in
    ``key_block``-token chunks with an online softmax (``_decode_attention_blocked``); otherwise the
    single-shot path below runs unchanged (same op sequence as the base configuration).
    Returns [B, Hq * v_head_dim] in q.dtype.
    """
    if key_block is not None and block_table.shape[1] * k_cache.shape[2] > key_block:
        return _decode_attention_blocked(q, k_cache, v_cache, block_table, positions, scale, v_head_dim,
                                         key_block, sliding_window=sliding_window, sink=sink,
                                         kv_pos_offset=kv_pos_offset)
    bsz, hq, _ = q.shape
    k, valid = gather_paged_kv(k_cache, block_table)        # [B, Hkv, L, D]
    v, _ = gather_paged_kv(v_cache, block_table)
    v = v[..., :v_head_dim]
    hkv, length = k.shape[1], k.shape[2]
    g = hq // hkv
    if g > 1:
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    kpos = torch.arange(length, device=q.device).unsqueeze(0).expand(bsz, length)
    if kv_pos_offset is not None:
        kpos = kpos + kv_pos_offset.view(bsz, 1).to(kpos.dtype)
    pos = positions.view(bsz, 1).to(kpos.dtype)
    visible = valid & (kpos <= pos)
    if sliding_window is not None:
        visible = visible & (pos - kpos < sliding_window)
    scores = torch.matmul(q.to(torch.float32).unsqueeze(2), k.to(torch.float32).transpose(-1, -2)) * scale
    scores = scores.masked_fill(~visible.view(bsz, 1, 1, length), _NEG)   # [B, Hq, 1, L]
    probs = _softmax_with_sink(scores, sink)
    out = torch.matmul(probs, v.to(torch.float32)).squeeze(2)            # [B, Hq, Dv]
    return out.reshape(bsz, -1).to(q.dtype)


# ── long context: key-blocked online softmax (never builds the full query x key score matrix) ──
def _online_softmax_update(state, s: torch.Tensor, v: torch.Tensor):
    """Merge one key block into the running (max ``m``, sum ``l``, weighted sum ``acc``), all FP32.

    s [..., Sq, Sk] fp32 scores with masked entries already set to ``_NEG``; v [..., Sk, Dv] fp32.
    A row whose keys are all masked so far has ``m == _NEG`` and accumulates junk with weight 1; the
    first block with a visible key rescales it by ``exp(_NEG - m_new) == 0``, so it never leaks.
    """
    m_blk = s.amax(dim=-1, keepdim=True)
    if state is None:
        p = torch.exp(s - m_blk)
        return m_blk, p.sum(dim=-1, keepdim=True), torch.matmul(p, v)
    m, l, acc = state
    m_new = torch.maximum(m, m_blk)
    alpha = torch.exp(m - m_new)
    p = torch.exp(s - m_new)
    return m_new, l * alpha + p.sum(dim=-1, keepdim=True), acc * alpha + torch.matmul(p, v)


def _online_softmax_finish(state, sink: torch.Tensor | None) -> torch.Tensor:
    """Normalise; the per-head sink logit is merged as one extra column (same maths as
    ``_softmax_with_sink``: the column joins the softmax denominator and contributes no value)."""
    m, l, acc = state
    if sink is not None:
        s = sink.to(torch.float32).view(*([1] * (m.dim() - 3)), -1, 1, 1)
        m_f = torch.maximum(m, s)
        alpha = torch.exp(m - m_f)
        l = l * alpha + torch.exp(s - m_f)
        acc = acc * alpha
    return acc / l


def _table_chunk(block_table: torch.Tensor, start: int, n: int) -> torch.Tensor:
    """Static column slice ``block_table[:, start:start+n]``, padded with -1 past the table width."""
    width = block_table.shape[1]
    stop = min(start + n, width)
    cols = block_table[:, start:stop]
    if stop - start < n:
        pad = torch.full((block_table.shape[0], n - (stop - start)), -1, dtype=block_table.dtype,
                         device=block_table.device)
        cols = torch.cat([cols, pad], dim=1)
    return cols


def _check_key_block(key_block: int, block_size: int, name: str) -> int:
    if key_block % block_size:
        raise ValueError(f"MiMoV2 Neuron: {name} key block ({key_block}) must be a multiple of the KV "
                         f"block size ({block_size})")
    return key_block // block_size


def segmented_prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor,
                                cached_seq_len: torch.Tensor, scale: float, v_head_dim: int, block_size: int,
                                key_block: int, sliding_window: int | None = None,
                                sink: torch.Tensor | None = None) -> torch.Tensor:
    """One prefill segment of a single request (vllm-neuron segmented prefill, docs/design-notes.md#segmented-prefill).

    q [Hq, S, Dqk], k [Hkv, S, Dqk], v [Hkv, S, Dv]: this segment (RoPE applied, V scaled; already
    written to the paged caches). Token i of the segment sits at absolute position
    ``cached_seq_len + i``. Keys = the ``cached_seq_len`` earlier tokens read from the paged cache
    (block_table row 0, -1 = no block) + this segment (causal). Scores are built one key block at a
    time (in-segment block [Hq, S, S], then prior blocks [Hq, S, key_block]) and merged with an
    online softmax in FP32.

    GA (``sliding_window`` None): the prior keys are split into ``ceil((W*bs - 1) / key_block)``
    static blocks (W = table width), enough for any ``cached_seq_len <= W*bs - 1``; blocks past the
    actual prefix are fully masked. SWA: only the last ``window`` prior keys matter, read with an
    ``index_select`` of ``ceil(window / bs) + 1`` table entries starting at
    ``max(0, cached_seq_len - window) // bs``. Returns [S, Hq * v_head_dim] in q.dtype.
    """
    if k_cache.shape[2] != block_size:
        raise ValueError(f"MiMoV2 Neuron: KV cache block dim ({k_cache.shape[2]}) != block_size ({block_size})")
    hq, seg, dqk = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    dev = q.device
    bt = block_table[:1]
    width = bt.shape[1]
    csl = cached_seq_len.reshape(-1)[:1].to(torch.long)                  # [1]
    qf = q.to(torch.float32)

    def rep(x):
        return x.repeat_interleave(g, dim=0) if g > 1 else x

    # 1) in-segment block (causal; SWA also windowed), indices relative to the segment
    qi = torch.arange(seg, device=dev).unsqueeze(1)
    ki = torch.arange(seg, device=dev).unsqueeze(0)
    masked = ki > qi
    if sliding_window is not None:
        masked = masked | (qi - ki >= sliding_window)
    s = torch.matmul(qf, rep(k).to(torch.float32).transpose(-1, -2)) * scale        # [Hq, S, S]
    s = s.masked_fill(masked.unsqueeze(0), _NEG)
    state = _online_softmax_update(None, s, rep(v).to(torch.float32))

    # 2) prior keys from the paged cache
    def prior_block(cols, kpos, extra_visible):
        kb, valid = gather_paged_kv(k_cache, cols)                        # [1, Hkv, n, D]
        vb, _ = gather_paged_kv(v_cache, cols)
        kb = rep(kb[0, :, :, :dqk]).to(torch.float32)
        vb = rep(vb[0, :, :, :v_head_dim]).to(torch.float32)
        visible = valid[0] & (kpos < csl)                                 # [n]
        visible = visible.unsqueeze(0)                                    # [1, n]
        if extra_visible is not None:
            visible = visible & extra_visible                             # [S, n]
        sc = torch.matmul(qf, kb.transpose(-1, -2)) * scale               # [Hq, S, n]
        sc = sc.masked_fill(~visible.unsqueeze(0), _NEG)
        return sc, vb

    if sliding_window is None and _use_nki_prefill(k_cache, block_size, width, dqk):
        # docs/design-notes.md#segmented-prefill: prior keys in an NKI flash kernel (runtime loop over the actual prefix, nki_attn.ga_prior_flash);
        # the keys past the last even number of whole 512-key tiles go through one masked torch block
        state = _ga_prior_nki(q, k_cache, v_cache, bt, csl, scale, block_size, state, prior_block)
    elif sliding_window is None:
        nkb = _check_key_block(key_block, block_size, "prefill")
        nb = max(0, -(-(width * block_size - 1) // key_block))
        for j in range(nb):
            cols = _table_chunk(bt, j * nkb, nkb)
            kpos = j * key_block + torch.arange(key_block, device=dev)
            sc, vb = prior_block(cols, kpos, None)
            state = _online_softmax_update(state, sc, vb)
    else:
        nsw = -(-sliding_window // block_size) + 1
        start = torch.clamp(csl - sliding_window, min=0) // block_size    # [1]
        idx = start + torch.arange(nsw, device=dev)                       # [nsw]
        in_table = (idx < width).repeat_interleave(block_size)            # [nsw*bs]
        cols = torch.index_select(bt, 1, torch.clamp(idx, max=width - 1).to(torch.long))
        kpos = start * block_size + torch.arange(nsw * block_size, device=dev)
        qpos = csl + torch.arange(seg, device=dev)                        # [S]
        win = (qpos.unsqueeze(1) - kpos.unsqueeze(0) < sliding_window) & (kpos >= 0).unsqueeze(0) \
            & in_table.unsqueeze(0)
        sc, vb = prior_block(cols, kpos, win)
        state = _online_softmax_update(state, sc, vb)

    out = _online_softmax_finish(state, sink)                             # [Hq, S, Dv]
    return out.transpose(0, 1).reshape(seg, -1).to(q.dtype)


_NKI_KT = 512   # keys per tile of nki_attn.ga_prior_flash
_NKI_PROGS = 2  # programs of the kernel launch (wrap_nki(...)[2], LNC=2): program p takes tiles p, p + 2, ...


def _nki_whole_tiles(csl: torch.Tensor) -> torch.Tensor:
    """Number of 512-key tiles handed to ``ga_prior_flash``: ``cached_seq_len // 512`` rounded down to a multiple of
    the program count, so both LNC programs run the same number of loop trips. An odd count (prefix-cache hits
    start at any 32-token boundary; cold segments always start on a 4096 boundary) hung the device execution
    (docs/design-notes.md#prefix-cache-odd-tiles)."""
    return (csl // (_NKI_KT * _NKI_PROGS)) * _NKI_PROGS


def _use_nki_prefill(k_cache: torch.Tensor, block_size: int, width: int, dqk: int) -> bool:
    """MIMO_PREFILL_ATTN=nki (the default on the device; torch in CPU mode): one KV head per rank, 192-wide K,
    512 % block_size == 0 (otherwise the torch path, per layer)."""
    if switches.get("MIMO_PREFILL_ATTN") != "nki":
        return False
    return k_cache.shape[1] == 1 and dqk == k_cache.shape[3] and _NKI_KT % block_size == 0


def _merge_states(a, b):
    """Merge two (max, sum, acc) online-softmax states over disjoint key sets."""
    m1, l1, acc1 = a
    m2, l2, acc2 = b
    m = torch.maximum(m1, m2)
    a1 = torch.exp(m1 - m)
    a2 = torch.exp(m2 - m)
    return m, l1 * a1 + l2 * a2, acc1 * a1 + acc2 * a2


def _nki_key_rows(bt: torch.Tensor, block_size: int, nt: int, mode: str = "gather") -> torch.Tensor:
    """Cache row (block * block_size + offset) of every key position 0 .. nt * 512 - 1 of block-table row 0 (-1 entries
    clamped to block 0; positions past the table repeat its last column). ``mode`` (MIMO_PREFILL_KIDX): "gather" =
    index_select of the table at every position (original code); "arith" (docs/design-notes.md#moe-hybrid): the same values by
    broadcasting the table over the block offsets, without the [nt * 512] index_select that the compiler lowers to
    ~8K indirect DMAs (~9 ms per GA layer per segment in the L3 profile, independent of the prefix length)."""
    dev = bt.device
    width = bt.shape[1]
    if mode == "arith":
        blk = torch.clamp(bt[0], min=0).to(torch.long)                       # [width]
        rows = (blk.unsqueeze(1) * block_size + torch.arange(block_size, device=dev).unsqueeze(0)).reshape(-1)
        pad = nt * _NKI_KT - width * block_size
        if pad:
            rows = torch.cat([rows, blk[-1] * block_size + torch.arange(pad, device=dev) % block_size])
        return rows
    if mode != "gather":
        raise ValueError(f"MIMO_PREFILL_KIDX must be gather or arith, got {mode!r}")
    pos = torch.arange(nt * _NKI_KT, device=dev)
    col = torch.clamp(pos // block_size, max=width - 1)
    blk = torch.clamp(torch.index_select(bt[0], 0, col), min=0).to(torch.long)
    return blk * block_size + pos % block_size


def _ga_prior_nki(q, k_cache, v_cache, bt, csl, scale, block_size, state, prior_block):
    """GA prior keys of one segment: whole 512-key tiles below ``cached_seq_len`` in the NKI kernel (an even number of
    them, ``_nki_whole_tiles``), the rest (``cached_seq_len % 1024`` keys) as one masked 1024-key torch block; merged
    into ``state`` (in-segment block)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .nki_attn import ga_prior_flash

    dev = q.device
    width = bt.shape[1]
    nt = -(-(width * block_size) // _NKI_KT)
    rows = _nki_key_rows(bt, block_size, nt, switches.get("MIMO_PREFILL_KIDX"))   # [nt * 512]
    kidx = rows.reshape(nt, _NKI_KT // 128, 128).transpose(1, 2).contiguous()  # [nt, 128, 4]
    kidx = kidx.to(torch.int32).to(torch.uint32)
    ntile = _nki_whole_tiles(csl).reshape(1, 1).to(torch.int32)
    qt = q[0].transpose(0, 1).contiguous()                                   # [192, S]
    m, l, acc = wrap_nki(ga_prior_flash)[_NKI_PROGS](qt=qt, k_cache=k_cache, v_cache=v_cache, kidx=kidx,
                                                     ntile=ntile, scale=float(scale))
    # keys [ntile * 512, cached_seq_len): one static 1024-key block, masked past cached_seq_len
    rem = _NKI_KT * _NKI_PROGS
    start = ntile.reshape(1).to(torch.long) * (_NKI_KT // block_size)        # [1] first table column
    idx = start + torch.arange(rem // block_size, device=dev)
    cols = torch.index_select(bt, 1, torch.clamp(idx, max=width - 1))
    kpos = start * block_size + torch.arange(rem, device=dev)
    sc, vb = prior_block(cols, kpos, None)
    state = _online_softmax_update(state, sc, vb)
    for p in range(m.shape[0]):
        state = _merge_states(state, (m[p].unsqueeze(0), l[p].unsqueeze(0), acc[p].unsqueeze(0)))
    return state


def _decode_attention_blocked(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                              block_table: torch.Tensor, positions: torch.Tensor, scale: float, v_head_dim: int,
                              key_block: int, sliding_window: int | None = None, sink: torch.Tensor | None = None,
                              kv_pos_offset: torch.Tensor | None = None) -> torch.Tensor:
    """``decode_attention`` with the block table split into static ``key_block``-token chunks (last
    chunk padded with -1); only one chunk is gathered / converted to FP32 at a time."""
    bsz, hq, dqk = q.shape
    bs = k_cache.shape[2]
    nkb = _check_key_block(key_block, bs, "decode")
    width = block_table.shape[1]
    nb = -(-width // nkb)
    qf = q.to(torch.float32).unsqueeze(2)                                 # [B, Hq, 1, D]
    pos = positions.view(bsz, 1).to(torch.long)
    state = None
    for j in range(nb):
        cols = _table_chunk(block_table, j * nkb, nkb)
        k, valid = gather_paged_kv(k_cache, cols)                         # [B, Hkv, KB, D]
        v, _ = gather_paged_kv(v_cache, cols)
        k = k[..., :dqk]
        v = v[..., :v_head_dim]
        g = hq // k.shape[1]
        if g > 1:
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
        kpos = (j * key_block + torch.arange(key_block, device=q.device)).unsqueeze(0).expand(bsz, key_block)
        if kv_pos_offset is not None:
            kpos = kpos + kv_pos_offset.view(bsz, 1).to(kpos.dtype)
        visible = valid & (kpos <= pos)
        if sliding_window is not None:
            visible = visible & (pos - kpos < sliding_window)
        s = torch.matmul(qf, k.to(torch.float32).transpose(-1, -2)) * scale   # [B, Hq, 1, KB]
        s = s.masked_fill(~visible.view(bsz, 1, 1, key_block), _NEG)
        state = _online_softmax_update(state, s, v.to(torch.float32))
    out = _online_softmax_finish(state, sink).squeeze(2)                  # [B, Hq, Dv]
    return out.reshape(bsz, -1).to(q.dtype)
