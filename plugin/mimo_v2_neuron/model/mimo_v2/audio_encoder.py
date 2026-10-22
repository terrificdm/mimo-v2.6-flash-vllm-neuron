# SPDX-License-Identifier: Apache-2.0
# Portions ported from the XiaomiMiMo/MiMo-V2.6-Flash-MOPD remote code (modeling_mimo_v2.py, revision b12f557;
#   Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team, Apache-2.0 file header):
#   AudioTokenizerEncoder, residual VQ and MiMoAudioEncoder, rewritten with plain tensor ops. See NOTICE.
"""MiMo audio towers for vllm-neuron (docs/design-notes.md#audio-input).

Two towers, ported from HF ``reference/hf/modeling_mimo_v2.py`` (``AudioTokenizerEncoder`` 1374-1497, RVQ 1143-1232,
``MiMoAudioEncoder`` 932-1036) and written with plain tensor ops so the same code runs on CPU and in a Neuron graph:

  1. AudioTokenizer encoder: log-mel ``[L, 128]`` -> conv1d(k3) + GELU -> conv1d(k3, s2) + GELU -> 24 pre-LN
     Transformer layers (d 1024, 16 heads, RoPE theta 1e4, causal; even layers also limited to the previous 128 keys;
     output of layer 3 added back after the last layer) -> LayerNorm -> conv(k2, s2, no bias) + GELU -> LayerNorm ->
     20-level residual VQ in FP32 (nearest code word by ``-(|x|^2 - 2 x E^T + |E|^2)``) -> codes ``[T2, 20]``;
  2. code embedding: codes padded to a multiple of 4 by repeating the last row, ``sum_j speech_embeddings[j]`` (BF16,
     added in order like HF), groups of 4 -> 6-layer Qwen2 transformer (full attention inside each group of 4, RoPE
     theta 6.4e5, q/k/v bias) -> RMSNorm -> ``[G, 4096]`` -> Linear(4096, 16384) + GELU + Linear(16384, 4096).

Fixed shapes: one item per call, mel padded to ``mel_bucket`` frames (default 3200 = 32 s; ``T1 = mel_bucket / 2``,
``T2 = T1 / 2`` codes, ``G = T2 / 4`` output tokens). Padding never reaches valid positions: attention is causal,
padded rows are zeroed before the stride-2 pooling, padded codes repeat the last valid code (as HF).

Tensor parallel inside the vision TP group (as the ViT): attention heads and FFN columns are split; the output
projections are row-parallel with FP32 partial sums and an FP32 all-reduce. Convolutions, norms, codebooks and the
speech embeddings are replicated. Weights keep the checkpoint dtypes (BF16; codebooks F32 storage holding BF16-rounded
values like vLLM on GPU, ``MIMO_AUDIO_CODEBOOK``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .routing import _topk_lastdim
from .vision_encoder import _all_reduce, _loader, _p, _rotate_half, _set_loader
from .vision_encoder import gelu as _gelu_erf

TOK = "audio_tokenizer.encoder."
ENC = "audio_encoder."
SPE = "speech_embeddings."


@dataclass
class MiMoAudioConfig:
    # tokenizer (audio_tokenizer/config.json)
    n_mels: int = 128
    d_model: int = 1024
    tok_layers: int = 24
    tok_heads: int = 16
    tok_ffn: int = 4096
    tok_window: int = 128
    swa_per_block: int = 2
    hybrid_attention: bool = True
    skip_layer_id: int = 3
    stride: int = 2
    avg_pooler: int = 2
    tok_rope_theta: float = 10000.0
    codebook_sizes: tuple = (1024, 1024, 256) + (128,) * 17
    ln_eps: float = 1e-5
    # code embedding + local transformer (config.json audio_config)
    channels: int = 20
    group: int = 4
    speech_vocab: int = 1280
    local_dim: int = 1024
    local_layers: int = 6
    local_heads: int = 16
    local_ffn: int = 4096
    local_rope_theta: float = 640000.0
    local_causal: bool = False
    rms_eps: float = 1e-6
    out_hidden: int = 4096
    # shapes
    mel_bucket: int = 3200

    @classmethod
    def from_configs(cls, audio_config: dict, tokenizer_config: dict, mel_bucket: int = 3200) -> "MiMoAudioConfig":
        a, t = dict(audio_config), dict(tokenizer_config)
        if a.get("projection_layers", 2) != 2 or not a.get("add_post_norm", True):
            raise ValueError("MiMo audio: only projection_layers=2 with add_post_norm is implemented")
        if t.get("ln_type", "LayerNorm") != "LayerNorm" or t.get("position_embedding_type", "rope") != "rope":
            raise ValueError("MiMo audio: tokenizer must use LayerNorm + RoPE")
        n_q = int(t.get("num_quantizers", 20))
        cbs = list(t.get("codebook_size", [1024]))
        cbs += [cbs[-1]] * (n_q - len(cbs))
        vocab = a.get("speech_vocab_size", 1280)
        if isinstance(vocab, str) and "-" in vocab:
            raise ValueError("MiMo audio: per-channel speech_vocab_size lists are not implemented")
        c = cls(n_mels=t["n_mels"], d_model=t["d_model"], tok_layers=t["encoder_layers"],
                tok_heads=t["encoder_attention_heads"], tok_ffn=t["encoder_ffn_dim"],
                tok_window=int(t.get("encoder_attn_window_size", [-1, -1])[0]),
                swa_per_block=int(t.get("swa_per_block", 2)), hybrid_attention=bool(t.get("hybrid_attention", False)),
                skip_layer_id=t.get("encoder_skip_layer_id"), stride=t["stride_size"], avg_pooler=t["avg_pooler"],
                tok_rope_theta=float(t.get("rope_theta", 10000)), codebook_sizes=tuple(cbs[:n_q]),
                channels=int(a["audio_channels"]), group=int(a["group_size"]), speech_vocab=int(vocab),
                local_dim=int(a["input_local_dim"]), local_layers=int(a["input_local_layers"]),
                local_heads=int(a["input_local_attn_heads"]), local_ffn=int(a["input_local_intermediate_size"]),
                local_rope_theta=float(a.get("rope_theta", 640000.0)),
                local_causal=not a.get("input_full_attention", True), out_hidden=int(a["out_hidden_size"]),
                mel_bucket=mel_bucket)
        if not t.get("encoder_causal", True):
            raise ValueError("MiMo audio: non-causal tokenizer encoder is not implemented")
        if c.channels != len(c.codebook_sizes):
            raise ValueError(f"MiMo audio: {c.channels} channels but {len(c.codebook_sizes)} quantizers")
        if mel_bucket % (c.stride * c.avg_pooler * c.group):
            raise ValueError(f"MiMo audio: mel_bucket {mel_bucket} must be a multiple of "
                             f"{c.stride * c.avg_pooler * c.group}")
        return c

    def window_layer(self, i: int) -> bool:
        if self.tok_window <= 0:
            return False
        return (i % self.swa_per_block < self.swa_per_block - 1) if self.hybrid_attention else True

    @property
    def t1(self) -> int:
        return (self.mel_bucket - 1) // self.stride + 1

    @property
    def t2(self) -> int:
        return math.ceil(self.t1 / self.avg_pooler)

    @property
    def g(self) -> int:
        return self.t2 // self.group


# ── lengths (same formulas as the HF processor / encoder) ───────────────────
def mel_to_lengths(mel_len: int, cfg: MiMoAudioConfig) -> tuple[int, int, int]:
    """mel frames -> (T1 after the stride-2 conv, T2 codes after pooling, G output tokens)."""
    t1 = (mel_len + 3 - 3 + 2 - 3) // cfg.stride + 1
    t2 = t1 // cfg.avg_pooler + int(t1 % cfg.avg_pooler != 0)
    return t1, t2, math.ceil(t2 / cfg.group)


def rope_cos_sin(n: int, head_dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """HF default RoPE: FP32 ``positions x inv_freq``, ``cat(f, f)``, cos / sin cast to the activation dtype."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    e = torch.cat([f, f], dim=-1)
    return e.cos().to(dtype), e.sin().to(dtype)


def prepare_inputs(mel: torch.Tensor, cfg: MiMoAudioConfig, dtype=torch.bfloat16) -> dict:
    """CPU pre-processing of one item: padded mel and the integer / mask inputs of :meth:`MiMoAudioModel.forward`."""
    L = int(mel.shape[0])
    if L > cfg.mel_bucket:
        raise ValueError(f"MiMo audio: {L} mel frames exceed the bucket of {cfg.mel_bucket} "
                         f"({cfg.mel_bucket / 100:.0f} s)")
    if L < 1:
        raise ValueError("MiMo audio: empty audio")
    t1, t2, g = mel_to_lengths(L, cfg)
    padded = torch.zeros(cfg.mel_bucket, cfg.n_mels, dtype=dtype)
    padded[:L] = mel.to(dtype)
    tok_cos, tok_sin = rope_cos_sin(cfg.t1, cfg.d_model // cfg.tok_heads, cfg.tok_rope_theta, dtype)
    loc_cos, loc_sin = rope_cos_sin(cfg.group, cfg.local_dim // cfg.local_heads, cfg.local_rope_theta, dtype)
    i = torch.arange(cfg.t1)
    causal = i[None, :] <= i[:, None]
    window = causal & ((i[:, None] - i[None, :]) <= cfg.tok_window)
    return {"mel": padded, "mel_valid": (torch.arange(cfg.mel_bucket) < L).to(dtype).unsqueeze(0),
            "tok_cos": tok_cos, "tok_sin": tok_sin, "loc_cos": loc_cos, "loc_sin": loc_sin,
            "mask_full": causal, "mask_window": window,
            "t1_valid": (torch.arange(cfg.t1) < t1).to(dtype).unsqueeze(-1),
            "code_index": torch.clamp(torch.arange(cfg.t2), max=t2 - 1).to(torch.int64),
            "lengths": (L, t1, t2, g)}


# ── TP loaders: HF Linear weights are [out, in] ─────────────────────────────
def _col_w(n_out: int, tp: int):
    """column parallel: rows [r*n, (r+1)*n) of ``[out, in]`` -> ``[in, n]``."""
    sh = n_out // tp
    return _loader(lambda s, rank: s[0][(rank % tp) * sh:(rank % tp + 1) * sh, :].t())


def _col_b(n_out: int, tp: int):
    sh = n_out // tp
    return _loader(lambda s, rank: s[0][(rank % tp) * sh:(rank % tp + 1) * sh])


def _row_w(n_in: int, tp: int):
    """row parallel: columns [r*n, (r+1)*n) of ``[out, in]`` -> ``[n, out]``."""
    sh = n_in // tp
    return _loader(lambda s, rank: s[0][:, (rank % tp) * sh:(rank % tp + 1) * sh].t())


def _rep():
    return _loader(lambda s, rank: s[0][:])


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU computed in FP32 and rounded once (what ``F.gelu`` / ``nn.GELU`` do on BF16 CPU tensors); the
    BF16 op-by-op form rounds five times and costs ~0.004 cosine on the 16384-wide projection (docs/design-notes.md#audio-criteria)."""
    return _gelu_erf(x.to(torch.float32)).to(x.dtype)


def _mm(x, w, b=None):
    """``x @ w (+ b)`` with ``w`` ``[in, out]``, one rounding (HF nn.Linear)."""
    return F.linear(x, w.t(), b)


def _row_parallel(x, w, b, group):
    out = torch.matmul(x.to(torch.float32), w.to(torch.float32))
    if group.world_size > 1:
        out = _all_reduce(group, out)
    if b is not None:
        out = out + b.to(torch.float32)
    return out.to(x.dtype)


def _layer_norm(x, w, b, eps):
    return F.layer_norm(x.to(torch.float32), (x.shape[-1],), w.to(torch.float32), b.to(torch.float32), eps).to(x.dtype)


def _rms_norm(x, w, eps):
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)                    # HF Qwen2RMSNorm: weight * normalised.to(input dtype)


def _rope(x, cos, sin):
    """HF apply_rotary_pos_emb in the activation dtype; x ``[..., S, heads, hd]``, cos/sin ``[S, hd]``."""
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return (x * c) + (_rotate_half(x) * s)


def _attend(q, k, v, mask, scale):
    """q/k/v ``[B, S, h, hd]``; boolean ``mask`` ``[S, S]`` (True = visible) or None; FP32 softmax."""
    q, k, v = (t.transpose(-3, -2) for t in (q, k, v))                         # [B, h, S, hd]
    s = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)) * scale
    if mask is not None:
        s = s.masked_fill(~mask, torch.finfo(torch.float32).min)
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v.to(torch.float32)).to(q.dtype)
    return o.transpose(-3, -2)                                                   # [B, S, h, hd]


class _TokLayer(nn.Module):
    def __init__(self, cfg: MiMoAudioConfig, i: int, tp_group, dtype):
        super().__init__()
        self.group, self.window = tp_group, cfg.window_layer(i)
        tp = tp_group.world_size
        d, f = cfg.d_model, cfg.tok_ffn
        self.h = cfg.tok_heads // tp
        self.hd = d // cfg.tok_heads
        self.eps = cfg.ln_eps
        n = self.h * self.hd
        self.q_w, self.q_b, self.k_w = _p(d, n, dtype=dtype), _p(n, dtype=dtype), _p(d, n, dtype=dtype)
        self.v_w, self.v_b = _p(d, n, dtype=dtype), _p(n, dtype=dtype)
        self.o_w, self.o_b = _p(n, d, dtype=dtype), _p(d, dtype=dtype)
        self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b = (_p(d, dtype=dtype) for _ in range(4))
        self.fc1_w, self.fc1_b = _p(d, f // tp, dtype=dtype), _p(f // tp, dtype=dtype)
        self.fc2_w, self.fc2_b = _p(f // tp, d, dtype=dtype), _p(d, dtype=dtype)
        for nm in ("q_w", "k_w", "v_w"):
            _set_loader(getattr(self, nm), _col_w(d, tp))
        for nm in ("q_b", "v_b"):
            _set_loader(getattr(self, nm), _col_b(d, tp))
        _set_loader(self.o_w, _row_w(d, tp))
        _set_loader(self.fc1_w, _col_w(f, tp))
        _set_loader(self.fc1_b, _col_b(f, tp))
        _set_loader(self.fc2_w, _row_w(f, tp))
        for nm in ("o_b", "ln1_w", "ln1_b", "ln2_w", "ln2_b", "fc2_b"):
            _set_loader(getattr(self, nm), _rep())

    @staticmethod
    def keys(i: int) -> dict:
        p = f"{TOK}layers.{i}."
        a = p + "self_attn."
        return {"q_w": a + "q_proj.weight", "q_b": a + "q_proj.bias", "k_w": a + "k_proj.weight",
                "v_w": a + "v_proj.weight", "v_b": a + "v_proj.bias", "o_w": a + "out_proj.weight",
                "o_b": a + "out_proj.bias", "ln1_w": p + "self_attn_layer_norm.weight",
                "ln1_b": p + "self_attn_layer_norm.bias", "ln2_w": p + "final_layer_norm.weight",
                "ln2_b": p + "final_layer_norm.bias", "fc1_w": p + "fc1.weight", "fc1_b": p + "fc1.bias",
                "fc2_w": p + "fc2.weight", "fc2_b": p + "fc2.bias"}

    def forward(self, x, cos, sin, mask_full, mask_window):
        t = x.shape[0]
        r = x
        h = _layer_norm(x, self.ln1_w, self.ln1_b, self.eps)
        q = _mm(h, self.q_w, self.q_b).view(1, t, self.h, self.hd)
        k = _mm(h, self.k_w).view(1, t, self.h, self.hd)
        v = _mm(h, self.v_w, self.v_b).view(1, t, self.h, self.hd)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = _attend(q, k, v, mask_window if self.window else mask_full, self.hd ** -0.5).reshape(t, self.h * self.hd)
        x = r + _row_parallel(o, self.o_w, self.o_b, self.group)
        h = _layer_norm(x, self.ln2_w, self.ln2_b, self.eps)
        h = gelu(_mm(h, self.fc1_w, self.fc1_b))
        return x + _row_parallel(h, self.fc2_w, self.fc2_b, self.group)


class _LocalLayer(nn.Module):
    """Qwen2 decoder layer (RMSNorm, q/k/v bias, no o_proj bias, SwiGLU)."""

    def __init__(self, cfg: MiMoAudioConfig, i: int, tp_group, dtype):
        super().__init__()
        self.group, self.causal = tp_group, cfg.local_causal
        tp = tp_group.world_size
        d, f = cfg.local_dim, cfg.local_ffn
        self.h, self.hd, self.eps = cfg.local_heads // tp, d // cfg.local_heads, cfg.rms_eps
        n = self.h * self.hd
        self.q_w, self.k_w, self.v_w = (_p(d, n, dtype=dtype) for _ in range(3))
        self.q_b, self.k_b, self.v_b = (_p(n, dtype=dtype) for _ in range(3))
        self.o_w = _p(n, d, dtype=dtype)
        self.ln1_w, self.ln2_w = _p(d, dtype=dtype), _p(d, dtype=dtype)
        self.gate_w, self.up_w = _p(d, f // tp, dtype=dtype), _p(d, f // tp, dtype=dtype)
        self.down_w = _p(f // tp, d, dtype=dtype)
        for nm in ("q_w", "k_w", "v_w"):
            _set_loader(getattr(self, nm), _col_w(d, tp))
        for nm in ("q_b", "k_b", "v_b"):
            _set_loader(getattr(self, nm), _col_b(d, tp))
        _set_loader(self.o_w, _row_w(d, tp))
        _set_loader(self.gate_w, _col_w(f, tp))
        _set_loader(self.up_w, _col_w(f, tp))
        _set_loader(self.down_w, _row_w(f, tp))
        _set_loader(self.ln1_w, _rep())
        _set_loader(self.ln2_w, _rep())

    @staticmethod
    def keys(i: int) -> dict:
        p = f"{ENC}input_local_transformer.layers.{i}."
        a = p + "self_attn."
        return {"q_w": a + "q_proj.weight", "k_w": a + "k_proj.weight", "v_w": a + "v_proj.weight",
                "q_b": a + "q_proj.bias", "k_b": a + "k_proj.bias", "v_b": a + "v_proj.bias",
                "o_w": a + "o_proj.weight", "ln1_w": p + "input_layernorm.weight",
                "ln2_w": p + "post_attention_layernorm.weight", "gate_w": p + "mlp.gate_proj.weight",
                "up_w": p + "mlp.up_proj.weight", "down_w": p + "mlp.down_proj.weight"}

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        r = x
        h = _rms_norm(x, self.ln1_w, self.eps)
        q = _mm(h, self.q_w, self.q_b).view(b, s, self.h, self.hd)
        k = _mm(h, self.k_w, self.k_b).view(b, s, self.h, self.hd)
        v = _mm(h, self.v_w, self.v_b).view(b, s, self.h, self.hd)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        mask = None
        if self.causal:
            i = torch.arange(s, device=x.device)
            mask = i[None, :] <= i[:, None]
        o = _attend(q, k, v, mask, self.hd ** -0.5).reshape(b, s, self.h * self.hd)
        x = r + _row_parallel(o, self.o_w, None, self.group)
        h = _rms_norm(x, self.ln2_w, self.eps)
        g, u = _mm(h, self.gate_w), _mm(h, self.up_w)
        a = (g.to(torch.float32) * torch.sigmoid(g.to(torch.float32))).to(g.dtype) * u
        return x + _row_parallel(a, self.down_w, None, self.group)


def _pool(m, h):
    """``[T1b, d]`` -> avg-pool conv (kernel = stride = avg_pooler) + GELU + LayerNorm -> ``[T2b, d]``."""
    cfg = m.cfg
    t1b = h.shape[0]
    if t1b % cfg.avg_pooler:
        h = F.pad(h, (0, 0, 0, cfg.avg_pooler - t1b % cfg.avg_pooler))
    h = gelu(F.conv1d(h.t().unsqueeze(0).contiguous(), m.down_w, None, stride=cfg.avg_pooler))[0].t()
    h = h.contiguous()                                                                 # [T2b, d]
    return _layer_norm(h, m.down_ln_w, m.down_ln_b, m.cfg.ln_eps)


def _rvq(m, h):
    """residual VQ over ``m.codebooks`` (FP32) -> codes ``[T2b, channels]`` (int64).

    ``MIMO_AUDIO_RVQ`` (docs/design-notes.md#audio-on-device) picks how the nearest codeword is selected and looked up:
      ``first_gather`` (default) first index of the maximum (compare + min, exact in FP32) + ``index_select``;
      ``onehot``       first index of the maximum + codeword as ``one_hot @ codebook`` (FP32, exact);
      ``gather``       ``_topk_lastdim`` + ``index_select`` (the first form tried);
      ``topk_onehot``  ``_topk_lastdim`` + one-hot product.
    The two top-k forms compile on trn2 but their NEFF fails to load (TDRV "invalid offset ...
    create_data_refill_rings"); all four give the same codes on the CPU (up to the order of exact ties)."""
    mode = os.environ.get("MIMO_AUDIO_RVQ", "first_gather")
    sel_first = mode in ("onehot", "first_gather")       # selection: first max index (else _topk_lastdim)
    lut_onehot = mode in ("onehot", "topk_onehot")       # codeword lookup: one_hot @ codebook (else index_select)
    res = h.to(torch.float32)
    codes = []
    for cb in m.codebooks:
        dist = -(res.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(res, cb.t()) + cb.pow(2).sum(1)[None, :])
        n = cb.shape[0]
        ar = torch.arange(n, device=res.device, dtype=torch.float32)[None, :] if (sel_first or lut_onehot) else None
        if sel_first:
            cand = torch.where(dist == dist.amax(1, keepdim=True), ar, torch.full_like(dist, n))
            idx = cand.amin(1)                                                       # exact integers in FP32
        else:
            idx = _topk_lastdim(dist, 1)[1][:, 0]
        if lut_onehot:
            res = res - torch.matmul((ar == idx.to(torch.float32)[:, None]).to(torch.float32), cb)
        else:
            res = res - torch.index_select(cb, 0, idx.to(torch.int64))
        codes.append(idx.to(torch.int64))
    return torch.stack(codes, dim=1)


class MiMoAudioModel(nn.Module):
    """Both towers for one TP rank of the vision TP group. ``forward`` -> (embeddings ``[G, out_hidden]``,
    codes ``[T2, channels]`` int64) for one padded item (inputs from :func:`prepare_inputs`)."""

    def __init__(self, cfg: MiMoAudioConfig, dtype: torch.dtype = torch.bfloat16, tp_group=None):
        super().__init__()
        if tp_group is None:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_tp_group

            tp_group = get_neuron_vision_tp_group()
        tp = tp_group.world_size
        for name, n in (("tok_heads", cfg.tok_heads), ("tok_ffn", cfg.tok_ffn), ("local_heads", cfg.local_heads),
                        ("local_ffn", cfg.local_ffn), ("projection", cfg.out_hidden * 4)):
            if n % tp:
                raise ValueError(f"MiMo audio: {name} {n} not divisible by vision tp {tp}")
        self.cfg, self.group, self.dtype = cfg, tp_group, dtype
        d = cfg.d_model
        self.conv1_w = _p(d, cfg.n_mels, 3, dtype=dtype)
        self.conv1_b = _p(d, dtype=dtype)
        self.conv2_w = _p(d, d, 3, dtype=dtype)
        self.conv2_b = _p(d, dtype=dtype)
        self.ln_w, self.ln_b = _p(d, dtype=dtype), _p(d, dtype=dtype)
        self.down_w = _p(d, d, cfg.avg_pooler, dtype=dtype)
        self.down_ln_w, self.down_ln_b = _p(d, dtype=dtype), _p(d, dtype=dtype)
        self.tok_layers = nn.ModuleList([_TokLayer(cfg, i, tp_group, dtype) for i in range(cfg.tok_layers)])
        self.codebooks = nn.ParameterList([_p(n, d, dtype=torch.float32) for n in cfg.codebook_sizes])
        self.speech_emb = nn.ParameterList([_p(cfg.speech_vocab, cfg.local_dim, dtype=dtype)
                                            for _ in range(cfg.channels)])
        self.local_layers = nn.ModuleList([_LocalLayer(cfg, i, tp_group, dtype) for i in range(cfg.local_layers)])
        self.local_norm_w = _p(cfg.local_dim, dtype=dtype)
        pin = cfg.local_dim * cfg.group
        self.proj1_w = _p(pin, pin * 4 // tp, dtype=dtype)
        self.proj2_w = _p(pin * 4 // tp, cfg.out_hidden, dtype=dtype)
        for nm in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "ln_w", "ln_b", "down_w", "down_ln_w", "down_ln_b",
                   "local_norm_w"):
            _set_loader(getattr(self, nm), _rep())
        # vLLM (GPU) loads the whole tokenizer with ``.to(bfloat16)`` and only casts the quantizer back to FP32 in
        # ``encode``, so its codebooks hold BF16-rounded values (mimo_audio.py:1273, HF load_audio_tokenizer the
        # same). Default follows that (docs/design-notes.md#audio-criteria); MIMO_AUDIO_CODEBOOK=fp32 keeps the checkpoint's F32 values.
        cb_bf16 = os.environ.get("MIMO_AUDIO_CODEBOOK", "bf16") != "fp32"
        for p in self.codebooks:
            _set_loader(p, _loader(lambda s, rank: s[0][:].to(torch.bfloat16).to(torch.float32)) if cb_bf16
                        else _rep())
        for p in self.speech_emb:
            _set_loader(p, _rep())
        _set_loader(self.proj1_w, _col_w(pin * 4, tp))
        _set_loader(self.proj2_w, _row_w(pin * 4, tp))

    def checkpoint_keys(self) -> dict:
        m = {"conv1_w": TOK + "conv1.weight", "conv1_b": TOK + "conv1.bias", "conv2_w": TOK + "conv2.weight",
             "conv2_b": TOK + "conv2.bias", "ln_w": TOK + "layer_norm.weight", "ln_b": TOK + "layer_norm.bias",
             "down_w": TOK + "down_sample_layer.0.weight", "down_ln_w": TOK + "down_sample_norm.weight",
             "down_ln_b": TOK + "down_sample_norm.bias", "local_norm_w": ENC + "input_local_transformer.norm.weight",
             "proj1_w": ENC + "projection.mlp.0.weight", "proj2_w": ENC + "projection.mlp.2.weight"}
        for i in range(self.cfg.tok_layers):
            m.update({f"tok_layers.{i}.{k}": v for k, v in _TokLayer.keys(i).items()})
        for j in range(self.cfg.channels):
            m[f"codebooks.{j}"] = f"{TOK}quantizer.vq.layers.{j}._codebook.embed"
            m[f"speech_emb.{j}"] = f"{SPE}{j}.weight"
        for i in range(self.cfg.local_layers):
            m.update({f"local_layers.{i}.{k}": v for k, v in _LocalLayer.keys(i).items()})
        return m

    # ── towers ─────────────────────────────────────────────────────────────
    def tokenize(self, mel, mel_valid, tok_cos, tok_sin, mask_full, mask_window, t1_valid):
        """padded mel ``[Lb, n_mels]`` -> RVQ codes ``[T2b, channels]`` (int64) and the pre-VQ features."""
        cfg = self.cfg
        x = mel.to(self.dtype).t().unsqueeze(0).contiguous()                           # [1, 128, Lb]
        # HF convolves exactly L frames (zero padding after them): zero conv1's outputs past L, otherwise the
        # stride-2 conv2 sees gelu(bias) at frame L in the last valid output
        x = gelu(F.conv1d(x, self.conv1_w, self.conv1_b, padding=1)) * mel_valid
        x = gelu(F.conv1d(x, self.conv2_w, self.conv2_b, stride=cfg.stride, padding=1))
        h = x[0].t().contiguous()                                                      # [T1b, d]
        skip = None
        for i, layer in enumerate(self.tok_layers):
            h = layer(h, tok_cos, tok_sin, mask_full, mask_window)
            if cfg.skip_layer_id is not None and i == cfg.skip_layer_id - 1:
                skip = h
        if skip is not None:
            h = h + skip
        h = _layer_norm(h, self.ln_w, self.ln_b, cfg.ln_eps)
        h = h * t1_valid                                                               # HF pads with zeros
        h = _pool(self, h)
        return _rvq(self, h), h

    def embed_codes(self, codes, code_index, loc_cos, loc_sin):
        """codes ``[T2b, C]`` -> ``[G, out_hidden]`` (rows past the valid length repeat the last valid code)."""
        cfg = self.cfg
        codes = torch.index_select(codes, 0, code_index)
        t2b = codes.shape[0]
        e = torch.zeros(t2b, cfg.local_dim, dtype=self.dtype, device=codes.device)
        for j, w in enumerate(self.speech_emb):
            e = e + torch.index_select(w, 0, codes[:, j])
        x = e.view(t2b // cfg.group, cfg.group, cfg.local_dim)
        for layer in self.local_layers:
            x = layer(x, loc_cos, loc_sin)
        x = _rms_norm(x, self.local_norm_w, cfg.rms_eps).reshape(t2b // cfg.group, cfg.group * cfg.local_dim)
        return _row_parallel(gelu(_mm(x, self.proj1_w)), self.proj2_w, None, self.group)

    def forward(self, mel, mel_valid, tok_cos, tok_sin, loc_cos, loc_sin, mask_full, mask_window, t1_valid,
                code_index):
        codes, _ = self.tokenize(mel, mel_valid, tok_cos, tok_sin, mask_full, mask_window, t1_valid)
        return self.embed_codes(codes, code_index, loc_cos, loc_sin), codes
