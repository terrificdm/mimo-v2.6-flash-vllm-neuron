# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 MoE routing (sigmoid scores + noaux_tc selection bias), pure PyTorch.

Matches HF ``MiMoV2MoEGate`` (modeling_mimo_v2.py:152-184) with n_group = topk_group = 1:
  logits = h.float() @ W.float().T ; s = sigmoid(logits)
  idx    = topk(s + bias, k)                  (bias only affects *selection*)
  w      = s.gather(idx) ; w /= w.sum(-1) + 1e-20 (norm_topk_prob) ; w *= routed_scaling_factor
"""

from __future__ import annotations

import torch


def _topk_lastdim(x: torch.Tensor, k: int):
    """``torch.topk(x, k, dim=-1)`` over [T, E], computed per 128-wide chunk and then over the candidates.

    On Neuron (device bug, tools/repro_topk.py), an in-graph ``torch.topk`` over the 256 router scores only returned indices < 128
    (it searched the first 128 columns). Chunking keeps every top-k call within 128 columns; the result
    equals ``torch.topk`` except for the order of exact ties.
    """
    t, e = x.shape
    if e <= 128 or e % 128 or k > 128:
        return torch.topk(x, k, dim=-1)
    c = e // 128
    v, i = torch.topk(x.reshape(t, c, 128), k, dim=-1)                          # [T, c, k]
    i = i + (torch.arange(c, device=x.device, dtype=i.dtype) * 128).view(1, c, 1)
    v2, j = torch.topk(v.reshape(t, c * k), k, dim=-1)
    return v2, torch.gather(i.reshape(t, c * k), 1, j)


def noaux_tc_route(hidden: torch.Tensor, router_weight: torch.Tensor, bias: torch.Tensor, top_k: int,
                   norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
    """Returns (dense_affinities [T, E] fp32, topk_idx [T, k] int64, topk_weight [T, k] fp32)."""
    logits = torch.matmul(hidden.to(torch.float32), router_weight.to(torch.float32).t())
    scores = torch.sigmoid(logits)
    choice = scores + bias.to(torch.float32).unsqueeze(0)
    _, topk_idx = _topk_lastdim(choice, top_k)
    topk_w = torch.gather(scores, 1, topk_idx)
    if top_k > 1 and norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * routed_scaling_factor
    dense = torch.zeros_like(scores).scatter(1, topk_idx, topk_w)
    return dense, topk_idx, topk_w
