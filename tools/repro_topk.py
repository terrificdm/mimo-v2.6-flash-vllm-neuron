#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device-bug repro (see also tools/repro_split.py): which top-k implementation selects the right experts on Neuron?

Router shape [T, 256] fp32, k=8 (MiMo-V2 MoE). Compares the selected index *sets* with CPU torch.topk.
Usage (instance, one free NeuronCore): NEURON_RT_VISIBLE_CORES=0 [REPRO_CHOICE=<captured .pt>] python tools/repro_topk.py
(torch.sort is not lowered on trn2: NCC_EVRF029; NF.topk without a process group fails in Dynamo.)
"""

import os

import torch

E, K = 256, 8


def t_torch(x):
    return torch.topk(x, K, dim=-1).indices.to(torch.int32)


def t_nf(x):
    import vllm_neuron.functional as NF

    return NF.topk(x, k=K, dim=-1, gather_dim=-1, process_group=None)[1].to(torch.int32)


def t_sort(x):
    return torch.sort(x, dim=-1, descending=True).indices[:, :K].to(torch.int32)


def t_argmax_loop(x):
    out = []
    neg = torch.full_like(x, torch.finfo(torch.float32).min)
    cols = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
    for _ in range(K):
        i = torch.argmax(x, dim=-1, keepdim=True)
        out.append(i)
        x = torch.where(cols == i, neg, x)
    return torch.cat(out, dim=-1).to(torch.int32)


def main():
    import vllm_neuron  # noqa: F401

    dev = torch.device("neuron:0")
    backend = os.environ.get("REPRO_BACKEND", "neuron_libtorch")
    torch.manual_seed(0)
    cases = [(T, torch.sigmoid(torch.randn(T, E)) + torch.randn(E) * 1e-3) for T in (1, 4, 16, 64)]
    cap = os.environ.get("REPRO_CHOICE")   # captured router choice [1, 256] from diag_capture (MIMO_DEBUG_CAPTURE)
    if cap:
        cases.insert(0, (1, torch.load(cap, weights_only=False).float().reshape(1, E)))
    for T, x in cases:
        ref = [set(r) for r in torch.topk(x, K, dim=-1).indices.tolist()]
        for fn in (t_torch, t_argmax_loop):
            torch._dynamo.reset()
            try:
                got = torch.compile(fn, backend=backend, fullgraph=True)(x.to(dev)).cpu()
                ok = sum(set(g) == r for g, r in zip(got.tolist(), ref))
                print(f"T={T:3d} {fn.__name__:14s} rows with correct set: {ok}/{T}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"T={T:3d} {fn.__name__:14s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
