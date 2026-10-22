#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device-bug repro (see also tools/repro_topk.py): does split-on-last-dim + reshape of a matmul output lower correctly on Neuron?

Compiles small functions with the libtorch_neuronx_lite torch.compile backend (neuron_libtorch) on one NeuronCore and compares
with eager CPU. Usage (instance, Neuron cores free): python tools/repro_split.py
"""

import os
import time

import torch

T, H, Q, K, V = 1024, 4096, 192, 192, 128


def f_split(h, w):
    qkv = torch.matmul(h, w)
    q, k, v = torch.split(qkv, [Q, K, V], dim=-1)
    return q.reshape(T, 1, Q) * 1.0, k.reshape(T, 1, K) * 1.0, v.reshape(T, 1, V) * 0.707


def f_split_contig(h, w):
    qkv = torch.matmul(h, w)
    q, k, v = torch.split(qkv, [Q, K, V], dim=-1)
    return (q.contiguous().reshape(T, 1, Q) * 1.0, k.contiguous().reshape(T, 1, K) * 1.0,
            v.contiguous().reshape(T, 1, V) * 0.707)


def f_tsplit_view(h, w):
    qkv = torch.matmul(h, w)
    q, k, v = torch.tensor_split(qkv, [Q, Q + K], dim=-1)
    return q.view(T, 1, Q) * 1.0, k.view(T, 1, K) * 1.0, v.view(T, 1, V) * 0.707


def f_wslice(h, w):
    q = torch.matmul(h, w[:, :Q])
    k = torch.matmul(h, w[:, Q:Q + K])
    v = torch.matmul(h, w[:, Q + K:])
    return q.reshape(T, 1, Q) * 1.0, k.reshape(T, 1, K) * 1.0, v.reshape(T, 1, V) * 0.707


def f_rowsplit(h, w):
    qkv_t = torch.matmul(w.t(), h.t())            # [Q+K+V, T]
    q, k, v = torch.split(qkv_t, [Q, K, V], dim=0)
    return q.t().reshape(T, 1, Q) * 1.0, k.t().reshape(T, 1, K) * 1.0, v.t().reshape(T, 1, V) * 0.707


def _rot_chunk(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rot_slice(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def f_rope_chunk(h, w):
    x = torch.matmul(h, w)[:, :Q].reshape(T, 1, Q)
    xr = x[..., :64]
    return x[..., 64:] * 1.0, _rot_chunk(xr) * 1.0, (xr * 0.5 + _rot_chunk(xr) * 0.25) * 1.0


def f_rope_slice(h, w):
    x = torch.matmul(h, w)[:, :Q].reshape(T, 1, Q)
    xr = x[..., :64]
    return x[..., 64:] * 1.0, _rot_slice(xr) * 1.0, (xr * 0.5 + _rot_slice(xr) * 0.25) * 1.0


def main():
    import vllm_neuron  # noqa: F401  (registers the backend / device)

    torch.manual_seed(0)
    h = torch.randn(T, H, dtype=torch.bfloat16)
    w = (torch.randn(H, Q + K + V, dtype=torch.bfloat16) / 64)
    dev = torch.device(os.environ.get("REPRO_DEVICE", "neuron:0"))
    hd, wd = h.to(dev), w.to(dev)
    for fn in (f_split, f_split_contig, f_tsplit_view, f_wslice, f_rowsplit, f_rope_chunk, f_rope_slice):
        ref = fn(h.float(), w.float())
        t0 = time.time()
        try:
            out = torch.compile(fn, backend=os.environ.get("REPRO_BACKEND", "neuron_libtorch"), fullgraph=True)(hd, wd)
            out = [o.cpu().float() for o in out]
            errs = [float((o - r).abs().max() / r.abs().max()) for o, r in zip(out, ref)]
            print(f"{fn.__name__:16s} rel max err q/k/v = " + " ".join(f"{e:.3g}" for e in errs)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__:16s} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
