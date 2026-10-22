#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the full BF16 conversion.

Checks, on the full output directory (``--full``):
  1. ``model.safetensors.index.json`` exists; every shard in the weight map exists; shard count =
     num_hidden_layers + 1 (one per layer + ``model-embed.safetensors``); no unreferenced shard files.
  2. ``metadata.total_size`` == sum of tensor byte sizes read from the shard headers, and every weight-map
     key is present in its shard (and vice versa).
  3. Every tensor is BF16 except ``*.mlp.gate.e_score_correction_bias`` (must be F32, one per MoE layer).
  4. config.json: num_hidden_layers / hybrid_layer_pattern / moe_layer_freq lengths agree, split layout,
     no quantization_config.
  5. Byte-for-byte: every tensor of full layers ``--l3-layers`` (original ids) equals the tensor of the L3
     directory's layer 0, 1, 2 ...; embed shard tensors equal too.
  6. Optional ``--patched-dir``: modeling/configuration files in --full and --l3 have the same md5 as
     the patched reference copies.
Prints a JSON summary (also written to ``--out``); exit 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def nbytes(meta: dict) -> int:
    n = DTYPE_BYTES[meta["dtype"]]
    for d in meta["shape"]:
        n *= d
    return n


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--l3-layers", required=True, help="original layer ids kept in the L3 dir, e.g. 0,1,5")
    ap.add_argument("--patched-dir", default=None, help="reference/hf_patched (md5 check of modeling files)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    res, errors = {"full": a.full, "l3": a.l3}, []

    idx = json.load(open(os.path.join(a.full, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    cfg = json.load(open(os.path.join(a.full, "config.json")))
    L = cfg["num_hidden_layers"]
    shards = sorted(set(wm.values()))
    expect_shards = sorted([f"model-layer-{i:02d}.safetensors" for i in range(L)] + ["model-embed.safetensors"])
    on_disk = sorted(f for f in os.listdir(a.full) if f.endswith(".safetensors"))
    res["num_hidden_layers"] = L
    res["num_shards"] = len(shards)
    res["num_tensors"] = len(wm)
    if shards != expect_shards:
        errors.append(f"weight-map shards {len(shards)} != expected {len(expect_shards)}")
    if on_disk != shards:
        errors.append(f"shard files on disk {sorted(set(on_disk) ^ set(shards))} differ from the weight map")

    # ---- headers: total_size, key coverage, dtypes
    total, bias, non_bf16, by_shard = 0, {}, [], {}
    for s in shards:
        h = header(os.path.join(a.full, s))
        by_shard[s] = h
        for k, m in h.items():
            total += nbytes(m)
            if wm.get(k) != s:
                errors.append(f"{s}: tensor {k} not mapped to this shard in the index")
            if k.endswith(".mlp.gate.e_score_correction_bias"):
                bias[k] = m["dtype"]
            elif m["dtype"] != "BF16":
                non_bf16.append((k, m["dtype"]))
    missing = [k for k, s in wm.items() if k not in by_shard.get(s, {})]
    res["total_size_index"] = idx["metadata"]["total_size"]
    res["total_size_headers"] = total
    res["total_size_gb"] = round(total / 1e9, 2)
    if total != idx["metadata"]["total_size"]:
        errors.append(f"total_size index {idx['metadata']['total_size']} != headers {total}")
    if missing:
        errors.append(f"{len(missing)} index keys missing from shards, e.g. {missing[:3]}")
    n_moe = sum(1 for f in cfg["moe_layer_freq"] if f)
    res["router_bias"] = {"count": len(bias), "moe_layers": n_moe, "dtypes": sorted(set(bias.values()))}
    if len(bias) != n_moe or set(bias.values()) != {"F32"}:
        errors.append(f"router bias: {len(bias)} tensors (expected {n_moe}), dtypes {set(bias.values())}")
    res["non_bf16_other"] = non_bf16[:10]
    if non_bf16:
        errors.append(f"{len(non_bf16)} non-BF16 tensors other than the router bias")

    # ---- config
    res["config"] = {k: cfg.get(k) for k in ("num_hidden_layers", "attention_projection_layout")}
    res["config"]["quantization_config"] = "quantization_config" in cfg
    if len(cfg["hybrid_layer_pattern"]) != L or len(cfg["moe_layer_freq"]) != L:
        errors.append("hybrid_layer_pattern / moe_layer_freq length != num_hidden_layers")
    if cfg.get("attention_projection_layout") != "split" or "quantization_config" in cfg:
        errors.append("config: attention_projection_layout must be split and quantization_config removed")

    # ---- byte-for-byte vs L3
    import torch
    from safetensors import safe_open

    orig = [int(x) for x in a.l3_layers.split(",")]
    l3_cfg = json.load(open(os.path.join(a.l3, "config.json")))
    for o in orig:
        if cfg["hybrid_layer_pattern"][o] != l3_cfg["hybrid_layer_pattern"][orig.index(o)] or \
                cfg["moe_layer_freq"][o] != l3_cfg["moe_layer_freq"][orig.index(o)]:
            errors.append(f"layer type of full layer {o} differs from L3 layer {orig.index(o)}")
    pairs = [(f"model-layer-{o:02d}.safetensors", f"model-layer-{n:02d}.safetensors", f"model.layers.{o}.",
              f"model.layers.{n}.") for n, o in enumerate(orig)]
    pairs.append(("model-embed.safetensors", "model-embed.safetensors", "", ""))
    cmp = {}
    for fs, ls, fp, lp in pairs:
        n_eq = n_bytes = 0
        with safe_open(os.path.join(a.full, fs), "pt") as F, safe_open(os.path.join(a.l3, ls), "pt") as G:
            fk = sorted(F.keys())
            gk = sorted(G.keys())
            mapped = sorted(lp + k[len(fp):] for k in fk) if fp else fk
            if mapped != gk:
                errors.append(f"{fs} vs L3 {ls}: key sets differ ({len(fk)} vs {len(gk)})")
            for k in fk:
                lk = lp + k[len(fp):] if fp else k
                if lk not in gk:
                    continue
                x, y = F.get_tensor(k), G.get_tensor(lk)
                same = x.dtype == y.dtype and x.shape == y.shape and torch.equal(
                    x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8))
                if not same:
                    errors.append(f"{k} != L3 {lk}")
                n_eq += int(same)
                n_bytes += x.numel() * x.element_size()
        cmp[fs] = {"l3_shard": ls, "tensors_equal": n_eq, "tensors": len(fk), "gb": round(n_bytes / 1e9, 2)}
        print(f"compared {fs} vs L3 {ls}: {n_eq}/{len(fk)} equal ({n_bytes / 1e9:.1f} GB)", flush=True)
    res["byte_compare_vs_l3"] = cmp

    # ---- patched HF files
    if a.patched_dir:
        md = {}
        for fn in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
            ref = os.path.join(a.patched_dir, fn)
            if not os.path.exists(ref):
                continue
            md[fn] = {d: md5(os.path.join(p, fn)) for d, p in (("ref", a.patched_dir), ("full", a.full),
                                                              ("l3", a.l3))}
            if len(set(md[fn].values())) != 1:
                errors.append(f"{fn}: md5 differs {md[fn]}")
        res["patched_md5"] = md

    res["errors"] = errors
    res["passed"] = not errors
    res["seconds"] = round(time.time() - t0, 1)
    s = json.dumps(res, indent=1, ensure_ascii=False)
    print(s)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(s)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
