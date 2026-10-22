#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Element-wise check of the trn2 FP8 conversion (``convert_checkpoint.py --fp8``).

  python tools/check_fp8_converted.py --src <orig dir> [--dst <fp8 dir>] [--layers 0,1,5] --out convert_check.json

For every converted layer, the dequantized FP8 tensors (``code * weight_scale``) are compared
with an independent dequantization of the original checkpoint (``tools/dequant.py``: FP8 block
dequant / fused-qkv split / MXFP4 LUT, the latter tested against the vllm-neuron oracle):

* MoE experts (MXFP4 -> FP8): must be bit-identical (0 differing elements, all layers / experts).
* FP8 linear (E4M3FN max 448 -> trn2 E4M3 max 240): differing elements are counted (underflow /
  rounding of subnormal codes), with max abs / relative error, per tensor kind and per layer.
* With ``--dst``: tensors are read from the written shards (``--layers`` gives the original layer
  ids in output order, as passed to the converter); non-FP8 tensors must equal the BF16 source.
  Without ``--dst`` the conversion is redone in memory (checks the whole checkpoint without writing).
* Diagnostic: how many expert elements would change if the 32-element scales were merged per row
  (NKI "row" FP8 MoE kernels) or per 128 columns.

Exit codes: 0 all checks pass, 1 a check failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import fp8_trn  # noqa: E402
from tools.convert_checkpoint import Src, attn_dims, convert_layer_fp8  # noqa: E402
from tools.dequant import dequant_fp8_block, dequant_mxfp4, split_fused_qkv  # noqa: E402

log = logging.getLogger("check_fp8")
_G = {}


def _init(src, dst, cfg, threads):
    torch.set_num_threads(threads)
    _G.update(src=Src(src), dst=dst, cfg=cfg)


def _read_dst(dst: str, new: int) -> dict:
    out = {}
    with safe_open(os.path.join(dst, f"model-layer-{new:02d}.safetensors"), "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _cmp(got: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = got != ref
    n = int(diff.sum())
    st = {"elements": ref.numel(), "changed": n, "underflow_to_zero": int((diff & (got == 0)).sum()),
          "max_abs_err": 0.0, "max_rel_err_changed": 0.0}
    if n:
        e = (got - ref).abs()
        st["max_abs_err"] = float(e.max())
        st["max_rel_err_changed"] = float((e[diff] / ref[diff].abs()).max())
    return st


def _add(acc: dict, kind: str, st: dict):
    acc[kind] = fp8_trn.add_stats(acc.get(kind, {}), st)


def _job(job):
    orig, new = job
    src, dst, cfg = _G["src"], _G["dst"], _G["cfg"]
    t0 = time.time()
    sp, dp = f"model.layers.{orig}.", f"model.layers.{new}."
    try:
        if dst:
            got = _read_dst(dst, new)
            conv_stats = None
        else:
            got, conv_stats = convert_layer_fp8(src, cfg, orig, new)
        res, coarse, errors = {}, {}, []

        def deq(name, group):
            w, s = got[dp + name + ".weight"], got[dp + name + ".weight_scale"]
            if w.dtype != torch.float8_e4m3fn or s.dtype != torch.float32:
                errors.append(f"{dp}{name}: dtypes {w.dtype}/{s.dtype}")
            if w.float().abs().max().item() > fp8_trn.TRN_FP8_MAX:
                errors.append(f"{dp}{name}: code above {fp8_trn.TRN_FP8_MAX}")
            return fp8_trn.dequant_rowgroup(w, s, group)

        d = attn_dims(cfg, orig)
        qkv = sp + "self_attn.qkv_proj.weight"
        refs = split_fused_qkv(src.get(qkv), src.get(qkv + "_scale_inv"), d["nh"], d["nkv"], d["hd"], d["vd"])
        for n, ref in zip("qkv", refs):
            _add(res, f"{n}_proj", _cmp(deq(f"self_attn.{n}_proj", fp8_trn.LINEAR_GROUP), ref))
        if cfg["moe_layer_freq"][orig]:
            for e in range(cfg["n_routed_experts"]):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    kk = f"mlp.experts.{e}.{proj}"
                    blocks, scales = src.get(sp + kk + ".weight"), src.get(sp + kk + ".weight_scale")
                    ref = dequant_mxfp4(blocks, scales).float()
                    _add(res, "experts", _cmp(deq(kk, fp8_trn.EXPERT_GROUP), ref))
                    codes = got[dp + kk + ".weight"]
                    _add(coarse, "per_row", fp8_trn.coarse_scale_stats(codes, scales, None))
                    _add(coarse, "per_128_cols", fp8_trn.coarse_scale_stats(codes, scales, 4))
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                kk = sp + f"mlp.{proj}.weight"
                ref = dequant_fp8_block(src.get(kk), src.get(kk + "_scale_inv"))
                _add(res, f"dense_{proj}", _cmp(deq(f"mlp.{proj}", fp8_trn.LINEAR_GROUP), ref))
        if dst:  # every non-FP8 tensor equals the source
            plain = [k for k in got if not k.endswith(".weight_scale") and got[k].dtype != torch.float8_e4m3fn]
            for k in plain:
                sk = sp + k[len(dp):]
                s = src.get(sk)
                want = s if s.dtype == torch.float32 else s.to(torch.bfloat16)
                if got[k].dtype != want.dtype or not torch.equal(got[k], want):
                    errors.append(f"{k}: differs from source {sk}")
        if res.get("experts", {}).get("changed", 0):
            errors.append(f"layer {orig}: {res['experts']['changed']} expert elements differ (must be 0)")
        if conv_stats is not None:  # converter's own count must agree with the independent one
            for kind, st in conv_stats.items():
                if kind != "experts" and st["changed"] != res[kind]["changed"]:
                    errors.append(f"layer {orig} {kind}: converter counted {st['changed']}, check {res[kind]['changed']}")
        return orig, new, res, coarse, errors, time.time() - t0
    except Exception as ex:  # reported by the parent
        return orig, new, None, None, [f"{type(ex).__name__}: {ex}"], time.time() - t0


def check_index(dst: str) -> dict:
    """index.json total_size == sum of tensor bytes in the shards; every indexed key is in its shard."""
    with open(os.path.join(dst, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    nbytes, missing, by_shard = 0, [], {}
    for k, fn in idx["weight_map"].items():
        by_shard.setdefault(fn, []).append(k)
    for fn, keys in by_shard.items():
        with safe_open(os.path.join(dst, fn), "pt") as f:
            have = set(f.keys())
            for k in keys:
                if k not in have:
                    missing.append(k)
                    continue
                sl = f.get_slice(k)
                n = 1
                for x in sl.get_shape():
                    n *= x
                nbytes += n * {"F32": 4, "BF16": 2, "F8_E4M3": 1}[sl.get_dtype()]
    return {"total_size": idx["metadata"]["total_size"], "tensor_bytes": nbytes, "shards": len(by_shard),
            "tensors": len(idx["weight_map"]), "missing": missing[:10],
            "ok": nbytes == idx["metadata"]["total_size"] and not missing}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--layers", default=None, help="original layer ids (output order); default all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.path.isfile(os.path.join(a.src, "model.safetensors.index.json")):
        ap.error(f"--src {a.src} has no index")
    if a.dst and not os.path.isfile(os.path.join(a.dst, "model.safetensors.index.json")):
        ap.error(f"--dst {a.dst} has no index (conversion not finished)")
    if not 1 <= a.workers <= 128:
        ap.error("--workers must be within 1..128")
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    ids = list(range(cfg["num_hidden_layers"])) if a.layers is None else [int(x) for x in a.layers.split(",")]
    jobs = [(o, n) for n, o in enumerate(ids)]
    threads = max(1, (os.cpu_count() or 8) // a.workers)
    per_layer, total, coarse_total, errors, t0 = {}, {}, {}, [], time.time()
    with mp.get_context("fork").Pool(min(a.workers, len(jobs)), initializer=_init,
                                     initargs=(a.src, a.dst, cfg, threads), maxtasksperchild=1) as pool:
        for orig, new, res, coarse, errs, dt in pool.imap_unordered(_job, jobs):
            errors += errs
            if res is None:
                log.error("layer %d failed: %s", orig, errs)
                continue
            per_layer[orig] = {"new_layer": new, "kinds": res, "expert_coarse_scale": coarse}
            for k, st in res.items():
                _add(total, k, st)
            for k, st in coarse.items():
                _add(coarse_total, k, st)
            log.info("layer %d checked in %.0fs (%d errors) [%d/%d]", orig, dt, len(errs), len(per_layer), len(jobs))
    lin = [k for k in total if k != "experts"]
    index = None
    if a.dst:
        index = check_index(a.dst)
        if not index["ok"]:
            errors.append(f"index check failed: {index}")
    lin_el = sum(total[k]["elements"] for k in lin)
    lin_ch = sum(total[k]["changed"] for k in lin)
    summary = {
        "src": a.src, "dst": a.dst, "mode": "written shards" if a.dst else "in-memory conversion",
        "layers": ids, "layers_checked": len(per_layer),
        "experts_mxfp4_to_fp8": {**total.get("experts", {}), "lossless": total.get("experts", {}).get("changed", -1) == 0},
        "linear_e4m3fn_to_trn_e4m3": {"elements": lin_el, "changed": lin_ch,
                                      "changed_fraction": lin_ch / lin_el if lin_el else None,
                                      "by_kind": {k: {**total[k], "changed_fraction": total[k]["changed"] / total[k]["elements"]}
                                                  for k in lin}},
        "expert_coarse_scale_diagnostic": {k: {**v, "changed_fraction": v["changed"] / v["elements"]}
                                           for k, v in coarse_total.items()},
        "per_layer": {str(k): per_layer[k] for k in sorted(per_layer)},
        "index_check": index,
        "errors": errors[:50], "n_errors": len(errors), "pass": not errors and len(per_layer) == len(jobs),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    log.info("experts lossless=%s; linear changed %d / %d (%.3g); errors %d; -> %s",
             summary["experts_mxfp4_to_fp8"]["lossless"], lin_ch, lin_el, lin_ch / max(lin_el, 1), len(errors), a.out)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
