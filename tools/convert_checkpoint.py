#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the MiMo-V2.6-Flash-MOPD checkpoint (FP8 block + MXFP4 + TP4 fused QKV)
into a plain BF16 checkpoint with split q/k/v projections (HF "split" layout).

  python tools/convert_checkpoint.py --src <orig dir> --dst <out dir> [--layers 0,1,5] [--workers 16] [--fp8]

* One output shard per decoder layer (``model-layer-XX.safetensors``) plus
  ``model-embed.safetensors`` (embed / final norm / lm_head).  Shards are written
  to ``*.tmp`` and atomically renamed; existing shards are skipped (resumable).
* ``*.mlp.gate.e_score_correction_bias`` is kept in F32; every other float
  tensor is written as BF16.
* MTP / vision / audio tensors are dropped.
* ``--fp8`` (default off, use a new ``-fp8`` directory): keep FP8 weights for trn2 instead of
  dequantizing (``tools/fp8_trn.py``): q/k/v, dense MLP and MoE experts become ``float8_e4m3fn``
  codes with ``|code| <= 240`` plus an F32 ``weight_scale`` ``[R, C // G]`` (G = 128 linear,
  32 experts; MXFP4 -> FP8 is bit-exact).  o_proj (BF16 in the checkpoint) and the rest stay
  BF16.  Per-shard rescale statistics go to ``fp8_stats/<shard>.json``; config.json gets a
  ``neuron_fp8_config`` block.
* ``model.safetensors.index.json`` is written last (atomic) and marks completion.

Exit codes: 0 ok / already converted, 1 conversion or validation error, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dequant import dequant_fp8_block, dequant_mxfp4, split_fused_qkv  # noqa: E402
from tools import fp8_trn  # noqa: E402

log = logging.getLogger("convert")
BIAS_SUFFIX = ".mlp.gate.e_score_correction_bias"
SCALE_SUFFIX = ".weight_scale"
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F8_E4M3": 1}
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "generation_config.json", "configuration_mimo_v2.py",
              "modeling_mimo_v2.py"]


class ConvertError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers
class Src:
    """Lazy tensor reader over the original sharded checkpoint."""

    def __init__(self, src_dir: str):
        self.dir = src_dir
        with open(os.path.join(src_dir, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, key: str) -> torch.Tensor:
        if key not in self.wm:
            raise ConvertError(f"missing source tensor {key}")
        fn = self.wm[key]
        h = self._handles.get(fn)
        if h is None:
            h = safe_open(os.path.join(self.dir, fn), "pt")
            self._handles[fn] = h
        return h.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.wm


def attn_dims(cfg: dict, orig_layer: int):
    if cfg["hybrid_layer_pattern"][orig_layer] == 1:
        return dict(swa=True, nh=cfg["swa_num_attention_heads"], nkv=cfg["swa_num_key_value_heads"],
                    hd=cfg["swa_head_dim"], vd=cfg["swa_v_head_dim"])
    return dict(swa=False, nh=cfg["num_attention_heads"], nkv=cfg["num_key_value_heads"],
                hd=cfg["head_dim"], vd=cfg["v_head_dim"])


def fp8_weight_keys(cfg: dict, orig_layer: int, new_layer: int) -> dict:
    """``--fp8``: FP8 weight key -> scale group size (q/k/v, dense MLP, experts)."""
    p = f"model.layers.{new_layer}."
    keys = {p + f"self_attn.{n}_proj.weight": fp8_trn.LINEAR_GROUP for n in "qkv"}
    if cfg["moe_layer_freq"][orig_layer]:
        for e in range(cfg["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys[p + f"mlp.experts.{e}.{proj}.weight"] = fp8_trn.EXPERT_GROUP
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys[p + f"mlp.{proj}.weight"] = fp8_trn.LINEAR_GROUP
    return keys


def expected_layer_shapes(cfg: dict, orig_layer: int, new_layer: int, fp8: bool = False) -> dict:
    H = cfg["hidden_size"]
    d = attn_dims(cfg, orig_layer)
    p = f"model.layers.{new_layer}."
    s = {
        p + "input_layernorm.weight": (H,),
        p + "post_attention_layernorm.weight": (H,),
        p + "self_attn.q_proj.weight": (d["nh"] * d["hd"], H),
        p + "self_attn.k_proj.weight": (d["nkv"] * d["hd"], H),
        p + "self_attn.v_proj.weight": (d["nkv"] * d["vd"], H),
        p + "self_attn.o_proj.weight": (H, d["nh"] * d["vd"]),
    }
    sink = (cfg.get("add_swa_attention_sink_bias") and d["swa"]) or \
           (cfg.get("add_full_attention_sink_bias") and not d["swa"])
    if sink:
        s[p + "self_attn.attention_sink_bias"] = (d["nh"],)
    if cfg["moe_layer_freq"][orig_layer]:
        E, I = cfg["n_routed_experts"], cfg["moe_intermediate_size"]
        s[p + "mlp.gate.weight"] = (E, H)
        s[p + "mlp.gate.e_score_correction_bias"] = (E,)
        for e in range(E):
            s[p + f"mlp.experts.{e}.gate_proj.weight"] = (I, H)
            s[p + f"mlp.experts.{e}.up_proj.weight"] = (I, H)
            s[p + f"mlp.experts.{e}.down_proj.weight"] = (H, I)
    else:
        I = cfg["intermediate_size"]
        s[p + "mlp.gate_proj.weight"] = (I, H)
        s[p + "mlp.up_proj.weight"] = (I, H)
        s[p + "mlp.down_proj.weight"] = (H, I)
    if fp8:
        for k, g in fp8_weight_keys(cfg, orig_layer, new_layer).items():
            r, c = s[k]
            s[k[:-len(".weight")] + SCALE_SUFFIX] = (r, c // g)
    return s


def validate(tensors: dict, expected: dict, fp8_keys: dict | None = None):
    fp8_keys = fp8_keys or {}
    missing = set(expected) - set(tensors)
    extra = set(tensors) - set(expected)
    if missing or extra:
        raise ConvertError(f"key mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    for k, t in tensors.items():
        if tuple(t.shape) != tuple(expected[k]):
            raise ConvertError(f"{k}: shape {tuple(t.shape)} != expected {tuple(expected[k])}")
        if k in fp8_keys:
            want = torch.float8_e4m3fn
        elif k.endswith(BIAS_SUFFIX) or k.endswith(SCALE_SUFFIX):
            want = torch.float32
        else:
            want = torch.bfloat16
        if t.dtype != want:
            raise ConvertError(f"{k}: dtype {t.dtype} != {want}")
        tf = t.float() if t.dtype == torch.float8_e4m3fn else t
        if not torch.isfinite(tf).all():
            raise ConvertError(f"{k}: contains NaN/Inf")
        if k in fp8_keys and tf.abs().max().item() > fp8_trn.TRN_FP8_MAX:
            raise ConvertError(f"{k}: FP8 code above {fp8_trn.TRN_FP8_MAX}")


def fp8_or_plain(src: Src, key: str) -> torch.Tensor:
    w = src.get(key)
    if w.dtype == torch.float8_e4m3fn:
        return dequant_fp8_block(w, src.get(key + "_scale_inv")).to(torch.bfloat16)
    return w.to(torch.bfloat16)


def convert_layer_fp8(src: Src, cfg: dict, orig: int, new: int) -> tuple[dict, dict]:
    """``--fp8`` layer conversion -> (tensors, stats by tensor kind)."""
    sp, dp = f"model.layers.{orig}.", f"model.layers.{new}."
    d = attn_dims(cfg, orig)
    out, stats = {}, {}

    def put(name, codes, scale, st, kind):
        out[dp + name + ".weight"] = codes
        out[dp + name + SCALE_SUFFIX] = scale
        stats[kind] = fp8_trn.add_stats(stats.get(kind, fp8_trn.empty_stats()), st)

    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        out[dp + n] = src.get(sp + n).to(torch.bfloat16)
    qkv_key = sp + "self_attn.qkv_proj.weight"
    w = src.get(qkv_key)
    if w.dtype != torch.float8_e4m3fn:
        raise ConvertError(f"{qkv_key}: expected float8_e4m3fn, got {w.dtype}")
    parts = fp8_trn.split_fused_qkv_fp8(w, src.get(qkv_key + "_scale_inv"), d["nh"], d["nkv"], d["hd"], d["vd"])
    for n, (codes, scale) in zip("qkv", parts):
        put(f"self_attn.{n}_proj", *fp8_trn.rescale_to_trn(codes, scale), f"{n}_proj")
    ow = src.get(sp + "self_attn.o_proj.weight")
    if ow.dtype == torch.float8_e4m3fn:
        raise ConvertError(f"{sp}self_attn.o_proj.weight: FP8 o_proj not expected (ignored_layers)")
    out[dp + "self_attn.o_proj.weight"] = ow.to(torch.bfloat16)
    if src.has(sp + "self_attn.attention_sink_bias"):
        out[dp + "self_attn.attention_sink_bias"] = src.get(sp + "self_attn.attention_sink_bias").to(torch.bfloat16)
    if cfg["moe_layer_freq"][orig]:
        out[dp + "mlp.gate.weight"] = src.get(sp + "mlp.gate.weight").to(torch.bfloat16)
        b = src.get(sp + "mlp.gate.e_score_correction_bias")
        if b.dtype != torch.float32:
            raise ConvertError(f"{sp}mlp.gate.e_score_correction_bias: source dtype {b.dtype}, expected F32")
        out[dp + "mlp.gate.e_score_correction_bias"] = b
        for e in range(cfg["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                kk = f"mlp.experts.{e}.{proj}"
                codes, scale = fp8_trn.mxfp4_to_fp8(src.get(sp + kk + ".weight"), src.get(sp + kk + ".weight_scale"))
                put(kk, codes, scale, {"elements": codes.numel(), "groups": scale.numel()}, "experts")
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            kk = sp + f"mlp.{proj}.weight"
            wd = src.get(kk)
            if wd.dtype != torch.float8_e4m3fn:
                raise ConvertError(f"{kk}: expected float8_e4m3fn, got {wd.dtype}")
            put(f"mlp.{proj}", *fp8_trn.fp8_linear(wd, src.get(kk + "_scale_inv")), f"dense_{proj}")
    return out, stats


def convert_layer(src: Src, cfg: dict, orig: int, new: int) -> dict:
    sp, dp = f"model.layers.{orig}.", f"model.layers.{new}."
    d = attn_dims(cfg, orig)
    out = {}
    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        out[dp + n] = src.get(sp + n).to(torch.bfloat16)
    # attention
    qkv_key = sp + "self_attn.qkv_proj.weight"
    w = src.get(qkv_key)
    if w.dtype != torch.float8_e4m3fn:
        raise ConvertError(f"{qkv_key}: expected float8_e4m3fn, got {w.dtype}")
    q, k, v = split_fused_qkv(w, src.get(qkv_key + "_scale_inv"), d["nh"], d["nkv"], d["hd"], d["vd"])
    out[dp + "self_attn.q_proj.weight"] = q.to(torch.bfloat16)
    out[dp + "self_attn.k_proj.weight"] = k.to(torch.bfloat16)
    out[dp + "self_attn.v_proj.weight"] = v.to(torch.bfloat16)
    out[dp + "self_attn.o_proj.weight"] = fp8_or_plain(src, sp + "self_attn.o_proj.weight")
    if src.has(sp + "self_attn.attention_sink_bias"):
        out[dp + "self_attn.attention_sink_bias"] = src.get(sp + "self_attn.attention_sink_bias").to(torch.bfloat16)
    # mlp
    if cfg["moe_layer_freq"][orig]:
        out[dp + "mlp.gate.weight"] = src.get(sp + "mlp.gate.weight").to(torch.bfloat16)
        b = src.get(sp + "mlp.gate.e_score_correction_bias")
        if b.dtype != torch.float32:
            raise ConvertError(f"{sp}mlp.gate.e_score_correction_bias: source dtype {b.dtype}, expected F32")
        out[dp + "mlp.gate.e_score_correction_bias"] = b
        for e in range(cfg["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                kk = f"mlp.experts.{e}.{proj}.weight"
                out[dp + kk] = dequant_mxfp4(src.get(sp + kk), src.get(sp + kk + "_scale"))
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[dp + f"mlp.{proj}.weight"] = fp8_or_plain(src, sp + f"mlp.{proj}.weight")
    return out


def convert_embed(src: Src, cfg: dict) -> tuple[dict, dict]:
    out = {k: src.get(k).to(torch.bfloat16)
           for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    return out, {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,), "lm_head.weight": (V, H)}


# --------------------------------------------------------------------------- worker
_G = {}


def _init_worker(src_dir, cfg, dst, threads, fp8=False):
    torch.set_num_threads(threads)
    _G.update(src=Src(src_dir), cfg=cfg, dst=dst, fp8=fp8)


def _write(tensors: dict, path: str):
    tmp = path + ".tmp"
    try:
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _job(job):
    kind, orig, new = job
    src, cfg, dst = _G["src"], _G["cfg"], _G["dst"]
    fn = "model-embed.safetensors" if kind == "embed" else f"model-layer-{new:02d}.safetensors"
    path = os.path.join(dst, fn)
    t0 = time.time()
    try:
        if os.path.exists(path):
            meta = {}
            with safe_open(path, "pt") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    n = 1
                    for x in sl.get_shape():
                        n *= x
                    meta[k] = (sl.get_dtype(), n * DTYPE_BYTES[sl.get_dtype()])
            return ("skip", fn, meta, time.time() - t0, None)
        fp8_keys, stats = None, None
        if kind == "embed":
            tensors, expected = convert_embed(src, cfg)
        elif _G.get("fp8"):
            tensors, stats = convert_layer_fp8(src, cfg, orig, new)
            expected = expected_layer_shapes(cfg, orig, new, fp8=True)
            fp8_keys = fp8_weight_keys(cfg, orig, new)
        else:
            tensors = convert_layer(src, cfg, orig, new)
            expected = expected_layer_shapes(cfg, orig, new)
        validate(tensors, expected, fp8_keys)
        meta = {k: (str(t.dtype), t.numel() * t.element_size()) for k, t in tensors.items()}
        if stats is not None:
            os.makedirs(os.path.join(dst, "fp8_stats"), exist_ok=True)
            with open(os.path.join(dst, "fp8_stats", fn + ".json"), "w") as f:
                json.dump({"orig_layer": orig, "new_layer": new, "stats": stats}, f, indent=1)
        _write(tensors, path)
        return ("ok", fn, meta, time.time() - t0, None)
    except Exception as e:  # reported by the parent, which exits 1
        return ("error", fn, None, time.time() - t0, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- main
def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", default=None, help="comma separated original layer ids (0-47)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--modeling-file", default=None, help="override modeling_mimo_v2.py copied into dst")
    ap.add_argument("--configuration-file", default=None, help="override configuration_mimo_v2.py copied into dst")
    ap.add_argument("--fp8", action="store_true",
                    help="keep FP8 weights for trn2 (tools/fp8_trn.py) instead of dequantizing to BF16")
    a = ap.parse_args(argv)
    if a.workers < 1 or a.workers > 128:
        ap.error("--workers must be within 1..128")
    if not os.path.isfile(os.path.join(a.src, "model.safetensors.index.json")):
        ap.error(f"--src {a.src} has no model.safetensors.index.json")
    if os.path.realpath(a.src) == os.path.realpath(a.dst):
        ap.error("--dst must differ from --src")
    if a.modeling_file and not os.path.isfile(a.modeling_file):
        ap.error(f"--modeling-file {a.modeling_file} not found")
    if a.configuration_file and not os.path.isfile(a.configuration_file):
        ap.error(f"--configuration-file {a.configuration_file} not found")
    with open(os.path.join(a.src, "config.json")) as f:
        n = json.load(f)["num_hidden_layers"]
    if a.layers is not None:
        try:
            ids = sorted({int(x) for x in a.layers.split(",") if x.strip() != ""})
        except ValueError:
            ap.error(f"--layers must be comma separated integers, got {a.layers!r}")
        if not ids or ids[0] < 0 or ids[-1] >= n:
            ap.error(f"--layers values must be within 0..{n - 1}")
        a.layer_ids = ids
    else:
        a.layer_ids = list(range(n))
    return a


def new_config(cfg: dict, ids: list, fp8: bool = False) -> dict:
    c = json.loads(json.dumps(cfg))
    q = c.pop("quantization_config", None)
    if fp8:
        c["neuron_fp8_config"] = {
            "format": "neuron_fp8", "fmt": "e4m3", "fp8_max": fp8_trn.TRN_FP8_MAX,
            "weight_scale": "per-row group: dequant[r, c] = code[r, c] * weight_scale[r, c // G]",
            "linear_group_size": fp8_trn.LINEAR_GROUP, "expert_group_size": fp8_trn.EXPERT_GROUP,
            "fp8_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "mlp.{gate,up,down}_proj (dense)", "mlp.experts.*.{gate,up,down}_proj"],
            "activation_scheme": (q or {}).get("activation_scheme"),
            "source_quantization_config": {k: v for k, v in (q or {}).items() if k != "ignored_layers"},
        }
    c["attention_projection_layout"] = "split"
    c["vision_config"] = {}
    c["audio_config"] = {}
    c["num_hidden_layers"] = len(ids)
    c["hybrid_layer_pattern"] = [cfg["hybrid_layer_pattern"][i] for i in ids]
    c["moe_layer_freq"] = [cfg["moe_layer_freq"][i] for i in ids]
    return c


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args(argv)
    os.makedirs(a.dst, exist_ok=True)
    idx_path = os.path.join(a.dst, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            shards = set(json.load(f)["weight_map"].values())
        if all(os.path.exists(os.path.join(a.dst, s)) for s in shards):
            log.info("already converted: %s (%d shards)", a.dst, len(shards))
            return 0
        os.remove(idx_path)
    with open(os.path.join(a.src, "config.json")) as f:
        cfg = json.load(f)
    jobs = [("embed", -1, -1)] + [("layer", o, n) for n, o in enumerate(a.layer_ids)]
    log.info("converting %s -> %s, layers %s, %d jobs, %d workers, fp8=%s", a.src, a.dst, a.layer_ids, len(jobs),
             a.workers, a.fp8)
    threads = max(1, (os.cpu_count() or 8) // a.workers)
    weight_map, total, t0, errors = {}, 0, time.time(), []
    ctx = mp.get_context("fork")
    with ctx.Pool(min(a.workers, len(jobs)), initializer=_init_worker,
                  initargs=(a.src, cfg, a.dst, threads, a.fp8), maxtasksperchild=1) as pool:
        for status, fn, meta, dt, err in pool.imap_unordered(_job, jobs):
            if status == "error":
                log.error("%s FAILED after %.1fs: %s", fn, dt, err)
                errors.append((fn, err))
                continue
            for k, (_, nbytes) in meta.items():
                weight_map[k] = fn
                total += nbytes
            log.info("%s %s in %.1fs (%d tensors) [%d/%d done, %.0fs elapsed]", fn,
                     "written" if status == "ok" else "exists, skipped", dt, len(meta),
                     len(set(weight_map.values())), len(jobs), time.time() - t0)
    if errors:
        log.error("%d shard(s) failed; index not written", len(errors))
        return 1
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(new_config(cfg, a.layer_ids, a.fp8), f, indent=2)
    for fn in COPY_FILES:
        s = os.path.join(a.src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.dst, fn))
    if a.modeling_file:
        shutil.copy2(a.modeling_file, os.path.join(a.dst, "modeling_mimo_v2.py"))
    if a.configuration_file:
        shutil.copy2(a.configuration_file, os.path.join(a.dst, "configuration_mimo_v2.py"))
    tmp = idx_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=1)
    os.replace(tmp, idx_path)
    log.info("done: %d tensors, total_size=%d bytes (%.1f GB), %.0fs", len(weight_map), total, total / 1e9,
             time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
