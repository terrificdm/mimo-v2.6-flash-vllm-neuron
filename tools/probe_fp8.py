#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FP8 / MXFP4 probe on one trn2 NeuronCore (see docs/gpu-alignment.md section 2).

  start_bg.sh p2_probe_fp8 env NEURON_RT_VISIBLE_CORES=0 VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT \
      bash scripts/venv_exec.sh python tools/probe_fp8.py --out $MIMO_WORK/results/fp8/probe.json

Compile path = the one vllm-neuron uses (torch.compile, backend "neuron_libtorch", compiler_args copied from
vllm_neuron/vllm/worker/neuron_model_runner.py:1398-1425). Compile variants for item 2:
  (a) default: --internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10. libtorch_neuronx_lite's
      backend._apply_platform_compiler_args always appends --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 on trn2;
      for (a) and (c) that function is replaced by the identity inside this process only (no installed file is
      changed), so (a) really compiles without the flag. The command actually run is read back from the cache
      directory's command.txt for every variant.
  (b) the runner's fp8 setting: ... --modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
  (c) like (a), plus NEURON_CC_FLAGS=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 in the environment.
Items: 1 E4M3FN host->device->host round trip (bit patterns); 2 on-device BF16->FP8->BF16 cast per variant;
3 FP8 dequant matmul (x_fp8.to(bf16) * scale) @ w vs CPU; 4 MXFP4 / MX support in NKI 0.6.0 (types, ISA
functions, documented hardware requirement, a tiny quantize_mx kernel attempt); 5 conclusions.
Every item records {"ok": false, "error": ...} on failure and the rest still runs. Exit code 0 when the JSON is
written. A failed neuronx-cc compile terminates the whole process (libtorch_neuronx_lite cache.py "terminating
process for cleanup"), so every item (and each compile of item 2) runs in its own child process (--child); the parent
keeps the error lines of a child that died.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback

import torch

VALUES = [448.0, -448.0, 300.0, -300.0, 256.0, 241.0, 240.0, -240.0, 239.0, 224.0, 1.0, -1.5,
          2.0 ** -6, 2.0 ** -9, 0.0]   # 2^-6 smallest normal, 2^-9 smallest subnormal of E4M3FN
HLO_DEFAULT = "--modular-flow-mac-threshold=10"
UNSAFE = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"


def compiler_args(hlo_opts: str):
    return ["--auto-cast=none", "--verbose=35", "-O1", f"--internal-hlo2tensorizer-options={hlo_opts}",
            "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]


VARIANTS = {
    "a_default": {"hlo": HLO_DEFAULT, "inject": False, "env": None},
    "b_runner_fp8": {"hlo": f"{HLO_DEFAULT} {UNSAFE}", "inject": True, "env": None},
    "c_neuron_cc_flags": {"hlo": HLO_DEFAULT, "inject": False, "env": UNSAFE},
}


def err(e):
    return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:3000]}", "traceback": traceback.format_exc()[-3000:]}


def classify(v: float) -> str:
    if v != v:
        return "nan"
    if math.isinf(v):
        return "inf"
    if abs(abs(v) - 448.0) < 1e-3:
        return "keep_448"
    if abs(abs(v) - 240.0) < 1e-3:
        return "saturate_240"
    return f"other_{v}"


class Compiler:
    """torch.compile with explicit options; records the cache hash and command.txt of each compile."""

    def __init__(self):
        import vllm_neuron  # noqa: F401  (registers the Neuron device and backends)
        import libtorch_neuronx_lite.compile.backend as B
        import libtorch_neuronx_lite.compile.cache as CA
        from libtorch_neuronx_lite import envs
        self.B, self.CA, self.envs = B, CA, envs
        self.orig_apply = B._apply_platform_compiler_args
        self.orig_hash = CA.create_cache_hash
        self.hashes = []

        def rec_hash(*a, **k):
            h = self.orig_hash(*a, **k)
            self.hashes.append(h)
            return h
        CA.create_cache_hash = rec_hash

    def run(self, fn, inputs, variant):
        v = VARIANTS[variant]
        self.B._apply_platform_compiler_args = self.orig_apply if v["inject"] else (lambda options: options)
        old = os.environ.get("NEURON_CC_FLAGS")
        if v["env"]:
            os.environ["NEURON_CC_FLAGS"] = v["env"]
        else:
            os.environ.pop("NEURON_CC_FLAGS", None)
        torch._dynamo.reset()
        n0 = len(self.hashes)
        t0 = time.time()
        try:
            cfn = torch.compile(fn, backend="neuron_libtorch", fullgraph=True,
                                options={"alias_meta_to_neuron": True, "compiler_args": compiler_args(v["hlo"])})
            out = cfn(*inputs)
            out = [o.cpu() for o in out] if isinstance(out, (tuple, list)) else out.cpu()
        finally:
            self.B._apply_platform_compiler_args = self.orig_apply
            if old is None:
                os.environ.pop("NEURON_CC_FLAGS", None)
            else:
                os.environ["NEURON_CC_FLAGS"] = old
        info = {"compile_and_run_s": round(time.time() - t0, 2), "hashes": self.hashes[n0:], "commands": []}
        root = self.envs.get_neuron_compile_cache_dir()
        for h in info["hashes"]:
            p = os.path.join(root, h, "command.txt")
            if os.path.exists(p):
                info["commands"].append(open(p).read().strip()[:2000])
        return out, info


def run_child(spec: str, timeout: int = 1200):
    """Run one compile in a child process; return its JSON, or the error lines when it died."""
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="probe_fp8_child_")
    os.close(fd)
    os.unlink(tmp)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", spec, "--out", tmp],
                           capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
        rc, text = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", str(e)
    if os.path.exists(tmp):
        with open(tmp) as f:
            r = json.load(f)
        os.unlink(tmp)
        return r
    lines = [ln for ln in text.splitlines() if any(k in ln for k in ("[ERROR]", "FATAL", "Error", "error:"))]
    return {"ok": False, "child_rc": rc, "child_s": round(time.time() - t0, 1),
            "error": "\n".join(lines)[-3000:] or text[-3000:]}


def item1_transfer(dev):
    x = torch.tensor(VALUES, dtype=torch.float32).to(torch.float8_e4m3fn)
    bits_cpu = x.view(torch.uint8).tolist()
    y = x.to(dev).cpu()
    bits_back = y.view(torch.uint8).tolist()
    return {"ok": True, "values": VALUES, "cpu_e4m3fn_value": x.float().tolist(), "bits_cpu": bits_cpu,
            "bits_back": bits_back, "identical": bits_cpu == bits_back}


def _cast_rt(x):
    return x.to(torch.float8_e4m3fn).to(torch.bfloat16)


def _cast_fp8(x):
    return x.to(torch.float8_e4m3fn)


def _dequant(xq):
    return xq.to(torch.bfloat16)


def item2_part(comp, dev, variant, part):
    xb = torch.tensor(VALUES, dtype=torch.bfloat16)
    ref = xb.to(torch.float8_e4m3fn).to(torch.bfloat16).float().tolist()
    if part == "rt":
        out, info = comp.run(_cast_rt, (xb.to(dev),), variant)
        got = out.float().tolist()
        return dict(ok=True, roundtrip_bf16=got, value_448=classify(got[0]), value_300=classify(got[2]),
                    value_241=classify(got[5]), matches_cpu=[g == c or (g != g and c != c) for g, c in zip(got, ref)],
                    compile=info)
    if part == "dq":   # checkpoint case: E4M3FN bits made on the host, converted to BF16 on the device
        xq = torch.tensor(VALUES, dtype=torch.float32).to(torch.float8_e4m3fn)
        out, info = comp.run(_dequant, (xq.to(dev),), variant)
        got = out.float().tolist()
        return dict(ok=True, host_bits=xq.view(torch.uint8).tolist(), host_value=xq.float().tolist(),
                    device_bf16=got, matches_cpu=[g == c or (g != g and c != c) for g, c in zip(got, xq.float().tolist())],
                    compile=info)
    out, info = comp.run(_cast_fp8, (xb.to(dev),), variant)
    return dict(ok=True, fp8_bits=out.view(torch.uint8).tolist(), fp8_bits_as_e4m3fn=out.float().tolist(), compile=info)


def item2_cast():
    xb = torch.tensor(VALUES, dtype=torch.bfloat16)
    ref = xb.to(torch.float8_e4m3fn).to(torch.bfloat16).float().tolist()
    res = {"ok": True, "input_bf16": xb.float().tolist(), "cpu_reference": ref, "variants": {}}
    for name in VARIANTS:
        r = run_child(f"item2:{name}:rt")
        f = run_child(f"item2:{name}:fp8")
        r["host_fp8_to_bf16"] = run_child(f"item2:{name}:dq")
        if f.get("ok"):
            r["fp8_bits"], r["fp8_bits_as_e4m3fn"], r["fp8_compile"] = f["fp8_bits"], f["fp8_bits_as_e4m3fn"], f["compile"]
        else:
            r["fp8_output"] = f
        res["variants"][name] = r
        print(time.strftime("%H:%M:%S"), "item2", name, "rt", r.get("ok"), "fp8", f.get("ok"), flush=True)
    return res


def _deq_mm(xq, scale, w):
    return (xq.to(torch.bfloat16) * scale) @ w


def item3_dequant_matmul(comp, dev):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(128, 256, generator=g) * 3
    scale = (x.abs().amax() / 200.0).reshape(1, 1)          # keep |x/scale| <= 200 (< 240, no saturation)
    xq = (x / scale).to(torch.float8_e4m3fn)
    w = (torch.randn(256, 128, generator=g) * 0.05).to(torch.bfloat16)
    sb = scale.to(torch.bfloat16)
    ref = (xq.float() * sb.float()) @ w.float()
    out, info = comp.run(_deq_mm, (xq.to(dev), sb.to(dev), w.to(dev)), "b_runner_fp8")
    rel = float((out.float() - ref).norm() / ref.norm())
    cpu_bf16 = _deq_mm(xq, sb, w).float()
    return {"ok": True, "rel_l2_vs_cpu_fp32": rel, "rel_l2_cpu_bf16_vs_cpu_fp32": float((cpu_bf16 - ref).norm() / ref.norm()),
            "max_abs_err": float((out.float() - ref).abs().max()), "variant": "b_runner_fp8", "compile": info}


def item4_mx():
    import nki
    import nki.isa as ni
    import nki.language as nl
    res = {"ok": True, "nki_version": getattr(nki, "__version__", None)}
    keys = ("fp4", "mx", "float4", "e2m1", "fp8", "float8", "e4m3", "e5m2", "e8m0")
    res["nl_types"] = [a for a in dir(nl) if any(k in a.lower() for k in keys)]
    res["isa_functions"] = [a for a in dir(ni) if any(k in a.lower() for k in ("mx", "quant"))]
    docs = {}
    for f in ("nc_matmul_mx", "quantize_mx"):
        d = (getattr(ni, f, None).__doc__ or "") if hasattr(ni, f) else ""
        docs[f] = {"exists": hasattr(ni, f), "first_lines": " ".join(d.split())[:400],
                   "requires_neuroncore_v4": "NeuronCore-v4" in d}
    res["isa_docs"] = docs
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target
        res["platform_target"] = get_platform_target()
    except Exception as e:  # noqa: BLE001
        res["platform_target"] = f"error: {e!r}"
    res["neuroncore_version_note"] = "trn2 = NeuronCore-v3; trn3 = NeuronCore-v4 (Neuron architecture docs)"
    # tiny kernel attempt: quantize a [128, 512] bf16 tile to MXFP8 (the simplest MX instruction)
    try:
        @nki.jit
        def mx_quant(x):
            out = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.shared_hbm)
            x_sb = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            ni.dma_copy(dst=x_sb, src=x)
            q_sb = nl.ndarray((128, 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            s_sb = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.sbuf)
            ni.quantize_mx(dst=q_sb, src=x_sb, dst_scale=s_sb)
            ni.dma_copy(dst=out, src=s_sb)
            return out
        x = torch.randn(128, 512, dtype=torch.bfloat16).to(torch.device("neuron:0"))
        y = mx_quant(x)
        res["kernel_attempt"] = {"ok": True, "scales_sample": y.cpu()[0, :8].tolist()}
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:1500]}"
        hw = any(k in msg for k in ("NeuronCore-v4", "trn3", "not supported", "unsupported", "Unsupported"))
        res["kernel_attempt"] = {"ok": False, "error": msg, "looks_like_hardware_limit": hw,
                                 "note": "an error that does not name the hardware may come from the kernel code"}
    return res


def conclusions(r):
    c = {}
    i2 = r.get("item2_cast", {})
    b = (i2.get("variants") or {}).get("b_runner_fp8", {})
    if b.get("ok"):
        c["e4m3fn_448_on_trn2_runner_flags"] = b.get("value_448")
    a = (i2.get("variants") or {}).get("a_default", {})
    if a.get("ok"):
        c["e4m3fn_448_without_flag"] = a.get("value_448")
    elif a:
        c["e4m3fn_448_without_flag"] = "compile rejected: " + (a.get("error") or "")[:400]
    # bf16->fp8->bf16 in one graph can be folded by the compiler, so judge by the stored FP8 bits and by the
    # host-E4M3FN -> device-BF16 conversion (the checkpoint case)
    bits = b.get("fp8_bits") or []
    if bits:
        c["device_bf16_to_fp8_bits_for_448_300_241"] = [bits[0], bits[2], bits[5]]
        c["device_fp8_encoding"] = ("E4M3 (IEEE-like, max 240): values >240 become 0x78 (E4M3 +Inf; would read as 256 in "
                                    "E4M3FN)" if bits[0] == 0x78 else
                                    "E4M3FN (0x7E for 448)" if bits[0] == 0x7E else f"0x{bits[0]:02x} for 448")
    dq = b.get("host_fp8_to_bf16") or {}
    if dq.get("ok"):
        got = dq["device_bf16"]
        c["host_e4m3fn_448_to_device_bf16"] = got[0]
        c["host_e4m3fn_dequant_all_match_cpu"] = all(dq["matches_cpu"])
        ok_small = all(m for m, v in zip(dq["matches_cpu"], dq["host_value"]) if abs(v) <= 240)
        c["host_e4m3fn_dequant_le240_match_cpu"] = ok_small
        c["checkpoint_fp8_direct_use"] = (
            "yes: E4M3FN bits up to 448 convert correctly on the device" if all(dq["matches_cpu"]) else
            "no: E4M3FN values above 240 are misread on trn2 (exponent-15 codes are Inf/NaN in E4M3); weights need "
            "rescaling into +-240 (e.g. halve the FP8 value, double the block scale) or BF16 dequantization")
    else:
        c["checkpoint_fp8_direct_use"] = "unknown: host_fp8_to_bf16 failed (see item2_cast)"
    c["fp8_kv_clamp"] = "clamp to +-240 (vllm_neuron.utils.dtype_utils FP8_CLAMP_MAX on trn2)"
    i4 = r.get("item4_mx", {})
    docs = i4.get("isa_docs") or {}
    c["mxfp4_matmul_on_trn2"] = ("not available: nc_matmul_mx / quantize_mx are documented NeuronCore-v4+ only "
                                 "(trn3); trn2 is NeuronCore-v3" if (docs.get("nc_matmul_mx") or {}).get(
                                     "requires_neuroncore_v4") else "see item4_mx")
    i3 = r.get("item3_dequant_matmul", {})
    if i3.get("ok"):
        c["dequant_matmul_rel_l2"] = i3.get("rel_l2_vs_cpu_fp32")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="1,2,3,4")
    ap.add_argument("--child", default=None, help="internal: item1, item2:<variant>:<rt|fp8|dq>, item3, item4")
    a = ap.parse_args(argv)
    if a.child:
        try:
            comp, dev = Compiler(), torch.device("neuron:0")
            if a.child == "item1":
                r = item1_transfer(dev)
            elif a.child == "item3":
                r = item3_dequant_matmul(comp, dev)
            elif a.child == "item4":
                r = item4_mx()
            else:
                _, variant, part = a.child.split(":")
                r = item2_part(comp, dev, variant, part)
        except Exception as e:  # noqa: BLE001
            r = err(e)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1, default=str)
        sys.stdout.flush()
        os._exit(0)
    items = set(a.items.split(","))
    res = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "env": {k: os.environ.get(k) for k in ("NEURON_RT_VISIBLE_CORES", "VLLM_CACHE_ROOT", "NEURON_CC_FLAGS")}}

    def write():
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(a.out + ".tmp", a.out)

    # the parent never opens the NeuronCore: one core has one owner, and each child needs it
    steps = [("1", "item1_transfer", lambda: run_child("item1")),
             ("2", "item2_cast", item2_cast),
             ("3", "item3_dequant_matmul", lambda: run_child("item3")),
             ("4", "item4_mx", lambda: run_child("item4"))]
    for key, name, f in steps:
        if key not in items:
            continue
        print(time.strftime("%H:%M:%S"), "running", name, flush=True)
        try:
            res[name] = f()
        except Exception as e:  # noqa: BLE001
            res[name] = err(e)
        print(time.strftime("%H:%M:%S"), name, "ok" if res[name].get("ok") else "FAILED", flush=True)
        write()
    res["item5_conclusions"] = conclusions(res)
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write()
    print(json.dumps(res["item5_conclusions"], indent=1), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
