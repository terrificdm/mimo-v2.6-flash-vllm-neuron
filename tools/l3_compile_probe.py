#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L3 compile probe for one context tier (see docs/design-notes.md#segmented-prefill).

Builds an offline vLLM ``LLM`` on the L3 model (same graph structure as the full model) with the tier's
max_model_len, segment size and KV block count, runs one 64-token prompt for 4 tokens, and records the
wall-clock times (engine start incl. compile, first request), the compile window (for
tools/neuroncc_hbm_summary.py --since/--until), the "Neuron HBM ... used" figures vLLM logs (read back from this
process's log file when --log is given) and one neuron-monitor memory snapshot taken while the engine is alive.

  start_bg.sh p2_l3_compile_32k $DEV_ENV bash scripts/venv_exec.sh python tools/l3_compile_probe.py \
      --max-model-len 32768 --kv-segment-size 4096 --num-gpu-blocks-override 8192 --out $R2/longctx/l3_probe_32k.json
Key blocks: --prefill-key-block / --decode-key-block set MIMO_ATTN_PREFILL_KEY_BLOCK / MIMO_ATTN_DECODE_KEY_BLOCK
before vLLM starts (worker processes inherit them). Exit codes: 0 ok, 3 config error, 4 exception (incl. compile
failure; the JSON is still written with ok=false and the error).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "accuracy"))
import lv_common as LV  # noqa: E402

PROMPT_LEN = 64
GEN = 4
_HBM = re.compile(r"Neuron HBM[^\n]*")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--kv-segment-size", type=int, default=0, choices=LV.SEGMENT_SIZES)
    ap.add_argument("--num-gpu-blocks-override", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=4, help="decode batch bucket (default 4, the serve value)")
    ap.add_argument("--prefill-key-block", type=int, default=None)
    ap.add_argument("--decode-key-block", type=int, default=None)
    ap.add_argument("--decode-moe-impl", default="tkg", choices=["tkg", "cte", "torch"])
    ap.add_argument("--model-dir", default=LV.DEFAULT_L3)
    ap.add_argument("--tp", type=int, default=64)
    ap.add_argument("--log", default=None, help="this job's log file (to read back vLLM's 'Neuron HBM' lines)")
    ap.add_argument("--no-monitor", action="store_true", help="skip the neuron-monitor snapshot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.kv_segment_size and a.kv_segment_size >= a.max_model_len:
        ap.error("--kv-segment-size must be < --max-model-len")
    for k in ("prefill_key_block", "decode_key_block"):
        v = getattr(a, k)
        if v is not None and (v <= 0 or v % 32):
            ap.error(f"--{k.replace('_', '-')} must be a positive multiple of the block size 32")
    return a


def apply_env(a) -> dict:
    LV.apply_decode_moe_impl(a.decode_moe_impl)
    if a.prefill_key_block:
        os.environ["MIMO_ATTN_PREFILL_KEY_BLOCK"] = str(a.prefill_key_block)
    if a.decode_key_block:
        os.environ["MIMO_ATTN_DECODE_KEY_BLOCK"] = str(a.decode_key_block)
    return {k: os.environ.get(k) for k in LV.ENV_KEYS + ("MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK")}


def monitor_snapshot(timeout: float = 20.0) -> dict:
    """One neuron-monitor report: per runtime, per NeuronCore memory usage breakdown (bytes and GiB)."""
    try:
        p = subprocess.Popen(["neuron-monitor"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError as e:
        return {"ok": False, "error": repr(e)}
    t0, line = time.time(), ""
    try:
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if line.strip().startswith("{"):
                break
    finally:
        p.kill()
    try:
        rep = json.loads(line)
    except ValueError as e:
        return {"ok": False, "error": f"no JSON from neuron-monitor: {e!r}"}
    cores = {}
    for rt in rep.get("neuron_runtime_data") or []:
        mem = (rt.get("report") or {}).get("memory_used") or {}
        bd = ((mem.get("neuron_runtime_used_bytes") or {}).get("usage_breakdown") or {}).get("neuroncore_memory_usage") or {}
        for core, parts in bd.items():
            tot = sum(v for v in parts.values() if isinstance(v, (int, float)))
            cores[f"{rt.get('pid')}:{core}"] = {"bytes": parts, "total_gib": round(tot / 2 ** 30, 4)}
    vals = [c["total_gib"] for c in cores.values()]
    return {"ok": True, "n_cores": len(cores), "max_core_gib": max(vals, default=None),
            "min_core_gib": min(vals, default=None), "cores": cores}


def hbm_lines(log_path):
    if not log_path or not os.path.exists(log_path):
        return []
    with open(log_path, errors="replace") as f:
        return _HBM.findall(f.read())[-8:]


def main(argv=None):
    a = parse_args(argv)
    env = apply_env(a)
    res = {"args": vars(a), "env": env, "ok": False, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    try:
        llm = LV.build_llm(a, max_num_seqs=a.max_num_seqs)
        res["engine_ready_s"] = round(time.time() - t0, 1)
        res["engine_ready_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        from vllm import SamplingParams
        ids = [(1000 + 37 * i) % 50000 + 100 for i in range(PROMPT_LEN)]
        t1 = time.time()
        out = llm.generate([{"prompt_token_ids": ids}],
                           SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True), use_tqdm=False)
        res["first_request_s"] = round(time.time() - t1, 2)
        res["generated"] = list(out[0].outputs[0].token_ids)
        res["monitor"] = {"ok": False, "error": "skipped"} if a.no_monitor else monitor_snapshot()
        res["ok"] = len(res["generated"]) == GEN
    except LV.ConfigError:
        raise
    except BaseException as e:  # noqa: BLE001
        import traceback
        res["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        res["traceback"] = traceback.format_exc()[-4000:]
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["total_s"] = round(time.time() - t0, 1)
    res["hbm_log_lines"] = hbm_lines(a.log)
    LV.write_json(a.out, res)
    LV.log(f"ok={res['ok']} engine_ready_s={res.get('engine_ready_s')} first_request_s={res.get('first_request_s')} "
           f"window {res['start_utc']} .. {res['end_utc']}")
    return 0 if res["ok"] else 4


if __name__ == "__main__":
    LV.run_main(main)
