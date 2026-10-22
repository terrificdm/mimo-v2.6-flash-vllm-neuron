#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send N long needle requests at the same time and record what the server does (queue, preempt or crash;
docs/design-notes.md#concurrency).

Builds the prompts with needle.py (different depths so the requests share no prefix), starts all of them in
parallel threads, and records per request: TTFT, end time, correctness, HTTP status/error. Before and after it
reads /metrics (vllm:num_preemptions_total, vllm:num_requests_running/waiting peak while running) and /health.

  python scripts/api_tests/concurrent_long.py --ctx 1048576 --length 600000 --n 2 --out $R/concurrency/t49_2x600k.json
Exit codes: 0 all requests correct and server healthy afterwards, 1 otherwise, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import needle as N  # noqa: E402


def metrics():
    try:
        with urllib.request.urlopen(C.base_url() + "/metrics", timeout=10) as r:
            text = r.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    out = {}
    for name in ("vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
                 "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        vals = [float(m.group(1)) for m in re.finditer(re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text)]
        if vals:
            out[name] = sum(vals)
    return out


def healthy():
    try:
        with urllib.request.urlopen(C.base_url() + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--length", type=int, required=True, help="content tokens per request")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tokenizer", default=N.DEFAULT_TOKENIZER)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.n < 1:
        ap.error("--n must be >= 1")
    depths = [round((i + 1) / (a.n + 1), 4) for i in range(a.n)]
    counter = N.make_counter(a.tokenizer)
    prompts = N.build_prompts(counter, a.ctx, [a.length], depths, a.seed)
    print(f"concurrent_long: {a.n} prompts of ~{prompts[0]['chat_prompt_tokens']} tokens", flush=True)
    m0, recs, peak = metrics(), [None] * a.n, {"running": 0.0, "waiting": 0.0}
    t0 = time.time()

    def worker(i):
        r = N.chat_stream(prompts[i]["messages"], a.timeout)
        r["finished_at_s"] = round(time.time() - t0, 3)
        recs[i] = r

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.n)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        m = metrics()
        peak["running"] = max(peak["running"], m.get("vllm:num_requests_running", 0.0))
        peak["waiting"] = max(peak["waiting"], m.get("vllm:num_requests_waiting", 0.0))
        time.sleep(5)
    for t in threads:
        t.join()
    m1, ok_health = metrics(), healthy()
    cases = []
    for p, r in zip(prompts, recs):
        correct = r["http_status"] == 200 and r["error"] is None and N.is_correct(r["content"], p["answer"])
        cases.append({"id": p["id"], "prompt_tokens": p["chat_prompt_tokens"], "correct": correct,
                      "ttft_s": r["ttft_s"], "finished_at_s": r["finished_at_s"], "http_status": r["http_status"],
                      "content": r["content"][:200], "error": (r["error"] or "")[:500] or None})
    pre = None
    if "vllm:num_preemptions_total" in m0 and "vllm:num_preemptions_total" in m1:
        pre = m1["vllm:num_preemptions_total"] - m0["vllm:num_preemptions_total"]
    doc = {"utc": C.now_utc(), "ctx": a.ctx, "length": a.length, "n": a.n, "wall_s": round(time.time() - t0, 1),
           "preemptions": pre, "peak_running": peak["running"], "peak_waiting": peak["waiting"],
           "healthy_after": ok_health, "metrics_before": m0, "metrics_after": m1, "cases": cases}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    for c in cases:
        print(f"  {c['id']}: correct={c['correct']} ttft={c['ttft_s']} done={c['finished_at_s']} "
              f"status={c['http_status']} err={c['error']}", flush=True)
    print(f"concurrent_long: preemptions={pre} peak_running={peak['running']} healthy_after={ok_health}", flush=True)
    return 0 if ok_health and all(c["correct"] for c in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
