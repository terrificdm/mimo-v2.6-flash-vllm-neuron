#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarise neuronx-cc HBM reports: one record per NEFF, sizes in GiB.

Reads ``<cache>/<hash>/log-neuron-cc.txt`` (any depth below --cache) whose mtime falls in [--since, --until]
(and/or the directories named by --hash), and parses:
  * the per physical core "DRAM Memory Usage" tables printed by ``[HBMUsage]`` (nc00 / nc01 of a logical core):
    Total, Model Constants, Unallocated Tensors (= I/O: weights, KV cache, inputs/outputs), Allocated Tensors
    (= internal scratchpad of that physical core), DMA ring sizes;
  * "Total estimated HBM usage is: X" (successful compile) or the NCC_EOOM002 line
    "Maximum peak HBM usage of X ... Y I/O tensors, Z internal (scratchpad) tensors" (failed compile).
Derived per NEFF:
  peak_gib    = EOOM "maximum peak" if present, else "Total estimated HBM usage"
  io_gib      = EOOM "I/O tensors" if present, else the sum of Unallocated over the physical cores
                (the compiler's EOOM figure is that sum: 14.12 = 2 x 7.06 in the BF16 4096 decode sample)
  scratch_gib = max(sum of Allocated over physical cores, peak_gib - io_gib)
The compiler prints sizes as B/KB/MB/GB with 1024 steps (its HBM limit "24.00GB" is the 24 GiB of one trn2
logical core), so its GB values are reported unchanged as GiB.
A trimmed copy of each log (command line, HBMUsage tables, un-allocated tensor tables, "Total estimated", error
lines) is written to ``<log-dir>/<task>_neuroncc_<hash>.log``.

  python tools/neuroncc_hbm_summary.py --cache $VLLM_CACHE_ROOT/neuron/compile_cache \
      --since 2026-10-19T11:08:00Z --until 2026-10-19T11:18:30Z --task bf16_4096 --out $MIMO_WORK/results/longctx/bf16_4096_hbm.json
Exit codes: 0 ok (at least one log parsed), 1 no matching log, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}
GIB = 1024 ** 3
_SIZE = r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB)"
_HDR = re.compile(r"\((nc\d+)/sgLnk\) \[HBMUsage\]:")
_ROW = re.compile(r"^│\s*([^│]+?)\s*│\s*" + _SIZE + r"\s*│\s*$")
_TOTAL_EST = re.compile(r"\[HBMUsage\]: Total estimated HBM usage is:\s*" + _SIZE)
_EOOM = re.compile(r"(NCC_E[A-Z]+\d+)\] Maximum peak HBM usage of " + _SIZE + r" exceeds HBM limit of " + _SIZE +
                   r"[^.]*\. This consists of " + _SIZE + r" model constants, " + _SIZE + r" I/O tensors, " + _SIZE +
                   r" internal \(scratchpad\) tensors")
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z ")
ROW_KEYS = {"Total:": "total", "Model Code": "model_code", "Model Constants": "model_constants",
            "Unallocated Tensors": "unallocated", "Allocated Tensors": "allocated", "DMA Ring IO": "dma_ring_io",
            "DMA Ring Spill": "dma_ring_spill"}


def to_gib(value: str, unit: str) -> float:
    return float(value) * UNITS[unit] / GIB


def _r(x):
    return None if x is None else round(x, 4)


def parse_log(lines):
    """Parse the lines of one log-neuron-cc.txt. Returns (record, keep) where keep are the line indices of the
    trimmed copy."""
    cores, keep = {}, set()
    cur, in_table = None, False
    total_est = eoom = None
    first_ts = last_ts = None
    for i, ln in enumerate(lines):
        m = _TS.match(ln)
        if m:
            last_ts = m.group(1)
            first_ts = first_ts or m.group(1)
        if i == 0 or "NCC_E" in ln or " ERROR " in ln[:40] or "Un-allocated" in ln or "un-allocated" in ln:
            keep.add(i)
        h = _HDR.search(ln)
        if h:
            cur, in_table = h.group(1), True
            keep.add(i)
            continue
        if in_table:
            keep.add(i)
            if ln.strip() == "":
                in_table = False
                continue
            r = _ROW.match(ln.strip())
            if r and r.group(1) in ROW_KEYS:
                cores.setdefault(cur, {})[ROW_KEYS[r.group(1)]] = to_gib(r.group(2), r.group(3))
            continue
        t = _TOTAL_EST.search(ln)
        if t:
            total_est = to_gib(t.group(1), t.group(2))
            keep.add(i)
        e = _EOOM.search(ln)
        if e:
            g = e.groups()
            eoom = {"code": g[0], "peak": to_gib(g[1], g[2]), "limit": to_gib(g[3], g[4]),
                    "model_constants": to_gib(g[5], g[6]), "io": to_gib(g[7], g[8]), "internal": to_gib(g[9], g[10])}
    # keep the BIRToNIR "un-allocated DRAM tensors by kind" tables (header line + table until blank line)
    for i, ln in enumerate(lines):
        if "un-allocated DRAM tensors by kind" in ln:
            j = i + 1
            while j < len(lines) and j < i + 14:
                keep.add(j)
                if lines[j].strip() == "" and j > i + 2:
                    break
                j += 1
    alloc = {c: v.get("allocated") for c, v in sorted(cores.items())}
    unalloc = {c: v.get("unallocated") for c, v in sorted(cores.items())}
    sum_alloc = sum(v for v in alloc.values() if v is not None) if cores else None
    sum_unalloc = sum(v for v in unalloc.values() if v is not None) if cores else None
    peak = eoom["peak"] if eoom else total_est
    io = eoom["io"] if eoom else sum_unalloc
    scratch = None
    cands = [x for x in (sum_alloc, (peak - io) if (peak is not None and io is not None) else None) if x is not None]
    if cands:
        scratch = max(cands)
    compile_s = None
    if first_ts and last_ts:
        f = dt.datetime.strptime(first_ts, "%Y-%m-%dT%H:%M:%S")
        g = dt.datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S")
        compile_s = (g - f).total_seconds()
    rec = {
        "eoom": bool(eoom), "eoom_code": eoom["code"] if eoom else None,
        "peak_gib": _r(peak), "io_gib": _r(io), "internal_gib": _r(eoom["internal"]) if eoom else None,
        "scratch_gib": _r(scratch), "sum_allocated_gib": _r(sum_alloc), "sum_unallocated_gib": _r(sum_unalloc),
        "unallocated_gib": _r(max((v for v in unalloc.values() if v is not None), default=None)),
        "allocated_gib_per_core": {c: _r(v) for c, v in alloc.items()},
        "total_gib_per_core": {c: _r(v.get("total")) for c, v in sorted(cores.items())},
        "total_estimated_gib": _r(total_est), "limit_gib": _r(eoom["limit"]) if eoom else None,
        "log_first_ts": first_ts, "log_last_ts": last_ts, "compile_s": compile_s,
    }
    return rec, sorted(keep)


def parse_utc(s: str) -> float:
    s = s.strip().rstrip("Z")
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def find_logs(cache: str, since: float | None, until: float | None, hashes):
    out = []
    for root, _dirs, files in os.walk(cache):
        if "log-neuron-cc.txt" not in files:
            continue
        p = os.path.join(root, "log-neuron-cc.txt")
        rel = os.path.relpath(root, cache)
        h = rel.split(os.sep)[0]
        mt = os.path.getmtime(p)
        if hashes and h in hashes:
            out.append((h, p, mt))
        elif since is not None and until is not None and since <= mt <= until:
            out.append((h, p, mt))
    return sorted(out, key=lambda x: x[2])


def summarise(cache, since, until, task, log_dir, hashes=()):
    recs = []
    for h, p, mt in find_logs(cache, since, until, set(hashes)):
        with open(p, errors="replace") as f:
            lines = f.read().splitlines()
        rec, keep = parse_log(lines)
        d = os.path.dirname(p)
        neff = [x for x in os.listdir(d) if x.endswith(".neff")]
        rec.update(hash=h, log=p, mtime_utc=dt.datetime.fromtimestamp(mt, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   neff_bytes=os.path.getsize(os.path.join(d, neff[0])) if neff else None,
                   compilation_complete=os.path.exists(os.path.join(d, ".compilation_complete")))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            tp = os.path.join(log_dir, f"{task}_neuroncc_{h}.log")
            with open(tp, "w") as f:
                f.write("\n".join(lines[i] for i in keep) + "\n")
            rec["trimmed_log"] = tp
        recs.append(rec)
    vals = lambda k: [r[k] for r in recs if r.get(k) is not None]  # noqa: E731
    return {
        "task": task, "cache": cache, "since": since and dt.datetime.fromtimestamp(since, dt.timezone.utc).isoformat(),
        "until": until and dt.datetime.fromtimestamp(until, dt.timezone.utc).isoformat(),
        "unit": "GiB (compiler GB, 1024-based)", "n_neff": len(recs), "n_eoom": sum(r["eoom"] for r in recs),
        "max_peak_gib": max(vals("peak_gib"), default=None), "max_scratch_gib": max(vals("scratch_gib"), default=None),
        "max_io_gib": max(vals("io_gib"), default=None), "neffs": recs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--since", default=None, help="UTC, e.g. 2026-10-19T11:08:00Z")
    ap.add_argument("--until", default=None)
    ap.add_argument("--hash", action="append", default=[], help="also include this cache directory (repeatable)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "logs"),
                    help="where the trimmed logs go ('' = do not write)")
    a = ap.parse_args(argv)
    if (a.since is None) != (a.until is None) or (a.since is None and not a.hash):
        ap.error("give --since and --until, and/or --hash")
    if not os.path.isdir(a.cache):
        ap.error(f"--cache {a.cache} is not a directory")
    since = parse_utc(a.since) if a.since else None
    until = parse_utc(a.until) if a.until else None
    doc = summarise(a.cache, since, until, a.task, a.log_dir, a.hash)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    for r in doc["neffs"]:
        print(f"{r['hash']} eoom={r['eoom']} peak={r['peak_gib']} io={r['io_gib']} scratch={r['scratch_gib']} "
              f"alloc={r['allocated_gib_per_core']} compile_s={r['compile_s']}")
    print(f"wrote {a.out}: {doc['n_neff']} NEFF, {doc['n_eoom']} EOOM, max peak {doc['max_peak_gib']} GiB")
    return 0 if doc["n_neff"] else 1


if __name__ == "__main__":
    sys.exit(main())
