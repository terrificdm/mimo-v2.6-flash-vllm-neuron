#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mechanical gate over smoke_test.sh output (stdlib only).

Reads <dir>/single.json and <dir>/concurrent4.json. Every entry must be HTTP 200; every entry whose `expect` is not
null must contain the expected string in its answer; and the expected set must be complete: '391' twice and '20'
twice (one each in single, one each in concurrent4). Other answers are only recorded.
Usage: smoke_gate.py <smoke_out_dir> [--out PREFIX]   exit 0 pass, 1 fail, 2 missing/unreadable files.
"""

import argparse
import json
import os
import sys

REQUIRED = {"391": 2, "20": 2}


def answer(r):
    try:
        c = r["response"]["choices"][0]
        return c["text"] if "text" in c else (c["message"].get("content") or "")
    except Exception:  # noqa: BLE001
        return ""


def gate(entries):
    """entries: list of (file_tag, entry). Returns (ok, rows, problems)."""
    rows, problems, seen = [], [], {k: 0 for k in REQUIRED}
    for tag, e in entries:
        exp = e.get("expect")
        a = answer(e)
        http_ok = e.get("http_status") == 200
        hit = None if exp is None else (exp in a)
        rows.append({"file": tag, "question": e.get("question"), "endpoint": e.get("endpoint"),
                     "http_status": e.get("http_status"), "expect": exp, "contains": hit, "answer": a[:600]})
        if not http_ok:
            problems.append(f"{tag}: HTTP {e.get('http_status')} for {e.get('question')!r}")
        if exp is not None:
            if hit:
                if exp in seen:
                    seen[exp] += 1
            else:
                problems.append(f"{tag}: answer to {e.get('question')!r} does not contain {exp!r}")
    for k, n in REQUIRED.items():
        if seen[k] < n:
            problems.append(f"expected {n} correct answers containing {k!r}, got {seen[k]}")
    return not problems, rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    entries = []
    for name in ("single", "concurrent4"):
        p = os.path.join(args.dir, f"{name}.json")
        try:
            with open(p) as f:
                entries += [(name, e) for e in json.load(f)]
        except (OSError, ValueError) as ex:
            print(f"smoke_gate: cannot read {p}: {ex}", file=sys.stderr)
            return 2
    ok, rows, problems = gate(entries)
    for r in rows:
        print(f"[{r['file']}] HTTP {r['http_status']} expect={r['expect']} contains={r['contains']} Q: {r['question']}")
    for p in problems:
        print("smoke_gate: FAIL " + p)
    print(f"smoke_gate: {'PASS' if ok else 'FAIL'} ({len(rows)} entries)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out + ".json", "w") as f:
            json.dump({"pass": ok, "problems": problems, "rows": rows, "smoke_dir": args.dir}, f, ensure_ascii=False,
                      indent=1)
        with open(args.out + ".md", "w") as f:
            f.write(f"# smoke gate: {'PASS' if ok else 'FAIL'}\n\nsmoke dir: {args.dir}\n\n"
                    "| file | HTTP | expect | contains | question |\n|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['file']} | {r['http_status']} | {r['expect']} | {r['contains']} | {r['question']} |\n")
            for p in problems:
                f.write(f"\n- FAIL: {p}")
            f.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
