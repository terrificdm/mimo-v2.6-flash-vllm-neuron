#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge test_params.py results of mode A (on-device sampling) and mode B (CPU sampling) into one parameter matrix
Outputs: <out>.json and <out>.md (Chinese table: 参数 / 模式 A / 模式 B / 结论 / 原因与证据).

Overall conclusion per case id:
  - modes that are missing or 'unverified' do not count; nothing left -> unverified
  - every counted mode supported -> supported
  - some but not all counted modes supported -> partial ("只在模式 X 生效")
  - otherwise any partial -> partial; else unsupported
--notes JSON maps case id -> str or list of str (code locations / log excerpts found while diagnosing);
a case's own `evidence` field (str or list) is included too.
Usage: param_matrix.py --a params_A.json [--b params_B.json] [--notes notes.json] --out PREFIX
"""

import argparse
import datetime as dt
import json
import os
import sys

ZH = {"supported": "支持", "partial": "部分支持", "unsupported": "不支持", "unverified": "未验证", "not_run": "未测"}
MODES = ("A", "B")


def overall(by_mode):
    """by_mode: {"A": support or None, "B": ...} -> (overall, only_in list)."""
    counted = {m: s for m, s in by_mode.items() if s in ("supported", "partial", "unsupported")}
    if not counted:
        return "unverified", []
    sup = [m for m, s in counted.items() if s == "supported"]
    if len(sup) == len(counted):
        return "supported", []
    if sup:
        return "partial", sorted(sup)
    if any(s == "partial" for s in counted.values()):
        return "partial", []
    return "unsupported", []


def _as_list(v):
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


def merge(docs, notes=None):
    """docs: {"A": result doc or None, "B": ...}. Returns the list of matrix rows (A order first, then B-only ids)."""
    notes = notes or {}
    cases = {m: {c["id"]: c for c in (d or {}).get("cases", [])} for m, d in docs.items()}
    order = []
    for m in MODES:
        for c in (docs.get(m) or {}).get("cases", []):
            if c["id"] not in order:
                order.append(c["id"])
    rows = []
    for cid in order:
        by_mode = {m: (cases.get(m, {}).get(cid) or {}).get("support") for m in MODES}
        ov, only = overall(by_mode)
        param = next((cases[m][cid].get("param") for m in MODES if cid in cases.get(m, {})), cid)
        ev = []
        for m in MODES:
            c = cases.get(m, {}).get(cid)
            if c:
                ev.append(f"{m}: {c.get('reason', '')}")
                ev += [f"{m}: {x}" for x in _as_list(c.get("evidence"))]
        ev += _as_list(notes.get(cid))
        rows.append({"id": cid, "param": param, "A": by_mode["A"] or "not_run", "B": by_mode["B"] or "not_run",
                     "overall": ov, "only_in": only, "evidence": ev})
    return rows


def _md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def to_markdown(rows, sources):
    lines = ["# 采样与 API 参数矩阵", "",
             "模式 A = 设备上采样（默认，on-device sampling）；模式 B = CPU 采样（`MIMO_SAMPLING=cpu`）。",
             "来源：" + "；".join(f"{m} = `{p}`" for m, p in sources.items() if p), "",
             "| 参数 | 模式 A | 模式 B | 结论 | 原因与证据 |", "|---|---|---|---|---|"]
    for r in rows:
        concl = ZH[r["overall"]] + (f"（只在模式 {'、'.join(r['only_in'])} 生效）" if r["only_in"] else "")
        ev = "<br>".join(_md(x)[:300] for x in r["evidence"])
        lines.append(f"| {_md(r['param'])} | {ZH[r['A']]} | {ZH[r['B']]} | {concl} | {ev} |")
    return "\n".join(lines) + "\n"


def _load(path):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        docs = {"A": _load(args.a), "B": _load(args.b)}
        notes = _load(args.notes) or {}
    except (OSError, ValueError) as e:
        print(f"param_matrix: cannot read input: {e}", file=sys.stderr)
        return 2
    rows = merge(docs, notes)
    sources = {"A": args.a, "B": args.b}
    doc = {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sources": sources,
           "modes": {m: {k: (d or {}).get(k) for k in ("tag", "gencfg", "started_utc", "finished_utc")}
                     for m, d in docs.items()}, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(to_markdown(rows, sources))
    print(f"param_matrix: {len(rows)} rows -> {args.out}.json/.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
