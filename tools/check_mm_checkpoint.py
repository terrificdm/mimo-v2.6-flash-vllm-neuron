#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a multimodal checkpoint directory built by make_mm_checkpoint.py (docs/design-notes.md#omni-architecture).

The mm-dir variant of tools/check_converted.py checks 1-3 (that tool's shard-count rule does not apply: the mm dir
has one extra ``model-visual.safetensors``):
  1. every shard in the weight map exists (text shards may be symlinks; their targets must exist);
  2. ``metadata.total_size`` == sum of tensor byte sizes from the shard headers, and the weight-map keys of each
     shard equal that shard's header keys;
  3. ``total_size`` == text dir ``total_size`` + visual shard bytes; 364 ``visual.*`` keys, all BF16, none of the
     merger / ln_q biases; config.json has ``vision_config`` and ``architectures = ["MiMoV2OmniForCausalLM"]``.
Usage: check_mm_checkpoint.py --mm <mm dir> --text <text dir> [--out summary.json]
Exit 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_converted import header, nbytes  # noqa: E402

BIAS_FREE = ("visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias", "visual.merger.ln_q.bias")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    idx = json.load(open(os.path.join(a.mm, "model.safetensors.index.json")))
    wm, total = idx["weight_map"], idx["metadata"]["total_size"]
    text_total = json.load(open(os.path.join(a.text, "model.safetensors.index.json")))["metadata"]["total_size"]
    errors, summed, visual_bytes, visual_dtypes = [], 0, 0, set()
    shards = sorted(set(wm.values()))
    for sh in shards:
        p = os.path.join(a.mm, sh)
        if not os.path.exists(p):
            errors.append(f"missing shard {sh}")
            continue
        h = header(p)
        want = {k for k, s in wm.items() if s == sh}
        if set(h) != want:
            errors.append(f"{sh}: header keys != weight-map keys (+{len(set(h) - want)} / -{len(want - set(h))})")
        b = sum(nbytes(m) for m in h.values())
        summed += b
        if sh == "model-visual.safetensors":
            visual_bytes = b
            visual_dtypes = {m["dtype"] for m in h.values()}
    vis = [k for k in wm if k.startswith("visual.")]
    if summed != total:
        errors.append(f"total_size {total} != header sum {summed}")
    if total != text_total + visual_bytes:
        errors.append(f"total_size {total} != text {text_total} + visual {visual_bytes}")
    if len(vis) != 364:
        errors.append(f"{len(vis)} visual.* keys (want 364)")
    if visual_dtypes != {"BF16"}:
        errors.append(f"visual dtypes {sorted(visual_dtypes)} (want BF16)")
    if any(k in wm for k in BIAS_FREE):
        errors.append("merger / ln_q bias keys present")
    cfg = json.load(open(os.path.join(a.mm, "config.json")))
    if not cfg.get("vision_config") or cfg.get("architectures") != ["MiMoV2OmniForCausalLM"]:
        errors.append(f"config: vision_config={bool(cfg.get('vision_config'))} architectures={cfg.get('architectures')}")
    out = {"mm": a.mm, "text": a.text, "shards": len(shards), "keys": len(wm), "visual_keys": len(vis),
           "total_size": total, "header_sum": summed, "text_total_size": text_total, "visual_bytes": visual_bytes,
           "processor_config": cfg.get("processor_config"), "ok": not errors, "errors": errors}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
