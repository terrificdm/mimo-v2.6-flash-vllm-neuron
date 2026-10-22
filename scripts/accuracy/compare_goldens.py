#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two goldens.pt files of the same prompts (docs/design-notes.md#fp8-alignment): GPU-simulation vs BF16 reference.

  python scripts/accuracy/compare_goldens.py --a $MIMO_WORK/goldens/L3 --b $MIMO_WORK/goldens/L3_gpusim \
      --out $MIMO_WORK/results/fp8/gpusim_vs_bf16.json

Per prompt and for both golden kinds (fp32 autoregressive, bf16 teacher-forced): whether the FP32 greedy tokens are
identical, top-1 agreement of the logits, top-5 relative error (max |b - a| over a's top-5 / |a's top-1|, the
logit-validation definition) and relative L2 of the whole logit vector, mean / max over the 16 steps.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def load(d):
    return torch.load(os.path.join(d, "goldens.pt"), map_location="cpu", weights_only=False)


def stats(la, lb):
    la, lb = la.float().reshape(la.shape[0], -1), lb.float().reshape(lb.shape[0], -1)
    top1 = (la.argmax(-1) == lb.argmax(-1)).float()
    idx = la.topk(5, dim=-1).indices
    va, vb = la.gather(1, idx), lb.gather(1, idx)
    k5 = ((vb - va).abs().amax(-1) / va[:, :1].abs().squeeze(-1).clamp(min=1e-6))
    l2 = (lb - la).norm(dim=-1) / la.norm(dim=-1)
    return {"top1_agreement": float(top1.mean()), "top5_rel_err_mean": float(k5.mean()),
            "top5_rel_err_max": float(k5.max()), "rel_l2_mean": float(l2.mean()), "rel_l2_max": float(l2.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="reference goldens dir (e.g. BF16 L3)")
    ap.add_argument("--b", required=True, help="compared goldens dir (e.g. L3_gpusim)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A, B = load(a.a), load(a.b)
    out = {"a": a.a, "b": a.b, "prompts": {}}
    for n in A["names"]:
        if n not in B["fp32_logits"]:
            continue
        out["prompts"][n] = {"fp32_tokens_identical": A["fp32_tokens"][n] == B["fp32_tokens"][n],
                             "fp32": stats(A["fp32_logits"][n], B["fp32_logits"][n]),
                             "bf16": stats(A["bf16_logits"][n], B["bf16_logits"][n])}
    for kind in ("fp32", "bf16"):
        v = [p[kind] for p in out["prompts"].values()]
        out[f"overall_{kind}"] = {k: (sum(x[k] for x in v) / len(v) if "mean" in k or "agreement" in k
                                      else max(x[k] for x in v)) for k in v[0]}
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    with open(os.path.expanduser(a.out), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("overall")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
