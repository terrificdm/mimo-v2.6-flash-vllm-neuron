#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-question agreement between two task_eval.py runs (e.g. native FP8 vs BF16) on the same questions
(docs/design-notes.md#task-eval).

Usage: python3 tools/task_eval_agreement.py <a.json> <b.json> [--out <json>]
Prints accuracy of each run, the share of questions with the same parsed answer, the share with the same
correctness, and the questions where only one run is correct.
"""
import argparse
import json


def load(path):
    d = json.load(open(path))
    return d["summary"], {it["idx"]: it for it in d["items"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out")
    args = ap.parse_args()
    sa, ia = load(args.a)
    sb, ib = load(args.b)
    common = sorted(set(ia) & set(ib))
    same_pred = sum(ia[i]["pred"] == ib[i]["pred"] for i in common)
    same_corr = sum(ia[i]["correct"] == ib[i]["correct"] for i in common)
    same_reply = sum(ia[i]["reply"] == ib[i]["reply"] for i in common)
    only_a = [i for i in common if ia[i]["correct"] and not ib[i]["correct"]]
    only_b = [i for i in common if ib[i]["correct"] and not ia[i]["correct"]]
    res = {
        "a": args.a, "b": args.b, "task": sa.get("task"), "n_common": len(common),
        "acc_a": sum(ia[i]["correct"] for i in common) / len(common),
        "acc_b": sum(ib[i]["correct"] for i in common) / len(common),
        "same_pred": same_pred / len(common), "same_correctness": same_corr / len(common),
        "same_reply_text": same_reply / len(common),
        "only_a_correct": [{"idx": i, "ref": ia[i]["ref"], "a": ia[i]["pred"], "b": ib[i]["pred"]} for i in only_a],
        "only_b_correct": [{"idx": i, "ref": ia[i]["ref"], "a": ia[i]["pred"], "b": ib[i]["pred"]} for i in only_b],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
