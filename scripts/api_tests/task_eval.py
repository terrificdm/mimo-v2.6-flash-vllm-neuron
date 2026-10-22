#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task-level accuracy on small GSM8K / MMLU subsets through the OpenAI chat API (stdlib only;
docs/design-notes.md#task-eval).

Protocol (fixed, written into the result file): 0-shot chat, enable_thinking=false, temperature 0 (greedy),
--workers concurrent requests. GSM8K: "solve step by step, final line 'Answer: <number>'", max_tokens 1024, score =
last number in the reply (after 'Answer:' when present) equals the reference. MMLU: "answer with the letter only",
max_tokens 16, score = first standalone A-D letter.
Data: results/task_eval_data/{gsm8k,mmlu}_test_200_seed0.jsonl (random 200 of the official test splits,
random.Random(0)).

Usage: python scripts/api_tests/task_eval.py --task gsm8k|mmlu [--n 200] [--workers 4] --out <prefix>
Writes <prefix>.json (per-item records + summary). Exit 0 when every request returned HTTP 200, 1 otherwise.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "task_eval_data")
OFF = {"enable_thinking": False}
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_prompt(it):
    return (it["question"] + "\n\nSolve the problem step by step. On the last line write 'Answer: ' followed by the "
            "final number only.")


def gsm8k_pred(text):
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text
    nums = NUM.findall(tail) or NUM.findall(text)
    if not nums:
        return None
    v = nums[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def mmlu_prompt(it):
    opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", it["choices"]))
    return (f"The following is a multiple choice question about {it['subject'].replace('_', ' ')}.\n\n"
            f"{it['question']}\n{opts}\n\nAnswer with the letter (A, B, C or D) only.")


def mmlu_pred(text):
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


TASKS = {"gsm8k": (gsm8k_prompt, gsm8k_pred, 1024), "mmlu": (mmlu_prompt, mmlu_pred, 16)}


def run_one(task, it):
    prompt, pred_fn, max_tokens = TASKS[task]
    body = C.chat_body([{"role": "user", "content": prompt(it)}], max_tokens=max_tokens, temperature=0,
                       chat_template_kwargs=OFF)
    r = C.post("/v1/chat/completions", body, timeout=1800)
    text, usage = "", {}
    if r["http_status"] == 200:
        ch = r["response"]["choices"][0]
        text = ch["message"].get("content") or ""
        usage = r["response"].get("usage", {})
    pred = pred_fn(text) if text else None
    ref = it["answer"]
    ok = pred is not None and (pred == ref if task == "mmlu" else _num_eq(pred, ref))
    return {"idx": it["idx"], "http_status": r["http_status"], "elapsed_s": r["elapsed_s"], "ref": ref, "pred": pred,
            "correct": bool(ok), "completion_tokens": usage.get("completion_tokens"), "reply": text}


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = os.path.join(DATA, f"{a.task}_test_200_seed0.jsonl")
    items = [json.loads(l) for l in open(path)][: a.n]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda it: run_one(a.task, it), items))
    wall = time.time() - t0
    ok = [r for r in recs if r["http_status"] == 200]
    summary = {"task": a.task, "n": len(recs), "http_ok": len(ok), "correct": sum(r["correct"] for r in recs),
               "accuracy": round(sum(r["correct"] for r in recs) / max(len(recs), 1), 4),
               "unparsed": sum(r["pred"] is None for r in ok), "wall_s": round(wall, 1), "workers": a.workers,
               "protocol": "0-shot chat, enable_thinking=false, temperature 0, max_tokens %d" % TASKS[a.task][2],
               "data": os.path.relpath(path, os.path.join(DATA, "..", "..", "..", "..")),
               "base_url": C.base_url(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summary, "items": recs}, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if len(ok) == len(recs) else 1


if __name__ == "__main__":
    sys.exit(main())
