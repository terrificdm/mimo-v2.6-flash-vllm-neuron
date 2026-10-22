#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vllm-project/vllm-neuron (Apache-2.0):
#   examples/vllm_neuron/accuracy/run_logit_validation_offline.py; modified. See NOTICE.
"""Neuron-side logit validation against the HF goldens (docs/design-notes.md#validation-rules).

Adapted from vllm-neuron ``examples/vllm_neuron/accuracy/run_logit_validation_offline.py``: the goldens
come from ``gen_goldens.py`` (FP32 autoregressive baseline + BF16 teacher-forced expected), the target is
this plugin running in vLLM, and ``multi_prompt_logit_validation`` does the three-way comparison with the
plugin defaults (DEFAULT_TOLERANCE_MAP, divergence 1 BF16 ULP, DEFAULT_AGGREGATE_CONFIG).

--batch-mode b1: max_num_seqs=1, only the prompt under test is submitted.
--batch-mode b4: max_num_seqs=4; the prompt under test is submitted together with the other 3 prompts of
                 the b4 set (different lengths, same SamplingParams) in one ``llm.generate`` call, and
                 only its own logits are returned. Each prompt takes the "under test" role in turn.
--batch-mode b2: as b4 with the 2 prompts of the b2 set (max_num_seqs=2; long prompts).
--max-model-len / --kv-segment-size: see lv_common.py (defaults = base configuration: 1024 tokens, no KV segments).
--top5-rtol-override R: replace the rtol of the top-5 tier of DEFAULT_TOLERANCE_MAP by R (default: not set,
                 the plugin defaults are used unchanged; multimodal last-resort rule only, docs/design-notes.md#validation-rules).

Example (device):
  start_bg.sh logit_b1 env -u VLLM_NEURON_CPU_MODE -u NKI_SIMULATOR ... bash scripts/venv_exec.sh \
    python scripts/accuracy/run_logit_validation.py --batch-mode b1 --decode-moe-impl tkg \
    --out $MIMO_WORK/results/logit/b1_tkg.json
Exit codes: see lv_common.py (0 pass, 1 criterion failed, 2 bad args, 3 config error, 4 crash).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common as C  # noqa: E402


def summarize_steps(token_results):
    steps = []
    for r in token_results:
        tw = r.get("three_way_metrics") or {}
        steps.append(C.to_py({
            "passed": r["passed"],
            "divergence": r["divergence"],
            "divergence_difference": r["divergence_difference"],
            "divergence_in_bounds": r["divergence_in_bounds"],
            "divergence_threshold": r["divergence_threshold"],
            "error_map": r["error_map"],
            "shift": r["shift"],
            "expected_top2_indices": r["expected_top2_indices"],
            "actual_top2_indices": r["actual_top2_indices"],
            "expected_top2_values": r["expected_top2_values"],
            "actual_top2_values": r["actual_top2_values"],
            "expected_top1_top2_diff": r["expected_top1_top2_diff"],
            "three_way": {k: tw.get(k) for k in ("base_linf", "tgt_linf", "base_l2", "tgt_l2", "bc", "passed")},
        }))
    return steps


def tolerance_map(top5_rtol_override, default_map: dict):
    """None (= library default) unless --top5-rtol-override is given; then a copy with the top-5 rtol replaced."""
    if top5_rtol_override is None:
        return None
    m = dict(default_map)
    atol, _ = m["5"]
    m["5"] = (atol, float(top5_rtol_override))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--top5-rtol-override", type=float, default=None,
                    help="rtol of the top-5 tolerance tier (default: not set = DEFAULT_TOLERANCE_MAP unchanged)")
    a = ap.parse_args(argv)
    # --decode-moe-impl is required and recorded so the environment is explicit: set before vLLM is imported / the LLM (and its worker processes) is created
    C.apply_decode_moe_impl(a.decode_moe_impl)
    env = C.env_snapshot()
    C.log("environment:", env)

    import torch
    # NOTE: import vllm before transformers (vllm-neuron example convention)
    import vllm  # noqa: F401
    from vllm_neuron.accuracy.constants import DEFAULT_TOLERANCE_MAP
    from vllm_neuron.accuracy.logit_validation import (DEFAULT_AGGREGATE_CONFIG, DEFAULT_DYNAMIC_THRESHOLD_CONFIG,
                                                       multi_prompt_logit_validation)

    names, ids, g = C.load_inputs(a)
    vocab = C.model_vocab(a.model_dir)
    if vocab != g["vocab_size"]:
        raise C.ConfigError(f"model vocab_size {vocab} != goldens vocab_size {g['vocab_size']}")
    max_num_seqs = C.BATCH_SEQS[a.batch_mode]
    tol_map = tolerance_map(a.top5_rtol_override, DEFAULT_TOLERANCE_MAP)
    C.log(f"mode={a.batch_mode} impl={a.decode_moe_impl} tp={a.tp} prompts="
          + ", ".join(f"{n}({len(ids[n])} tok)" for n in names))

    t0 = time.time()
    llm = C.build_llm(a, max_num_seqs)
    t_init = time.time() - t0
    C.log(f"LLM ready in {t_init:.0f}s")
    sp = C.sampling_params(vocab, C.STEPS)
    calls = {"n": 0, "seconds": 0.0}

    def under_test(input_ids):
        """Name of the prompt whose (possibly teacher-extended) ids were passed in."""
        hits = [n for n in names if input_ids[:len(ids[n])] == ids[n]]
        if not hits:
            raise C.ConfigError("generate_fn got input ids that match no prompt")
        return max(hits, key=lambda n: len(ids[n]))

    def generate_fn(input_ids):
        if len(input_ids) != 1:
            raise C.ConfigError(f"expected batch size 1 per validated prompt, got {len(input_ids)}")
        cur = list(input_ids[0])
        if len(cur) + C.STEPS > a.max_model_len:
            raise C.ConfigError(f"prompt length {len(cur)} + {C.STEPS} > {a.max_model_len}")
        if a.batch_mode == "b1":
            batch, slot, tag = [cur], 0, under_test(cur)
        else:
            tag = under_test(cur)
            batch = [cur if n == tag else ids[n] for n in names]   # companions keep their original prompts
            slot = names.index(tag)
        t = time.time()
        outs = llm.generate([{"prompt_token_ids": b} for b in batch], sp, use_tqdm=False)
        calls["n"] += 1
        calls["seconds"] += time.time() - t
        per = [C.completion_logits(o.outputs[0], vocab, C.STEPS, f"{tag}/batch{i}") for i, o in enumerate(outs)]
        logits, toks = per[slot]
        C.log(f"generate #{calls['n']} {tag} (len {len(cur)}, batch {len(batch)}) tokens {toks} "
              f"in {time.time() - t:.1f}s")
        return logits.unsqueeze(1), torch.tensor([toks], dtype=torch.long)   # [16,1,V], [1,16]

    res = multi_prompt_logit_validation(
        prompts_input_ids=[[list(ids[n])] for n in names],
        generate_fn=generate_fn,
        prompts_expected_logits=[g["bf16_logits"][n].float() for n in names],
        prompts_baseline_logits=[g["fp32_logits"][n].float() for n in names],
        tol_map=tol_map,
        colorize=False,
    )

    per_prompt = {}
    for n, (ok, results) in zip(names, res.per_prompt_results):
        steps = summarize_steps(results[0]) if results else []
        per_prompt[n] = {
            "prompt_tokens": len(ids[n]),
            "passed": bool(ok),
            "failed_steps": [i for i, s in enumerate(steps) if not s["passed"]],
            "divergent_steps": [i for i, s in enumerate(steps) if s["divergence"]],
            "fp32_golden_tokens": list(g["fp32_tokens"][n]),
            "steps": steps,
        }
    out = {
        "script": "run_logit_validation.py",
        "batch_mode": a.batch_mode,
        "decode_moe_impl": a.decode_moe_impl,
        "environment": env,
        "model_dir": a.model_dir,
        "goldens": a.goldens,
        "tp": a.tp,
        "max_num_seqs": max_num_seqs,
        "max_model_len": a.max_model_len,
        "kv_segment_size": a.kv_segment_size,
        "prompts": names,
        # main criterion = the plugin's overall verdict (all prompts pass two-way tolerances, or the
        # aggregate sigma-ratio <= 1.0, or the aggregate per-token BC passes)
        "passed": bool(res.passed),
        "all_prompts_passed": all(p["passed"] for p in per_prompt.values()),
        "aggregate": C.to_py(res.aggregate_metrics, max_len=256),
        "tolerances": {"tol_map": tol_map or DEFAULT_TOLERANCE_MAP,
                       "top5_rtol_override": a.top5_rtol_override, "divergence_n_ulps": 1, "mantissa_bits": 7,
                       "dynamic_threshold_config": DEFAULT_DYNAMIC_THRESHOLD_CONFIG,
                       "aggregate_config": DEFAULT_AGGREGATE_CONFIG},
        "per_prompt": per_prompt,
        "timing": {"llm_init_s": round(t_init, 1), "generate_calls": calls["n"],
                   "generate_s": round(calls["seconds"], 1), "total_s": round(time.time() - t0, 1)},
    }
    C.write_json(a.out, out)
    C.log(f"RESULT {a.batch_mode}/{a.decode_moe_impl}: passed={out['passed']} "
          f"all_prompts_passed={out['all_prompts_passed']} sigma_ratio={out['aggregate'].get('agg_sigma_ratio')} "
          f"agg_bc={out['aggregate'].get('agg_bc')} failed_steps="
          + str({n: p["failed_steps"] for n, p in per_prompt.items()}))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    C.run_main(main)
