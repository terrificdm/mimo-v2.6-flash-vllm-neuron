#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mechanical fallback criterion (docs/design-notes.md#validation-rules) — only meaningful after the main criterion failed.

1. Trigger precondition: for every failed step s of every prompt in the main-criterion result JSON
   (logits from position p = len(prompt) - 1 + s), the HF-BF16 vs HF-FP32 routing flip set F (top-8
   expert sets differ in at least one MoE layer, recorded by gen_goldens.py --record-routing on
   prompt + 16 FP32 tokens) must contain a position q with len(prompt) - 1 <= q <= p (generation
   region only). Any failed step without such a q -> precondition not met -> criterion failed.
   Extension (approved by the user on 2026-10-19 after the first L3 B=1 result):
   with ``--neuron-routing`` (capture_routing.py output) a failed step is also covered when, at some
   MoE layer and position q with len(prompt) - 1 <= q <= p, the Neuron top-8 set differs from HF-FP32
   and the Neuron 8th-9th score gap is < 1e-3 (Neuron-side near-tie). q only counts if the Neuron run's
   tokens up to q equal the FP32 golden tokens (same context as the reference).
2. Teacher-forced Neuron logits: for every prompt and s = 0..15 the prefix prompt + fp32_tokens[:s] is run
   with max_tokens=1 (raw logits); all prompts of the batch set for the same s are submitted in one
   ``llm.generate`` call (b4: 4 prefixes of different lengths in the same batch).
3. Criteria (all three must hold): (a) top-1 agreement Neuron vs HF-BF16 >= 95% over all prompts x 16
   steps; (b) at every step, on the HF-BF16 top-5 token ids, |N - H| <= 1e-5 + 0.05 * |H|; (c) at every
   top-1 mismatch step, HF-BF16 raw top1 - top2 < 0.1.

Exit codes: see lv_common.py (0 pass, 1 criterion failed, 2 bad args, 3 config error, 4 crash).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common as C  # noqa: E402

TOP1_AGREEMENT_MIN = 0.95
TOP5_RTOL, TOP5_ATOL = 0.05, 1e-5
NEURON_GAP89_MAX = 1e-3   # extended precondition (user-approved 2026-10-19)
NEAR_TIE_MAX = 0.1


def routing_flips(routing, names, ids):
    """Per prompt: flip pairs (layer, position) where the HF-FP32 and HF-BF16 top-8 sets differ."""
    out = {}
    for n in names:
        P = len(ids[n])
        pairs = []
        for li, (f, b) in enumerate(zip(routing[n]["fp32"], routing[n]["bf16"])):
            fs, bs = f.sort(-1).values, b.sort(-1).values
            for q in (fs != bs).any(-1).nonzero().flatten().tolist():
                pairs.append((li, q))
        pos = sorted({q for _, q in pairs})
        gen_lo, gen_hi = P - 1, P - 1 + C.STEPS - 1
        out[n] = {
            "num_moe_layers": len(routing[n]["fp32"]),
            "flip_pairs": pairs,
            "positions": pos,
            "prompt_region_positions": [q for q in pos if q < gen_lo],
            "generation_region_positions": [q for q in pos if gen_lo <= q <= gen_hi],
            "generation_region": [gen_lo, gen_hi],
            "flips_per_layer": {str(li): sum(1 for l, _ in pairs if l == li)
                                for li in range(len(routing[n]["fp32"]))},
        }
    return out


def neuron_near_tie(P: int, p: int, neuron_prompt: dict, hf_fp32: list, moe_layers: list,
                    gap_max: float = NEURON_GAP89_MAX):
    """Earliest (q, layer, gap89) with P-1 <= q <= p where the Neuron top-8 set != HF-FP32 and the Neuron
    8th-9th gap < gap_max, considering only q whose context equals the golden one; else None.

    neuron_prompt: capture_routing.py per-prompt entry; hf_fp32: per MoE layer (in ``moe_layers`` order)
    [P + 16, 8] HF-FP32 top-8 ids.
    """
    nt, gt = neuron_prompt["neuron_tokens"], neuron_prompt["fp32_golden_tokens"]
    for q in range(P - 1, p + 1):
        if q >= P and list(nt[:q - P + 1]) != list(gt[:q - P + 1]):
            return None   # Neuron context diverged from the golden sequence
        r = neuron_prompt["routing"].get(str(q))
        if r is None:
            continue
        for li, layer in enumerate(moe_layers):
            e = r.get(str(layer))
            if e is None:
                continue
            if set(e["idx"]) != set(int(x) for x in hf_fp32[li][q].tolist()) and e["gap89"] < gap_max:
                return {"position": q, "layer": layer, "gap89": e["gap89"]}
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--main-result", default=None,
                    help="main-criterion JSON (default: <dir of --out>/<batch-mode>_<impl>.json)")
    ap.add_argument("--neuron-routing", default=None,
                    help="capture_routing.py JSON; enables the user-approved Neuron near-tie precondition")
    C.add_mm_args(ap)
    a = ap.parse_args(argv)
    # --decode-moe-impl is required and recorded so the environment is explicit: set before vLLM is imported / the LLM (and its worker processes) is created
    C.apply_decode_moe_impl(a.decode_moe_impl)
    env = C.env_snapshot()
    C.log("environment:", env)

    import json
    import torch
    import vllm  # noqa: F401  (import vllm before transformers)

    names, ids, g = C.load_inputs(a)
    main_path = a.main_result or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                              f"{a.batch_mode}_{a.decode_moe_impl}.json")
    if not os.path.exists(main_path):
        raise C.ConfigError(f"main-criterion result {main_path} not found (run run_logit_validation.py first)")
    main_res = json.load(open(main_path))
    if main_res.get("batch_mode") != a.batch_mode or main_res.get("decode_moe_impl") != a.decode_moe_impl:
        raise C.ConfigError(f"{main_path} is for {main_res.get('batch_mode')}/{main_res.get('decode_moe_impl')}, "
                            f"not {a.batch_mode}/{a.decode_moe_impl}")
    rpath = os.path.join(a.goldens, "routing.pt")
    if not os.path.exists(rpath):
        raise C.ConfigError(f"{rpath} not found (gen_goldens.py --record-routing)")
    hf_routing = torch.load(rpath, weights_only=False)
    flips = routing_flips(hf_routing, names, ids)
    nroute = None
    if a.neuron_routing:
        nroute = json.load(open(a.neuron_routing))
        if nroute.get("batch_mode") != a.batch_mode or nroute.get("decode_moe_impl") != a.decode_moe_impl:
            raise C.ConfigError(f"{a.neuron_routing} is for {nroute.get('batch_mode')}/{nroute.get('decode_moe_impl')}")
    moe_layers = None
    if nroute is not None:
        layers = sorted({int(l) for pp in nroute["per_prompt"].values() for r in pp["routing"].values() for l in r})
        if len(layers) != len(hf_routing[names[0]]["fp32"]):
            raise C.ConfigError(f"Neuron routing layers {layers} vs {len(hf_routing[names[0]]['fp32'])} HF MoE layers")
        moe_layers = layers

    # ---- 1. trigger precondition
    pre = {"main_result": main_path, "main_passed": main_res.get("passed"), "per_prompt": {}, "satisfied": True,
           "neuron_near_tie_extension": None if nroute is None else {
               "approved": "user, 2026-10-19, after the first L3 B=1 result", "gap89_max": NEURON_GAP89_MAX,
               "neuron_routing": a.neuron_routing}}
    for n in names:
        P = len(ids[n])
        gen_pos = flips[n]["generation_region_positions"]
        entries = []
        for s in main_res["per_prompt"][n]["failed_steps"]:
            p = P - 1 + s
            qs = [q for q in gen_pos if P - 1 <= q <= p]
            tie = None
            if not qs and nroute is not None:
                tie = neuron_near_tie(P, p, nroute["per_prompt"][n], hf_routing[n]["fp32"], moe_layers)
            covered = bool(qs) or tie is not None
            entries.append({"step": s, "position": p, "earliest_generation_flip": qs[0] if qs else None,
                            "neuron_near_tie": tie,
                            "covered_by": "hf_flip" if qs else ("neuron_near_tie" if tie else None),
                            "covered": covered})
            if not covered:
                pre["satisfied"] = False
        pre["per_prompt"][n] = {"failed_steps": entries}
    n_failed = sum(len(v["failed_steps"]) for v in pre["per_prompt"].values())
    pre["num_failed_steps"] = n_failed
    if n_failed == 0:
        pre["note"] = "main result has no failed steps; precondition vacuously satisfied"
    C.log(f"precondition: failed steps={n_failed} satisfied={pre['satisfied']}")

    # ---- 2. teacher-forced Neuron logits (max_tokens=1)
    vocab = C.model_vocab(a.model_dir)
    if vocab != g["vocab_size"]:
        raise C.ConfigError(f"model vocab_size {vocab} != goldens vocab_size {g['vocab_size']}")
    max_num_seqs = C.BATCH_SEQS[a.batch_mode]
    t0 = time.time()
    llm = C.build_llm(a, max_num_seqs, **C.mm_llm_kwargs(a))
    req = C.request_builder(a, g)
    t_init = time.time() - t0
    sp = C.sampling_params(vocab, 1)
    neuron = {n: [] for n in names}
    for s in range(C.STEPS):
        batch = [ids[n] + list(g["fp32_tokens"][n][:s]) for n in names]
        t = time.time()
        outs = llm.generate([req(n, b) for n, b in zip(names, batch)], sp, use_tqdm=False)
        for n, o in zip(names, outs):
            lg, _ = C.completion_logits(o.outputs[0], vocab, 1, f"{n}/prefix{s}")
            neuron[n].append(lg[0])
        C.log(f"step {s}: {len(batch)} prefixes in {time.time() - t:.1f}s")

    # ---- 3. criteria
    total = agree = 0
    b_fail, c_fail, max_rel = [], [], 0.0
    per_prompt = {}
    for n in names:
        rows = []
        for s in range(C.STEPS):
            N = neuron[n][s]
            H = g["bf16_logits"][n][s, 0].float()
            n1, h1 = int(N.argmax()), int(H.argmax())
            top5 = H.topk(5).indices
            diff = (N[top5] - H[top5]).abs()
            rel = float((diff / H[top5].abs().clamp(min=1e-10)).max())
            ok_b = bool((diff <= TOP5_ATOL + TOP5_RTOL * H[top5].abs()).all())
            h_top2 = H.topk(2).values
            gap = float(h_top2[0] - h_top2[1])
            total += 1
            agree += int(n1 == h1)
            max_rel = max(max_rel, rel)
            if not ok_b:
                b_fail.append([n, s])
            if n1 != h1 and not gap < NEAR_TIE_MAX:
                c_fail.append([n, s])
            rows.append({"step": s, "neuron_top1": n1, "hf_bf16_top1": h1, "top1_match": n1 == h1,
                         "top5_max_rel_err": rel, "top5_ok": ok_b, "hf_bf16_top1_top2_gap": gap})
        per_prompt[n] = {"prompt_tokens": len(ids[n]), "steps": rows}
    rate = agree / total
    crit = {
        "a_top1_agreement": {"value": rate, "threshold": TOP1_AGREEMENT_MIN, "passed": rate >= TOP1_AGREEMENT_MIN},
        "b_top5_rtol": {"max_rel_err": max_rel, "rtol": TOP5_RTOL, "atol": TOP5_ATOL,
                        "failed_steps": b_fail, "passed": not b_fail},
        "c_near_tie": {"threshold": NEAR_TIE_MAX, "failed_steps": c_fail, "passed": not c_fail},
    }
    passed = pre["satisfied"] and all(v["passed"] for v in crit.values())
    out = {
        "script": "fallback_check.py",
        "batch_mode": a.batch_mode,
        "decode_moe_impl": a.decode_moe_impl,
        "environment": env,
        "model_dir": a.model_dir,
        "goldens": a.goldens,
        "tp": a.tp,
        "prompts": names,
        "passed": passed,
        "precondition": pre,
        "criteria": crit,
        "routing_flips": {n: {k: v for k, v in f.items() if k != "flip_pairs"}
                          | {"num_flip_pairs": len(f["flip_pairs"]),
                             "num_prompt_region": len(f["prompt_region_positions"]),
                             "num_generation_region": len(f["generation_region_positions"])}
                          for n, f in flips.items()},
        "per_prompt": per_prompt,
        "timing": {"llm_init_s": round(t_init, 1), "total_s": round(time.time() - t0, 1)},
    }
    C.write_json(a.out, C.to_py(out, max_len=4096))
    C.log(f"RESULT fallback {a.batch_mode}/{a.decode_moe_impl}: passed={passed} precondition={pre['satisfied']} "
          f"(a) {rate:.3f} (b) max_rel {max_rel:.4f} fails {len(b_fail)} (c) fails {len(c_fail)}")
    return 0 if passed else 1


if __name__ == "__main__":
    C.run_main(main)
