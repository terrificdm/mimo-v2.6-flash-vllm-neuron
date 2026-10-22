#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record the Neuron MoE routing of the main-criterion generation (input to fallback_check.py).

Extended fallback precondition (approved by the user on 2026-10-19 *after* the first L3 B=1 result,
see docs/design-notes.md#validation-rules): a failed step may also be covered by a Neuron-side routing near-tie, i.e. the
Neuron top-8 set differs from HF-FP32 at some MoE layer and generation-region position q <= p, and the
Neuron 8th/9th score gap there is < 1e-3. This script produces the Neuron side of that check.

It repeats the main-criterion generation (same LLM arguments, same batches: b1 = each prompt alone,
b4 = the 4 prompts of the set in one ``llm.generate``; greedy 16 tokens) with
``MIMO_DEBUG_CAPTURE=1`` + ``neuron_config.tensor_capture``, so the model's router writes, per call,
``l<layer>.t<T>.moe_idx`` (top-8 ids) and ``l<layer>.t<T>.moe_choice`` (sigmoid + bias scores).
Prefill captures are per-rank SP shards (stitched in rank order), decode captures are replicated
(rank 0). Rows are mapped to prompts via the capture metadata (request ids, positions).

Output JSON: per prompt, the generated tokens and, for every position q >= len(prompt) - 1, per MoE
layer index (model layer id) the sorted top-8 ids and the 8th-9th score gap.

  start_bg.sh route_b1 <DEV_ENV> bash scripts/venv_exec.sh python scripts/accuracy/capture_routing.py \
    --batch-mode b1 --decode-moe-impl tkg --num-gpu-blocks-override 1024 \
    --capture-dir $MIMO_WORK/captures/route_b1_tkg --out .../results/logit/neuron_routing_b1_tkg.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common as C  # noqa: E402

_CAP = re.compile(r"manual\.l(\d+)\.t(\d+)\.moe_(idx|choice)$")


def _phase_tensors(phase_dir: str, prefill: bool):
    """{layer: {"idx": [rows, 8], "choice": [rows, E]}} for one capture call."""
    import torch

    out = {}
    for d in sorted(glob.glob(os.path.join(phase_dir, "manual.l*.t*.moe_*"))):
        m = _CAP.search(os.path.basename(d))
        if not m:
            continue
        layer, kind = int(m.group(1)), m.group(3)
        if prefill:
            files = sorted(glob.glob(os.path.join(d, "rank*.pt")), key=lambda f: int(os.path.basename(f)[4:-3]))
            t = torch.cat([torch.load(f, weights_only=False) for f in files], 0)
        else:
            t = torch.load(os.path.join(d, "rank0.pt"), weights_only=False)
        out.setdefault(layer, {})[kind] = t
    return out


def parse_captures(capture_dir: str, req_to_name: dict, prompt_len: dict) -> dict:
    """-> {name: {q: {layer: {"idx": [...], "gap89": float}}}} for q >= len(prompt) - 1."""
    res = {n: {} for n in prompt_len}
    for ph in sorted(glob.glob(os.path.join(capture_dir, "dp0", "*"))):
        base = os.path.basename(ph)
        metas = glob.glob(os.path.join(ph, "*_meta.json"))
        if not metas:
            continue
        meta = json.load(open(metas[0]))
        prefill = base.startswith("prefill")
        tens = _phase_tensors(ph, prefill)
        if not tens:
            continue
        if prefill:
            rid = meta["req_ids"][0]
            n = req_to_name.get(int(rid.split("-")[0]))
            if n is None:
                continue
            # real rows have consecutive positions from the first one (0, or cached_seq_len for a
            # segmented-prefill segment); a prefill only holds prompt tokens, so only q = len(prompt) - 1
            q0 = meta["positions"][0]
            rows = [(i, q) for i, q in enumerate(meta["positions"]) if q == q0 + i and q == prompt_len[n] - 1]
            items = [(n, i, q) for i, q in rows]
        else:
            items = []
            for i, (rid, q) in enumerate(zip(meta["req_ids"], meta["positions"])):
                n = req_to_name.get(int(rid.split("-")[0]))
                if n is not None and q >= prompt_len[n] - 1:
                    items.append((n, i, q))
        for n, i, q in items:
            for layer, t in tens.items():
                ch = t["choice"][i].float()
                s = ch.sort(descending=True).values
                res[n].setdefault(int(q), {})[layer] = {
                    "idx": sorted(int(x) for x in t["idx"][i].flatten().tolist()),
                    "gap89": float(s[7] - s[8]),
                }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--capture-dir", required=True, help="must not exist yet")
    C.add_mm_args(ap)
    a = ap.parse_args(argv)
    C.apply_decode_moe_impl(a.decode_moe_impl)
    os.environ["MIMO_DEBUG_CAPTURE"] = "1"   # read by the plugin at import (worker processes inherit it)
    C.log("environment:", C.env_snapshot(), "MIMO_DEBUG_CAPTURE=1")
    if os.path.exists(a.capture_dir):
        raise C.ConfigError(f"--capture-dir {a.capture_dir} already exists")

    import vllm  # noqa: F401

    names, ids, g = C.load_inputs(a)
    vocab = C.model_vocab(a.model_dir)
    max_num_seqs = C.BATCH_SEQS[a.batch_mode]
    llm = C.build_llm(a, max_num_seqs, {"tensor_capture": {"modules": ["lm_head"], "capture_dir": a.capture_dir}},
                      **C.mm_llm_kwargs(a))
    req = C.request_builder(a, g)
    sp = C.sampling_params(vocab, C.STEPS)
    batches = [[n] for n in names] if a.batch_mode == "b1" else [names]
    req_to_name, tokens, counter = {}, {}, 0
    for b in batches:
        outs = llm.generate([req(n, ids[n]) for n in b], sp, use_tqdm=False)
        for n, o in zip(b, outs):
            req_to_name[int(o.request_id.split("-")[0])] = n
            tokens[n] = list(o.outputs[0].token_ids)
            counter += 1
        C.log("generated", {n: tokens[n] for n in b})
    routing = parse_captures(a.capture_dir, req_to_name, {n: len(ids[n]) for n in names})
    out = {
        "script": "capture_routing.py", "batch_mode": a.batch_mode, "decode_moe_impl": a.decode_moe_impl,
        "environment": C.env_snapshot(), "model_dir": a.model_dir, "tp": a.tp, "capture_dir": a.capture_dir,
        "prompts": names,
        "per_prompt": {n: {"prompt_tokens": len(ids[n]), "neuron_tokens": tokens[n],
                           "fp32_golden_tokens": list(g["fp32_tokens"][n]),
                           "routing": {str(q): {str(l): v for l, v in sorted(r.items())}
                                       for q, r in sorted(routing[n].items())}}
                       for n in names},
    }
    for n in names:
        C.log(f"{n}: positions with routing {len(routing[n])} (expected {C.STEPS})")
        if len(routing[n]) < C.STEPS:
            raise C.ConfigError(f"{n}: captured routing for {len(routing[n])} positions, expected {C.STEPS}")
    C.write_json(a.out, out)
    return 0


if __name__ == "__main__":
    C.run_main(main)
