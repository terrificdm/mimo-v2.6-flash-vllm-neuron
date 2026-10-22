#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate HF reference goldens for logit validation (instance CPU).

* FP32 model: autoregressive greedy, 16 steps -> fp32_logits[name] [16, 1, V]
* BF16 model: teacher-forced on the FP32 tokens -> bf16_logits[name] [16, 1, V]
  (routing bias e_score_correction_bias is restored to FP32 after loading)
* --record-routing: forward hooks on every MiMoV2MoEGate record top-8 expert ids for
  prompt + 16 FP32 tokens under both models; FP32-vs-BF16 set differences are the
  trigger precondition of the fallback criterion (docs/design-notes.md#validation-rules).

Outputs: <out>/goldens.pt, <out>/routing.pt (with --record-routing),
         $MIMO_WORK/results/goldens/<basename(out)>_summary.json
Exit codes: 0 ok, 2 bad args, 3 config/assertion error.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

STEPS = 16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fail(msg):
    log("ERROR:", msg)
    sys.exit(3)


def load_model(model_dir, dtype):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype,
                                             attn_implementation="eager")
    m.eval()
    log(f"loaded {dtype} model in {time.time() - t0:.0f}s ({type(m).__name__})")
    return m


def restore_fp32_bias(model, model_dir):
    """HF casts the F32 routing bias to the load dtype; restore it from the checkpoint."""
    from safetensors import safe_open
    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    n = 0
    for i, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and hasattr(gate, "e_score_correction_bias"):
            key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
            with safe_open(os.path.join(model_dir, wm[key]), "pt") as f:
                t = f.get_tensor(key)
            if t.dtype != torch.float32:
                fail(f"{key} in checkpoint is {t.dtype}, expected float32")
            gate.e_score_correction_bias.data = t.float()
            n += 1
    return n


def assert_fp32_bias(model, tag):
    gates = [l.mlp.gate for l in model.model.layers if hasattr(l.mlp, "gate")]
    bad = [g.e_score_correction_bias.dtype for g in gates if g.e_score_correction_bias.dtype != torch.float32]
    if bad:
        fail(f"{tag}: e_score_correction_bias not float32: {bad}")
    log(f"{tag}: e_score_correction_bias float32 on all {len(gates)} MoE gates: OK")


class RoutingRecorder:
    def __init__(self, model):
        self.records, self.handles = [], []
        for name, mod in model.named_modules():
            if type(mod).__name__ == "MiMoV2MoEGate":
                self.handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        self.records.append(out[0].detach().clone())   # topk_idx [T, 8]

    def take(self):
        r, self.records = self.records, []
        return r

    def close(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def last_logits(model, ids):
    return model(torch.tensor([ids]), use_cache=False, return_dict=True).logits[:, -1, :].float()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dry-run", action="store_true", help="1 prompt, 2 tokens, nothing saved")
    ap.add_argument("--record-routing", action="store_true")
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=1024,
                    help="length limit checked per prompt: len(prompt) + 32 <= this (default 1024, base configuration)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    vocab = cfg.vocab_size
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    spec = json.load(open(a.prompts))
    names = sorted(set().union(*(spec["sets"][k] for k in spec["sets"])))   # every prompt of every set
    steps = 2 if a.dry_run else STEPS
    if a.dry_run:
        names = names[:1]
    ids = {}
    for n in names:
        ids[n] = tok(spec["prompts"][n], add_special_tokens=False)["input_ids"]
        log(f"{n}: {len(ids[n])} prompt tokens")
        if len(ids[n]) + 2 * STEPS > a.max_model_len:
            fail(f"{n}: len(prompt)={len(ids[n])} + 32 > {a.max_model_len}")

    # ---- FP32 autoregressive
    m = load_model(a.model_dir, torch.float32)
    assert_fp32_bias(m, "fp32")
    rec = RoutingRecorder(m) if a.record_routing else None
    fp32_logits, fp32_tokens, routing = {}, {}, {}
    for n in names:
        cur, outs = list(ids[n]), []
        t0 = time.time()
        for _ in range(steps):
            lg = last_logits(m, cur)
            if lg.shape[-1] != vocab:
                fail(f"logits last dim {lg.shape[-1]} != vocab_size {vocab}")
            outs.append(lg)
            cur.append(int(lg.argmax(-1)))
        if rec:
            rec.take()
        fp32_logits[n] = torch.stack(outs, 0)                 # [steps, 1, V]
        fp32_tokens[n] = cur[len(ids[n]):]
        log(f"{n}: fp32 tokens {fp32_tokens[n]} -> {tok.decode(fp32_tokens[n])!r} ({time.time() - t0:.0f}s)")
        if rec:
            with torch.inference_mode():
                m(torch.tensor([cur]), use_cache=False, return_dict=True)
            routing[n] = {"fp32": rec.take()}
    if rec:
        rec.close()
    del m
    gc.collect()

    # ---- BF16 teacher-forced
    m = load_model(a.model_dir, torch.bfloat16)
    nb = restore_fp32_bias(m, a.model_dir)
    log(f"bf16: restored FP32 routing bias on {nb} gates")
    assert_fp32_bias(m, "bf16")
    rec = RoutingRecorder(m) if a.record_routing else None
    bf16_logits = {}
    for n in names:
        cur, outs = list(ids[n]), []
        for s in range(steps):
            outs.append(last_logits(m, cur))
            cur.append(fp32_tokens[n][s])
        if rec:
            rec.take()
        bf16_logits[n] = torch.stack(outs, 0)
        agree = (bf16_logits[n].argmax(-1) == fp32_logits[n].argmax(-1)).float().mean().item()
        log(f"{n}: bf16 teacher-forced top-1 agreement with fp32 = {agree:.3f}")
        if rec:
            with torch.inference_mode():
                m(torch.tensor([cur]), use_cache=False, return_dict=True)
            routing[n]["bf16"] = rec.take()
    if rec:
        rec.close()
    del m
    gc.collect()

    if a.dry_run:
        log("dry-run OK")
        return 0

    # ---- routing flips
    summary = {"model_dir": a.model_dir, "vocab_size": vocab, "steps": STEPS, "prompts": {}}
    for n in names:
        P = len(ids[n])
        ent = {"prompt_tokens": P, "fp32_tokens": fp32_tokens[n], "fp32_text": tok.decode(fp32_tokens[n]),
               "bf16_top1_agreement": (bf16_logits[n].argmax(-1) == fp32_logits[n].argmax(-1)).float().mean().item()}
        if a.record_routing:
            flips = []          # (layer, position)
            for li, (f, b) in enumerate(zip(routing[n]["fp32"], routing[n]["bf16"])):
                fs, bs = f.sort(-1).values, b.sort(-1).values
                for q in torch.nonzero((fs != bs).any(-1)).flatten().tolist():
                    flips.append((li, q))
            pos = sorted({q for _, q in flips})
            gen_lo, gen_hi = P - 1, P - 1 + STEPS - 1
            ent["routing"] = {
                "num_moe_layers": len(routing[n]["fp32"]),
                "flip_pairs": flips,
                "flip_positions": pos,
                "prompt_region_positions": [q for q in pos if q < gen_lo],
                "generation_region_positions": [q for q in pos if gen_lo <= q <= gen_hi],
                "generation_region": [gen_lo, gen_hi],
                "flips_per_layer": {str(li): sum(1 for l, _ in flips if l == li)
                                    for li in range(len(routing[n]["fp32"]))},
            }
            log(f"{n}: routing flips prompt-region={len(ent['routing']['prompt_region_positions'])} "
                f"gen-region={len(ent['routing']['generation_region_positions'])}")
        summary["prompts"][n] = ent

    os.makedirs(a.out, exist_ok=True)
    torch.save({"names": names, "input_ids": {n: [ids[n]] for n in names}, "fp32_logits": fp32_logits,
                "bf16_logits": bf16_logits, "fp32_tokens": fp32_tokens, "vocab_size": vocab},
               os.path.join(a.out, "goldens.pt"))
    if a.record_routing:
        torch.save(routing, os.path.join(a.out, "routing.pt"))
    res = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "results", "goldens", f"{os.path.basename(os.path.normpath(a.out))}_summary.json")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    json.dump(summary, open(res, "w"), ensure_ascii=False, indent=1)
    log(f"saved goldens to {a.out}, summary {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
