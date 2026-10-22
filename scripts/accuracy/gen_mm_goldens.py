#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF multimodal goldens for the L3-mm logit validation (instance CPU).

Model: an ``-mm`` directory from ``tools/make_mm_checkpoint.py`` (e.g. MiMo-V2.6-Flash-MOPD-bf16-L3-mm), loaded
with the patched HF remote code (``MiMoV2ForCausalLM`` builds ``visual`` from ``vision_config``); merger / ln_q
biases are zeroed (not in the checkpoint), the ViT ``inv_freq`` buffer is recomputed (left
uninitialised by from_pretrained) and the FP32 routing bias is restored in the BF16 model.

Inputs: each test image (448x448 from make_mm_assets.py -> 196 image tokens) + one question, in chat format,
processed by **vLLM's MiMo-Omni processor** (the tensors the server feeds the model, docs/design-notes.md#mm-reference-inputs): the expanded prompt
ids, ``pixel_values`` (BF16 as served; the FP32 model gets the same values as FP32) and ``image_grid_thw``.

* FP32: autoregressive greedy, 16 steps (image re-encoded every step, full forward) -> fp32_logits [16, 1, V]
* BF16: teacher-forced on the FP32 tokens -> bf16_logits [16, 1, V]
* vision embeddings (HF ``visual`` output) of both dtypes -> vit_fp32 / vit_bf16 [196, 4096] (ViT check, docs/design-notes.md#vit-accuracy)

Output ``<out>/goldens.pt`` uses the gen_goldens.py layout (names, input_ids, fp32_logits, bf16_logits,
fp32_tokens, vocab_size) plus ``mm[name] = {"image", "question", "grid_thw", "pixel_values", "vit_fp32",
"vit_bf16"}``, and ``<out>/prompts.json`` (set b1 = all names) for run_mm_validation.py.
Exit codes: 0 ok, 2 bad args, 3 config / assertion error.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STEPS = 16
IMAGE_TOKEN = 151655
DEFAULT_ITEMS = [("red_square", "red_square.png", "What color is the square in the image?"),
                 ("number_739", "number_739.png", "What number is written in the image?")]
MERGER = ("merger.mlp.0", "merger.mlp.2", "merger.ln_q")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fail(msg):
    log("ERROR:", msg)
    sys.exit(3)


def chat_prompt(question: str) -> str:
    return (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n")


def restore_vit_inv_freq(m):
    """from_pretrained leaves the ViT's non-persistent ``inv_freq`` buffer uninitialised (values up to 5e33,
    ViT cos(bf16, fp32) 0.70); recompute it as ``MiMoVisionRotaryEmbedding.__init__`` does (theta 10000, FP32)."""
    rot = m.visual.rotary_pos_emb
    n = rot.inv_freq.numel()
    rot.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 2 * n, 2, dtype=torch.float) / (2 * n)))


def load(model_dir, dtype):
    from gen_goldens import assert_fp32_bias, load_model, restore_fp32_bias

    m = load_model(model_dir, dtype)
    if not hasattr(m, "visual"):
        fail(f"{model_dir}: HF model has no visual tower (vision_config empty?)")
    with torch.no_grad():
        for name in MERGER:
            mod = m.visual.get_submodule(name)
            if mod.bias is not None:
                mod.bias.zero_()
    restore_vit_inv_freq(m)
    if dtype == torch.bfloat16:
        log(f"bf16: restored FP32 routing bias on {restore_fp32_bias(m, model_dir)} gates")
    assert_fp32_bias(m, str(dtype))
    return m


@torch.inference_mode()
def last_logits(m, ids, pv, thw):
    return m(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_thw=thw, use_cache=False,
             return_dict=True).logits[:, -1, :].float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--assets", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--dry-run", action="store_true", help="1 image, 2 tokens, nothing saved")
    ap.add_argument("--record-routing", action="store_true",
                    help="routing.pt (gen_goldens.py layout) for fallback_check.py --mm")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    import mm_prep
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer

    vocab = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True).vocab_size
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    proc = mm_prep.vllm_processor(a.model_dir)
    items = DEFAULT_ITEMS[:1] if a.dry_run else DEFAULT_ITEMS
    steps = 2 if a.dry_run else STEPS
    inp = {}
    for name, fn, q in items:
        path = os.path.join(a.assets, fn)
        img = Image.open(path).convert("RGB")
        out = mm_prep.run_vllm(proc, chat_prompt(q), {"image": [img]})
        ids = list(out["prompt_token_ids"])
        pv = mm_prep.item_field(out, "image", 0, "pixel_values").to(torch.bfloat16)   # as served
        thw = mm_prep.item_field(out, "image", 0, "image_grid_thw").reshape(1, 3)
        n_img = ids.count(IMAGE_TOKEN)
        if n_img != int(thw.prod()) // 4:
            fail(f"{name}: {n_img} image tokens != grid {thw.tolist()} / 4")
        inp[name] = dict(ids=ids, pv=pv, thw=thw, image=path, question=q)
        log(f"{name}: {len(ids)} prompt tokens ({n_img} image tokens), grid {thw.tolist()}")

    res = {n: {} for n in inp}
    routing = {n: {} for n in inp}
    from gen_goldens import RoutingRecorder
    for dtype in (torch.float32, torch.bfloat16):
        m = load(a.model_dir, dtype)
        rec = RoutingRecorder(m) if a.record_routing else None
        tag = "fp32" if dtype == torch.float32 else "bf16"
        for n, d in inp.items():
            pv = d["pv"].to(dtype)
            with torch.inference_mode():
                res[n][f"vit_{tag}"] = m.visual(pixel_values=pv, grid_thw=d["thw"]).float()
            cur, outs = list(d["ids"]), []
            t0 = time.time()
            for s in range(steps):
                lg = last_logits(m, cur, pv, d["thw"])
                if lg.shape[-1] != vocab:
                    fail(f"logits last dim {lg.shape[-1]} != vocab_size {vocab}")
                outs.append(lg)
                cur.append(int(lg.argmax(-1)) if tag == "fp32" else res[n]["fp32_tokens"][s])
            res[n][f"{tag}_logits"] = torch.stack(outs, 0)
            if rec:   # one forward over prompt + the 16 FP32 tokens (same as gen_goldens.py)
                rec.take()
                seq = list(d["ids"]) + (cur[len(d["ids"]):] if tag == "fp32" else list(res[n]["fp32_tokens"]))
                with torch.inference_mode():
                    m(input_ids=torch.tensor([seq]), pixel_values=pv, image_grid_thw=d["thw"], use_cache=False,
                      return_dict=True)
                routing[n][tag] = rec.take()
            if tag == "fp32":
                res[n]["fp32_tokens"] = cur[len(d["ids"]):]
                log(f"{n}: fp32 tokens {res[n]['fp32_tokens']} -> {tok.decode(res[n]['fp32_tokens'])!r} "
                    f"({time.time() - t0:.0f}s)")
            else:
                agree = (res[n]["bf16_logits"].argmax(-1) == res[n]["fp32_logits"].argmax(-1)).float().mean().item()
                v32, v16 = res[n]["vit_fp32"], res[n]["vit_bf16"]
                cos = torch.nn.functional.cosine_similarity(v32.flatten(), v16.flatten(), dim=0).item()
                res[n]["bf16_top1_agreement"], res[n]["vit_cos_bf16_fp32"] = agree, cos
                log(f"{n}: bf16 teacher-forced top-1 agreement {agree:.3f}, ViT cos(bf16, fp32) {cos:.5f}")
                if cos < 0.99:   # docs/design-notes.md#vit-accuracy: HF BF16 vs FP32 is 0.9976-0.9992; far below means a broken reference
                    fail(f"{n}: ViT cos(bf16, fp32) {cos:.5f} < 0.99 (uninitialised buffer?)")
        if rec:
            rec.close()
        del m
        gc.collect()

    if a.dry_run:
        log("dry-run OK")
        return 0
    names = list(inp)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"names": names, "input_ids": {n: [inp[n]["ids"]] for n in names},
                "fp32_logits": {n: res[n]["fp32_logits"] for n in names},
                "bf16_logits": {n: res[n]["bf16_logits"] for n in names},
                "fp32_tokens": {n: res[n]["fp32_tokens"] for n in names}, "vocab_size": vocab,
                "mm": {n: {"image": inp[n]["image"], "question": inp[n]["question"],
                           "grid_thw": inp[n]["thw"].tolist(), "pixel_values": inp[n]["pv"],
                           "vit_fp32": res[n]["vit_fp32"], "vit_bf16": res[n]["vit_bf16"]} for n in names}},
               os.path.join(a.out, "goldens.pt"))
    if a.record_routing:
        torch.save(routing, os.path.join(a.out, "routing.pt"))
    with open(os.path.join(a.out, "prompts.json"), "w") as f:
        json.dump({"sets": {"b1": names}, "prompts": {n: inp[n]["question"] for n in names},
                   "images": {n: inp[n]["image"] for n in names}}, f, indent=1)
    summary = {"model_dir": a.model_dir, "vocab_size": vocab, "steps": STEPS,
               "prompts": {n: {"prompt_tokens": len(inp[n]["ids"]), "image": inp[n]["image"],
                               "grid_thw": inp[n]["thw"].tolist(), "fp32_tokens": res[n]["fp32_tokens"],
                               "fp32_text": tok.decode(res[n]["fp32_tokens"]),
                               "bf16_top1_agreement": res[n]["bf16_top1_agreement"],
                               "vit_cos_bf16_fp32": res[n]["vit_cos_bf16_fp32"]} for n in names}}
    rp = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "results", "goldens", f"{os.path.basename(os.path.normpath(a.out))}_summary.json")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    json.dump(summary, open(rp, "w"), ensure_ascii=False, indent=1)
    log(f"saved goldens to {a.out}, summary {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
