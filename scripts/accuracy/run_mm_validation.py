#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multimodal logit validation on the L3-mm model.

Same comparison as run_logit_validation.py (``multi_prompt_logit_validation``, FP32 autoregressive baseline + BF16
teacher-forced expected, plugin default tolerances, docs/design-notes.md#validation-rules) but every request carries its image: the goldens
(``gen_mm_goldens.py``) hold the processor-expanded prompt ids; they are passed to vLLM as token ids with the
196-token image run collapsed to one ``<|image_pad|>`` (vLLM re-expands token prompts; found in the tiny-model drill) plus the
image as ``multi_modal_data``. The engine's expanded prompt must equal the golden ids (checked every call).

--capture-vision: sets ``MIMO_VIT_DUMP_DIR`` so the model's ``embed_multimodal`` (TP rank 0) saves the merger
output rows it wrote to the encoder cache; they are compared with the HF BF16 / FP32 ViT outputs stored in the
goldens (docs/design-notes.md#vit-accuracy: main criterion cos(device, HF BF16) >= 0.999, fallback three-way +
cos(device, HF FP32) >= cos(HF BF16, HF FP32) - 0.002).

Example (device ViT check):
  start_bg.sh p2_vit_device $DEV_ENV bash scripts/venv_exec.sh python scripts/accuracy/run_mm_validation.py \
    --batch-mode b1 --decode-moe-impl tkg --tp 64 --model-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16-L3-mm \
    --goldens $MIMO_WORK/goldens/L3_mm --prompts $MIMO_WORK/goldens/L3_mm/prompts.json \
    --num-gpu-blocks-override 1024 --capture-vision --out $R2/mm/l3mm_b1.json
Exit codes: see lv_common.py (0 pass, 1 criterion failed, 2 bad args, 3 config error, 4 crash).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common as C  # noqa: E402
from run_logit_validation import summarize_steps, tolerance_map  # noqa: E402

IMAGE_TOKEN = 151655
AUDIO_TOKEN = 151669


def collapse_image_tokens(ids, image_token=IMAGE_TOKEN):
    out = []
    for t in ids:
        if not (t == image_token and out and out[-1] == image_token):
            out.append(t)
    return out


def vit_compare(dump_dir: str, g: dict, names: list) -> dict:
    """Device merger rows (first dump per image) vs HF BF16 / FP32 ViT outputs from the goldens."""
    import torch
    import torch.nn.functional as F
    from vllm_neuron.accuracy.testing import assert_close_three_way

    files = sorted(glob.glob(os.path.join(dump_dir, "vit_*.pt")))
    if not files:
        raise C.ConfigError(f"--capture-vision: no ViT dumps in {dump_dir}")
    dumps = [torch.load(f) for f in files]
    out, ok_all = {}, True
    for n in names:   # match each golden image to a dumped item by the sum of its pixel values
        mm = g["mm"][n]
        want = float(mm["pixel_values"].float().sum())
        cand = [(abs(s - want), d, j, f) for f, d in zip(files, dumps) for j, s in enumerate(d["pv_sums"])]
        err, d, j, fname = min(cand, key=lambda c: c[0])
        if err > 1e-3 * abs(want) + 1e-2:
            raise C.ConfigError(f"{n}: no ViT dump matches the image (pixel sum {want}, closest diff {err})")
        if list(d["grid_thw"][j]) != list(mm["grid_thw"][0]):
            raise C.ConfigError(f"{n}: dump grid {d['grid_thw'][j]} != golden grid {mm['grid_thw']}")
        dev = d["embeds"][j].float()
        f32, b16 = mm["vit_fp32"].float(), mm["vit_bf16"].float()
        if dev.shape != f32.shape:
            raise C.ConfigError(f"{n}: device ViT rows {tuple(dev.shape)} != HF {tuple(f32.shape)}")
        # float64: float32 accumulation over a 14080 x 4096 video gives cosines > 1
        cos = lambda x, y: F.cosine_similarity(x.double().flatten(), y.double().flatten(), dim=0).item()  # noqa: E731
        r = assert_close_three_way([f32], [b16], [dev], name=f"vit/{n}")
        rec = {"cos_dev_hf_bf16": cos(dev, b16), "cos_dev_hf_fp32": cos(dev, f32), "cos_hf_bf16_fp32": cos(b16, f32),
               "three_way_passed": bool(r.passed), "dump": fname}
        rec["main_passed"] = rec["cos_dev_hf_bf16"] >= 0.999
        rec["fallback_passed"] = bool(r.passed) and rec["cos_dev_hf_fp32"] >= rec["cos_hf_bf16_fp32"] - 0.002
        rec["passed"] = rec["main_passed"] or rec["fallback_passed"]
        ok_all = ok_all and rec["passed"]
        out[n] = rec
        C.log(f"ViT {n}: cos(dev,HF bf16)={rec['cos_dev_hf_bf16']:.5f} cos(dev,HF fp32)={rec['cos_dev_hf_fp32']:.5f} "
              f"cos(HF bf16,fp32)={rec['cos_hf_bf16_fp32']:.5f} three_way={r.passed} -> passed={rec['passed']}")
    return {"passed": ok_all, "items": out, "num_dumps": len(files)}


def audio_compare(dump_dir: str, g: dict, names: list) -> dict:
    """Device audio rows (first dump per item) vs HF BF16 / FP32 audio embeddings (docs/design-notes.md#audio-criteria: cos(dev, HF fp32) >=
    cos(HF bf16, HF fp32) - 0.01; cos(dev, HF bf16) reported)."""
    import torch
    import torch.nn.functional as F

    files = sorted(glob.glob(os.path.join(dump_dir, "audio_*.pt")))
    if not files:
        raise C.ConfigError(f"--capture-audio: no audio dumps in {dump_dir}")
    dumps = [torch.load(f) for f in files]
    out, ok_all = {}, True
    for n in names:
        mm = g["mm"][n]
        want = float(mm["mel"].float().sum())
        err, d, fname = min(((abs(x["mel_sum"] - want), x, f) for f, x in zip(files, dumps)), key=lambda c: c[0])
        if err > 1e-3 * abs(want) + 1e-2:
            raise C.ConfigError(f"{n}: no audio dump matches (mel sum {want}, closest diff {err})")
        dev, f32, b16 = d["embeds"].float(), mm["audio_fp32"].float(), mm["audio_bf16"].float()
        if dev.shape != f32.shape:
            raise C.ConfigError(f"{n}: device audio rows {tuple(dev.shape)} != HF {tuple(f32.shape)}")
        cos = lambda x, y: F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()  # noqa: E731
        rec = {"cos_dev_hf_bf16": cos(dev, b16), "cos_dev_hf_fp32": cos(dev, f32), "cos_hf_bf16_fp32": cos(b16, f32),
               "dump": fname}
        rec["passed"] = rec["cos_dev_hf_fp32"] >= rec["cos_hf_bf16_fp32"] - 0.01
        ok_all = ok_all and rec["passed"]
        out[n] = rec
        C.log(f"audio {n}: cos(dev,HF bf16)={rec['cos_dev_hf_bf16']:.5f} cos(dev,HF fp32)={rec['cos_dev_hf_fp32']:.5f} "
              f"cos(HF bf16,fp32)={rec['cos_hf_bf16_fp32']:.5f} -> passed={rec['passed']}")
    return {"passed": ok_all, "items": out, "num_dumps": len(files)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--top5-rtol-override", type=float, default=None)
    ap.add_argument("--capture-vision", action="store_true", help="dump device ViT rows and compare with HF")
    ap.add_argument("--audio", action="store_true", help="goldens from gen_audio_lm_goldens.py (audio prompts)")
    ap.add_argument("--capture-audio", action="store_true", help="dump device audio rows and compare with HF")
    ap.add_argument("--vision-block", type=int, default=4096, help="vision_attention_block_size / bucket")
    ap.add_argument("--vision-tp", type=int, default=8)
    ap.add_argument("--vision-dp", type=int, default=8)
    ap.add_argument("--vision-cache-blocks", type=int, default=17,
                    help="encoder_cache_num_blocks (auto-derived pool is too small when every image takes a block, docs/design-notes.md#encoder-cache-blocks)")
    a = ap.parse_args(argv)
    C.apply_decode_moe_impl(a.decode_moe_impl)
    dump_dir = None
    if a.capture_vision:
        dump_dir = tempfile.mkdtemp(prefix="mimo_vit_dump_", dir=os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")))
        os.environ["MIMO_VIT_DUMP_DIR"] = dump_dir   # before the LLM (workers inherit it)
    audio_dump = None
    if a.capture_audio:
        audio_dump = tempfile.mkdtemp(prefix="mimo_audio_dump_", dir=os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")))
        os.environ["MIMO_AUDIO_DUMP_DIR"] = audio_dump
    env = C.env_snapshot()
    env["MIMO_VIT_DUMP_DIR"] = dump_dir
    env["MIMO_AUDIO_DUMP_DIR"] = audio_dump
    C.log("environment:", env)

    import torch
    import vllm  # noqa: F401
    from PIL import Image
    from vllm_neuron.accuracy.constants import DEFAULT_TOLERANCE_MAP
    from vllm_neuron.accuracy.logit_validation import (DEFAULT_AGGREGATE_CONFIG, DEFAULT_DYNAMIC_THRESHOLD_CONFIG,
                                                       multi_prompt_logit_validation)

    names, ids, g = C.load_inputs(a)
    if "mm" not in g:
        raise C.ConfigError(f"{a.goldens}/goldens.pt has no 'mm' entry (use gen_mm_goldens.py)")
    if a.batch_mode != "b1":
        raise C.ConfigError("multimodal validation supports --batch-mode b1 only (one request per prefill)")
    vocab = C.model_vocab(a.model_dir)
    if vocab != g["vocab_size"]:
        raise C.ConfigError(f"model vocab_size {vocab} != goldens vocab_size {g['vocab_size']}")
    if a.audio:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_tests"))
        from make_audio_assets import read_wav

        def _wav(p):
            with open(p, "rb") as f:
                return read_wav(f.read())
        media = {n: _wav(g["mm"][n]["audio"]) for n in names}
        kind, mtok = "audio", AUDIO_TOKEN
    else:
        media = {n: Image.open(g["mm"][n]["image"]).convert("RGB") for n in names}
        kind, mtok = "image", IMAGE_TOKEN
    tol_map = tolerance_map(a.top5_rtol_override, DEFAULT_TOLERANCE_MAP)
    C.log("prompts: " + ", ".join(f"{n}({len(ids[n])} tok, {kind} {g['mm'][n][kind]})" for n in names))

    t0 = time.time()
    vnc = {"num_vision_tokens_buckets": [a.vision_block], "vision_attention_block_size": a.vision_block,
           "tp_size": a.vision_tp, "dp_size": a.vision_dp, "encoder_cache_num_blocks": a.vision_cache_blocks}
    llm = C.build_llm(a, 1, extra_additional_config={"vision_neuron_config": vnc},
                      extra_kwargs={"limit_mm_per_prompt": {"image": 0 if a.audio else 1, "video": 0,
                                                            "audio": 1 if a.audio else 0}})
    t_init = time.time() - t0
    C.log(f"LLM ready in {t_init:.0f}s")
    sp = C.sampling_params(vocab, C.STEPS)
    calls = {"n": 0, "seconds": 0.0}

    def under_test(input_ids):
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
        tag = under_test(cur)
        t = time.time()
        req = {"prompt_token_ids": collapse_image_tokens(cur, mtok), "multi_modal_data": {kind: media[tag]}}
        o = llm.generate([req], sp, use_tqdm=False)[0]
        if list(o.prompt_token_ids) != cur:
            raise C.ConfigError(f"{tag}: engine prompt ids (len {len(o.prompt_token_ids)}) != golden ids (len {len(cur)})")
        calls["n"] += 1
        calls["seconds"] += time.time() - t
        logits, toks = C.completion_logits(o.outputs[0], vocab, C.STEPS, tag)
        C.log(f"generate #{calls['n']} {tag} (len {len(cur)}) tokens {toks} in {time.time() - t:.1f}s")
        return logits.unsqueeze(1), torch.tensor([toks], dtype=torch.long)

    res = multi_prompt_logit_validation(
        prompts_input_ids=[[list(ids[n])] for n in names], generate_fn=generate_fn,
        prompts_expected_logits=[g["bf16_logits"][n].float() for n in names],
        prompts_baseline_logits=[g["fp32_logits"][n].float() for n in names], tol_map=tol_map, colorize=False)

    per_prompt = {}
    for n, (ok, results) in zip(names, res.per_prompt_results):
        steps = summarize_steps(results[0]) if results else []
        per_prompt[n] = {"prompt_tokens": len(ids[n]), kind: g["mm"][n][kind], "passed": bool(ok),
                         "failed_steps": [i for i, s in enumerate(steps) if not s["passed"]],
                         "divergent_steps": [i for i, s in enumerate(steps) if s["divergence"]],
                         "fp32_golden_tokens": list(g["fp32_tokens"][n]), "steps": steps}
    vit = vit_compare(dump_dir, g, names) if dump_dir else None
    audio = audio_compare(audio_dump, g, names) if audio_dump else None
    out = {"script": "run_mm_validation.py", "batch_mode": a.batch_mode, "decode_moe_impl": a.decode_moe_impl,
           "environment": env, "model_dir": a.model_dir, "goldens": a.goldens, "tp": a.tp,
           "vision_neuron_config": vnc, "max_model_len": a.max_model_len, "prompts": names,
           "passed": bool(res.passed), "all_prompts_passed": all(p["passed"] for p in per_prompt.values()),
           "aggregate": C.to_py(res.aggregate_metrics, max_len=256),
           "tolerances": {"tol_map": tol_map or DEFAULT_TOLERANCE_MAP, "top5_rtol_override": a.top5_rtol_override,
                          "dynamic_threshold_config": DEFAULT_DYNAMIC_THRESHOLD_CONFIG,
                          "aggregate_config": DEFAULT_AGGREGATE_CONFIG},
           "per_prompt": per_prompt, "vit": vit, "audio": audio,
           "timing": {"llm_init_s": round(t_init, 1), "generate_calls": calls["n"],
                      "generate_s": round(calls["seconds"], 1), "total_s": round(time.time() - t0, 1)}}
    C.write_json(a.out, out)
    C.log(f"RESULT mm/{a.batch_mode}: passed={out['passed']} all_prompts_passed={out['all_prompts_passed']} "
          f"sigma_ratio={out['aggregate'].get('agg_sigma_ratio')} vit_passed={vit and vit['passed']} "
          f"audio_passed={audio and audio['passed']} failed_steps="
          + str({n: p["failed_steps"] for n, p in per_prompt.items()}))
    return 0 if out["passed"] and (vit is None or vit["passed"]) and (audio is None or audio["passed"]) else 1


if __name__ == "__main__":
    C.run_main(main)
