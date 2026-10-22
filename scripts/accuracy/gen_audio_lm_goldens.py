#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF goldens for the L3 audio logit validation (docs/design-notes.md#audio-on-device; instance CPU). Same layout as gen_mm_goldens.py.

  start_bg.sh p2_goldens_l3audio env OMP_NUM_THREADS=96 bash scripts/venv_exec.sh python \
      scripts/accuracy/gen_audio_lm_goldens.py --model-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16-L3-mm-audio \
      --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --out $MIMO_WORK/goldens/L3_audio --record-routing

Inputs: each WAV asset + a question in chat format, through vLLM's MiMo-Omni processor (the served prompt ids and
log-mel). The HF audio towers (``audio_ref.hf_towers``: tokenizer + code embedding, both dtypes) give the audio
embeddings, which the HF L3 model receives as ``audio_embeds`` (its ``<|audio_pad|>`` rows):
  * FP32: towers FP32, model FP32, autoregressive greedy 16 steps -> fp32_logits [16, 1, V];
  * BF16: towers BF16 (codebooks BF16-rounded like vLLM), model BF16 teacher-forced -> bf16_logits.
``mm[name] = {"audio", "question", "token_len", "mel", "audio_fp32", "audio_bf16"}`` (+ routing.pt with
--record-routing) for run_mm_validation.py --audio.
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
AUDIO_TOKEN = 151669
ITEMS = [("jfk", "jfk.wav", "Transcribe this audio."),
         ("beeps_3", "beeps_3.wav", "How many beeps are there in this audio? Answer with a number.")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chat_prompt(q: str) -> str:
    return (f"<|im_start|>user\n<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>{q}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n")


@torch.inference_mode()
def last_logits(m, ids, emb):
    return m(input_ids=torch.tensor([ids]), audio_embeds=emb, use_cache=False, return_dict=True).logits[:, -1, :].float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--src", required=True, help="original checkpoint (audio towers for the HF reference)")
    ap.add_argument("--assets", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets/audio"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--record-routing", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    import audio_ref
    import mm_prep
    from gen_goldens import RoutingRecorder
    from gen_mm_goldens import load
    from make_audio_assets import read_wav
    from transformers import AutoConfig, AutoTokenizer

    vocab = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True).vocab_size
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    proc = mm_prep.vllm_processor(a.model_dir)
    M, se, enc, tw = audio_ref.hf_towers(a.src)
    inp = {}
    for name, fn, q in ITEMS:
        path = os.path.join(a.assets, fn)
        with open(path, "rb") as f:
            x, sr = read_wav(f.read())
        out = mm_prep.run_vllm(proc, chat_prompt(q), {"audio": [(x, sr)]})
        ids = list(out["prompt_token_ids"])
        it = out["mm_kwargs"]["audio"][0]
        mel, tl = it["audio_features"].data, int(it["audio_token_lens"].data)
        if ids.count(AUDIO_TOKEN) != tl:
            log(f"ERROR: {name}: {ids.count(AUDIO_TOKEN)} audio tokens != audio_token_lens {tl}")
            return 3
        emb = {dt: audio_ref.hf_run(M, se, enc, tw, mel, dt)["embeds"] for dt in (torch.float32, torch.bfloat16)}
        inp[name] = dict(ids=ids, mel=mel, tl=tl, audio=path, question=q, emb=emb)
        log(f"{name}: {len(ids)} prompt tokens ({tl} audio tokens)")
    res = {n: {} for n in inp}
    routing = {n: {} for n in inp}
    for dtype in (torch.float32, torch.bfloat16):
        m = load(a.model_dir, dtype)
        rec = RoutingRecorder(m) if a.record_routing else None
        tag = "fp32" if dtype == torch.float32 else "bf16"
        for n, d in inp.items():
            emb = d["emb"][dtype].to(dtype)
            cur, outs = list(d["ids"]), []
            t0 = time.time()
            for s in range(STEPS):
                lg = last_logits(m, cur, emb)
                outs.append(lg)
                cur.append(int(lg.argmax(-1)) if tag == "fp32" else res[n]["fp32_tokens"][s])
            res[n][f"{tag}_logits"] = torch.stack(outs, 0)
            if rec:
                rec.take()
                seq = list(d["ids"]) + (cur[len(d["ids"]):] if tag == "fp32" else list(res[n]["fp32_tokens"]))
                with torch.inference_mode():
                    m(input_ids=torch.tensor([seq]), audio_embeds=emb, use_cache=False, return_dict=True)
                routing[n][tag] = rec.take()
            if tag == "fp32":
                res[n]["fp32_tokens"] = cur[len(d["ids"]):]
                log(f"{n}: fp32 tokens -> {tok.decode(res[n]['fp32_tokens'])!r} ({time.time() - t0:.0f}s)")
            else:
                agree = (res[n]["bf16_logits"].argmax(-1) == res[n]["fp32_logits"].argmax(-1)).float().mean().item()
                res[n]["bf16_top1_agreement"] = agree
                log(f"{n}: bf16 teacher-forced top-1 agreement {agree:.3f}")
        if rec:
            rec.close()
        del m
        gc.collect()
    names = list(inp)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"names": names, "input_ids": {n: [inp[n]["ids"]] for n in names},
                "fp32_logits": {n: res[n]["fp32_logits"] for n in names},
                "bf16_logits": {n: res[n]["bf16_logits"] for n in names},
                "fp32_tokens": {n: res[n]["fp32_tokens"] for n in names}, "vocab_size": vocab,
                "mm": {n: {"audio": inp[n]["audio"], "question": inp[n]["question"], "token_len": inp[n]["tl"],
                           "mel": inp[n]["mel"], "audio_fp32": inp[n]["emb"][torch.float32].float(),
                           "audio_bf16": inp[n]["emb"][torch.bfloat16].float()} for n in names}},
               os.path.join(a.out, "goldens.pt"))
    if a.record_routing:
        torch.save(routing, os.path.join(a.out, "routing.pt"))
    with open(os.path.join(a.out, "prompts.json"), "w") as f:
        json.dump({"sets": {"b1": names}, "prompts": {n: inp[n]["question"] for n in names},
                   "audios": {n: inp[n]["audio"] for n in names}}, f, indent=1)
    summary = {"model_dir": a.model_dir, "vocab_size": vocab, "steps": STEPS,
               "prompts": {n: {"prompt_tokens": len(inp[n]["ids"]), "audio": inp[n]["audio"],
                               "audio_tokens": inp[n]["tl"], "fp32_tokens": res[n]["fp32_tokens"],
                               "fp32_text": tok.decode(res[n]["fp32_tokens"]),
                               "bf16_top1_agreement": res[n]["bf16_top1_agreement"]} for n in names}}
    rp = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "results", "goldens", f"{os.path.basename(os.path.normpath(a.out))}_summary.json")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    json.dump(summary, open(rp, "w"), ensure_ascii=False, indent=1)
    log(f"saved goldens to {a.out}, summary {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
