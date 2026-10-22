#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audio feasibility checks on the instance CPU (docs/design-notes.md#audio-input).

  start_bg.sh p2_audio_prep env OMP_NUM_THREADS=32 bash scripts/venv_exec.sh python scripts/accuracy/audio_prep.py \
      --model-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --assets $MIMO_WORK/mm_assets/audio \
      --out $MIMO_WORK/results/audio/prep.json

1. vLLM front end: can ``vllm.multimodal.media.audio.load_audio`` read a WAV in this venv (soundfile / av missing)?
   Does the MiMo-Omni processor (vLLM's, unmodified) turn a ``(waveform, sr)`` item into ``audio_features`` (log-mel
   ``[T, 128]``) and how many ``<|audio_pad|>`` tokens (``audio_token_lens``) does it emit? Does
   ``get_mm_max_tokens_per_item`` cover audio (encoder budget)?
2. HF reference pipeline on CPU, both towers, timed: ``MiMoAudioTokenizer`` encoder (mel -> 20 RVQ codes per 40 ms),
   then ``speech_embeddings`` + ``MiMoAudioEncoder`` (6-layer local transformer on groups of 4 + projection) ->
   ``[tokens, 4096]``; checks the token count equals the processor's ``audio_token_lens``; BF16 vs FP32 code agreement
   (how discrete RVQ codes react to BF16 noise, which bounds what a device port can match).
Writes JSON; exit 0 when written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "plugin"), os.path.join(ROOT, "scripts", "accuracy"), os.path.join(ROOT, "scripts", "api_tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROMPT = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>What is in this audio?"


def err(e):
    return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:1500]}", "traceback": traceback.format_exc()[-2000:]}


def check_loader(path):
    from vllm.multimodal.media import audio as A
    try:
        x, sr = A.load_audio(path, sr=None)
        return {"ok": True, "samples": int(len(x)), "sr": sr}
    except Exception as e:  # noqa: BLE001
        return err(e)


def front_end(model_dir, wavs):
    from make_audio_assets import read_wav
    from mm_prep import run_vllm, vllm_processor

    proc = vllm_processor(model_dir)
    res = {"ok": True, "processor": type(proc).__name__, "items": {}}
    try:
        res["max_tokens_per_item"] = dict(proc.info.get_mm_max_tokens_per_item(8192, {"audio": 1, "image": 1}))
    except Exception as e:  # noqa: BLE001
        res["max_tokens_per_item"] = err(e)
    res["data_parser_resampler"] = getattr(getattr(proc.info.get_data_parser(), "audio_resampler", None), "method", None)
    mels = {}
    for name, path in wavs.items():
        with open(path, "rb") as f:
            x, sr = read_wav(f.read())
        try:
            out = run_vllm(proc, PROMPT, {"audio": [(x, sr)]})
            mel = out["mm_kwargs"]["audio"][0]["audio_features"].data
            tl = int(out["mm_kwargs"]["audio"][0]["audio_token_lens"].data)
            ids = out["prompt_token_ids"]
            res["items"][name] = {"ok": True, "seconds": len(x) / sr, "mel_shape": list(mel.shape),
                                  "mel_dtype": str(mel.dtype), "audio_token_lens": tl,
                                  "audio_pad_in_prompt": int(sum(1 for t in ids if t == 151669)),
                                  "tokens_per_second": tl / (len(x) / sr)}
            mels[name] = (mel, tl)
        except Exception as e:  # noqa: BLE001
            res["items"][name] = err(e)
    return res, mels


def hf_towers(model_dir):
    from safetensors import safe_open

    from tests.hf_ref import load as load_hf
    M, _ = load_hf()
    with open(os.path.join(model_dir, "config.json")) as f:
        ac = json.load(f)["audio_config"]
    acn = M._as_namespace(ac)
    se = M._build_speech_embeddings(acn)
    enc = M.MiMoAudioEncoder(acn)
    sd_se, sd_enc = {}, {}
    with safe_open(os.path.join(model_dir, "model_pp0_ep0_shard0.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("speech_embeddings."):
                sd_se[k[len("speech_embeddings."):]] = f.get_tensor(k)
            elif k.startswith("audio_encoder."):
                sd_enc[k[len("audio_encoder."):]] = f.get_tensor(k)
    m1 = se.load_state_dict(sd_se, strict=True)
    m2 = enc.load_state_dict(sd_enc, strict=False)
    tok_dir = os.path.join(model_dir, "audio_tokenizer")
    with open(os.path.join(tok_dir, "config.json")) as f:
        tcfg = M.MiMoAudioTokenizerConfig(**json.load(f))
    tok = M.MiMoAudioTokenizer(tcfg)
    with safe_open(os.path.join(tok_dir, "model.safetensors"), "pt") as f:
        sd_tok = {k: f.get_tensor(k) for k in f.keys() if k.startswith("encoder.")}
    m3 = tok.load_state_dict(sd_tok, strict=False)
    info = {"speech_embeddings_missing": list(m1.missing_keys), "audio_encoder_missing": list(m2.missing_keys),
            "audio_encoder_unexpected": list(m2.unexpected_keys)[:10],
            "tokenizer_missing": [k for k in m3.missing_keys if k.startswith("encoder.")][:20],
            "param_count_m": {"tokenizer_encoder": sum(p.numel() for p in tok.encoder.parameters()) / 1e6,
                              "speech_embeddings": sum(p.numel() for p in se.parameters()) / 1e6,
                              "audio_encoder": sum(p.numel() for p in enc.parameters()) / 1e6}}
    return M, se, enc, tok, info


def run_towers(M, se, enc, tok, mel, dtype):
    se, enc, tok = se.to(dtype), enc.to(dtype), tok.to(dtype)
    t0 = time.time()
    codes = M.tokenize_audio_batch([mel.to(dtype)], tok.encoder, segment_size=enc.audio_segment_size,
                                   device=torch.device("cpu"))[0]
    t1 = time.time()
    with torch.no_grad():
        emb = enc(speech_embeddings=se, audio_codes=codes)
    t2 = time.time()
    return codes, emb, {"tokenizer_s": round(t1 - t0, 3), "encoder_s": round(t2 - t1, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    wavs = {n: os.path.join(a.assets, n) for n in sorted(os.listdir(a.assets)) if n.endswith(".wav")}
    res = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": torch.get_num_threads()}

    def write():
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)

    res["vllm_load_audio"] = {n: check_loader(p) for n, p in wavs.items()}
    try:
        res["front_end"], mels = front_end(a.model_dir, wavs)
    except Exception as e:  # noqa: BLE001
        res["front_end"], mels = err(e), {}
    write()
    try:
        M, se, enc, tok, info = hf_towers(a.model_dir)
        res["hf_towers"] = info
        res["hf_runs"] = {}
        for name, (mel, tl) in mels.items():
            r = {}
            c32, e32, t32 = run_towers(M, se, enc, tok, mel.float(), torch.float32)
            cbf, ebf, tbf = run_towers(M, se, enc, tok, mel.float(), torch.bfloat16)
            r["codes_shape"] = list(c32.shape)
            r["embeds_shape"] = list(e32.shape)
            r["tokens_match_processor"] = int(e32.shape[0]) == tl
            r["codes_bf16_eq_fp32_frac"] = float((c32 == cbf).float().mean())
            r["codes_bf16_eq_fp32_frac_by_quantizer"] = [round(float(v), 4) for v in (c32 == cbf).float().mean(0)]
            r["embeds_cos_bf16_vs_fp32"] = float(torch.nn.functional.cosine_similarity(
                e32.float().flatten(), ebf.float().flatten(), dim=0))
            r["time_fp32"], r["time_bf16"] = t32, tbf
            res["hf_runs"][name] = r
            write()
    except Exception as e:  # noqa: BLE001
        res["hf_towers"] = err(e)
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write()
    print(json.dumps({k: v for k, v in res.items() if k != "hf_towers"}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
