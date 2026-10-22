# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MiMo audio reference and tests (docs/design-notes.md#audio-criteria).

  AudioCkpt(src)               audio tensors of the original checkpoint under the plugin's key names:
                               ``audio_tokenizer.encoder.*`` (audio_tokenizer/model.safetensors, prefix added),
                               ``audio_encoder.*`` / ``speech_embeddings.*`` (model_pp0_ep0_shard0.safetensors)
  audio_config(src)            plugin ``MiMoAudioConfig`` from config.json audio_config + audio_tokenizer/config.json
  hf_towers(src)               HF modules (``tests/hf_ref.py`` patched remote code): speech embeddings, MiMoAudioEncoder,
                               MiMoAudioTokenizer (same as scripts/accuracy/audio_prep.py)
  hf_run(M, se, enc, tok, mel, dtype)  -> dict(features [T2, d], codes [T2, C], embeds [G, H]) for one mel
  our_audio(src, tp_group)     plugin ``MiMoAudioModel`` for one TP rank, weights through the plugin loaders
  served_mels(src, assets)     the mels vLLM's MiMo-Omni processor produces for the WAV assets
"""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "plugin"), os.path.join(ROOT, "scripts", "accuracy"),
           os.path.join(ROOT, "scripts", "api_tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOK_PREFIX = "audio_tokenizer."


class AudioCkpt:
    def __init__(self, src: str):
        from safetensors import safe_open

        self.src, self._open = src, safe_open
        tok = os.path.join(src, "audio_tokenizer", "model.safetensors")
        with open(os.path.join(src, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        self.map = {k: os.path.join(src, v) for k, v in wm.items()
                    if k.startswith(("audio_encoder.", "speech_embeddings.", TOK_PREFIX))}
        if os.path.isfile(tok):
            with safe_open(tok, "pt") as f:
                for k in f.keys():
                    if k.startswith("encoder."):
                        self.map.setdefault(TOK_PREFIX + k, tok)
        self._files = {}

    def _f(self, key):
        fn = self.map[key]
        if fn not in self._files:
            self._files[fn] = self._open(fn, "pt")
        return self._files[fn]

    def _k(self, key):
        fn = self.map[key]
        return key[len(TOK_PREFIX):] if fn.endswith(os.path.join("audio_tokenizer", "model.safetensors")) else key

    def slice(self, key):
        return self._f(key).get_slice(self._k(key))

    def tensor(self, key):
        return self._f(key).get_tensor(self._k(key))


def audio_config(src: str, mel_bucket: int = 3200):
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import MiMoAudioConfig

    with open(os.path.join(src, "config.json")) as f:
        ac = json.load(f).get("audio_config") or {}
    tok_cfg = os.path.join(src, "audio_tokenizer", "config.json")
    with open(tok_cfg) as f:
        tc = json.load(f)
    return MiMoAudioConfig.from_configs(ac, tc, mel_bucket=mel_bucket)


def hf_towers(src: str):
    from audio_prep import hf_towers as _h

    M, se, enc, tok, _ = _h(src)
    return M, se.eval(), enc.eval(), tok.eval()


@torch.no_grad()
def hf_run(M, se, enc, tok, mel: torch.Tensor, dtype) -> dict:
    import copy

    # copies: casting the shared FP32 modules in place would leave BF16-rounded codebooks for a later FP32 run
    se, enc, tok = (copy.deepcopy(m).to(dtype) for m in (se, enc, tok))
    e = tok.encoder
    L = mel.shape[0]
    lens = torch.tensor([L])
    feats = M._at_unpack_hidden_states(mel.to(dtype), lens)
    h, out_len, *_ = e.get_features(input_features=feats.transpose(1, 2), output_length=e.get_output_length(lens))
    e.quantizer.float()
    codes = e.quantizer.encode(h.float()).transpose(0, 1).contiguous()        # [T2, C]
    emb = enc(speech_embeddings=se, audio_codes=codes)
    return {"features": h, "codes": codes, "embeds": emb}


def our_audio(src: str, tp_group, ckpt: AudioCkpt | None = None, mel_bucket: int = 3200):
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import MiMoAudioModel
    from vllm_neuron.utils.weight_loader import get_weight_loader

    ckpt = ckpt or AudioCkpt(src)
    cfg = audio_config(src, mel_bucket)
    m = MiMoAudioModel(cfg, torch.bfloat16, tp_group=tp_group)
    params = dict(m.named_parameters())
    mapping = m.checkpoint_keys()
    if set(mapping) != set(params):
        raise ValueError(f"unmapped params: {sorted(set(params) ^ set(mapping))[:5]}")
    for pname, key in mapping.items():
        t = get_weight_loader(params[pname]).load([ckpt.slice(key)], tp_group.rank_in_group)
        if tuple(t.shape) != tuple(params[pname].shape):
            raise ValueError(f"{pname}: loaded {tuple(t.shape)} != param {tuple(params[pname].shape)}")
        if t.dtype != params[pname].dtype:
            raise ValueError(f"{pname}: checkpoint dtype {t.dtype} != param {params[pname].dtype}")
        params[pname].data = t
    return m.eval(), cfg


def served_mels(src: str, assets: str) -> dict:
    from audio_prep import PROMPT
    from make_audio_assets import read_wav
    from mm_prep import run_vllm, vllm_processor

    proc = vllm_processor(src)
    out = {}
    for n in sorted(os.listdir(assets)):
        if not n.endswith(".wav"):
            continue
        with open(os.path.join(assets, n), "rb") as f:
            x, sr = read_wav(f.read())
        o = run_vllm(proc, PROMPT, {"audio": [(x, sr)]})
        it = o["mm_kwargs"]["audio"][0]
        out[n] = (it["audio_features"].data, int(it["audio_token_lens"].data))
    return out
