# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for run_logit_validation.py and fallback_check.py (docs/design-notes.md#validation-rules).

Environment handling: ``--decode-moe-impl`` is a required argument and is written to
``os.environ["MIMO_DECODE_MOE_IMPL"]`` by :func:`apply_decode_moe_impl` *before* vLLM is imported or
the ``LLM`` is created, so the engine worker processes inherit it. Do not rely on a shell ``export``.

Exit codes used by both scripts: 0 pass, 1 accuracy criterion failed, 2 bad arguments (argparse),
3 configuration error (wrong logprob count, wrong number of generated tokens, length limits, ...),
4 unexpected exception (traceback in the log).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

STEPS = 16
MAX_MODEL_LEN = 1024          # default --max-model-len of the first L3 graphs (docs/design-notes.md#validation-rules)
SEGMENT_SIZES = (0, 512, 1024, 2048, 4096)   # --kv-segment-size: 0 = no segmented prefill
BATCH_SEQS = {"b1": 1, "b2": 2, "b4": 4}     # --batch-mode -> max_num_seqs / prompts per batch
DEFAULT_L3 = os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16-L3")
DEFAULT_GOLDENS = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "goldens/L3")
DEFAULT_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
ENV_KEYS = ("MIMO_DECODE_MOE_IMPL", "VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR", "VLLM_NEURON_DISABLE_NKI_KERNELS",
            "VLLM_CACHE_ROOT", "VLLM_NEURON_COMPILATION_TIMEOUT", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
            "VLLM_ENGINE_READY_TIMEOUT_S")


class ConfigError(RuntimeError):
    """A configuration problem (not an accuracy failure); scripts exit with code 3."""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--batch-mode", required=True, choices=list(BATCH_SEQS),
                    help="b1: each prompt alone; b2 / b4: the 2 / 4 prompts of the set in one batch")
    ap.add_argument("--decode-moe-impl", required=True, choices=["tkg", "cte", "torch"],
                    help="written to os.environ['MIMO_DECODE_MOE_IMPL'] before the LLM is created")
    ap.add_argument("--model-dir", default=DEFAULT_L3)
    ap.add_argument("--goldens", default=DEFAULT_GOLDENS, help="directory with goldens.pt (gen_goldens.py)")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--tp", type=int, default=64)
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--num-gpu-blocks-override", type=int, default=None,
                    help="cap the KV cache size: needed for CPU-mode drills (CPU mode sizes the KV cache from host "
                         "memory) and on device for the L3 model (otherwise vLLM fills the free HBM "
                         "with KV blocks and the decode graph exceeds the 24 GB per-core HBM limit)")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                    help=f"max_model_len (default {MAX_MODEL_LEN}, base configuration)")
    ap.add_argument("--kv-segment-size", type=int, default=0, choices=SEGMENT_SIZES,
                    help="segmented prefill: max_num_batched_tokens, num_batched_tokens_buckets and "
                         "kv_segment_size_buckets set to this value (default 0 = off, base configuration)")


def apply_decode_moe_impl(impl: str) -> None:
    os.environ["MIMO_DECODE_MOE_IMPL"] = impl


def env_snapshot() -> dict:
    return {k: os.environ.get(k) for k in ENV_KEYS}


def load_inputs(a):
    """Returns (names, ids{name: [int]}, goldens dict)."""
    spec = json.load(open(a.prompts))
    if a.batch_mode not in spec["sets"]:
        raise ConfigError(f"{a.prompts} has no prompt set {a.batch_mode!r} (sets: {list(spec['sets'])})")
    names = list(spec["sets"][a.batch_mode])
    if a.batch_mode != "b1" and len(names) != BATCH_SEQS[a.batch_mode]:
        raise ConfigError(f"{a.batch_mode} set must have {BATCH_SEQS[a.batch_mode]} prompts, got {names}")
    g = torch.load(os.path.join(a.goldens, "goldens.pt"), weights_only=False)
    missing = [n for n in names if n not in g["names"]]
    if missing:
        raise ConfigError(f"goldens in {a.goldens} have no entry for prompts {missing}")
    ids = {n: list(g["input_ids"][n][0]) for n in names}
    for n in names:
        if len(ids[n]) + 2 * STEPS > a.max_model_len:
            raise ConfigError(f"{n}: len(prompt)={len(ids[n])} + {2 * STEPS} > {a.max_model_len}")
        for key in ("fp32_logits", "bf16_logits"):
            shp = tuple(g[key][n].shape)
            if shp != (STEPS, 1, g["vocab_size"]):
                raise ConfigError(f"goldens {key}[{n}] shape {shp} != ({STEPS}, 1, {g['vocab_size']})")
    return names, ids, g


def build_llm(a, max_num_seqs: int, extra_neuron_config: dict | None = None, extra_additional_config: dict | None = None,
              extra_kwargs: dict | None = None):
    """extra_additional_config / extra_kwargs: multimodal validation (vision_neuron_config, limit_mm_per_prompt)."""
    from vllm import LLM
    seg = a.kv_segment_size
    if seg and seg >= a.max_model_len:
        raise ConfigError(f"--kv-segment-size {seg} must be < --max-model-len {a.max_model_len}")
    nbt = seg or a.max_model_len
    seg_cfg = {"kv_segment_size_buckets": [seg]} if seg else {}
    kw = dict(
        model=a.model_dir, trust_remote_code=True,
        tensor_parallel_size=a.tp,
        # vllm-neuron builds EP groups only for ep_degree > 1, so TP=1 runs without expert parallelism
        enable_expert_parallel=a.tp > 1,
        max_model_len=a.max_model_len, max_num_batched_tokens=nbt, max_num_seqs=max_num_seqs,
        block_size=32, enable_prefix_caching=False, async_scheduling=False,
        max_logprobs=-1, logprobs_mode="raw_logits",
        additional_config={"neuron_config": {
            "on_device_sampling_config": {},
            "num_batched_tokens_buckets": [nbt],
            "num_seqs_buckets": [max_num_seqs], **seg_cfg, **(extra_neuron_config or {})},
            **(extra_additional_config or {})},
        **(extra_kwargs or {}),
    )
    if a.num_gpu_blocks_override is not None:
        kw["num_gpu_blocks_override"] = a.num_gpu_blocks_override
    log("LLM kwargs:", json.dumps({k: v for k, v in kw.items()}, default=str))
    return LLM(**kw)


# ── multimodal goldens (gen_mm_goldens.py) for capture_routing.py / fallback_check.py --mm (docs/design-notes.md#vit-accuracy) ──
IMAGE_TOKEN = 151655
AUDIO_TOKEN = 151669


def add_mm_args(ap: argparse.ArgumentParser) -> None:
    """Same vision settings as run_mm_validation.py (its defaults, explicit encoder cache blocks: docs/design-notes.md#encoder-cache-blocks)."""
    ap.add_argument("--mm", action="store_true", help="goldens from gen_mm_goldens.py: every request carries its image")
    ap.add_argument("--audio", action="store_true",
                    help="goldens from gen_audio_lm_goldens.py: every request carries its audio (implies --mm)")
    ap.add_argument("--vision-block", type=int, default=4096)
    ap.add_argument("--vision-tp", type=int, default=8)
    ap.add_argument("--vision-dp", type=int, default=8)
    ap.add_argument("--vision-cache-blocks", type=int, default=17)


def mm_llm_kwargs(a) -> dict:
    """build_llm keyword arguments for --mm (empty dict without it)."""
    if not (getattr(a, "mm", False) or getattr(a, "audio", False)):
        return {}
    audio = getattr(a, "audio", False)
    vnc = {"num_vision_tokens_buckets": [a.vision_block], "vision_attention_block_size": a.vision_block,
           "tp_size": a.vision_tp, "dp_size": a.vision_dp, "encoder_cache_num_blocks": a.vision_cache_blocks}
    return {"extra_additional_config": {"vision_neuron_config": vnc},
            "extra_kwargs": {"limit_mm_per_prompt": {"image": 0 if audio else 1, "video": 0, "audio": 1 if audio else 0}}}


def collapse_image_tokens(ids, image_token: int = IMAGE_TOKEN) -> list:
    """Expanded prompt ids -> one placeholder per image run (vLLM re-expands token prompts)."""
    out = []
    for t in ids:
        if not (t == image_token and out and out[-1] == image_token):
            out.append(t)
    return out


def request_builder(a, g):
    """fn(name, token_ids) -> llm.generate input; with --mm adds the prompt's image (golden ``mm`` entry)."""
    if not (getattr(a, "mm", False) or getattr(a, "audio", False)):
        return lambda name, token_ids: {"prompt_token_ids": list(token_ids)}
    if "mm" not in g:
        raise ConfigError(f"{a.goldens}/goldens.pt has no 'mm' entry (use gen_mm_goldens.py)")
    if getattr(a, "audio", False):
        import wave

        import numpy as np

        def _wav(p):
            with wave.open(p, "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
                return x, float(w.getframerate())
        audios = {n: _wav(e["audio"]) for n, e in g["mm"].items()}
        return lambda name, token_ids: {"prompt_token_ids": collapse_image_tokens(token_ids, AUDIO_TOKEN),
                                        "multi_modal_data": {"audio": audios[name]}}
    from PIL import Image

    images = {n: Image.open(e["image"]).convert("RGB") for n, e in g["mm"].items()}
    return lambda name, token_ids: {"prompt_token_ids": collapse_image_tokens(token_ids),
                                    "multi_modal_data": {"image": images[name]}}


def model_vocab(model_dir: str) -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_dir, trust_remote_code=True).vocab_size


def sampling_params(vocab: int, max_tokens: int):
    from vllm import SamplingParams
    # ignore_eos: vLLM 0.24 adds generation_config eos ids to the stop set; the HF golden never stops
    return SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=vocab, detokenize=False)


def completion_logits(completion, vocab: int, max_tokens: int, tag: str):
    """Dense raw logits [max_tokens, vocab] and token ids for one completion; ConfigError on any mismatch."""
    toks = list(completion.token_ids)
    lps = completion.logprobs or []
    if len(toks) != max_tokens or len(lps) != max_tokens:
        raise ConfigError(f"{tag}: expected {max_tokens} generated tokens with logprobs, "
                          f"got {len(toks)} tokens / {len(lps)} logprob steps")
    rows = []
    for s, d in enumerate(lps):
        if len(d) != vocab:
            raise ConfigError(f"{tag} step {s}: {len(d)} logprob entries != vocab_size {vocab}")
        ids = torch.tensor(list(d.keys()), dtype=torch.long)
        if int(ids.max()) >= vocab:
            raise ConfigError(f"{tag} step {s}: token id {int(ids.max())} >= vocab_size {vocab}")
        row = torch.full((vocab,), -float("inf"))
        row[ids] = torch.tensor([lp.logprob for lp in d.values()], dtype=torch.float32)
        rows.append(row)
    return torch.stack(rows, 0), toks


def to_py(x, max_len: int = 64):
    """JSON-friendly conversion; drops large arrays (full-vocab logits / error vectors)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.tolist() if x.numel() <= max_len else None
    if np is not None and isinstance(x, np.ndarray):
        return x.tolist() if x.size <= max_len else None
    if np is not None and isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            v2 = to_py(v, max_len)
            if v2 is not None or v is None:
                out[str(k)] = v2
        return out
    if isinstance(x, (list, tuple)):
        if len(x) > max_len and not all(isinstance(e, (int, float, str, bool)) for e in x):
            return None
        return [to_py(e, max_len) for e in x]
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return x


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    log(f"wrote {path}")


def run_main(main) -> None:
    """Run ``main`` -> exit code; ConfigError -> 3. os._exit avoids hangs on vLLM non-daemon threads."""
    import traceback
    try:
        rc = main()
    except SystemExit as e:  # argparse errors -> 2
        rc = e.code if isinstance(e.code, int) else 2
    except ConfigError as e:
        log("ERROR (config):", e)
        rc = 3
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        rc = 4
    log(f"exit code {rc}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
