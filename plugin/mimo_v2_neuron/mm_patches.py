# SPDX-License-Identifier: Apache-2.0
"""Make vllm-neuron's vision helpers aware of the out-of-tree ``MiMoV2OmniForCausalLM`` (docs/design-notes.md#omni-architecture).

``vllm_neuron.utils.vision_utils.get_vision_token_merge_factor`` / ``get_max_pixels_token_count`` look the model
class up in vllm-neuron's own hard-coded model list (``vllm_neuron/model/registry.py:get_models``), not in vLLM's
``ModelRegistry``, so for a plugin model they return 1 / None. The runner then sizes encoder-cache blocks in raw
patches instead of merged tokens (tiny drill: ``shape mismatch: value tensor of shape [1, 256, 512] cannot be
broadcast to indexing result of shape [1, 1024, 512]``), and the platform's request validation and the
scheduler's encoder budget use the wrong factor.

Fix (installed files are not edited): wrap the two module functions so that, for ``MiMoV2OmniForCausalLM`` only,
they return the MiMo values (``spatial_merge_size ** 2`` and ``max_pixels // patch_size ** 2``, same maths as
Qwen3-VL); every other architecture goes to the original function. All callers import these names inside their
function bodies (``platform.py``, ``core/scheduler.py``, ``worker/neuron_model_runner.py``), so replacing the
module attributes is enough. Applied from the plugin's ``register()`` in every vLLM process.

It also patches ``vllm.multimodal.media.audio.load_audio``: the venv has neither soundfile nor PyAV, so WAV
requests fall back to a stdlib ``wave`` reader (:func:`patch_audio_loading`). With soundfile installed
outside the venv (``$MIMO_WORK/pydeps``) MP3 / FLAC / Ogg decode too; the WAV reader stays the fallback
(docs/design-notes.md#audio-input).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
ARCHS = ("MiMoV2OmniForCausalLM",)


def _arch(hf_config):
    archs = getattr(hf_config, "architectures", None) or [None]
    return archs[0]


def _vc_get(hf_config, key, default):
    vc = getattr(hf_config, "vision_config", None) or {}
    v = vc.get(key) if isinstance(vc, dict) else getattr(vc, key, None)
    return default if v is None else v


def merge_factor(hf_config) -> int:
    return int(_vc_get(hf_config, "spatial_merge_size", 2)) ** 2


def max_pixels_token_count(hf_config, max_pixels: int) -> int:
    return int(max_pixels) // int(_vc_get(hf_config, "patch_size", 16)) ** 2


def read_wav(src) -> tuple:
    """PCM WAV (8/16/32-bit int, mono or multi-channel) -> (float32 mono waveform, sample rate). Python stdlib +
    numpy only: the venv has neither soundfile nor PyAV, which vLLM's ``load_audio`` needs."""
    import io
    import wave

    import numpy as np

    f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with wave.open(f if hasattr(f, "read") else str(f), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw in (2, 4):
        x = np.frombuffer(raw, dtype="<i2" if sw == 2 else "<i4").astype(np.float32) / float(2 ** (8 * sw - 1))
    else:
        raise ValueError(f"unsupported WAV sample width {sw} bytes")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, float(sr)


def _head(path, n: int = 4) -> bytes:
    if hasattr(path, "read"):
        pos = path.tell()
        b = path.read(n)
        path.seek(pos)
        return b
    with open(path, "rb") as f:
        return f.read(n)


def sf_allowed(head: bytes) -> bool:
    """Containers handed to soundfile/libsndfile : WAV (RIFF/RIFX/RF64), FLAC, Ogg, MP3 (ID3 tag or MPEG
    frame sync). Anything else skips libsndfile, whose header parsers for rarer formats have known read-path CVEs
    (e.g. CVE-2025-52194, IRCAM, libsndfile 1.2.2 as bundled in soundfile 0.14.0)."""
    return head[:4] in (b"RIFF", b"RIFX", b"RF64", b"fLaC", b"OggS") or head[:3] == b"ID3" \
        or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)


def patch_audio_loading() -> list[str]:
    """``vllm.multimodal.media.audio.load_audio``: decode with the original (soundfile, from
    ``$MIMO_WORK/pydeps`` via serve.sh's PYTHONPATH) at the native rate, resampling here with
    ``resample_audio_scipy`` as before (PyAV, which the original resamples with, is not installed). When
    soundfile is missing, the container is not one of :func:`sf_allowed`, or decoding fails, read WAV with
    :func:`read_wav`; if that fails too the original error is raised. The duration limit is enforced on
    every path. ``AudioMediaIO`` looks the function up in its module at call time, so replacing the module
    attribute is enough."""
    try:
        import vllm.multimodal.media.audio as A
    except Exception as e:  # vllm not importable (pure unit tests)
        logger.debug("mimo audio loading patch skipped (%s)", e)
        return []
    orig = A.load_audio
    if getattr(orig, "_mimo_patched", False):
        return []

    def load_audio(path, *, sr=22050, mono=True, max_duration_s=None):
        err = None
        try:
            if not sf_allowed(_head(path)):
                raise ValueError("audio container not handed to soundfile")
            x, native = orig(path, sr=None, mono=mono, max_duration_s=max_duration_s)
        except Exception as e:
            if "maximum allowed duration" in str(e):
                raise
            err = e
        if err is not None:
            if hasattr(path, "seek"):
                path.seek(0)
            try:
                x, native = read_wav(path.read() if hasattr(path, "read") else path)
            except Exception:
                raise err
            if max_duration_s is not None and len(x) / native > max_duration_s:
                raise ValueError(f"Audio exceeds maximum allowed duration of {max_duration_s}s")
        if sr is not None and sr != native:
            from vllm.multimodal.audio import resample_audio_scipy
            x = resample_audio_scipy(x, orig_sr=native, target_sr=sr)
            native = float(sr)
        return x, native

    load_audio._mimo_patched = True
    A.load_audio = load_audio
    return ["load_audio"]


def apply() -> list[str]:
    done = patch_audio_loading()
    try:
        import vllm_neuron.utils.vision_utils as vu
    except Exception as e:  # vllm-neuron not importable (pure unit tests)
        logger.debug("mimo mm patches skipped (%s)", e)
        return done
    orig = vu.get_vision_token_merge_factor
    if not getattr(orig, "_mimo_patched", False):
        def get_vision_token_merge_factor(hf_config):
            if _arch(hf_config) not in ARCHS:
                return orig(hf_config)
            # The worker's runner calls this in __init__ (_init_encoder_cache), i.e. before load_model: the earliest
            # point to install the runner patches so that the load_model wrapper (audio compile hand-off, docs/design-notes.md#audio-on-device)
            # is in place when the model is loaded. No-op in processes that never imported the runner.
            from .runner_patches import apply as apply_runner_patches
            apply_runner_patches(import_if_missing=False)
            return merge_factor(hf_config)

        get_vision_token_merge_factor._mimo_patched = True
        vu.get_vision_token_merge_factor = get_vision_token_merge_factor
        done.append("get_vision_token_merge_factor")
    orig_px = vu.get_max_pixels_token_count
    if not getattr(orig_px, "_mimo_patched", False):
        def get_max_pixels_token_count(hf_config, max_pixels):
            return max_pixels_token_count(hf_config, max_pixels) if _arch(hf_config) in ARCHS \
                else orig_px(hf_config, max_pixels)

        get_max_pixels_token_count._mimo_patched = True
        vu.get_max_pixels_token_count = get_max_pixels_token_count
        done.append("get_max_pixels_token_count")
    done += patch_platform_vision_buckets()
    return done


VISION_BUCKETS_ENV = "MIMO_MM_VISION_BUCKETS"


def derive_vision_buckets_env(vllm_config) -> str | None:
    """When ``MIMO_MM_VISION_BUCKETS`` is unset, take it from ``--additional-config``
    ``vision_neuron_config.num_vision_tokens_buckets`` so a plain ``vllm serve`` gets the API-side video budget and
    per-request block check (docs/design-notes.md#vision-request-blocks) that serve.sh enabled by exporting the variable. The single bucket [4096] keeps
    the old behaviour (no variable, as serve.sh without MIMO_MM_VISION_BUCKETS), and so do the layouts the checks do
    not cover (``vision_attention_block_size`` other than 4096, buckets that are not multiples of 4096) and the vLLM
    CPU mode (tiny drills keep the old defaults, ``switches``). Exported to ``os.environ`` because
    the engine core and the workers are separate processes started after the config is built (they inherit it).
    Returns the value set, or None."""
    import os

    from .switches import cpu_mode

    if os.environ.get(VISION_BUCKETS_ENV, "").strip() or cpu_mode():
        return None
    mc = getattr(vllm_config, "model_config", None)
    if mc is None or _arch(mc.hf_config) not in ARCHS:
        return None
    vnc = (getattr(vllm_config, "additional_config", None) or {}).get("vision_neuron_config")
    b = vnc.get("num_vision_tokens_buckets") if isinstance(vnc, dict) else \
        getattr(vnc, "num_vision_tokens_buckets", None)
    if not b:
        return None
    bs = vnc.get("vision_attention_block_size") if isinstance(vnc, dict) else \
        getattr(vnc, "vision_attention_block_size", None)
    b = sorted(int(x) for x in b)
    if b == [4096] or int(bs or 4096) != 4096 or any(x % 4096 for x in b):
        return None
    val = ",".join(str(x) for x in b)
    os.environ[VISION_BUCKETS_ENV] = val
    logger.info("MiMoV2 Omni Neuron: %s=%s (from --additional-config vision_neuron_config.num_vision_tokens_buckets)",
                VISION_BUCKETS_ENV, val)
    return val


def patch_platform_vision_buckets() -> list[str]:
    """Wrap ``NeuronPlatform.check_and_update_config`` (runs in the API process when the vLLM config is built) to
    call :func:`derive_vision_buckets_env` after vllm-neuron has resolved ``vision_neuron_config``."""
    try:
        import vllm_neuron.vllm.platform as pm
    except Exception as e:  # vllm-neuron not importable (pure unit tests)
        logger.debug("mimo platform patch skipped (%s)", e)
        return []
    cls = getattr(pm, "NeuronPlatform", None)
    raw = cls.__dict__.get("check_and_update_config") if cls is not None else None
    func = getattr(raw, "__func__", None)
    if func is None or getattr(func, "_mimo_patched", False):
        return []

    def check_and_update_config(klass, vllm_config):
        out = func(klass, vllm_config)
        try:
            derive_vision_buckets_env(vllm_config)
        except Exception as e:  # never block startup; serve.sh users can still export the variable
            logger.warning("MiMoV2 Omni Neuron: could not derive %s (%s)", VISION_BUCKETS_ENV, e)
        return out

    check_and_update_config._mimo_patched = True
    cls.check_and_update_config = classmethod(check_and_update_config)
    return ["NeuronPlatform.check_and_update_config"]
