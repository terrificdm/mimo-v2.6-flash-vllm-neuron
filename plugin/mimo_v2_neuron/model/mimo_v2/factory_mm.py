# SPDX-License-Identifier: Apache-2.0
"""Factory for ``MiMoV2OmniForCausalLM`` (MiMo-V2 text + image / video input) on vllm-neuron.

See docs/design-notes.md#omni-architecture. Registered by the plugin next to the text-only ``MiMoV2ForCausalLM``
(which is unchanged). vLLM's front end resolves this class too (the plugin overrides the architecture name in
every process), so the class carries what the front end needs:

  * vLLM's own MiMo-Omni processor / processing info / dummy-input builder
    (``vllm.model_executor.models.mimo_v2_omni``, imported, not modified) registered on this class, so the API
    server and the worker build identical inputs (``pixel_values`` / ``image_grid_thw``,
    ``pixel_values_videos`` / ``video_grid_thw``);
  * ``SupportsMultiModal`` (``supports_multimodal = True``, placeholder strings identical to vLLM's) so the
    vllm-neuron runner enables its on-device encoder-cache path;
  * vllm-neuron ``SupportsSpatialMerge`` / ``SupportsMaxPixels`` class methods (bucket and request validation).

The worker only calls :meth:`from_configs`, which validates the text config exactly like the text factory and
the vision config (vision TP must divide the 32 Q / 8 KV heads, buckets / block size), then builds
:class:`~.model_mm.MiMoV2OmniForCausalLMNeuron`.
"""

from __future__ import annotations

import json
import logging
import os

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mimo_v2_omni import (
    MiMoV2OmniDummyInputsBuilder,
    MiMoV2OmniMultiModalProcessor,
    MiMoV2OmniProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .factory import MiMoV2ForCausalLM, _runtime_settings, log_switches

logger = logging.getLogger(__name__)
_P = "MiMoV2 Omni Neuron: "

# serve.sh keeps vision_attention_block_size at 4096 patches (1024 merged tokens per encoder-cache block;
# docs/design-notes.md#vision-bucket)
VISION_BLOCK = 4096


def vision_buckets_env() -> list | None:
    """``MIMO_MM_VISION_BUCKETS`` (serve.sh) as an increasing list of patch counts, or None when unset.
    Unset keeps the default single 4096-patch bucket (buckets [4096]): no video budget override and no per-request
    block check (docs/design-notes.md#vision-request-blocks). A plain ``vllm serve`` gets it from
    ``--additional-config`` (``mm_patches.derive_vision_buckets_env``)."""
    s = os.environ.get("MIMO_MM_VISION_BUCKETS", "").strip()
    if not s:
        return None
    try:
        b = [int(x) for x in s.split(",")]
    except ValueError:
        raise ValueError(_P + f"MIMO_MM_VISION_BUCKETS must be comma-separated integers; got {s!r}") from None
    if any(x <= 0 or x % VISION_BLOCK for x in b) or b != sorted(set(b)):
        raise ValueError(_P + f"MIMO_MM_VISION_BUCKETS must be increasing multiples of {VISION_BLOCK}; got {s!r}")
    return b


def max_request_blocks(buckets) -> int:
    """Encoder-cache blocks the text prefill graph takes per request (vllm-neuron
    ``max_vision_blocks_per_request = ceil(max_bucket / merge / cache_block)`` with cache_block = VISION_BLOCK / merge)."""
    return -(-max(buckets) // VISION_BLOCK)


def item_blocks(modality: str, grid) -> int:
    """Encoder-cache blocks of one item: images / videos per :func:`model_mm.vision_block_plan`, audio 1."""
    if modality == "audio":
        return 1
    from .model_mm import vision_block_plan

    rows = [tuple(int(v) for v in r) for r in grid.reshape(-1, 3).tolist()]
    return sum(len(p) for p in vision_block_plan(rows, VISION_BLOCK, 4))


def check_request_blocks(blocks_by_item: list, limit: int) -> None:
    """``blocks_by_item``: [(modality, blocks)]; raises ValueError (HTTP 400) when the request needs more encoder-cache
    blocks than one text prefill step can take (vllm-neuron would raise ``Vision block truncation`` in the worker)."""
    total = sum(n for _, n in blocks_by_item)
    if total > limit:
        desc = ", ".join(f"{m} {n}" for m, n in blocks_by_item)
        raise ValueError(f"this request needs {total} vision/audio encoder blocks ({desc}; 1 block = "
                         f"{VISION_BLOCK // 4} tokens per image / video time group, 1 per audio) but the compiled "
                         f"configuration takes at most {limit} per request; send fewer or smaller images / videos")


def _audio_config_dict(hf_config):
    ac = getattr(hf_config, "audio_config", None)
    if not ac:
        return None
    return ac if isinstance(ac, dict) else ac.to_dict() if hasattr(ac, "to_dict") else dict(vars(ac))


def audio_model_config(hf_config):
    """``MiMoAudioConfig`` when config.json has an ``audio_config`` (``make_mm_checkpoint.py --audio``), else None.
    The tokenizer config is ``<model dir>/audio_tokenizer/config.json``."""
    ac = _audio_config_dict(hf_config)
    if ac is None:
        return None
    from .audio_encoder import MiMoAudioConfig

    path = os.path.join(getattr(hf_config, "_name_or_path", "") or "", "audio_tokenizer", "config.json")
    if not os.path.isfile(path):
        raise ValueError(_P + f"audio_config is set but {path} is missing (make_mm_checkpoint.py --audio copies it)")
    with open(path) as f:
        return MiMoAudioConfig.from_configs(ac, json.load(f))


class MiMoV2NeuronProcessingInfo(MiMoV2OmniProcessingInfo):
    """vLLM's processing info + audio (docs/design-notes.md#audio-on-device): the encoder budget gets an ``audio`` entry (vLLM's only lists image
    and video) and audio resampling uses scipy (the default PyAV is not installed)."""

    def get_data_parser(self):
        from vllm.multimodal.parse import MultiModalDataParser

        return MultiModalDataParser(target_sr=24000.0, audio_resample_method="scipy")

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        out = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        if _audio_config_dict(self.get_hf_config()) is not None:
            from .audio_encoder import MiMoAudioConfig

            out["audio"] = MiMoAudioConfig().g          # tokens of the largest mel bucket (32 s)
        buckets = vision_buckets_env()
        if buckets is not None and "video" in out:
            # vLLM's video estimate (14 frames of 1024x1024, ~14336 tokens) sets the encoder budget / cache size;
            # videos between that and the request limit max_bucket / merge would pass validation but never be
            # scheduled for encoding (docs/design-notes.md#vision-request-blocks).
            from ...mm_patches import merge_factor

            out["video"] = max(int(out["video"]), max(buckets) // merge_factor(self.get_hf_config()))
        return out


class MiMoV2NeuronDummyInputsBuilder(MiMoV2OmniDummyInputsBuilder):
    """vLLM's dummy inputs + a 30 s silent audio item per audio slot (profiling only)."""

    AUDIO_DUMMY_SAMPLES = 30 * 24000

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        d = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
        n = mm_counts.get("audio", 0)
        if n:
            d["audio"] = self._get_dummy_audios(length=self.AUDIO_DUMMY_SAMPLES, num_audios=n,
                                                overrides=(mm_options or {}).get("audio"))
        return d


class MiMoV2NeuronMultiModalProcessor(MiMoV2OmniMultiModalProcessor):
    """Rejects audio longer than the Neuron mel bucket in the API process (HTTP 400) instead of in the worker.
    With ``MIMO_MM_VISION_BUCKETS`` set it also rejects requests whose images / videos / audios need more
    encoder-cache blocks than one text prefill step takes (docs/design-notes.md#vision-request-blocks)."""

    _BLOCK_MEMO_MAX = 4096

    def apply(self, *args, **kwargs):
        out = super().apply(*args, **kwargs)
        buckets = vision_buckets_env()
        if buckets is not None:
            check_request_blocks(self._request_blocks(out), max_request_blocks(buckets))
        return out

    def _request_blocks(self, out) -> list:
        """[(modality, blocks)] for every item of a processed request. Items served from the sender cache come back
        as None (no grid): their count is remembered by mm hash, else bounded below by ceil(embeds / block tokens)."""
        memo = self.__dict__.setdefault("_mimo_block_memo", {})
        kw, hashes, ph = out.get("mm_kwargs") or {}, out.get("mm_hashes") or {}, out.get("mm_placeholders") or {}
        res = []
        for modality in ("image", "video", "audio"):
            items = list(kw.get(modality) or [])
            hs = list(hashes.get(modality) or [None] * len(items))
            pos = list(ph.get(modality) or [])
            for i, item in enumerate(items):
                h = hs[i] if i < len(hs) else None
                if item is not None:
                    n = 1 if modality == "audio" else item_blocks(modality, item.get_data()[f"{modality}_grid_thw"])
                    if h is not None:
                        if len(memo) >= self._BLOCK_MEMO_MAX:
                            memo.pop(next(iter(memo)))
                        memo[h] = n
                elif h in memo:
                    n = memo[h]
                elif modality == "audio":
                    n = 1
                else:
                    emb = pos[i].get_num_embeds() if i < len(pos) else 0
                    n = max(1, -(-int(emb) // (VISION_BLOCK // 4)))
                res.append((modality, n))
        return res

    def _call_hf_processor(self, *args, **kwargs):
        out = super()._call_hf_processor(*args, **kwargs)
        lens = out.get("audio_token_lens") if hasattr(out, "get") else None
        if lens is not None:
            from .audio_encoder import MiMoAudioConfig

            cfg = MiMoAudioConfig()
            for t in (lens.tolist() if hasattr(lens, "tolist") else list(lens)):
                if int(t) > cfg.g:
                    raise ValueError(f"audio of {int(t)} tokens exceeds the Neuron limit of {cfg.g} tokens "
                                     f"({cfg.mel_bucket / 100:.0f} s)")
        return out


def _vision_config(hf_config):
    vc = getattr(hf_config, "vision_config", None)
    if not vc:
        raise ValueError(_P + "config.json has no vision_config; build the directory with tools/make_mm_checkpoint.py")
    return vc


@MULTIMODAL_REGISTRY.register_processor(MiMoV2NeuronMultiModalProcessor, info=MiMoV2NeuronProcessingInfo,
                                        dummy_inputs=MiMoV2NeuronDummyInputsBuilder)
class MiMoV2OmniForCausalLM(MiMoV2ForCausalLM, SupportsMultiModal):
    """vLLM ModelRegistry entry for ``MiMoV2OmniForCausalLM`` (front-end interface + Neuron factory)."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs):
        raise NotImplementedError(_P + "the Neuron runner calls embed_multimodal on the built model")

    # vllm-neuron SupportsSpatialMerge / SupportsMaxPixels
    # (vllm-neuron looks these up in its own model list only; ``mm_patches`` routes its helpers here)
    @classmethod
    def get_vision_token_merge_factor(cls, hf_config) -> int:
        from ...mm_patches import merge_factor

        _vision_config(hf_config)
        return merge_factor(hf_config)

    @classmethod
    def get_max_pixels_token_count(cls, hf_config, max_pixels: int) -> int:
        from ...mm_patches import max_pixels_token_count

        _vision_config(hf_config)
        return max_pixels_token_count(hf_config, max_pixels)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, *, text_neuron_config=None, vision_neuron_config=None):
        from ...runner_patches import apply as apply_runner_patches
        from .config import MiMoV2Config
        from .model_mm import MiMoV2OmniForCausalLMNeuron
        from .vision_encoder import MiMoVisionConfig

        if neuron_config is None:
            neuron_config = text_neuron_config
        rt = _runtime_settings()
        cfg = MiMoV2Config.from_configs(hf_config, neuron_config, max_model_len=rt["max_model_len"])
        cls._validate_config(cfg, neuron_config, rt)
        vcfg = MiMoVisionConfig.from_hf(_vision_config(hf_config))
        cls._validate_vision(vcfg, vision_neuron_config, cfg)
        acfg = audio_model_config(hf_config)
        apply_runner_patches()
        logger.info(_P + "building model (layers=%d, decode_moe_impl=%s, vision depth=%d, audio=%s)",
                    cfg.num_hidden_layers, cfg.decode_moe_impl, vcfg.depth, acfg is not None)
        log_switches(cfg, neuron_config, {"vision_buckets": list(vision_neuron_config.num_vision_tokens_buckets),
                                          "MIMO_MM_VISION_BUCKETS": os.environ.get("MIMO_MM_VISION_BUCKETS") or "<unset>",
                                          "audio": acfg is not None})
        return MiMoV2OmniForCausalLMNeuron(cfg, vcfg, vision_neuron_config, audio_cfg=acfg)

    @staticmethod
    def _validate_vision(vcfg, vnc, cfg) -> None:
        def bad(msg):
            logger.error(_P + msg)
            raise ValueError(_P + msg)

        if vnc is None:
            bad("vision_neuron_config is missing (the Neuron platform creates it for models with vision_config)")
        if vcfg.out_hidden_size != cfg.hidden_size:
            bad(f"vision out_hidden_size {vcfg.out_hidden_size} != text hidden_size {cfg.hidden_size}")
        tp = vnc.tp_size
        if vcfg.num_heads % tp or (tp % vcfg.num_key_value_heads and vcfg.num_key_value_heads % tp):
            bad(f"vision tp_size {tp} must divide num_heads {vcfg.num_heads} and divide / be divided by "
                f"num_key_value_heads {vcfg.num_key_value_heads}")
        bs = vnc.vision_attention_block_size
        if bs % vcfg.merge_unit:
            bad(f"vision_attention_block_size {bs} must be a multiple of {vcfg.merge_unit}")
        if not vnc.num_vision_tokens_buckets:
            bad("num_vision_tokens_buckets is not set")
        if quant := getattr(vnc, "quantization", None):
            bad(f"vision quantization {quant!r} is not supported (BF16 only)")
        env_buckets = vision_buckets_env()
        # the API process sizes its checks from the env, the graphs from vnc (docs/design-notes.md#vision-request-blocks)
        if env_buckets is not None:
            if sorted(int(b) for b in vnc.num_vision_tokens_buckets) != env_buckets:
                bad(f"MIMO_MM_VISION_BUCKETS {env_buckets} != vision_neuron_config.num_vision_tokens_buckets "
                    f"{vnc.num_vision_tokens_buckets}")
            if bs != VISION_BLOCK:
                bad(f"MIMO_MM_VISION_BUCKETS assumes vision_attention_block_size {VISION_BLOCK}, got {bs}")
