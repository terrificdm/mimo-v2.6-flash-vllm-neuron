# SPDX-License-Identifier: Apache-2.0
"""Factory for the MiMo-V2 Neuron model: validates the configuration, then builds the model."""

from __future__ import annotations

import logging
import os

import torch.nn as nn

from ... import switches
from .config import DECODE_MOE_IMPLS, KEY_BLOCK_ENVS, MiMoV2Config

logger = logging.getLogger(__name__)
_P = "MiMoV2 Neuron: "
SEGMENT_SIZES = (512, 1024, 2048, 4096)   # segmented prefill sizes accepted (vllm-neuron bucket_utils range)


def _runtime_settings() -> dict:
    """Settings that live outside NeuronConfig (read from the current vLLM config)."""
    from vllm.config import get_current_vllm_config

    vc = get_current_vllm_config()
    try:
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_degree
        ep = get_neuron_ep_degree()
    except Exception:  # parallel state not initialised (unit tests)
        ep = 1
    return {
        "tp": vc.parallel_config.tensor_parallel_size,
        "ep": ep,
        "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
        "speculative_config": vc.speculative_config,
        "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
        "max_model_len": vc.model_config.max_model_len if vc.model_config is not None else None,
        "block_size": vc.cache_config.block_size,
    }


def log_switches(cfg: MiMoV2Config, neuron_config, extra: dict | None = None) -> None:
    """One log line with every plugin switch (value and source) and what the model actually uses."""
    fp8 = bool(getattr(cfg, "fp8_weights", False))
    used = {"fp8_weights": fp8,
            "fp8_linear": (cfg.fp8_act or {}).get("impl") if fp8 else "n/a (BF16 weights)",
            "fp8_experts": cfg.fp8_expert_impl if fp8 else "n/a (BF16 weights)",
            "moe_cte": cfg.moe_cte_impl if fp8 and cfg.fp8_expert_impl == "row" else "n/a",
            "kv_segment": getattr(neuron_config, "kv_segment_size_buckets", None),
            "prefill_buckets": getattr(neuron_config, "num_batched_tokens_buckets", None)}
    used.update(extra or {})
    switches.log_summary(used)


class MiMoV2ForCausalLM(nn.Module):
    """vLLM ModelRegistry entry for ``MiMoV2ForCausalLM`` (factory pattern, see onboarding guide 1c).

    The Neuron runner only calls ``from_configs``. vLLM's front end, however, inspects the registered
    class while building ``ModelConfig`` (``is_text_generation_model``: ``__init__`` accepting
    ``vllm_config``, callable ``embed_input_ids`` / ``compute_logits``, ``forward(input_ids, positions)``)
    and rejects it with "does not support `--runner generate`" otherwise. The built-in vllm-neuron models
    pass because their architectures are registered in the front end by vLLM's own GPU classes; this
    architecture name is overridden by the plugin in every process, so the factory carries the interface
    itself. The stubs are never called on Neuron.
    """

    def __init__(self, hf_config=None, neuron_config=None, *, vllm_config=None, prefix: str = "") -> None:
        super().__init__()
        if hf_config is None and vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
        self._model = self.from_configs(hf_config, neuron_config)

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        raise NotImplementedError(_P + "embed_input_ids is not used by the Neuron runner")

    def compute_logits(self, hidden_states):
        raise NotImplementedError(_P + "compute_logits is not used by the Neuron runner")

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, *, text_neuron_config=None,
                     vision_neuron_config=None) -> nn.Module:
        # The HF MiMoV2Config always has a ``vision_config`` attribute (``{}`` after conversion), so the
        # Neuron platform creates a default vision_neuron_config and the runner calls
        # ``from_configs(hf_config=..., text_neuron_config=..., vision_neuron_config=...)``. This port is
        # text-only (the vision tower is dropped at conversion) and ignores the vision config.
        if neuron_config is None:
            neuron_config = text_neuron_config
        rt = _runtime_settings()
        try:
            cfg = MiMoV2Config.from_configs(hf_config, neuron_config, max_model_len=rt["max_model_len"])
        except ValueError as e:   # e.g. non-integer MIMO_ATTN_*_KEY_BLOCK
            logger.error(str(e))
            raise
        cls._validate_config(cfg, neuron_config, rt)
        from ...runner_patches import apply as apply_runner_patches
        from .model import MiMoV2ForCausalLMNeuron

        apply_runner_patches()

        logger.info("MiMoV2 Neuron: building model (layers=%d, decode_moe_impl=%s, prefill_key_block=%d, "
                    "decode_key_block=%d, fp8_expert_impl=%s, moe_cte_impl=%s)", cfg.num_hidden_layers,
                    cfg.decode_moe_impl, cfg.prefill_key_block, cfg.decode_key_block, cfg.fp8_expert_impl,
                    cfg.moe_cte_impl)
        log_switches(cfg, neuron_config)
        return MiMoV2ForCausalLMNeuron(cfg)

    @staticmethod
    def _validate_config(cfg: MiMoV2Config, neuron_config, rt: dict) -> None:
        def bad(msg):
            logger.error(_P + msg)
            raise ValueError(_P + msg)

        if cfg.decode_moe_impl not in DECODE_MOE_IMPLS:
            bad(f"MIMO_DECODE_MOE_IMPL must be one of tkg/cte/torch, got {cfg.decode_moe_impl!r}")
        if cfg.has_quantization_config:
            bad("checkpoint still has quantization_config; convert it to BF16 first with "
                "tools/convert_checkpoint.py")
        if cfg.attention_projection_layout != "split":
            bad(f"attention_projection_layout must be 'split' (got {cfg.attention_projection_layout!r}); "
                "use the checkpoint produced by tools/convert_checkpoint.py")
        tp = rt["tp"]
        for name, nh, nkv in (("full", cfg.num_attention_heads, cfg.num_key_value_heads),
                              ("swa", cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads)):
            if nh % tp:
                bad(f"{name} attention heads ({nh}) must be divisible by tensor_parallel_size ({tp})")
            if tp % nkv and nkv % tp:
                bad(f"tensor_parallel_size ({tp}) and {name} KV heads ({nkv}) must divide one another")
        if cfg.n_routed_experts % rt["ep"]:
            bad(f"n_routed_experts ({cfg.n_routed_experts}) must be divisible by the EP degree ({rt['ep']})")
        segs = getattr(neuron_config, "kv_segment_size_buckets", None) if neuron_config is not None else None
        if rt["enable_prefix_caching"] and not segs:   # prefix caching (docs/limitations.md): the segmented prefill path reads any cached prefix
            bad("prefix caching needs segmented prefill (kv_segment_size_buckets); otherwise pass "
                "--no-enable-prefix-caching")
        if segs:
            segs = list(segs)
            if len(segs) != 1 or segs[0] not in SEGMENT_SIZES:
                bad(f"kv_segment_size_buckets must be a single value in {list(SEGMENT_SIZES)}, got {segs}")
            nbt = getattr(neuron_config, "num_batched_tokens_buckets", None)
            pb = switches.get("MIMO_PREFILL_BUCKETS")
            if pb == "auto":   # default: 512 + segment for segmented prefill (switches.default_prefill_buckets)
                auto = switches.default_prefill_buckets(segs[0], nbt, rt["max_model_len"])
                pb = ",".join(str(v) for v in auto) if auto else None
            if pb:   # MIMO_PREFILL_BUCKETS: smaller prefill buckets below the segment size (attention is shape-generic)
                try:
                    vals = sorted({int(x) for x in pb.split(",") if x.strip()})
                except ValueError:
                    bad(f"MIMO_PREFILL_BUCKETS must be comma-separated integers, got {pb!r}")
                if not vals or vals[-1] != segs[0] or any(v not in SEGMENT_SIZES for v in vals):
                    bad(f"MIMO_PREFILL_BUCKETS ({pb}) must be values in {list(SEGMENT_SIZES)} ending with the "
                        f"segment size {segs[0]}")
                if nbt is None or list(nbt) != segs:
                    bad(f"num_batched_tokens_buckets ({nbt}) must equal kv_segment_size_buckets ({segs}) "
                        "before MIMO_PREFILL_BUCKETS is applied")
                neuron_config.num_batched_tokens_buckets = vals
                logger.info(_P + "prefill buckets %s (kv_segment_size %d; MIMO_PREFILL_BUCKETS)", vals, segs[0])
                nbt = segs
            if nbt is None or list(nbt) != segs:
                bad(f"num_batched_tokens_buckets ({nbt}) must equal kv_segment_size_buckets ({segs}) "
                    "for segmented prefill")
        if rt["speculative_config"] is not None:
            bad("speculative decoding / MTP is not supported in this version")
        mnbt, mml = rt["max_num_batched_tokens"], rt["max_model_len"]
        if mnbt is not None and mml is not None and mnbt < mml and not segs:
            bad(f"max_num_batched_tokens ({mnbt}) must be >= max_model_len ({mml}) unless segmented "
                "prefill (kv_segment_size_buckets) is enabled")
        bs = rt.get("block_size") or 1
        for name, val in zip(KEY_BLOCK_ENVS, (cfg.prefill_key_block, cfg.decode_key_block)):
            lo = max(128, bs)
            if val % bs or not lo <= val <= 65536:
                bad(f"{name} ({val}) must be a multiple of the KV block size ({bs}) within [{lo}, 65536]")
