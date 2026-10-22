# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: registers the Neuron-native MiMoV2ForCausalLM (and, for multimodal input, MiMoV2OmniForCausalLM).

vLLM calls ``register()`` (entry point group ``vllm.general_plugins``) in every
process before the model class is resolved, so the architecture string from
config.json (``MiMoV2ForCausalLM``) resolves to this implementation instead of
vLLM's built-in GPU one. ``MiMoV2OmniForCausalLM`` (image / video input) is only used by checkpoint directories
built with ``tools/make_mm_checkpoint.py``; the text-only registration is unchanged.
"""

ARCH = "MiMoV2ForCausalLM"
# Point at the defining module: vLLM caches the inspected model interfaces keyed by the hash of this
# module's file ($VLLM_CACHE_ROOT/modelinfos), so the key must change whenever factory.py changes.
MODEL_CLS = "mimo_v2_neuron.model.mimo_v2.factory:MiMoV2ForCausalLM"
MM_ARCH = "MiMoV2OmniForCausalLM"
MM_MODEL_CLS = "mimo_v2_neuron.model.mimo_v2.factory_mm:MiMoV2OmniForCausalLM"


# Timeouts the 1M configuration needs (a cached start takes ~18 min, a first compile several hours;
# vLLM's defaults are 600 s / 300 s / 600 s). Set only when the environment does not already have them, so
# ``vllm serve`` works without exporting them (serve.sh still exports its own values).
TIMEOUT_DEFAULTS = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "21600",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "7200",
    "VLLM_NEURON_COMPILATION_TIMEOUT": "7200",
}


def default_timeouts() -> list[str]:
    """Fill in :data:`TIMEOUT_DEFAULTS`; returns the names set by this call. ``vllm.v1.engine.core_client`` copies
    VLLM_ENGINE_READY_TIMEOUT_S into a module global at import time (usually before plugins load), so that global
    is updated too."""
    import os
    import sys

    done = []
    for k, v in TIMEOUT_DEFAULTS.items():
        if os.environ.get(k, "") == "":
            os.environ[k] = v
            done.append(k)
    if "VLLM_ENGINE_READY_TIMEOUT_S" in done:
        cc = sys.modules.get("vllm.v1.engine.core_client")
        if cc is not None and hasattr(cc, "VLLM_ENGINE_READY_TIMEOUT_S"):
            cc.VLLM_ENGINE_READY_TIMEOUT_S = int(TIMEOUT_DEFAULTS["VLLM_ENGINE_READY_TIMEOUT_S"])
    return done


def register() -> None:
    import logging

    timeouts = default_timeouts()

    from vllm import ModelRegistry

    # vLLM warns when overriding its built-in MiMoV2ForCausalLM; that override is intended.
    reg_logger = logging.getLogger("vllm.model_executor.models.registry")
    prev = reg_logger.level
    reg_logger.setLevel(logging.ERROR)
    try:
        ModelRegistry.register_model(ARCH, MODEL_CLS)
        ModelRegistry.register_model(MM_ARCH, MM_MODEL_CLS)
    finally:
        reg_logger.setLevel(prev)
    from .mm_patches import apply as apply_mm_patches

    apply_mm_patches()   # vllm-neuron vision helpers: MiMoV2OmniForCausalLM only, other archs unchanged
    from .runner_patches import apply_scheduler_patch

    apply_scheduler_patch()   # MIMO_PREFILL_BUCKETS (default auto on the device) / MIMO_KV_CONCURRENCY
    if timeouts:
        logging.getLogger(__name__).info("MiMoV2 Neuron: default %s", " ".join(
            f"{k}={TIMEOUT_DEFAULTS[k]}" for k in timeouts))
