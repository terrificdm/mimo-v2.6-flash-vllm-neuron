"""Plugin switches: defaults are the validated recommended configuration (docs/design-notes.md#plugin-defaults).

Originally every optimisation was off unless the serve command exported a ``MIMO_*`` variable, so the validated
service needed about a dozen of them. Now an unset switch takes the validated value, so a plain ``vllm serve``
(only vLLM command-line arguments and ``--additional-config``) runs the validated configuration. Setting a
variable still wins, so every old implementation stays reachable for troubleshooting (for example
``MIMO_FP8_IMPL=emul``, ``MIMO_KV_WRITE=index_put``, ``MIMO_PREFILL_BUCKETS=off``).

Defaults that only make sense in part of the configurations follow "auto" rules at the read site and fall back to
the old implementation (with a warning) when the rule does not hold:

- FP8 linear / expert implementations: FP8 checkpoints only (the switches are not read for BF16 weights);
  ``MIMO_FP8_IMPL`` falls back to ``emul`` when ``MIMO_FP8_ACT_MAX`` is not 448.
- ``MIMO_MOE_CTE_IMPL``: only used by the FP8 ``row`` experts.
- ``MIMO_PREFILL_ATTN`` / ``MIMO_PREFILL_KIDX``: segmented prefill only; the kernel is skipped per layer when the
  shapes do not fit (one KV head per rank, 192-wide K, 512 % block_size == 0).
- ``MIMO_PREFILL_BUCKETS``: segmented prefill with a segment > 512 and < max_model_len gets ``512,<segment>``.
- ``MIMO_KV_VNOPAD``: only when head_dim + v_head_dim is even.
- ``MIMO_VIT_DP_GROUP``: only read when the checkpoint has a vision tower.
- ``MIMO_CC_MODULAR_MAC`` (+ ``MIMO_CC_MODULAR_MAC_MAXTOK``): decode graphs only (token-id first input <= 64 rows).

The vLLM CPU mode (``VLLM_NEURON_CPU_MODE=1``: CPU tests and tiny drills) keeps the old defaults, because the NKI
kernels and compiler options only exist on the device. :func:`summary` is the one log line with every value.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_P = "MiMoV2 Neuron: "

# name -> (validated default on the device, old default = CPU mode). None = unset (feature off).
DEFAULTS: dict[str, tuple[str | None, str | None]] = {
    "MIMO_FP8_IMPL": ("nki", "emul"),
    "MIMO_FP8_EXPERT_IMPL": ("row", "expand"),
    "MIMO_MOE_CTE_IMPL": ("hybrid", "shard_on_i"),
    "MIMO_KV_WRITE": ("nki", "index_put"),
    "MIMO_KV_VNOPAD": ("1", "0"),
    "MIMO_PREFILL_ATTN": ("nki", "torch"),
    "MIMO_PREFILL_KIDX": ("arith", "gather"),
    "MIMO_PREFILL_BUCKETS": ("auto", None),
    "MIMO_VIT_DP_GROUP": ("world", "0"),
    "MIMO_CC_MODULAR_MAC": ("100000000000000", None),
    "MIMO_CC_MODULAR_MAC_MAXTOK": ("64", None),
    "MIMO_DECODE_MOE_IMPL": ("tkg", "tkg"),
}
OFF = "off"   # explicit value that turns a default-on feature off (PREFILL_BUCKETS, CC_MODULAR_MAC, MAXTOK)


def cpu_mode() -> bool:
    """vllm-neuron CPU mode (``VLLM_NEURON_CPU_MODE``, parsed like ``vllm_neuron.envs``: int -> bool)."""
    v = os.environ.get("VLLM_NEURON_CPU_MODE")
    try:
        return bool(int(v)) if v not in (None, "") else False
    except ValueError:
        return False


def is_set(name: str) -> bool:
    return os.environ.get(name, "") != ""


def get(name: str) -> str | None:
    """The environment value when set, else the validated default (device) or the old default (CPU mode).
    ``off`` comes back as None for the switches whose old default is "unset"."""
    v = os.environ.get(name, "")
    if v == "":
        dev, old = DEFAULTS[name]
        v = old if cpu_mode() else dev
    if v == OFF and DEFAULTS[name][1] is None:
        return None
    return v


def source(name: str) -> str:
    if is_set(name):
        return "env"
    return "cpu-mode default" if cpu_mode() else "default"


def default_prefill_buckets(segment: int | None, num_batched_tokens_buckets, max_model_len: int | None):
    """``MIMO_PREFILL_BUCKETS=auto``: [512, segment] for segmented prefill whose segment is above 512 and below
    max_model_len and whose num_batched_tokens_buckets are still [segment]; None otherwise (no small bucket)."""
    if not segment or segment <= 512:
        return None
    if max_model_len is not None and segment >= max_model_len:
        return None
    if num_batched_tokens_buckets is None or list(num_batched_tokens_buckets) != [segment]:
        return None
    return [512, segment]


def summary(extra: dict | None = None) -> str:
    """``NAME=value(source)`` for every switch in :data:`DEFAULTS` plus ``extra`` (already resolved values)."""
    parts = []
    for n in DEFAULTS:
        v = get(n)
        parts.append(f"{n}={'<off>' if v is None else v}({source(n)})")
    for k, v in (extra or {}).items():
        parts.append(f"{k}={v}")
    return " ".join(parts)


def log_summary(extra: dict | None = None) -> None:
    logger.info(_P + "switches: %s", summary(extra))
