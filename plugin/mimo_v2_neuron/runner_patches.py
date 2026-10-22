# SPDX-License-Identifier: Apache-2.0
# Patches vllm-project/vllm-neuron 0.24.0.1.1.0 (Apache-2.0) at runtime; contains short source excerpts of
#   vllm_neuron/vllm/worker/neuron_model_runner.py used to verify the patched code. See NOTICE.
"""Runtime fixes for three vllm-neuron 0.24.0.1.1.0 NeuronModelRunner crashes (engine-crash fixes).

All three crashes kill the whole EngineCore on a single valid OpenAI request, so they are patched at runtime
(installed package files are never edited). The patches are applied only when the MiMo-V2 model is built
(``MiMoV2ForCausalLM.from_configs``), so other models served from the same venv are unaffected.

1. ``seed`` -> ``_update_states`` calls ``torch.Generator(device=self.device)`` with the Neuron device,
   which torch-neuronx does not implement (NotImplementedError ``getNewGenerator`` for PrivateUse1).
   Fix: while ``_update_states`` runs, the module's ``torch`` resolves ``Generator`` to a factory that
   creates the generator on CPU. With CPU sampling (logits copied to CPU before the vLLM Sampler) the seed
   then takes effect; with on-device sampling the generator is unused (the device sampler has no RNG input)
   and the seed is ignored instead of crashing the engine.
2. on-device sampling + ``logprobs`` + temperature > 0 -> ``_sample`` copies the gathered logits to CPU and
   runs the full vLLM Sampler on them with the device-resident ``sampling_metadata``;
   ``Sampler.apply_temperature`` then fails with "Expected all tensors to be on the same device".
   That path also used the CPU sampler's own (re-sampled) token for the logprobs instead of the token the
   device emitted. Fix: compute the logprobs exactly as ``Sampler.forward`` does (raw logprobs / raw logits
   of the unmodified logits) but gather them for the device-sampled token, without sampling again.
3. CPU sampling (``on_device_sampling_config: null``, mode B) + temperature > 0 (or penalties, min_p,
   logit_bias, min_tokens) -> ``_sample`` runs the vLLM Sampler on the CPU logits (``_execute_model_forward``
   moves them to CPU) with the same device-resident ``sampling_metadata`` (the InputBatch is built with
   ``device=self.device``) and fails the same way in ``apply_temperature``. Fix: run the Sampler with a CPU
   copy of the metadata (tensor fields and the logits processors' tensors moved to CPU per step); the
   seeded generators are already on CPU (fix 1), so ``seed`` takes effect in this mode. In the same mode the
   CPU logits keep the decode padding rows (e.g. 2 requests in the 4-row bucket) while the metadata and the
   structured-output bitmask have one row per real request: the padding rows are dropped before sampling and
   the bitmask is padded with all-allowed rows in ``_apply_grammar_bitmask_cpu``.
4. (not a crash; docs/design-notes.md#cpu-resample) on-device sampling has no RNG input and no penalty inputs, so ``seed`` and presence /
   frequency / repetition penalties were silently ignored. When a batch contains such a request, ``_sample``
   re-samples it on CPU from the gathered logits with the vLLM Sampler and a CPU copy of the metadata (the
   mode-B path); other batches keep the device tokens. ``MIMO_ODS_CPU_RESAMPLE=0`` restores the old behaviour.
5. (diagnostics only) ``MIMO_DEBUG_METADATA=hit`` logs the ``cached_seq_len`` of prefill steps that
   start off a 4096-token boundary (prefix-cache hits) together with the request's block-table extent.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import os
import sys

import torch

from . import switches

logger = logging.getLogger(__name__)
_P = "MiMoV2 Neuron: "
RUNNER_MODULE = "vllm_neuron.vllm.worker.neuron_model_runner"
_SEED_SNIPPET = "torch.Generator(device=self.device)"
_LOGPROBS_SNIPPET = 'logits_cpu = self._on_device_logits.to("cpu")'
_CPU_SAMPLER_SNIPPET = 'logger.info("Using vLLM Sampler")'
_BITMASK_SNIPPET = 'return logits.masked_fill(~bitmask, float("-inf"))'
_RAW_MODES = ("raw_logprobs", "raw_logits")


def cpu_generator(device=None):
    """``torch.Generator`` replacement: any non-CPU/CUDA device (Neuron/XLA) gets a CPU generator."""
    if device is not None and torch.device(device).type not in ("cpu", "cuda"):
        device = "cpu"
    return torch.Generator() if device is None else torch.Generator(device=device)


class _TorchWithCPUGenerator:
    """Stand-in for the ``torch`` module inside the runner module: identical except ``Generator``."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def Generator(device=None):  # noqa: N802 (mirrors torch.Generator)
        return cpu_generator(device)


def device_token_logprobs(sampler, logits: torch.Tensor, token_ids: list[int], num_logprobs: int):
    """Logprobs for the device-sampled tokens, matching ``vllm.v1.sample.sampler.Sampler.forward``.

    ``logits``: gathered logits (any device), one row per entry of ``token_ids``; ``num_logprobs`` >= 0.
    """
    rows = logits[: len(token_ids)].to("cpu")
    if sampler.logprobs_mode == "raw_logprobs":
        raw = sampler.compute_logprobs(rows)
    else:  # raw_logits
        raw = rows.to(torch.float32)
    tok = torch.tensor(token_ids, dtype=torch.int64)
    return sampler.gather_logprobs(raw, num_logprobs, token_ids=tok)


def _wants_fixed_logprobs(runner, spec_decode_metadata, md) -> bool:
    n = getattr(md, "max_num_logprobs", None)
    return (
        spec_decode_metadata is None
        and runner.on_device_sampling
        and not runner.use_async_scheduling
        and runner._on_device_logits is not None
        and n is not None and n >= 0
        and not getattr(md, "logprob_token_ids", None)
        and getattr(runner.sampler, "logprobs_mode", None) in _RAW_MODES
    )


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t if t.device.type == "cpu" else t.to("cpu")


def _cpu_value(v):
    """Tensor -> CPU tensor; tuple of tensors -> tuple of CPU tensors; anything else unchanged (same object)."""
    if isinstance(v, torch.Tensor):
        return _cpu(v)
    if isinstance(v, tuple) and v and all(isinstance(x, torch.Tensor) for x in v):
        return tuple(_cpu(x) for x in v)
    return v


def _cpu_logitsproc(p):
    """Shallow copy of a logits processor whose tensor attributes are on CPU (used for one ``apply`` only;
    the original keeps receiving ``update_state``)."""
    changed = {k: nv for k, v in vars(p).items() if (nv := _cpu_value(v)) is not v}
    if not changed:
        return p
    q = copy.copy(p)
    for k, v in changed.items():
        setattr(q, k, v)
    return q


def cpu_sampling_metadata(md):
    """Copy of a vLLM ``SamplingMetadata`` with every tensor (incl. logits processors' state) on CPU."""
    changes = {}
    for f in dataclasses.fields(md):
        v = getattr(md, f.name)
        nv = _cpu_value(v)
        if nv is not v:
            changes[f.name] = nv
    lps = md.logitsprocs
    if lps is not None and hasattr(lps, "argmax_invariant"):
        new = copy.copy(lps)
        new.argmax_invariant = [_cpu_logitsproc(p) for p in lps.argmax_invariant]
        new.non_argmax_invariant = [_cpu_logitsproc(p) for p in lps.non_argmax_invariant]
        changes["logitsprocs"] = new
    return dataclasses.replace(md, **changes) if changes else md


def _wants_cpu_metadata(runner, spec_decode_metadata, logits) -> bool:
    return (
        spec_decode_metadata is None
        and not runner.on_device_sampling
        and not runner.use_async_scheduling
        and isinstance(logits, torch.Tensor) and logits.device.type == "cpu"
    )


def _wants_cpu_resample(runner, spec_decode_metadata, md) -> bool:
    """docs/design-notes.md#cpu-resample: on-device sampling has no RNG input and no penalties, so a batch with a seeded request or any
    presence / frequency / repetition penalty is re-sampled on CPU from the gathered logits with the vLLM Sampler
    (the same code path as mode B). ``MIMO_ODS_CPU_RESAMPLE=0`` turns this off (device tokens, seed / penalties
    ignored as before)."""
    return (
        os.environ.get("MIMO_ODS_CPU_RESAMPLE", "1") != "0"
        and spec_decode_metadata is None
        and runner.on_device_sampling
        and not runner.use_async_scheduling
        and runner._on_device_logits is not None
        and (bool(getattr(md, "generators", None)) or getattr(md, "no_penalties", True) is False)
        and not getattr(md, "logprob_token_ids", None)
    )


def _make_sample(orig, sampler_output_cls, cpu_fix=True):
    def _sample(self, model_output_tensor, spec_decode_metadata=None):
        md = self.input_batch.sampling_metadata
        if cpu_fix and _wants_cpu_metadata(self, spec_decode_metadata, model_output_tensor):
            # decode batches are padded up to the num_seqs bucket (e.g. 3 requests -> 4 logits rows); the
            # metadata has one row per real request, so drop the trailing padding rows
            n = getattr(self.input_batch, "num_reqs", None)
            logits = model_output_tensor[:n] if n and model_output_tensor.shape[0] > n else model_output_tensor
            return self.sampler(logits=logits, sampling_metadata=cpu_sampling_metadata(md))
        if cpu_fix and _wants_cpu_resample(self, spec_decode_metadata, md):
            tokens = model_output_tensor.tolist()
            n = getattr(self.input_batch, "num_reqs", None) or len(tokens)
            # copy: the Sampler edits logits in place, and the logprobs below read the unmodified rows. Move to the
            # CPU first, then cast: a device -> CPU copy with a dtype change fails on Neuron ("Expected
            # self.dtype() == dst.dtype()", 1M regression 14:57Z)
            logits = self._on_device_logits[:n].to("cpu").to(torch.float32).clone()
            out = self.sampler(logits=logits, sampling_metadata=dataclasses.replace(
                cpu_sampling_metadata(md), max_num_logprobs=None))
            tokens[:n] = out.sampled_token_ids.flatten().tolist()   # padding rows keep the device tokens
            n_lp = getattr(md, "max_num_logprobs", None)
            logprobs = None
            if n_lp is not None and n_lp >= 0 and getattr(self.sampler, "logprobs_mode", None) in _RAW_MODES:
                logprobs = device_token_logprobs(self.sampler, self._on_device_logits, tokens, n_lp)
            return sampler_output_cls([[x] for x in tokens], logprobs)
        if not _wants_fixed_logprobs(self, spec_decode_metadata, md):
            return orig(self, model_output_tensor, spec_decode_metadata)
        tokens = model_output_tensor.tolist()
        logprobs = device_token_logprobs(self.sampler, self._on_device_logits, tokens, md.max_num_logprobs)
        return sampler_output_cls([[x] for x in tokens], logprobs)

    _sample._mimo_patched = True
    return _sample


def _make_apply_grammar_bitmask_cpu(orig):
    def _apply_grammar_bitmask_cpu(self, logits, bitmask):
        # the bitmask has one row per real request, the CPU logits keep the decode padding rows (e.g. 2 requests
        # in the 4-row bucket); pad the mask with all-allowed rows (the padding rows are dropped later)
        if bitmask.dim() == 2 and logits.dim() == 2 and bitmask.shape[0] < logits.shape[0]:
            bitmask = torch.nn.functional.pad(bitmask, (0, 0, 0, logits.shape[0] - bitmask.shape[0]), value=True)
        return orig(self, logits, bitmask)

    _apply_grammar_bitmask_cpu._mimo_patched = True
    return _apply_grammar_bitmask_cpu


def _make_update_states(orig, mod):
    def _update_states(self, *args, **kwargs):
        real = mod.torch
        mod.torch = _TorchWithCPUGenerator(real)
        try:
            return orig(self, *args, **kwargs)
        finally:
            mod.torch = real

    _update_states._mimo_patched = True
    return _update_states


def _tune_compiler_args(opts) -> None:
    """``MIMO_CC_MODULAR_MAC=<n>`` (docs/design-notes.md#kv-write) replaces vllm-neuron's ``--modular-flow-mac-threshold=10``
    (``none`` drops it, leaving the compiler default). The runner's threshold makes neuronx-cc split every graph
    into modular-flow partitions; a KV cache tensor that crosses a partition boundary becomes a separate
    intermediate buffer and is copied whole (L3 device profile: decode time grows with the KV pool size).
    The default on the device is ``100000000000000`` (no partitioning); ``off`` (or CPU mode) leaves the
    runner's arguments untouched. Edits ``opts["compiler_args"]`` (a list) in place."""
    val = switches.get("MIMO_CC_MODULAR_MAC")
    if not val or not opts or "compiler_args" not in opts:
        return
    args = opts["compiler_args"]
    flag = "--modular-flow-mac-threshold=10"
    for i, a in enumerate(args):
        if flag in a:
            new = " --modular-flow-mac-threshold=" + val if val != "none" else ""
            a2 = " ".join(a.replace(flag, new.strip()).split()).replace("options= ", "options=")
            if a2.endswith("options="):
                args.pop(i)
            else:
                args[i] = a2
            logger.info(_P + "compiler args: %r -> %r (MIMO_CC_MODULAR_MAC)", a, a2)
            return
    logger.warning(_P + "MIMO_CC_MODULAR_MAC ignored: no %r in compiler args %r", flag, args)


def _modular_mac_applies(example_inputs) -> bool:
    """docs/design-notes.md#kv-write: ``MIMO_CC_MODULAR_MAC_MAXTOK=<n>`` limits ``MIMO_CC_MODULAR_MAC`` to graphs whose first input (the
    token ids) has at most ``n`` rows, i.e. the decode graphs; the prefill graphs keep the runner's partitioning
    (an unpartitioned full-model prefill graph takes ~1 h to compile). Default 64 on the device; ``off``
    (or unset in CPU mode): every graph.

    Only a 1-D integer first input counts as token ids: the vision tower's first input is BF16 [8, 4096, 1536]
    (8 rows), and compiling it unpartitioned drove walrus_driver to 135 GB RSS and a host OOM (docs/design-notes.md#host-oom)."""
    lim = switches.get("MIMO_CC_MODULAR_MAC_MAXTOK")
    if not lim:
        return True
    try:
        first = example_inputs[0]
        if first.dim() != 1 or first.dtype.is_floating_point or first.dtype.is_complex:
            logger.info(_P + "MIMO_CC_MODULAR_MAC skipped for a graph whose first input is %s %s (not token ids)",
                        first.dtype, tuple(first.shape))
            return False
        ntok = first.shape[0]
    except Exception:  # unknown caller layout: leave this graph's arguments untouched
        return False
    ok = ntok <= int(lim)
    logger.info(_P + "MIMO_CC_MODULAR_MAC %s for a graph with %d tokens (MAXTOK=%s)",
                "applied" if ok else "skipped", ntok, lim)
    return ok


BACKEND_MODULE = "libtorch_neuronx_lite.compile.backend"


def _make_platform_compiler_args(orig):
    """Wrap the backend's ``_apply_platform_compiler_args`` (runs at compile time, before the cache key is hashed)
    so ``MIMO_CC_MODULAR_MAC`` reaches every graph compiled in this process (the runner sets its options inside
    ``load_model``, which is already running when the plugin patches are applied)."""
    def _apply_platform_compiler_args(options):
        out = orig(options)
        args = out.get("compiler_args") if isinstance(out, dict) else None
        if isinstance(args, list) and _modular_mac_applies(sys._getframe(1).f_locals.get("example_inputs")):
            out = {**out, "compiler_args": list(args)}
            _tune_compiler_args(out)
        return out

    _apply_platform_compiler_args._mimo_patched = True
    return _apply_platform_compiler_args


def _make_load_model(orig):
    """After the runner's ``load_model``: give an audio-capable model a compile function with the runner's own
    backend and compile options (the runner only compiles ``visual`` separately; docs/design-notes.md#audio-on-device). CPU / eager / debug modes
    leave it unset, so the audio graph runs eagerly."""
    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        inner = getattr(self.model, "_orig_mod", self.model)
        if getattr(inner, "audio", None) is not None:
            from vllm_neuron import envs

            opts = getattr(self, "compile_options", None)
            eager = (envs.VLLM_NEURON_CPU_MODE or envs.VLLM_NEURON_DEBUG_MODE
                     or self.vllm_config.model_config.enforce_eager or opts is None)
            if not eager:
                backend = envs.get_compile_backend_name()
                inner.audio_compile = lambda fn: torch.compile(fn, backend=backend, fullgraph=True, options=opts)
            logger.info(_P + "audio towers: %s", "eager" if eager else f"compiled on first use ({backend})")
        return out

    load_model._mimo_patched = True
    return load_model


WORKER_MODULE = "vllm_neuron.vllm.worker.neuron_worker"


def _make_warmup_vision(orig):
    """After the worker's vision warmup: warm up the audio graph too."""
    def _warmup_vision_encoder(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        m = self.model_runner.model
        m = getattr(m, "_orig_mod", m)
        if getattr(m, "audio", None) is not None and hasattr(m, "warmup_audio"):
            m.warmup_audio(self.model_runner.encoder_cache)
        return out

    _warmup_vision_encoder._mimo_patched = True
    return _warmup_vision_encoder


def _block_table_summary(runner, csl: int) -> list[dict]:
    """Row 0 of each KV group's CPU block table (0 = vLLM null block): width, allocated entries, max id and the
    null entries below the column of ``csl`` (MIMO_DEBUG_METADATA=hit)."""
    out = []
    for g, bt in enumerate(runner.input_batch.block_table.block_tables):
        row = bt.block_table.cpu[0]
        col = min(csl // bt.block_size, row.shape[0])
        out.append({"group": g, "block_size": int(bt.block_size), "width": int(row.shape[0]),
                    "allocated": int((row > 0).sum()), "max_id": int(row.max()),
                    "null_below_csl": int((row[:col] == 0).sum())})
    return out


def _make_compute_cached_seq_len(orig):
    """MIMO_DEBUG_METADATA=hit (logging only): a prefill step whose ``cached_seq_len`` is not a
    multiple of 4096 (a prefix-cache hit; cold segments always start on a 4096 boundary) logs the value, the GA
    prior kernel's 512-key tile count ``cached_seq_len // 512`` and the request's block tables."""
    def _compute_cached_seq_len(self):
        csl = orig(self)
        try:
            if csl % 4096 and not self._is_decode():
                logger.warning(_P + "prefix hit: cached_seq_len=%d (mod 4096 = %d, 512-key tiles = %d, %s) "
                               "block tables %s", csl, csl % 4096, csl // 512,
                               "odd" if (csl // 512) % 2 else "even", _block_table_summary(self, int(csl)))
        except Exception as e:  # noqa: BLE001 (diagnostics must never break a step)
            logger.warning(_P + "MIMO_DEBUG_METADATA=hit: summary failed (%s)", e)
        return csl

    _compute_cached_seq_len._mimo_patched = True
    return _compute_cached_seq_len


def apply(import_if_missing: bool = True) -> list[str]:
    """Install the patches once per process. Returns the names of the methods patched by this call.
    ``import_if_missing=False``: only patch when the runner module is already imported (worker processes)."""
    mod = sys.modules.get(RUNNER_MODULE)
    if mod is None and not import_if_missing:
        return []
    if mod is None:
        try:
            import importlib
            mod = importlib.import_module(RUNNER_MODULE)
        except Exception as e:  # vllm-neuron not importable (pure unit tests)
            logger.debug(_P + "runner patches skipped (%s)", e)
            return []
    cls = getattr(mod, "NeuronModelRunner", None)
    if cls is None:
        logger.warning(_P + "runner patches skipped: %s has no NeuronModelRunner", RUNNER_MODULE)
        return []
    done = []
    us = cls._update_states
    if not getattr(us, "_mimo_patched", False):
        if _SEED_SNIPPET in inspect.getsource(us):
            cls._update_states = _make_update_states(us, mod)
            done.append("_update_states")
        else:
            logger.warning(_P + "seed patch skipped: _update_states no longer contains %r", _SEED_SNIPPET)
    sm = cls._sample
    if not getattr(sm, "_mimo_patched", False):
        if _LOGPROBS_SNIPPET in inspect.getsource(sm) and hasattr(mod, "SamplerOutput"):
            cpu_fix = _CPU_SAMPLER_SNIPPET in inspect.getsource(sm)
            if not cpu_fix:
                logger.warning(_P + "CPU-sampling metadata patch skipped: _sample no longer contains %r",
                               _CPU_SAMPLER_SNIPPET)
            cls._sample = _make_sample(sm, mod.SamplerOutput, cpu_fix=cpu_fix)
            done.append("_sample")
        else:
            logger.warning(_P + "logprobs patch skipped: _sample no longer matches vllm-neuron 0.24.0.1.1.0")
    lm = cls.load_model
    if not getattr(lm, "_mimo_patched", False):
        cls.load_model = _make_load_model(lm)
        done.append("load_model")
    wmod = sys.modules.get(WORKER_MODULE)
    wcls = getattr(wmod, "NeuronWorker", None) if wmod is not None else None
    wv = getattr(wcls, "_warmup_vision_encoder", None) if wcls is not None else None
    if wv is not None and not getattr(wv, "_mimo_patched", False):
        wcls._warmup_vision_encoder = _make_warmup_vision(wv)
        done.append("NeuronWorker._warmup_vision_encoder")
    if switches.get("MIMO_CC_MODULAR_MAC"):
        bmod = sys.modules.get(BACKEND_MODULE)
        if bmod is None:
            try:
                import importlib
                bmod = importlib.import_module(BACKEND_MODULE)
            except Exception as e:
                logger.warning(_P + "MIMO_CC_MODULAR_MAC ignored: %s not importable (%s)", BACKEND_MODULE, e)
        pa = getattr(bmod, "_apply_platform_compiler_args", None) if bmod is not None else None
        if pa is not None and not getattr(pa, "_mimo_patched", False):
            bmod._apply_platform_compiler_args = _make_platform_compiler_args(pa)
            done.append("backend._apply_platform_compiler_args")
    cs = getattr(cls, "_compute_cached_seq_len", None)
    if os.environ.get("MIMO_DEBUG_METADATA") == "hit" and cs is not None and not getattr(cs, "_mimo_patched", False):
        cls._compute_cached_seq_len = _make_compute_cached_seq_len(cs)
        done.append("_compute_cached_seq_len")
    gb = getattr(cls, "_apply_grammar_bitmask_cpu", None)
    if gb is not None and not getattr(gb, "_mimo_patched", False):
        if _BITMASK_SNIPPET in inspect.getsource(gb):
            cls._apply_grammar_bitmask_cpu = _make_apply_grammar_bitmask_cpu(gb)
            done.append("_apply_grammar_bitmask_cpu")
        else:
            logger.warning(_P + "bitmask patch skipped: _apply_grammar_bitmask_cpu no longer contains %r",
                           _BITMASK_SNIPPET)
    if done:
        logger.info(_P + "patched NeuronModelRunner.%s (seeded generators on CPU; on-device-sampling "
                    "logprobs gathered for the device-sampled token; CPU sampling with CPU sampling "
                    "metadata)", "/".join(done))
    return done


SCHEDULER_MODULE = "vllm_neuron.vllm.core.scheduler"


def _make_scheduler_init(orig, vals: list[int] | str | None, kv_conc: int | None = None):
    """Small prefill buckets (MIMO_PREFILL_BUCKETS): the NeuronScheduler pads every prefill chunk to one of its own ``num_batched_tokens_buckets``
    (``num_scheduled_tokens_padded``, which the runner uses as the graph shape). With segmented prefill those
    are ``[segment size]``, so a short prompt is padded to the full segment even when the worker compiled a
    smaller bucket (``MIMO_PREFILL_BUCKETS``, factory). Give the scheduler the same list. ``vals="auto"`` (the
    default): ``[512, segment]`` by :func:`switches.default_prefill_buckets`, the rule the worker uses.

    docs/design-notes.md#concurrency: ``kv_conc`` (``MIMO_KV_CONCURRENCY=<n>``) replaces the scheduler's admission cap ``_max_kv_concurrent``
    (KV tokens // max_model_len, i.e. 1 for the 1M config with 36864 blocks), capped at max_num_seqs. Requests
    whose total context exceeds the KV pool then rely on vLLM preemption (not validated on Neuron)."""
    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        cur = list(getattr(self, "num_batched_tokens_buckets", None) or [])
        if vals == "auto":   # default: same rule as the worker (factory._validate_config)
            auto = switches.default_prefill_buckets(cur[0] if len(cur) == 1 else None, cur,
                                                    getattr(self, "max_model_len", None))
            if auto:
                self.num_batched_tokens_buckets = auto
                logger.info(_P + "scheduler prefill buckets %s -> %s (MIMO_PREFILL_BUCKETS default)", cur, auto)
        elif vals:
            if cur == [vals[-1]]:
                self.num_batched_tokens_buckets = list(vals)
                logger.info(_P + "scheduler prefill buckets %s -> %s (MIMO_PREFILL_BUCKETS)", cur, vals)
            else:
                logger.warning(_P + "MIMO_PREFILL_BUCKETS %s ignored by the scheduler: its buckets are %s",
                               vals, cur)
        if kv_conc and hasattr(self, "_max_kv_concurrent"):
            old = self._max_kv_concurrent
            self._max_kv_concurrent = min(kv_conc, getattr(self, "max_num_seqs", kv_conc))
            logger.warning(_P + "scheduler KV concurrency cap %s -> %s (MIMO_KV_CONCURRENCY)", old,
                           self._max_kv_concurrent)

    __init__._mimo_patched = True
    return __init__


def apply_scheduler_patch() -> bool:
    """Install the scheduler patch when ``MIMO_PREFILL_BUCKETS`` (``auto`` by default on the device, ``off``
    disables it) or ``MIMO_KV_CONCURRENCY`` is set (any process; no-op otherwise)."""
    pb = switches.get("MIMO_PREFILL_BUCKETS")
    kc = os.environ.get("MIMO_KV_CONCURRENCY")
    if not pb and not kc:
        return False
    try:
        vals = ("auto" if pb == "auto" else sorted({int(x) for x in pb.split(",") if x.strip()})) if pb else None
        kv_conc = int(kc) if kc else None
        import importlib
        mod = importlib.import_module(SCHEDULER_MODULE)
    except Exception as e:  # bad value (the factory reports it) or vllm-neuron not importable
        logger.warning(_P + "scheduler patch skipped (%s)", e)
        return False
    cls = getattr(mod, "NeuronScheduler", None)
    if cls is None or getattr(cls.__init__, "_mimo_patched", False):
        return False
    cls.__init__ = _make_scheduler_init(cls.__init__, vals, kv_conc)
    return True
