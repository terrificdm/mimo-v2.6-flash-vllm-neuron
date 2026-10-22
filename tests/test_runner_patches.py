# SPDX-License-Identifier: Apache-2.0
"""CPU tests for plugin/mimo_v2_neuron/runner_patches.py (engine-crash fixes: seed / ODS-logprobs)."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from mimo_v2_neuron import runner_patches as rp

sampler_mod = pytest.importorskip("vllm.v1.sample.sampler")


def test_cpu_generator_maps_device_to_cpu():
    assert rp.cpu_generator(torch.device("xla:0")).device.type == "cpu"
    assert rp.cpu_generator("cpu").device.type == "cpu"
    assert rp.cpu_generator().device.type == "cpu"
    g1, g2 = rp.cpu_generator(torch.device("xla:0")), rp.cpu_generator("cpu")
    g1.manual_seed(7), g2.manual_seed(7)
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))


def test_update_states_wrapper_swaps_generator_only_during_call():
    fake = types.SimpleNamespace(torch=torch)

    def orig(self, x):
        assert fake.torch.zeros(1).dtype == torch.float32  # other attributes still forward to torch
        return fake.torch.Generator(device=torch.device("xla:0"))

    wrapped = rp._make_update_states(orig, fake)
    g = wrapped(None, 1)
    assert g.device.type == "cpu" and fake.torch is torch

    def boom(self):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        rp._make_update_states(boom, fake)(None)
    assert fake.torch is torch


@pytest.mark.parametrize("mode", ["raw_logprobs", "raw_logits"])
def test_device_token_logprobs_matches_sampler_forward_layout(mode):
    torch.manual_seed(0)
    logits = torch.randn(3, 50, dtype=torch.bfloat16)
    sampler = sampler_mod.Sampler(logprobs_mode=mode)
    toks = [int(logits[0].argmax()), 5, 17]  # row 0 greedy, rows 1-2 arbitrary "device-sampled" tokens
    lp = rp.device_token_logprobs(sampler, logits, toks, 5)
    raw = sampler.compute_logprobs(logits) if mode == "raw_logprobs" else logits.float()
    ref = sampler.gather_logprobs(raw, 5, token_ids=torch.tensor(toks))
    assert torch.equal(lp.logprob_token_ids, ref.logprob_token_ids)
    assert torch.equal(lp.logprobs, ref.logprobs)
    assert lp.logprob_token_ids[:, 0].tolist() == toks  # column 0 = the emitted token
    assert lp.selected_token_ranks[0].item() == 1
    # padded rows beyond the number of tokens are dropped
    lp2 = rp.device_token_logprobs(sampler, torch.cat([logits, logits]), toks, 0)
    assert lp2.logprobs.shape == (3, 1)


def _fake_runner(n_logprobs, **kw):
    md = SimpleNamespace(max_num_logprobs=n_logprobs, logprob_token_ids=None,
                         temperature=torch.ones(2, device="meta"))  # device tensor: must never be touched
    base = dict(input_batch=SimpleNamespace(sampling_metadata=md), on_device_sampling=True,
                use_async_scheduling=False, _on_device_logits=torch.randn(2, 40),
                sampler=sampler_mod.Sampler())
    base.update(kw)
    return SimpleNamespace(**base)


class _Out:
    def __init__(self, toks, lp):
        self.sampled_token_ids, self.logprobs_tensors = toks, lp


def test_sample_wrapper_uses_device_tokens_and_falls_back():
    calls = []

    def orig(self, t, spec=None):
        calls.append(spec)
        return "orig"

    s = rp._make_sample(orig, _Out)
    out = s(_fake_runner(3), torch.tensor([4, 9]))
    assert out.sampled_token_ids == [[4], [9]] and out.logprobs_tensors.logprob_token_ids[:, 0].tolist() == [4, 9]
    assert not calls
    assert s(_fake_runner(0), torch.tensor([4, 9])).logprobs_tensors.logprobs.shape == (2, 1)
    # cases left to vllm-neuron: no logprobs, -1 (full vocab), spec decode, CPU sampling without the CPU fix,
    # async, token-id logprobs
    for r, spec in [(_fake_runner(None), None), (_fake_runner(-1), None), (_fake_runner(3), object()),
                    (_fake_runner(3, use_async_scheduling=True), None), (_fake_runner(3, _on_device_logits=None), None)]:
        assert s(r, torch.tensor([1, 2]), spec) == "orig"
    assert rp._make_sample(orig, _Out, cpu_fix=False)(_fake_runner(3, on_device_sampling=False),
                                                       torch.tensor([1, 2])) == "orig"
    r = _fake_runner(3)
    r.input_batch.sampling_metadata.logprob_token_ids = {0: [1]}
    assert s(r, torch.tensor([1, 2])) == "orig"


def _real_metadata(n=2, temperature=1.0, seed=None, dev="cpu", freq=0.0):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.logits_processor.builtin import (LogitBiasLogitsProcessor, MinPLogitsProcessor,
                                                          MinTokensLogitsProcessor)
    from vllm.v1.sample.metadata import SamplingMetadata

    vcfg = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    lps = LogitsProcessors([MinPLogitsProcessor(vcfg, "cpu", False), LogitBiasLogitsProcessor(None, "cpu", False),
                            MinTokensLogitsProcessor(vcfg, "cpu", False)])
    gens = {}
    if seed is not None:
        for i in range(n):
            gens[i] = torch.Generator().manual_seed(seed)
    t = lambda x, dt=torch.float32: torch.full((n,), x, dtype=dt, device=dev)  # noqa: E731
    return SamplingMetadata(temperature=t(temperature), all_greedy=temperature == 0, all_random=temperature > 0,
                            top_p=t(0.95), top_k=None, generators=gens, max_num_logprobs=None,
                            no_penalties=freq == 0, prompt_token_ids=torch.zeros((n, 3), dtype=torch.int64,
                                                                                 device=dev),
                            frequency_penalties=t(freq), presence_penalties=t(0.0), repetition_penalties=t(1.0),
                            output_token_ids=[[] for _ in range(n)], allowed_token_ids_mask=None,
                            bad_words_token_ids={}, logitsprocs=lps)


def test_cpu_sampling_metadata_moves_every_tensor(monkeypatch):
    moved = []

    def fake_cpu(t):
        if t.device.type == "cpu":
            return t
        moved.append(tuple(t.shape))
        return torch.zeros(t.shape, dtype=t.dtype)

    monkeypatch.setattr(rp, "_cpu", fake_cpu)
    md = _real_metadata(dev="meta", seed=3)
    lb = md.logitsprocs.non_argmax_invariant[0]
    lb.logits_slice = (torch.zeros(2, dtype=torch.int32, device="meta"),) * 2
    out = rp.cpu_sampling_metadata(md)
    for f in ("temperature", "top_p", "prompt_token_ids", "frequency_penalties", "presence_penalties",
              "repetition_penalties"):
        assert getattr(out, f).device.type == "cpu" and getattr(md, f).device.type == "meta"
    new_lb = out.logitsprocs.non_argmax_invariant[0]
    assert new_lb is not lb and all(x.device.type == "cpu" for x in new_lb.logits_slice)
    assert lb.logits_slice[0].device.type == "meta"  # the original processor keeps its state
    assert out.generators is md.generators and out.output_token_ids is md.output_token_ids
    assert len(moved) >= 8
    cpu_md = _real_metadata()
    assert rp.cpu_sampling_metadata(cpu_md).temperature is cpu_md.temperature  # nothing to move: same tensors


def test_sample_wrapper_cpu_sampling_seeded():
    sampler = sampler_mod.Sampler()

    def runner(md):
        return SimpleNamespace(input_batch=SimpleNamespace(sampling_metadata=md), on_device_sampling=False,
                               use_async_scheduling=False, _on_device_logits=None, sampler=sampler)

    def orig(self, t, spec=None):
        raise AssertionError("CPU sampling must not reach the unpatched path")

    s = rp._make_sample(orig, _Out)
    torch.manual_seed(0)
    logits = torch.randn(2, 64)
    a = s(runner(_real_metadata(seed=11)), logits.clone()).sampled_token_ids.tolist()
    b = s(runner(_real_metadata(seed=11)), logits.clone()).sampled_token_ids.tolist()
    assert a == b  # same seed -> same tokens
    outs = {tuple(s(runner(_real_metadata(seed=k)), logits.clone()).sampled_token_ids.flatten().tolist())
            for k in range(12)}
    assert len(outs) > 1  # different seeds -> different samples
    g = s(runner(_real_metadata(temperature=0.0)), logits.clone()).sampled_token_ids.flatten().tolist()
    assert g == logits.argmax(-1).tolist()
    # decode padding: 2 requests in a 4-row bucket -> only the 2 real rows are sampled
    r = runner(_real_metadata(seed=5))
    r.input_batch.num_reqs = 2
    padded = torch.cat([logits, torch.randn(2, 64)])
    ref = s(runner(_real_metadata(seed=5)), logits.clone()).sampled_token_ids.tolist()
    assert s(r, padded).sampled_token_ids.tolist() == ref


def test_sample_wrapper_device_sampling_resamples_seeded_and_penalised(monkeypatch):
    """docs/design-notes.md#cpu-resample: mode A + seed / penalty -> CPU re-sample from the gathered logits; plain batches keep device tokens."""
    sampler = sampler_mod.Sampler()

    def runner(md, logits, n=None):
        return SimpleNamespace(input_batch=SimpleNamespace(sampling_metadata=md, num_reqs=n), on_device_sampling=True,
                               use_async_scheduling=False, _on_device_logits=logits, sampler=sampler)

    def orig(self, t, spec=None):
        return "orig"

    s = rp._make_sample(orig, _Out)
    torch.manual_seed(0)
    logits = torch.randn(2, 64)
    dev = torch.tensor([1, 2])
    a = s(runner(_real_metadata(seed=11), logits), dev).sampled_token_ids
    b = s(runner(_real_metadata(seed=11), logits), dev).sampled_token_ids
    assert a == b and a != [[1], [2]]
    outs = {tuple(x[0] for x in s(runner(_real_metadata(seed=k), logits), dev).sampled_token_ids) for k in range(12)}
    assert len(outs) > 1
    # greedy + frequency penalty: the CPU sampler applies the penalty (argmax of the penalised logits)
    md = _real_metadata(temperature=0.0, freq=100.0)
    md.output_token_ids[0].append(int(logits[0].argmax()))
    out = s(runner(md, logits), dev).sampled_token_ids
    assert out[0][0] != int(logits[0].argmax()) and out[1][0] == int(logits[1].argmax())
    # padding row (2 requests in a 3-row bucket) keeps the device token
    padded = torch.cat([logits, torch.randn(1, 64)])
    assert s(runner(_real_metadata(seed=11), padded, n=2), torch.tensor([1, 2, 7])).sampled_token_ids == a + [[7]]
    # no seed, no penalty -> unpatched path; switch off -> unpatched path
    assert s(runner(_real_metadata(), logits), dev) == "orig"
    monkeypatch.setenv("MIMO_ODS_CPU_RESAMPLE", "0")
    assert s(runner(_real_metadata(seed=11), logits), dev) == "orig"


def test_grammar_bitmask_cpu_wrapper_pads_mask_rows():
    def orig(self, logits, bitmask):
        return logits.masked_fill(~bitmask, float("-inf"))

    f = rp._make_apply_grammar_bitmask_cpu(orig)
    logits = torch.zeros(4, 6)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 1] = True
    out = f(None, logits, mask)
    assert out.shape == (4, 6)
    assert torch.isinf(out[:2]).sum() == 10 and out[:2, 1].eq(0).all()  # real rows: only token 1 allowed
    assert not torch.isinf(out[2:]).any()  # padding rows untouched
    full = torch.ones(4, 6, dtype=torch.bool)
    assert torch.equal(f(None, logits, full), logits)


def test_apply_patches_real_runner_once():
    pytest.importorskip(rp.RUNNER_MODULE)
    first = rp.apply()
    mod = sys.modules[rp.RUNNER_MODULE]
    assert mod.NeuronModelRunner._sample._mimo_patched and mod.NeuronModelRunner._update_states._mimo_patched
    assert mod.NeuronModelRunner._apply_grammar_bitmask_cpu._mimo_patched
    assert mod.NeuronModelRunner.load_model._mimo_patched     # audio compile hand-off (docs/design-notes.md#audio-on-device)
    assert set(first) <= {"_sample", "_update_states", "_apply_grammar_bitmask_cpu", "load_model",
                          "NeuronWorker._warmup_vision_encoder"}
    assert rp.apply() == []  # idempotent


def test_debug_hit_logs_only_off_boundary_prefill(caplog):
    """MIMO_DEBUG_METADATA=hit : logs prefill steps whose cached_seq_len is off a 4096 boundary, with
    the 512-key tile parity and block-table extent; never changes the returned value."""
    table = torch.zeros(1, 64, dtype=torch.int32)
    table[0, :20] = torch.arange(5, 25)
    table[0, 2] = 0                                   # one null block below the hit
    bt = SimpleNamespace(block_size=32, block_table=SimpleNamespace(cpu=table))

    class R:
        decode = False
        csl = 576
        input_batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[bt]))

        def _is_decode(self):
            return self.decode

        def _compute_cached_seq_len(self):
            return self.csl

    f = rp._make_compute_cached_seq_len(R._compute_cached_seq_len)
    r = R()
    with caplog.at_level("WARNING", logger=rp.__name__):
        assert f(r) == 576
        r.csl = 8192
        assert f(r) == 8192                           # cold segment boundary: silent
        r.csl, r.decode = 600, True
        assert f(r) == 600                            # decode: silent
    msgs = [m.getMessage() for m in caplog.records]
    assert len(msgs) == 1 and "cached_seq_len=576" in msgs[0] and "512-key tiles = 1, odd" in msgs[0]
    assert "'allocated': 19" in msgs[0] and "'null_below_csl': 1" in msgs[0] and "'max_id': 24" in msgs[0]


def test_modular_mac_maxtok_only_token_id_graphs(monkeypatch):
    """Only a 1-D integer first input (token ids) with <= MAXTOK rows gets the no-partition args; the
    vision tower (BF16 [8, 4096, 1536]) and prefill graphs keep the runner's partitioning."""
    monkeypatch.delenv("MIMO_CC_MODULAR_MAC_MAXTOK", raising=False)
    assert rp._modular_mac_applies([torch.zeros(8, 4096, 1536, dtype=torch.bfloat16)])
    monkeypatch.setenv("MIMO_CC_MODULAR_MAC_MAXTOK", "64")
    assert rp._modular_mac_applies([torch.zeros(4, dtype=torch.int64)])
    assert not rp._modular_mac_applies([torch.zeros(4096, dtype=torch.int64)])
    assert not rp._modular_mac_applies([torch.zeros(8, 4096, 1536, dtype=torch.bfloat16)])
    assert not rp._modular_mac_applies([torch.zeros(8, 2, dtype=torch.int32)])
    assert not rp._modular_mac_applies(None)


def test_scheduler_prefill_buckets_patch():
    """MIMO_PREFILL_BUCKETS: the scheduler's padding buckets become the smaller list when they are [segment]."""
    class S:
        def __init__(self, cur):
            self.num_batched_tokens_buckets = cur

    init = rp._make_scheduler_init(S.__init__, [512, 4096])
    s = S.__new__(S)
    init(s, [4096])
    assert s.num_batched_tokens_buckets == [512, 4096]
    s2 = S.__new__(S)
    init(s2, [1024, 2048])          # not the segmented single bucket: left alone
    assert s2.num_batched_tokens_buckets == [1024, 2048]


def test_scheduler_kv_concurrency_patch():
    """MIMO_KV_CONCURRENCY replaces the admission cap, capped at max_num_seqs."""
    class S:
        def __init__(self):
            self._max_kv_concurrent, self.max_num_seqs = 1, 4
            self.num_batched_tokens_buckets = [4096]

    s = S.__new__(S)
    rp._make_scheduler_init(S.__init__, None, 8)(s)
    assert s._max_kv_concurrent == 4 and s.num_batched_tokens_buckets == [4096]
