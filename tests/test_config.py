# SPDX-License-Identifier: Apache-2.0
import json
import os
import types

import pytest

from mimo_v2_neuron.model.mimo_v2 import factory
from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hf_dict():
    d = json.load(open(os.path.join(ROOT, "reference/hf/config.json")))
    d.pop("quantization_config")
    d["attention_projection_layout"] = "split"
    return d


GOOD_RT = dict(tp=64, ep=64, enable_prefix_caching=False, speculative_config=None,
               max_num_batched_tokens=4096, max_model_len=4096)


def _validate(d, rt=None, nc=None, impl=None, monkeypatch=None):
    if impl is not None:
        monkeypatch.setenv("MIMO_DECODE_MOE_IMPL", impl)
    cfg = MiMoV2Config.from_configs(d, nc)
    factory.MiMoV2ForCausalLM._validate_config(cfg, nc, dict(GOOD_RT, **(rt or {})))
    return cfg


def test_from_configs_fields(monkeypatch):
    monkeypatch.delenv("MIMO_DECODE_MOE_IMPL", raising=False)
    cfg = _validate(_hf_dict())
    assert cfg.num_hidden_layers == 48 and cfg.vocab_size == 152576
    assert cfg.swa_num_key_value_heads == 8 and cfg.num_key_value_heads == 4
    assert cfg.rope_dim(0) == 64 and cfg.is_swa(1) and not cfg.is_swa(0)
    assert cfg.is_moe(1) and not cfg.is_moe(0)
    assert cfg.routed_scaling_factor == 1.0
    assert cfg.decode_moe_impl == "tkg"
    assert cfg.attn_dims(1)["has_sink"] and not cfg.attn_dims(0)["has_sink"]


@pytest.mark.parametrize("impl", ["tkg", "cte", "torch"])
def test_decode_impl_valid(impl, monkeypatch):
    assert _validate(_hf_dict(), impl=impl, monkeypatch=monkeypatch).decode_moe_impl == impl


def _expect(match, **kw):
    with pytest.raises(ValueError, match="^MiMoV2 Neuron: .*" + match):
        _validate(**kw)


def test_invalid_impl(monkeypatch):
    _expect("MIMO_DECODE_MOE_IMPL must be one of tkg/cte/torch", d=_hf_dict(), impl="foo", monkeypatch=monkeypatch)


def test_quantization_config_rejected():
    d = _hf_dict()
    d["quantization_config"] = {"quant_method": "fp8"}
    _expect("quantization_config", d=d)


def test_fused_layout_rejected():
    d = _hf_dict()
    d["attention_projection_layout"] = "fused_qkv"
    _expect("attention_projection_layout", d=d)


@pytest.mark.parametrize("rt,match", [
    ({"tp": 48}, "divisible by tensor_parallel_size"),
    ({"tp": 6, "ep": 1}, "divisible by tensor_parallel_size|must divide one another"),
    ({"ep": 3}, "EP degree"),
    ({"enable_prefix_caching": True}, "prefix caching"),
    ({"speculative_config": object()}, "speculative"),
    ({"max_num_batched_tokens": 1024}, "max_num_batched_tokens"),
])
def test_runtime_rejections(rt, match):
    _expect(match, d=_hf_dict(), rt=rt)


def test_segmented_prefill_rejected():
    nc = types.SimpleNamespace(kv_segment_size_buckets=[1024])
    _expect("segmented prefill", d=_hf_dict(), nc=nc)
