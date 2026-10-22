"""Plugin switch defaults = final validated configuration on the device, old defaults in the
vLLM CPU mode, environment always wins, auto rules fall back safely."""

import os
from types import SimpleNamespace

import pytest

from mimo_v2_neuron import switches as sw

VALIDATED = {   # the MIMO_* plugin switches the validated 1M + multimodal serve command used to export
    "MIMO_FP8_IMPL": "nki", "MIMO_FP8_EXPERT_IMPL": "row", "MIMO_MOE_CTE_IMPL": "hybrid", "MIMO_KV_WRITE": "nki",
    "MIMO_KV_VNOPAD": "1", "MIMO_PREFILL_ATTN": "nki", "MIMO_PREFILL_KIDX": "arith", "MIMO_VIT_DP_GROUP": "world",
    "MIMO_CC_MODULAR_MAC": "100000000000000", "MIMO_CC_MODULAR_MAC_MAXTOK": "64", "MIMO_DECODE_MOE_IMPL": "tkg",
}


@pytest.fixture
def clean(monkeypatch):
    for n in list(sw.DEFAULTS) + ["MIMO_FP8_ACT_MAX", "MIMO_MM_VISION_BUCKETS"]:
        monkeypatch.delenv(n, raising=False)
    return monkeypatch


def test_device_defaults_equal_validated_command(clean):
    clean.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    for n, v in VALIDATED.items():
        assert sw.get(n) == v, n
        assert sw.source(n) == "default"
    assert sw.get("MIMO_PREFILL_BUCKETS") == "auto"


def test_cpu_mode_keeps_old_defaults(clean):
    clean.setenv("VLLM_NEURON_CPU_MODE", "1")
    assert sw.cpu_mode()
    for n, (_, old) in sw.DEFAULTS.items():
        assert sw.get(n) == old, n
        assert sw.source(n) == "cpu-mode default"


def test_env_wins_and_off(clean):
    clean.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    clean.setenv("MIMO_FP8_IMPL", "emul")
    clean.setenv("MIMO_PREFILL_BUCKETS", "off")
    clean.setenv("MIMO_CC_MODULAR_MAC", "off")
    assert sw.get("MIMO_FP8_IMPL") == "emul" and sw.source("MIMO_FP8_IMPL") == "env"
    assert sw.get("MIMO_PREFILL_BUCKETS") is None
    assert sw.get("MIMO_CC_MODULAR_MAC") is None
    clean.setenv("MIMO_KV_WRITE", "")          # empty = unset
    assert sw.get("MIMO_KV_WRITE") == "nki"
    line = sw.summary({"x": 1})
    assert "MIMO_FP8_IMPL=emul(env)" in line and "MIMO_CC_MODULAR_MAC=<off>(env)" in line and line.endswith("x=1")


def test_default_prefill_buckets():
    f = sw.default_prefill_buckets
    assert f(4096, [4096], 1048576) == [512, 4096]
    assert f(512, [512], 1048576) is None            # segment already 512
    assert f(4096, [4096], 4096) is None             # not segmented (segment >= max_model_len)
    assert f(4096, [512, 4096], 1048576) is None     # buckets already customised
    assert f(None, [1024, 4096], 4096) is None


def test_act_settings_device_default_and_fallback(clean):
    from mimo_v2_neuron.model.mimo_v2 import fp8_linear as F

    clean.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    assert F.act_settings()["impl"] == "nki"
    clean.setenv("MIMO_FP8_ACT_MAX", "240")
    assert F.act_settings()["impl"] == "emul"         # default nki needs 448: falls back
    clean.setenv("MIMO_FP8_IMPL", "nki")
    with pytest.raises(ValueError):
        F.act_settings()                              # explicit nki + 240 is still an error


def test_config_fp8_defaults(clean):
    from mimo_v2_neuron.model.mimo_v2.config import MiMoV2Config

    clean.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    hf = {"neuron_fp8_config": {"format": "neuron_fp8"}, "num_hidden_layers": 2}
    cfg = MiMoV2Config.from_configs(hf, None)
    assert (cfg.fp8_expert_impl, cfg.moe_cte_impl, cfg.fp8_act["impl"]) == ("row", "hybrid", "nki")
    bf16 = MiMoV2Config.from_configs({"num_hidden_layers": 2}, None)
    assert not bf16.fp8_weights and bf16.fp8_expert_impl == "expand"   # BF16 weights: FP8 switches not read
    clean.setenv("VLLM_NEURON_CPU_MODE", "1")
    cfg = MiMoV2Config.from_configs(hf, None)
    assert (cfg.fp8_expert_impl, cfg.moe_cte_impl, cfg.fp8_act["impl"]) == ("expand", "shard_on_i", "emul")


def test_scheduler_auto_prefill_buckets():
    from mimo_v2_neuron import runner_patches as rp

    class S:
        def __init__(self, cur, mml):
            self.num_batched_tokens_buckets, self.max_model_len = cur, mml

    init = rp._make_scheduler_init(S.__init__, "auto")
    for cur, mml, want in (([4096], 1048576, [512, 4096]), ([4096], 32768, [512, 4096]),
                           ([1024, 4096], 4096, [1024, 4096]), ([512], 1048576, [512])):
        s = S.__new__(S)
        init(s, list(cur), mml)
        assert s.num_batched_tokens_buckets == want, (cur, mml)


def test_derive_vision_buckets_env(clean):
    from mimo_v2_neuron import mm_patches as mp

    def vc(buckets, arch="MiMoV2OmniForCausalLM"):
        return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
                               additional_config={"vision_neuron_config": {"num_vision_tokens_buckets": buckets}})

    clean.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    clean.setenv("MIMO_MM_VISION_BUCKETS", "")    # empty = unset; monkeypatch restores the original afterwards
    assert mp.derive_vision_buckets_env(vc([1024, 2048])) is None               # tiny layouts: not 4096 multiples
    assert mp.derive_vision_buckets_env(vc([4096])) is None                     # old single bucket: no checks
    assert mp.derive_vision_buckets_env(vc([4096, 65536], "Other")) is None     # other models untouched
    assert os.environ["MIMO_MM_VISION_BUCKETS"] == ""
    assert mp.derive_vision_buckets_env(vc([65536, 4096])) == "4096,65536"
    assert os.environ["MIMO_MM_VISION_BUCKETS"] == "4096,65536"
    clean.setenv("MIMO_MM_VISION_BUCKETS", "")
    clean.setenv("VLLM_NEURON_CPU_MODE", "1")
    assert mp.derive_vision_buckets_env(vc([4096, 65536])) is None              # CPU mode: old behaviour
    clean.delenv("VLLM_NEURON_CPU_MODE")
    clean.setenv("MIMO_MM_VISION_BUCKETS", "4096,32768")                       # explicit value wins
    assert mp.derive_vision_buckets_env(vc([4096, 65536])) is None
    assert os.environ["MIMO_MM_VISION_BUCKETS"] == "4096,32768"


def test_default_timeouts(monkeypatch):
    import mimo_v2_neuron as m

    for k in m.TIMEOUT_DEFAULTS:
        monkeypatch.setenv(k, "")   # empty = unset; monkeypatch restores the original afterwards
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "123")
    done = m.default_timeouts()
    assert "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" not in done
    assert os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "123"
    assert os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] == "21600"
    assert os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] == "7200"
