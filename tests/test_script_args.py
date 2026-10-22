# SPDX-License-Identifier: Apache-2.0
"""The accuracy scripts take a required --decode-moe-impl and write it to
os.environ["MIMO_DECODE_MOE_IMPL"] before the vLLM ``LLM`` (and its worker processes) is created.

``LLM`` is replaced by a fake that records the environment at construction time and then stops the
script; goldens / model config loading is stubbed, so no model or goldens are needed.
"""

import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "scripts", "accuracy")
SCRIPTS = ["run_logit_validation", "fallback_check"]
VOCAB = 152576


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_acc_{name}", os.path.join(ACC, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stop(Exception):
    pass


@pytest.mark.parametrize("script", SCRIPTS)
def test_missing_decode_moe_impl_exits_2(script, tmp_path):
    mod = _load(script)
    with pytest.raises(SystemExit) as e:
        mod.main(["--batch-mode", "b1", "--out", str(tmp_path / "x.json")])
    assert e.value.code == 2


@pytest.mark.parametrize("script", SCRIPTS)
def test_invalid_decode_moe_impl_exits_2(script, tmp_path):
    mod = _load(script)
    with pytest.raises(SystemExit) as e:
        mod.main(["--batch-mode", "b1", "--decode-moe-impl", "foo", "--out", str(tmp_path / "x.json")])
    assert e.value.code == 2


@pytest.mark.parametrize("impl", ["tkg", "cte", "torch"])
@pytest.mark.parametrize("script", SCRIPTS)
def test_env_set_before_llm_is_created(script, impl, tmp_path, monkeypatch):
    import torch
    import vllm

    mod = _load(script)
    C = mod.C
    # a stale value from the calling shell must be overridden by the argument
    monkeypatch.setenv("MIMO_DECODE_MOE_IMPL", "stale-shell-value")
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen["env"] = os.environ.get("MIMO_DECODE_MOE_IMPL")
            seen["kw"] = kw
            raise _Stop

    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    monkeypatch.setattr(C, "load_inputs", lambda a: (["P1"], {"P1": [1, 2, 3]}, {"vocab_size": VOCAB}))
    monkeypatch.setattr(C, "model_vocab", lambda d: VOCAB)
    out = tmp_path / "res.json"
    argv = ["--batch-mode", "b1", "--decode-moe-impl", impl, "--model-dir", str(tmp_path), "--goldens",
            str(tmp_path), "--tp", "64", "--out", str(out)]
    if script == "fallback_check":
        main_res = tmp_path / f"b1_{impl}.json"
        main_res.write_text(json.dumps({"batch_mode": "b1", "decode_moe_impl": impl, "passed": False,
                                        "per_prompt": {"P1": {"failed_steps": []}}}))
        (tmp_path / "routing.pt").write_bytes(b"")
        monkeypatch.setattr(torch, "load", lambda *a, **k: {})
        monkeypatch.setattr(mod, "routing_flips", lambda r, names, ids: {
            n: {"generation_region_positions": [], "flip_pairs": [], "prompt_region_positions": []} for n in names})

    with pytest.raises(_Stop):
        mod.main(argv)
    assert seen["env"] == impl
    assert os.environ["MIMO_DECODE_MOE_IMPL"] == impl
    kw = seen["kw"]
    assert kw["tensor_parallel_size"] == 64 and kw["enable_expert_parallel"] is True
    assert kw["max_model_len"] == 1024 and kw["max_num_batched_tokens"] == 1024 and kw["max_num_seqs"] == 1
    assert kw["max_logprobs"] == -1 and kw["logprobs_mode"] == "raw_logits"
    assert kw["enable_prefix_caching"] is False and kw["async_scheduling"] is False
    nc = kw["additional_config"]["neuron_config"]
    assert nc == {"on_device_sampling_config": {}, "num_batched_tokens_buckets": [1024], "num_seqs_buckets": [1]}


# ---------------------------------------------------------------------------------------------
# Long context: --max-model-len / --kv-segment-size / --batch-mode b2 /
# --top5-rtol-override. Defaults must keep the base-configuration behaviour (asserted above and below).
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("script", SCRIPTS)
def test_segmented_long_args(script, tmp_path, monkeypatch):
    import torch
    import vllm

    mod = _load(script)
    C = mod.C
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen["kw"] = kw
            raise _Stop

    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    monkeypatch.setattr(C, "load_inputs", lambda a: (["L1", "L2"], {"L1": [1, 2, 3], "L2": [4, 5]}, {"vocab_size": VOCAB}))
    monkeypatch.setattr(C, "model_vocab", lambda d: VOCAB)
    argv = ["--batch-mode", "b2", "--decode-moe-impl", "tkg", "--model-dir", str(tmp_path), "--goldens",
            str(tmp_path), "--max-model-len", "16384", "--kv-segment-size", "4096", "--num-gpu-blocks-override",
            "2048", "--out", str(tmp_path / "res.json")]
    if script == "fallback_check":
        (tmp_path / "b2_tkg.json").write_text(json.dumps({"batch_mode": "b2", "decode_moe_impl": "tkg", "passed": False,
                                                          "per_prompt": {"L1": {"failed_steps": []},
                                                                         "L2": {"failed_steps": []}}}))
        (tmp_path / "routing.pt").write_bytes(b"")
        monkeypatch.setattr(torch, "load", lambda *a, **k: {})
        monkeypatch.setattr(mod, "routing_flips", lambda r, names, ids: {
            n: {"generation_region_positions": [], "flip_pairs": [], "prompt_region_positions": []} for n in names})
    with pytest.raises(_Stop):
        mod.main(argv)
    kw = seen["kw"]
    assert kw["max_model_len"] == 16384 and kw["max_num_batched_tokens"] == 4096 and kw["max_num_seqs"] == 2
    assert kw["num_gpu_blocks_override"] == 2048
    assert kw["additional_config"]["neuron_config"] == {
        "on_device_sampling_config": {}, "num_batched_tokens_buckets": [4096], "num_seqs_buckets": [2],
        "kv_segment_size_buckets": [4096]}


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("bad", [["--kv-segment-size", "300"], ["--batch-mode", "b3"]])
def test_new_args_invalid_exit_2(script, bad, tmp_path):
    mod = _load(script)
    argv = ["--batch-mode", "b1", "--decode-moe-impl", "tkg", "--out", str(tmp_path / "x.json")] + bad
    with pytest.raises(SystemExit) as e:
        mod.main(argv)
    assert e.value.code == 2


def test_common_arg_defaults_are_base():
    import argparse
    C = _load("run_logit_validation").C
    ap = argparse.ArgumentParser()
    C.add_common_args(ap)
    a = ap.parse_args(["--batch-mode", "b4", "--decode-moe-impl", "tkg", "--out", "x"])
    assert a.max_model_len == 1024 and a.kv_segment_size == 0
    assert C.BATCH_SEQS == {"b1": 1, "b2": 2, "b4": 4}


def test_segment_must_be_below_max_model_len():
    import argparse
    C = _load("run_logit_validation").C
    a = argparse.Namespace(kv_segment_size=4096, max_model_len=4096, model_dir="x", tp=64,
                           num_gpu_blocks_override=None)
    with pytest.raises(C.ConfigError):
        C.build_llm(a, 1)


def test_top5_rtol_override():
    mod = _load("run_logit_validation")
    default = {"5": (1e-5, 0.011), "50": (1e-5, 0.02), "1000": (1e-5, 0.03), "all": (1e-5, 0.05)}
    before = dict(default)
    assert mod.tolerance_map(None, default) is None          # not set: library default used unchanged
    m = mod.tolerance_map(0.02, default)
    assert m["5"] == (1e-5, 0.02) and {k: v for k, v in m.items() if k != "5"} == {k: v for k, v in default.items()
                                                                                  if k != "5"}
    assert default == before


# prompts_long.json (two >10K-token private documents) is not shipped in this repo; the tests that read it
# are skipped when it is absent.
needs_prompts_long = pytest.mark.skipif(not os.path.isfile(os.path.join(ACC, "prompts_long.json")),
                                        reason="scripts/accuracy/prompts_long.json not shipped")


@needs_prompts_long
def test_load_inputs_missing_set_and_lengths(tmp_path):
    import argparse
    import torch
    C = _load("run_logit_validation").C
    a = argparse.Namespace(prompts=os.path.join(ACC, "prompts.json"), batch_mode="b2", goldens=str(tmp_path),
                           max_model_len=1024)
    with pytest.raises(C.ConfigError, match="no prompt set 'b2'"):
        C.load_inputs(a)
    # a 12K-token prompt is rejected at the base default and accepted with --max-model-len 16384
    V = 8
    g = {"names": ["L1", "L2"], "vocab_size": V, "input_ids": {"L1": [list(range(12000))], "L2": [list(range(11000))]},
         "fp32_logits": {n: torch.zeros(16, 1, V) for n in ("L1", "L2")},
         "bf16_logits": {n: torch.zeros(16, 1, V) for n in ("L1", "L2")}}
    torch.save(g, tmp_path / "goldens.pt")
    a.prompts = os.path.join(ACC, "prompts_long.json")
    with pytest.raises(C.ConfigError, match="> 1024"):
        C.load_inputs(a)
    a.max_model_len = 16384
    names, ids, _ = C.load_inputs(a)
    assert names == ["L1", "L2"] and len(ids["L1"]) == 12000


@needs_prompts_long
def test_prompts_long_sets():
    spec = json.load(open(os.path.join(ACC, "prompts_long.json")))
    assert set(spec["sets"]) == {"b1", "b2"}
    assert spec["sets"]["b1"] == spec["sets"]["b2"] == ["L1", "L2"]
    assert all(len(spec["prompts"][n]) > 10000 for n in ("L1", "L2"))


# ---------------------------------------------------------------------------------------------
# scripts/serve.sh switches. SERVE_DRY_RUN=1 + SERVE_VENV_ACTIVATE=0,
# so no venv / device is needed. With no switch set the command must equal the base-configuration
# command recorded by the authors byte for byte.
# ---------------------------------------------------------------------------------------------
import subprocess  # noqa: E402

SERVE = os.path.join(ROOT, "scripts", "serve.sh")
BASE_CMD = (
    "serve.sh: command: vllm serve /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-bf16 --served-model-name mimo-v2.6-flash "
    "--trust-remote-code --tensor-parallel-size 64 --enable-expert-parallel --max-model-len 4096 "
    "--max-num-batched-tokens 4096 --max-num-seqs 4 --block-size 32 --no-enable-prefix-caching --host 127.0.0.1 "
    "--port 8000 --additional-config "
    r"\{\"neuron_config\":\ \{\"num_batched_tokens_buckets\":\ \[1024\,\ 4096\]\,\ \"num_seqs_buckets\":\ \[1\,\ 4\]\}\}"
    " --num-gpu-blocks-override 4096 "
)


def _serve(**env):
    e = {k: v for k, v in os.environ.items() if not (k.startswith("MIMO_") or k.startswith("SERVE_"))}
    e.pop("MODEL_DIR", None)
    e.update({"SERVE_DRY_RUN": "1", "SERVE_VENV_ACTIVATE": "0"})
    e.update(env)
    p = subprocess.run(["bash", SERVE], env=e, capture_output=True, text=True)
    cmd = [ln for ln in p.stdout.splitlines() if ln.startswith("serve.sh: command:")]
    return p.returncode, (cmd[0] if cmd else None), p.stdout + p.stderr


def test_serve_default_equals_base():
    rc, cmd, _ = _serve()
    assert rc == 0 and cmd == BASE_CMD


def test_serve_extra_blocks_not_duplicated():
    rc, cmd, out = _serve(SERVE_EXTRA="--num-gpu-blocks-override 4096",
                          MODEL_DIR="/mnt/nvme/models/MiMo-V2.6-Flash-MOPD-bf16")
    assert rc == 0 and cmd == BASE_CMD
    assert cmd.count("--num-gpu-blocks-override") == 1
    assert "WARNING" in out


@pytest.mark.parametrize("env,expect", [
    ({"MIMO_API": "1"}, ["--reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --no-async-scheduling",
                         r"\"enable_structured_outputs\":\ true", "--generation-config vllm"]),
    ({"MIMO_API": "1", "MIMO_SO": "0"}, ["--enable-auto-tool-choice --generation-config vllm"]),
    ({"MIMO_API": "1", "MIMO_REASONING_PARSER": "qwen3"}, ["--reasoning-parser qwen3"]),
    ({"MIMO_API": "1", "MIMO_GENCFG": "auto"}, ["--generation-config auto"]),
    ({"MIMO_GENCFG": "vllm"}, ["--generation-config vllm"]),
    ({"MIMO_SAMPLING": "cpu"}, [r"\"on_device_sampling_config\":\ null"]),
    ({"MIMO_CTX": "2048"}, ["--max-model-len 2048 --max-num-batched-tokens 2048", r"\[1024\,\ 2048\]"]),
    ({"MIMO_CTX": "32768"}, ["--max-model-len 32768 --max-num-batched-tokens 4096",
                             r"\"kv_segment_size_buckets\":\ \[4096\]", "--num-gpu-blocks-override 8192 "]),
    ({"MIMO_CTX": "131072"}, ["--num-gpu-blocks-override 12288 "]),
    ({"MIMO_CTX": "524288"}, ["--num-gpu-blocks-override 20480 "]),
    ({"MIMO_CTX": "1048576"}, ["--max-model-len 1048576", "--num-gpu-blocks-override 36864 "]),
    ({"MIMO_CTX": "32768", "MIMO_KV_BLOCKS": "10000"}, ["--num-gpu-blocks-override 10000 "]),
    ({"MIMO_CTX": "524288", "MIMO_KV_SEG": "1024"}, ["--max-model-len 524288 --max-num-batched-tokens 1024",
                                                     r"\"num_batched_tokens_buckets\":\ \[1024\]",
                                                     r"\"kv_segment_size_buckets\":\ \[1024\]"]),
    ({"MIMO_CTX": "1048576", "MIMO_KV_SEG": "512"}, ["--max-num-batched-tokens 512",
                                                     r"\"kv_segment_size_buckets\":\ \[512\]"]),
    ({"MIMO_KV_DTYPE": "fp8"}, ["--kv-cache-dtype fp8"]),
    ({"MIMO_DECODE_CTX_BUCKETS": "32768,65536"}, [r"\"decode_context_length_buckets\":\ \[32768\,\ 65536\]"]),
    ({"MIMO_MM": "1"}, ["vllm serve /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-bf16-mm ",
                        r"\"vision_neuron_config\":\ \{\"tp_size\":\ 8",
                        r"\"encoder_cache_num_blocks\":\ 65\}",
                        "--allowed-local-media-path " + os.path.join(os.path.expanduser("~"), "mimo-work", "mm_assets"),
                        "--limit-mm-per-prompt",
                        "--mm-processor-kwargs"]),
    ({"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "4096,65536"},
     [r"\"num_vision_tokens_buckets\":\ \[4096\,\ 65536\]\,\ \"vision_attention_block_size\":\ 4096\,"
      r"\ \"encoder_cache_num_blocks\":\ 65\}"]),
    ({"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "4096,32768"}, [r"\"num_vision_tokens_buckets\":\ \[4096\,\ 32768\]"]),
])
def test_serve_switch_values(env, expect):
    rc, cmd, out = _serve(**env)
    assert rc == 0, out
    for s in expect:
        assert s in cmd, (s, cmd)
    assert cmd.count("--num-gpu-blocks-override") <= 1


def test_serve_ctx2048_has_no_kv_blocks_and_off_disables():
    assert "--num-gpu-blocks-override" not in _serve(MIMO_CTX="2048")[1]
    assert "--num-gpu-blocks-override" not in _serve(MIMO_KV_BLOCKS="off")[1]


def test_serve_mm_keeps_explicit_model_dir():
    _, cmd, _ = _serve(MIMO_MM="1", MODEL_DIR="/x/L3-mm")
    assert "vllm serve /x/L3-mm " in cmd


def test_serve_mm_default_vision_buckets_unchanged():
    # without MIMO_MM_VISION_BUCKETS the vision config keeps the single [4096] bucket
    rc, cmd, out = _serve(MIMO_MM="1")
    assert rc == 0 and "MIMO_MM_VISION_BUCKETS=<unset>" in out
    assert (r"\"num_vision_tokens_buckets\":\ \[4096\]\,\ \"vision_attention_block_size\":\ 4096\,"
            r"\ \"encoder_cache_num_blocks\":\ 65\}") in cmd


def test_serve_key_blocks_printed():
    rc, _, out = _serve(MIMO_ATTN_PREFILL_KEY_BLOCK="8192", MIMO_ATTN_DECODE_KEY_BLOCK="32768")
    assert rc == 0 and "MIMO_ATTN_PREFILL_KEY_BLOCK=8192" in out and "MIMO_ATTN_DECODE_KEY_BLOCK=32768" in out


@pytest.mark.parametrize("ctx,timeout", [("4096", "14400"), ("131072", "14400"), ("524288", "21600"),
                                         ("1048576", "21600")])
def test_serve_ready_timeout(ctx, timeout):
    rc, _, out = _serve(MIMO_CTX=ctx)
    assert rc == 0 and f"VLLM_ENGINE_READY_TIMEOUT_S={timeout}" in out


@pytest.mark.parametrize("env", [
    {"MIMO_API": "2"}, {"MIMO_CTX": "8192"}, {"MIMO_SO": "x"}, {"MIMO_REASONING_PARSER": "deepseek"},
    {"MIMO_GENCFG": "hf"}, {"MIMO_SAMPLING": "gpu"}, {"MIMO_KV_DTYPE": "fp16"}, {"MIMO_MM": "yes"},
    {"MIMO_KV_BLOCKS": "-1"}, {"MIMO_KV_BLOCKS": "0"}, {"MIMO_KV_BLOCKS": "abc"},
    {"MIMO_DECODE_CTX_BUCKETS": "32768,"}, {"MIMO_DECODE_CTX_BUCKETS": "a"},
    {"MIMO_ATTN_PREFILL_KEY_BLOCK": "x"}, {"MIMO_ATTN_DECODE_KEY_BLOCK": "-5"},
    {"MIMO_CTX": "32768", "MIMO_KV_SEG": "3000"}, {"MIMO_KV_SEG": "1024"},
    {"MIMO_MM_VISION_BUCKETS": "4096,65536"}, {"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "4096,6000"},
    {"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "65536,4096"}, {"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "4096,131072"},
    {"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "4096,"}, {"MIMO_MM": "1", "MIMO_MM_VISION_BUCKETS": "x"},
])
def test_serve_invalid_exits_2(env):
    rc, cmd, _ = _serve(**env)
    assert rc == 2 and cmd is None
