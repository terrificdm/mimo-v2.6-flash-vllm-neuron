# Tests

CPU / NKI-simulator regression tests for the `mimo_v2_neuron` plugin, the conversion tools and the scripts.
They run on the trn2 instance inside the Neuron vLLM venv; no NeuronCore is used.

```bash
bash scripts/prepare_hf_reference.sh            # once: reference/hf + reference/hf_patched (see NOTICE)
bash scripts/venv_exec.sh bash scripts/run_tests.sh
```

`run_tests.sh` runs two pytest processes: CPU mode (`VLLM_NEURON_CPU_MODE=1`, `-m "not nkisim"`), then CPU mode
plus `NKI_SIMULATOR=1` for the four `*_sim.py` files. Tests whose data is missing are skipped, not failed.

## What each test needs

| Test | Needs | Without it |
|---|---|---|
| `test_api_common.py`, `test_script_args.py`, `test_long_tools.py` | Python + pytest (some cases torch / vLLM) | torch / vLLM cases skipped or fail to import outside the venv |
| `test_config.py`, `test_routing.py`, `test_attention_long.py` (part), `test_vision_cpu.py` (config) | `reference/hf/config.json`, `reference/hf_patched/` from `scripts/prepare_hf_reference.sh` | error (file not found) |
| `test_fallback_precondition.py` | torch | import error outside the venv |
| `test_runner_patches.py` | Neuron vLLM venv (vllm, vllm-neuron) | skipped (`importorskip`) |
| `test_vision_buckets.py` | Neuron vLLM venv: imports vllm / `factory_mm` at module load (`VLLM_NEURON_CPU_MODE=1`, no checkpoint) | import error outside the venv |
| `test_dequant.py`, `test_fp8_convert.py` | original checkpoint for some cases (`MIMO_SRC`, default `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD`) | those cases skipped |
| `test_audio_cpu.py`, `test_vision_cpu.py`, `test_mm_processor_parity.py` | original checkpoint (`MIMO_SRC_DIR` / `MIMO_MM_MODEL_DIR`) + assets (`MIMO_MM_ASSETS` / `MIMO_AUDIO_ASSETS`, default `$MIMO_WORK/mm_assets[/audio]`); ViT goldens optional (`MIMO_VIT_GOLDENS`) | skipped |
| `test_modules_cpu.py`, `test_kv_vnopad.py`, `test_attention_long.py` (part), `test_fp8_linear.py` | 3-layer checkpoints (`MIMO_L3_DIR`, `MIMO_FP8_L3_DIR`, default `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD-{bf16,fp8}-L3`) | skipped |
| `test_tiny_e2e.py`, `test_tiny_mm_e2e.py` | tiny random checkpoint from `tools/make_tiny_checkpoint.py` (`MIMO_TINY_DIR`, `MIMO_TINY_MM_DIR`, default `$MIMO_WORK/tiny/tiny3[_mm]`); logs go to `$MIMO_WORK/logs/e2e` | skipped |
| `test_moe_tkg_sim.py`, `test_fp8_nki_sim.py`, `test_kv_write_nki_sim.py`, `test_prefill_attn_nki_sim.py` | `VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1` (second pass of `run_tests.sh`) | skipped |

`MIMO_MODELS` defaults to `/mnt/nvme/models` and `MIMO_WORK` to `~/mimo-work`.

## Not included

- In `test_script_args.py`, `test_load_inputs_missing_set_and_lengths` and `test_prompts_long_sets` are skipped
  because scripts/accuracy/prompts_long.json (long prompts built from the authors' private documents) is not shipped.
- `test_long_tools.py::test_hbm_parse_eoom_sample` uses a full compiler log that is not shipped; the
  trimmed copy `tests/data/neuroncc_eoom_sample.log` is used instead and the full-log case is skipped.
