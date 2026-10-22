# Configuration reference

Every environment variable read by the scripts and the plugin in this repository, with its default and the file that
reads it. Defaults are the values in the code; when a variable is unset the behavior equals the validated defaults.

Two rules that matter in practice:

- Background jobs run in tmux (`scripts/start_bg.sh`), and **tmux does not inherit `export`ed variables**. Put every
  variable into the command with an `env` prefix, as in the README commands.
- `scripts/serve.sh` validates its own switches and exits with code 2 on an invalid value. Variables marked
  "passed through" are only printed by `serve.sh` and read by the plugin inside the vLLM workers.

## 1. Paths

| Variable | Default | Purpose | Read by |
|---|---|---|---|
| `MIMO_VENV` | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0` (Neuron DLAMI) | venv activated by the scripts | `serve.sh`, `bench.sh`, `download_weights.sh`; fallback for `VENV` in `venv_exec.sh`, `api_tests/run_suite.sh` |
| `VENV` | `$MIMO_VENV` | venv override | `venv_exec.sh`, `api_tests/run_suite.sh` |
| `MIMO_MODELS` | `/mnt/nvme/models` | root of all model directories | all scripts, tools and tests |
| `MODEL_DIR` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16` (`-bf16-mm` when `MIMO_MM=1` and `MODEL_DIR` is unset) | model directory to serve / benchmark / tokenize | `serve.sh`, `bench.sh`, `api_tests/needle.py` (tokenizer), `api_tests/offline_parser_check.py` |
| `MIMO_WORK` | `$HOME/mimo-work` | logs, status files, results, pydeps, goldens, tiny models, test media | all scripts; the README links it to `/mnt/nvme/mimo-work` |
| `MIMO_MM_ASSETS` | `$MIMO_WORK/mm_assets` | `--allowed-local-media-path` of the server; default `--assets` / `--out` of the media tools; the larger test media of [development.md](development.md#larger-test-media) go to `$MIMO_MM_ASSETS/large` | `serve.sh`, `api_tests/mm_qa.py`, `api_tests/make_mm_assets.py`, `api_tests/vision_limits_check.py`, `api_tests/mm_long_timing.py`, tests |
| `VLLM_CACHE_ROOT` | `/mnt/nvme/vllm-cache` | vLLM / Neuron compile cache (`$VLLM_CACHE_ROOT/neuron/compile_cache`) | `serve.sh` (exported) |
| `HF_HOME` | `/mnt/nvme/hf-cache` | Hugging Face cache during download | `download_weights.sh` |
| `DST` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD` (`download_weights.sh`), `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16` (`convert_full.sh`) | output directory | `download_weights.sh`, `convert_full.sh` |
| `SRC` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD` | original checkpoint | `convert_full.sh` |
| `WORKERS` | `32` | converter worker processes | `convert_full.sh` |
| `MIMO_SRC_DIR` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD` | downloaded original checkpoint (only `*.py`, `*.json` needed) | `prepare_hf_reference.sh`; `tests/test_audio_cpu.py` |
| `MIMO_REPO` | the repository that contains `start_bg.sh` | working directory of background jobs | `start_bg.sh` |
| `MIMO_PYDEPS` | `$MIMO_WORK/pydeps` (`off` disables) | packages installed outside the venv, appended to the end of `PYTHONPATH` if the directory exists: soundfile 0.14.0 for MP3 / FLAC (MP3 / FLAC / Ogg validated end to end with the recommended configuration; install: README Step 1). Without it only PCM WAV is read | `serve.sh` |
| `HF_TOKEN` | unset | Hugging Face token, only if your account needs one | `hf download` in `download_weights.sh` |

## 2. `scripts/serve.sh` switches

`scripts/serve.sh` is optional: it only assembles a `vllm serve` command like the one in README Step 4 (print it
with `SERVE_DRY_RUN=1`). The deployment itself does not need it.

| Variable | Default | Effect |
|---|---|---|
| `MIMO_CTX` | `4096` | `--max-model-len`; one of `2048`, `4096`, `32768`, `131072`, `524288`, `1048576`. ≥ 32768 turns on segmented prefill |
| `MIMO_KV_SEG` | `4096` | prefill segment length for `MIMO_CTX` ≥ 32768 (`4096`, `2048`, `1024`, `512`): sets `--max-num-batched-tokens`, `num_batched_tokens_buckets` and `kv_segment_size_buckets` |
| `MIMO_KV_BLOCKS` | by context: 4096 → 4096, 32K → 8192, 128K → 12288, 512K → 20480, 1M → 36864, 2048 → not set; `off` removes it | `--num-gpu-blocks-override` (block size 32). Required: without it vLLM sizes the KV pool from free HBM and the decode graph runs out of HBM at compile time |
| `MIMO_API` | `0` | `1`: `--reasoning-parser $MIMO_REASONING_PARSER --tool-call-parser mimo --enable-auto-tool-choice` |
| `MIMO_REASONING_PARSER` | `mimo` | `mimo` or `qwen3` |
| `MIMO_SO` | `1` | with `MIMO_API=1`: `--no-async-scheduling` + `neuron_config.enable_structured_outputs` (JSON object / schema outputs) |
| `MIMO_GENCFG` | `vllm` when `MIMO_API=1`, otherwise not set | `--generation-config` (`auto` or `vllm`) |
| `MIMO_SAMPLING` | `device` | `cpu`: `on_device_sampling_config: null` (all sampling on CPU, "mode B") |
| `MIMO_KV_DTYPE` | `bf16` | `fp8`: `--kv-cache-dtype fp8`. **Not validated** with the final configuration (task skipped) |
| `MIMO_KV_SCALES` | unset | printed only; not read by the plugin in this version |
| `MIMO_MM` | `0` | `1`: `--limit-mm-per-prompt` (2 images, 1 video), `--allowed-local-media-path $MIMO_MM_ASSETS`, `--mm-processor-kwargs`, and `vision_neuron_config` (TP 8, DP = `SERVE_TP` ÷ 8, vision buckets `[4096]` or `MIMO_MM_VISION_BUCKETS`, block 4096) |
| `MIMO_AUDIO` | `0` | `1` (needs `MIMO_MM=1` and a directory built with `make_mm_checkpoint.py --audio`): allows 1 audio per request |
| `MIMO_MM_PROCESSOR_KWARGS` | `{"max_pixels": 1048576}` | `--mm-processor-kwargs` JSON. The MiMo processor ignores it; the effective pixel cap is written into the `-mm` directory's config.json by `make_mm_checkpoint.py --max-pixels` |
| `MIMO_MM_VISION_BUCKETS` | unset (= `[4096]`) | needs `MIMO_MM=1`; increasing multiples of 4096, at most 65536, otherwise exit 2. Sets `vision_neuron_config.num_vision_tokens_buckets` and is exported to the API process and the workers. With `4096,65536` (recommended configuration; a plain `vllm serve` reads the same list from `--additional-config`) a video can have up to 16,384 vision tokens (largest bucket ÷ 4) and a request up to 16 encoder blocks (largest bucket ÷ 4096); the API process rejects larger requests with HTTP 400, and the workers check that the value matches `vision_neuron_config` ([design-notes](design-notes.md#vision-request-blocks)). Single images stay at ≤ 1024 tokens. Unset: one block per image / video, ≤ 1024 tokens, no request check |
| `MIMO_MM_CACHE_BLOCKS` | `65` | `vision_neuron_config.encoder_cache_num_blocks` (8 MiB per block). The automatic value fails on the 5th distinct image with "Encoder cache full" |
| `MIMO_MAX_SEQS` | `4` (must be ≥ 4) | `--max-num-seqs`; > 4 adds a third decode batch bucket `[1, 4, n]` |
| `MIMO_PREFIX_CACHE` | `0` | `1`: `--enable-prefix-caching` (only for `MIMO_CTX` ≥ 32768). Validated up to 940K tokens; an earlier version hung on long cache hits and is fixed, see [limitations.md](limitations.md) |
| `MIMO_DECODE_CTX_BUCKETS` | unset | `neuron_config.decode_context_length_buckets`, e.g. `4096,32768,131072,524288` |
| `MIMO_ATTN_PREFILL_KEY_BLOCK`, `MIMO_ATTN_DECODE_KEY_BLOCK` | unset (plugin: prefill by context 4096–32768, decode 32768) | key block sizes of the torch attention path (passed through; `plugin/.../config.py`) |
| `MIMO_DECODE_MOE_IMPL` | `tkg` | decode MoE implementation `tkg` / `cte` / `torch` (exported by `serve.sh`; `plugin/.../config.py`). Use `tkg` |
| `SERVE_TP` | `64` | `--tensor-parallel-size` (`--enable-expert-parallel` when > 1) |
| `SERVE_PORT` | `8000` | `--port`. The server always binds `--host 127.0.0.1` |
| `SERVE_EXTRA` | empty | extra `vllm serve` arguments (word-split) |
| `SERVE_DRY_RUN` | `0` | `1`: print the effective environment and the full `vllm serve` command, then exit |
| `SERVE_VENV_ACTIVATE` | `1` | `0`: do not activate the venv (dry runs on machines without it) |
| `SERVE_ALLOW_CPU` | `0` | `1`: re-export `VLLM_NEURON_CPU_MODE=1` (tiny CPU drills only) |
| `MIMO_PROFILE_DIR`, `MIMO_PROFILE_ACTIVITIES` | unset / `"device_profile", "system_profile"` | adds `neuron_profiler` and `--profiler-config` (`/start_profile`, `/stop_profile`). At 1M the device profiler does not fit in HBM; use `MIMO_PROFILE_ACTIVITIES='"system_profile"'` |

The plugin fills in three vLLM timeouts when they are unset (`VLLM_ENGINE_READY_TIMEOUT_S=21600`,
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`, `VLLM_NEURON_COMPILATION_TIMEOUT=7200`), so a plain `vllm serve` does not
time out while loading or compiling. Exported by `serve.sh` (not configurable): `VLLM_NEURON_COMPILATION_TIMEOUT=7200`,
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200` (vLLM default 300 s), `VLLM_ENGINE_READY_TIMEOUT_S=14400` (21600 for
`MIMO_CTX` ≥ 524288; vLLM default 600 s). `serve.sh` also unsets `VLLM_NEURON_CPU_MODE`, `NKI_SIMULATOR` and
`VLLM_NEURON_DISABLE_NKI_KERNELS`, and the README commands additionally pass `env -u` for them.

<a id="plugin-switches"></a>
## 3. Plugin switches

Read by the plugin inside the workers (and the API process), not by vLLM. **Leave them unset**: the default is the
validated value (first value in the Default column; [design note](design-notes.md#plugin-defaults)). The second value
(“CPU mode”) applies with `VLLM_NEURON_CPU_MODE=1` (CPU tests) and is the older implementation; set a variable only to
switch back to it for troubleshooting. Each worker logs one `MiMoV2 Neuron: switches: NAME=value(source) ...` line.

| Variable | Default | Effect | File (`plugin/mimo_v2_neuron/...`) |
|---|---|---|---|
| `MIMO_FP8_IMPL` | `nki` (CPU mode `emul`) | linear-layer FP8: `nki` = native FP8 NKI kernel, `emul` = BF16 emulation. FP8 checkpoints only; falls back to `emul` when `MIMO_FP8_ACT_MAX` is not 448 | `model/mimo_v2/fp8_linear.py` |
| `MIMO_FP8_EXPERT_IMPL` | `row` (CPU mode `expand`) | expert FP8: `row` = nkilib MoE kernels read FP8 directly, `expand` = expand to BF16 each step | `model/mimo_v2/config.py` |
| `MIMO_FP8_ACT` | `1` | emulation path only: `0` disables activation quantization | `model/mimo_v2/fp8_linear.py` |
| `MIMO_FP8_ACT_MAX` | `448` | emulation path only: activation FP8 range, `448` (GPU E4M3FN) or `240` | `model/mimo_v2/fp8_linear.py` |
| `MIMO_FP8_MM` | `bf16` | emulation path only: matmul operand type (`fp32` returns NaN on the device) | `model/mimo_v2/fp8_linear.py` |
| `MIMO_KV_VNOPAD` | `1` (CPU mode `0`) | `1`: store V with 128 dims instead of padding to 192 (same layout as the GPU DIFFKV backend; saves about 2.55 GiB compiler peak per core at 1M) | `model/mimo_v2/model.py` |
| `MIMO_KV_WRITE` | `nki` (CPU mode `index_put`) | `nki`: write KV in place with an NKI scatter kernel (avoids a full KV-pool copy per step) | `model/mimo_v2/model.py` |
| `MIMO_MOE_CTE_IMPL` | `hybrid` (CPU mode `shard_on_i`) | prefill expert kernel of the FP8 `row` experts: `hybrid` = nkilib `shard_on_i_hybrid`, loops over non-empty expert blocks only (940K first token 324.7 → 204.2 s); `shard_on_i` = static loop over all blocks | `model/mimo_v2/config.py` |
| `MIMO_PREFILL_KIDX` | `arith` (CPU mode `gather`) | key row indices of the NKI prefix kernel: `arith` = broadcast arithmetic, `gather` = 1M-element `index_select` (same values) | `model/mimo_v2/attention_torch.py` |
| `MIMO_KV_GATHER` | `xla` | `nki`: gather KV blocks with indirect DMA (no gain measured; not used) | `model/mimo_v2/attention_torch.py` |
| `MIMO_PREFILL_ATTN` | `nki` (CPU mode `torch`) | `nki`: GA-layer prefix attention of segmented prefill uses the NKI flash kernel (allows segment 4096); layers whose shapes do not fit use torch | `model/mimo_v2/attention_torch.py` |
| `MIMO_PREFILL_BUCKETS` | `auto` (CPU mode unset) | extra prefill buckets below the segment length, e.g. `512,4096`; the last value must equal the segment length. `auto` = `512,<segment>` when the segment is > 512 and < `--max-model-len`; `off` = none. Patches both the runner and the scheduler | `model/mimo_v2/factory.py`, `runner_patches.py` |
| `MIMO_CC_MODULAR_MAC` | `100000000000000` (CPU mode unset) | replaces the runner's fixed `--modular-flow-mac-threshold=10` for neuronx-cc (`<n>` or `none`; `off` = keep 10) | `runner_patches.py` |
| `MIMO_CC_MODULAR_MAC_MAXTOK` | `64` (CPU mode unset) | applies the previous setting only to graphs with ≤ n tokens (the decode graphs). Keep `64`: un-partitioned prefill graphs exhaust host memory | `runner_patches.py` |
| `MIMO_KV_CONCURRENCY` | unset | replaces the scheduler's concurrency cap (1 at 1M). The server hangs if the running requests together exceed the KV pool; not a default | `runner_patches.py` |
| `MIMO_ODS_CPU_RESAMPLE` | `1` | with device sampling, batches that use `seed` / `presence_penalty` / `frequency_penalty` are re-sampled on CPU (`0` turns this off; these parameters are then ignored) | `runner_patches.py` |
| `MIMO_VIT_DP_GROUP` | `world` (CPU mode `0`) | vision data parallel: `0` = each vision TP group encodes all blocks; `world` = split blocks and all-gather over all 64 ranks; `1` = column groups (fails to load on trn2, CPU tests only) | `model/mimo_v2/model_mm.py` |
| `MIMO_VIT_SINKS` | `1` | `0` disables the attention sinks of the windowed ViT layers (diagnostics only) | `model/mimo_v2/vision_encoder.py` |
| `MIMO_AUDIO_DEVICE` | `neuron` | audio towers on the device or on worker CPU (`cpu`) | `model/mimo_v2/model_mm.py` |
| `MIMO_AUDIO_TP` | `replicated` | audio tower sharding (full copy per rank, no collectives) | `model/mimo_v2/model_mm.py` |
| `MIMO_AUDIO_RVQ` | `first_gather` | RVQ nearest-codeword selection: `first_gather` = first max index via compare + min, then `index_select`. The `topk` forms (`gather`, `topk_onehot`) compile but fail to load on trn2 (out-of-bounds DMA) | `model/mimo_v2/audio_encoder.py` |
| `MIMO_AUDIO_CODEBOOK` | `bf16` | round codebooks to BF16 like vLLM (`fp32` keeps them) | `model/mimo_v2/audio_encoder.py` |
| `MIMO_AUDIO_CPU_THREADS` | `2` | threads of the CPU audio fallback | `model/mimo_v2/model_mm.py` |
| `MIMO_DEBUG_CAPTURE`, `MIMO_DEBUG_METADATA`, `MIMO_VIT_DUMP_DIR`, `MIMO_AUDIO_DUMP_DIR`, `MIMO_AUDIO_EMBED_OVERRIDE` | unset | validation / debugging hooks used by `scripts/accuracy/*`; do not set when serving | `model/mimo_v2/model.py`, `model/mimo_v2/model_mm.py` |

## 4. Other scripts

| Variable | Default | Purpose | Read by |
|---|---|---|---|
| `PORT` | `8000` | server port used by clients | `bench.sh`, `smoke_test.sh`, `api_tests/common.py`, `api_tests/run_suite.sh` |
| `BENCH_OUT` | `$MIMO_WORK/results/bench` | `vllm bench serve` result directory | `bench.sh` |
| `SMOKE_OUT` | `$MIMO_WORK/results/smoke` | smoke request / response files | `smoke_test.sh` |
| `SMOKE_MAX_TOKENS` | unset (completions 128, chat 512) | override `max_tokens` (the authors used `128` for ≥ 512K servers) | `smoke_test.sh` |
| `MIMO_RESULTS` | `$MIMO_WORK/results` | API suite output root (`$MIMO_RESULTS/api/...`) | `api_tests/run_suite.sh` |
| `RUN_SUITE_NO_VENV` | `0` | `1`: skip venv activation (unit tests) | `api_tests/run_suite.sh` |
| `MIMO_API_BASE` | `http://127.0.0.1:$PORT` | API base URL | `api_tests/common.py` |
| `MIMO_API_MODEL` | `mimo-v2.6-flash` | served model name | `api_tests/common.py` |
| `MEM_GUARD_INTERVAL` | `30` | seconds between samples | `mem_guard.sh` |
| `MEM_GUARD_MIN_GB` | `200` | kill compiler processes when available host memory drops below this | `mem_guard.sh` |
| `REPRO_BACKEND`, `REPRO_DEVICE`, `REPRO_CHOICE` | `neuron_libtorch`, `neuron`, unset | device-bug reproducers | `tools/repro_split.py`, `tools/repro_topk.py` |

## 5. Tests

| Variable | Default | Read by |
|---|---|---|
| `MIMO_SRC` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD` | `tests/test_dequant.py`, `tests/test_fp8_convert.py` |
| `MIMO_SRC_DIR` | same | `tests/test_audio_cpu.py` |
| `MIMO_L3_DIR`, `MIMO_FP8_L3_DIR` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16-L3`, `...-fp8-L3` | `tests/test_modules_cpu.py`, `tests/test_attention_long.py`, `tests/test_fp8_linear.py` |
| `MIMO_MM_MODEL_DIR` | `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD` (original checkpoint) | `tests/test_vision_cpu.py`, `tests/test_mm_processor_parity.py` |
| `MIMO_MM_ASSETS`, `MIMO_AUDIO_ASSETS` | `$MIMO_WORK/mm_assets`, `$MIMO_WORK/mm_assets/audio` | multimodal tests |
| `MIMO_VIT_GOLDENS` | `$MIMO_WORK/goldens/mm/vit_goldens.pt` | `tests/test_vision_cpu.py` |
| `MIMO_VIT_TEST_TP` | `8` | `tests/test_vision_cpu.py` |
| `MIMO_TINY_DIR`, `MIMO_TINY_MM_DIR` | `$MIMO_WORK/tiny/tiny3`, `$MIMO_WORK/tiny/tiny3_mm` | `tests/test_tiny_e2e.py`, `tests/test_tiny_mm_e2e.py` |
| `VLLM_NEURON_CPU_MODE`, `NKI_SIMULATOR` | set by `scripts/run_tests.sh` | CPU-mode and NKI-simulator passes |

See [tests/README.md](../tests/README.md) for what each test needs.
