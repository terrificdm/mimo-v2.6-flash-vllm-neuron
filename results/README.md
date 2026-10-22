# Representative results

A small, curated subset of the measurements behind this tutorial, taken by the authors with the code in this repository. Large logs, profiles, compiled
artifacts, goldens, per-question task outputs and the authors' own test media are not included.

- [bench_summary.md](bench_summary.md): performance numbers with their test conditions.
- [accuracy_summary.md](accuracy_summary.md): accuracy numbers with their test conditions.
- The decisions behind the configurations: [docs/design-notes.md](../docs/design-notes.md).

Sanitization applied to every raw result file: absolute instance paths were replaced with the tutorial's variables
(`/mnt/nvme/models` → `$MIMO_MODELS`, `/mnt/nvme/mimo-work` and `/home/ubuntu/mimo-work` → `$MIMO_WORK`,
`/mnt/nvme/vllm-cache` → `$VLLM_CACHE_ROOT`, `/home/ubuntu` → `~`). JSON files still parse. Output paths inside
raw files (for example `$MIMO_WORK/results/...`) point to the authors' working directories and are not part of this
repository.
Some files name the authors' own test media (for example space_woaudio.mp4, a 3-minute 720p recording,
beach_dog.jpeg, a 2048 × 1365 photo, and the MP3 / FLAC clips), which are not redistributed. An internal decision
label printed by the offline audio check was removed.

Configurations:

- **Recommended** (`recommended_*` files, `multimodal/ask_recommended/`): the recommended configuration (1M context +
  image / video / audio + prefix caching). Two ways of starting the same compiled graphs were measured:
  - `recommended_vllm_serve_*` and `recommended_32k_*`: exactly the README Step 4 command with this version of the
    code (plugin defaults, no `MIMO_*` variables); `recommended_32k_*` is the 32K-context variant of Step 4.
  - the other `recommended_*` files: the same settings started through `scripts/serve.sh` with the plugin switches set
    as `MIMO_*` environment variables, on the code just before the plugin defaults were changed to these values.
    Same compiled graphs; the Step 4 command re-run gave the same numbers within run-to-run noise
    ([bench_summary.md](bench_summary.md)).

  Full regression with `--quick` tool / parameter / reasoning subsets
  ([docs/limitations.md](../docs/limitations.md#validation-scope-of-the-recommended-configuration)).
- **Config A** (`config_a_*` files and the other `multimodal/` files): an earlier version of the same command, before
  the long-prefill speed-ups (expert prefill kernel `shard_on_i`, gather-based key indices) and with prefix caching
  off. A reduced regression.
- **Without the video bucket** (`no_video_buckets_*` files): Config A with `num_vision_tokens_buckets [4096]` only and
  without soundfile (WAV-only audio); the full regression.

Case IDs such as `I1`–`I4`, `B1`, `V1`, `U1`–`U3` (multimodal), `T1`–`T10` and `T2b` (tool calling), `R1`–`R8`
(reasoning) are test-case IDs defined in the suite scripts under `scripts/api_tests/`.

## Files

| File | What it is |
|---|---|
| `bench/recommended_c1.json` | `vllm bench serve` c1 (512 in / 128 out, 8 requests), recommended (serve.sh with switches) |
| `bench/recommended_vllm_serve_c1.json` | Same benchmark, exactly the README Step 4 command |
| `bench/recommended_32k_c1.json` | Same benchmark, the 32K-context variant of Step 4 |
| `api/recommended_vllm_serve_needle_940k.md` | Needle at 940K (depth 0.5), exactly the Step 4 command |
| `api/recommended_vllm_serve_prefix_cache_940k.json`, `api/recommended_vllm_serve_prefix_cache_30k.json` | Prefix-cache check (first request, repeat, follow-up), exactly the Step 4 command |
| `api/recommended_32k_needle_30k.md`, `api/recommended_32k_mm.md` | Needle at 30K (3 depths) and the image / video / audio regression, 32K variant |
| `api/recommended_needle_940k.md`, `api/recommended_needle_30k.md` | Needle at 940K (depth 0.5) and 30K (3 depths), recommended |
| `api/recommended_prefix_cache_940k.json`, `api/recommended_prefix_cache_30k.json` | Same needle prompt sent twice and with a follow-up: first request, cache hit, answers |
| `api/recommended_decode_30k.json` | Decode time per token at about 30K context |
| `api/recommended_mm.md` | Image / video / audio Q&A regression, recommended |
| `api/recommended_smoke.md`, `api/recommended_tools_quick.md`, `api/recommended_params_mode_a_quick.md`, `api/recommended_reasoning_quick.md` | Smoke gate and `--quick` tool / parameter / reasoning regressions, recommended |
| `multimodal/recommended_vision_limits_check.json` | `scripts/api_tests/vision_limits_check.py` on the recommended command |
| `multimodal/recommended_hbm_runtime_summary.json` | Per-core HBM at run time, recommended (summary of a neuron-monitor snapshot) |
| `multimodal/ask_recommended/*.txt` | `scripts/api_tests/ask.py` outputs on the recommended command |
| `accuracy/l3_fp8_recommended_b1_vs_gpusim.json` | L3 logit validation with the plugin defaults vs GPU-algorithm reference |
| `bench/bf16_4096_c1.json`, `bench/bf16_4096_c4.json` | `vllm bench serve` output, BF16 weights, 4096 context (BF16 deployment, no switches) |
| `bench/fp8_native_4096_c1.json`, `bench/fp8_native_4096_c4.json` | Same benchmark, native FP8 weights, 4096 context (earlier version, native FP8 switches) |
| `bench/no_video_buckets_1m_mm_c1.json`, `bench/no_video_buckets_1m_mm_c4.json` | Same benchmark, 1M + multimodal without the video bucket |
| `bench/config_a_c1.json`, `bench/config_a_c1_run2.json` | Same benchmark, c1 only, two runs on Config A |
| `bench/1m_kv_concurrency4_c{1,2,4}.json` | 1M config (without `MIMO_VIT_DP_GROUP=world`) plus `MIMO_KV_CONCURRENCY=4` (not a default) |
| `bench/vit_timing_dp_default.json`, `bench/vit_timing_dp_world.json` | Image request latency, `scripts/api_tests/mm_encode_timing.py --sizes 448,896 --n 5`, vision DP off vs `world` |
| `api/no_video_buckets_mm.md` | Image / video / audio Q&A regression without the video bucket |
| `api/no_video_buckets_needle_940k.md`, `api/no_video_buckets_needle_30k.md` | Needle-in-a-haystack at 940K and 30K tokens, without the video bucket |
| `api/no_video_buckets_params_mode_a.md` | 17 sampling / API parameter cases, without the video bucket |
| `api/no_video_buckets_reasoning_quick.md`, `api/no_video_buckets_tools_quick.md` | Quick reasoning-parser and tool-calling regression, without the video bucket |
| `api/no_video_buckets_smoke.md` | Smoke gate (10 requests), without the video bucket |
| `api/api_matrix.zh.md` | Full API capability matrix (in Chinese) |
| `api/config_a_mm.md` | Image / video / audio Q&A regression on Config A |
| `api/config_a_needle_30k.md`, `api/config_a_needle_940k_depth50.md` | Needle at 30K (3 depths) and 940K (depth 0.5 only), Config A |
| `api/config_a_smoke.md`, `api/config_a_tools_quick.md`, `api/config_a_params_mode_a_quick.md`, `api/config_a_reasoning_quick.md` | Smoke gate and `--quick` tool / parameter / reasoning regressions, Config A |
| `multimodal/vision_limits_check.json` | `scripts/api_tests/vision_limits_check.py` on Config A: text, 3-minute 720p video, image, two images, rejected 18-block request (HTTP 400), request after it |
| `multimodal/video_16384_check.json` | 1024 × 1024, 64-frame synthetic video (16,384 vision tokens, the limit) |
| `multimodal/long_video_timing.json` | `scripts/api_tests/mm_long_timing.py`: long video / large photo / test video against text of the same length |
| `multimodal/large_image_timing.json` | `scripts/api_tests/mm_encode_timing.py --sizes 1024,2048 --n 3` on Config A |
| `multimodal/vit_compare.json` | Device ViT vs HF on the long video and the photo (`scripts/accuracy/compare_vit_dumps.py`) |
| `multimodal/hbm_runtime_summary.json` | Per-core HBM at run time, Config A (summary extracted from a neuron-monitor snapshot) |
| `multimodal/audio_offline.txt` | Offline audio decoding check with soundfile (MP3, FLAC, 7 WAV files) |
| `multimodal/ask/*.txt` | `scripts/api_tests/ask.py` outputs on Config A: images, videos, WAV / MP3 / FLAC, a rejected 17-block request |
| `accuracy/l3_fp8_final_b1_vs_gpusim.json` | L3 logit validation, Config A kernel switches (earlier version), vs GPU-algorithm reference |
| `accuracy/l3_fp8_seg4k_torch_b1_vs_gpusim.json` | Same, control run: segment 4096 with torch attention and no 512 bucket |
| `accuracy/task_eval_gsm8k_fp8_vs_bf16.json`, `accuracy/task_eval_mmlu_fp8_vs_bf16.json` | Per-question agreement, FP8 1M without the video bucket vs BF16 4096 server (`tools/task_eval_agreement.py`) |
| `accuracy/task_eval_item_ids.json` | Only the `idx` field of the 200 + 200 sampled questions (question text not redistributed; dataset licenses not checked) |
| `conversion/fp8_convert_check_fulldir.json` | Element-wise check of the full FP8 conversion (`tools/check_fp8_converted.py`) |
