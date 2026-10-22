# Performance summary

All numbers were measured on one trn2.48xlarge (TP=64, expert parallel 64, decode MoE `tkg`) with the software
versions in the README "Prerequisites" section. **"Recommended"** is the recommended configuration: the rows marked "Step 4 command" ran exactly the README
Step 4 command with this version of the code; the other "Recommended" rows ran the same settings through
`scripts/serve.sh` with the plugin switches as environment variables (same compiled graphs, code just before the
plugin defaults were changed to these values). The other rows are earlier versions, kept for comparison: **"Config A"** is the same command before the two
long-prefill speed-ups (expert prefill kernel `shard_on_i` instead of `hybrid`, gather-based key indices) and before
prefix caching was enabled; **"without the video bucket"** is Config A with `num_vision_tokens_buckets [4096]` only and
WAV-only audio (same text path). Files shipped here are listed in [README.md](README.md); decisions are explained in
[docs/design-notes.md](../docs/design-notes.md).

There is no GPU (H100) latency or throughput measurement for this model in the authors' runs, so no GPU
comparison is given.

## 1. Short requests (`scripts/bench.sh`)

Conditions: `vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos`,
two warm-up requests first. c1 = concurrency 1, 8 requests; c2 = concurrency 2, 8 requests; c4 = concurrency 4,
16 requests. Values are means from the `vllm bench serve` JSON (median in brackets where it differs noticeably).
`--max-num-seqs 4` in every row.

| Config | c1 TTFT (ms) | c1 TPOT (ms) | c1 out tok/s | c4 TTFT (ms) | c4 TPOT (ms) | c4 out tok/s | File |
|---|---|---|---|---|---|---|---|
| BF16 weights, 4096 context | 270.3 | 46.11 | 20.89 | 696.3 | 54.14 | 67.57 | `bench/bf16_4096_c{1,4}.json` |
| Native FP8 weights, 4096 context (earlier version) | 291.3 | 43.09 | 22.21 | 742.7 (739.1) | 52.14 | 69.48 | `bench/fp8_native_4096_c{1,4}.json` |
| **Recommended, 1M + image/video/audio + prefix caching** | **198.3** | **47.15** | 20.69 | not re-run | not re-run | not re-run | `bench/recommended_c1.json` |
| Recommended, Step 4 command (re-run) | 199.1 (198.8) | 47.46 | 20.56 | not re-run | not re-run | not re-run | `bench/recommended_vllm_serve_c1.json` |
| Recommended, 32K-context variant of Step 4 | 126.2 (125.9) | 38.35 | 25.62 | not re-run | not re-run | not re-run | `bench/recommended_32k_c1.json` |
| Config A, 1M + image/video/audio, run 1 | 308.2 | 47.56 | 20.16 | not re-run | not re-run | not re-run | `bench/config_a_c1.json` |
| Config A, run 2 | 308.8 | 47.72 | 20.10 | not re-run | not re-run | not re-run | `bench/config_a_c1_run2.json` |
| 1M + image/video/audio without the video bucket | 306.7 | 46.83 | 20.46 | 16687 (19020) | 46.82 | 20.50 | `bench/no_video_buckets_1m_mm_c{1,4}.json` |
| 1M + `MIMO_KV_CONCURRENCY=4` (not default; run before `MIMO_VIT_DP_GROUP=world` was added) | 306.0 | 46.87 | 20.45 | 755.0 | 73.36 | 50.83 | `bench/1m_kv_concurrency4_c{1,2,4}.json` |

- The recommended short-request TTFT is lower than Config A's because the expert prefill kernel no longer loops over
  empty expert blocks; TPOT is unchanged (decode graphs are the same).
- Effect of the video bucket: TTFT +0.5 / +0.7 %, TPOT +1.6 / +1.9 % in these runs (gate +5 %). A later same-day
  comparison put it at +0.6–1.6 % (about 0.3–0.7 ms per token) and traced it to the larger vision graphs loaded at
  start-up (more HBM per core); the decode graph itself is the same
  ([docs/limitations.md](../docs/limitations.md#multimodal)).
- c4 on the 1M config is serialized: vllm-neuron admits `KV tokens // max_model_len` = 36864 × 32 / 1,048,576
  = 1 running request, so the other three wait
  ([design-notes.md](../docs/design-notes.md#concurrency)).
- c2 with `MIMO_KV_CONCURRENCY=4`: TTFT 456.0 ms, TPOT 71.0 ms, 27.02 tok/s (`bench/1m_kv_concurrency4_c2.json`).
  TPOT rises because decode is compiled only for batch 1 and 4, so two requests are padded to 4.
- FP8 with BF16 emulation (the older implementation, `MIMO_FP8_IMPL=emul`, `MIMO_FP8_EXPERT_IMPL=expand`), 4096 context:
  c1 TTFT 455 ms (median), TPOT 89.5 ms; c4 36.6 tok/s (raw file not shipped).
- 4096 context, native FP8, `MIMO_MAX_SEQS=8`: c4 61.8 tok/s, c8 65.8 tok/s with 9352 ms TTFT; little gain because
  prefill cannot be batched with decode (raw files not shipped).

## 2. Long context (needle-in-a-haystack)

Conditions: `scripts/api_tests/run_suite.sh needle` with fixed prompts (`needle.py` builds them deterministically), 3 needle depths
(10 %, 50 %, 90 %), one request at a time, segment length 4096.

| Config | Prompt length | Result | TTFT | Per 4096-token prefill segment | File |
|---|---|---|---|---|---|
| **Recommended** | about 30K tokens | 3/3 correct | 3.83–3.84 s cold (depth 0.1 was a cache hit of the step before: 0.27 s) | 0.48 s | `api/recommended_needle_30k.md` |
| **Recommended** | about 940K tokens | 1/1 correct (depth 0.5; 3 depths on Config A and on an earlier prefix-cache build) | 204.5 s | 0.889 s | `api/recommended_needle_940k.md` |
| Recommended, Step 4 command (re-run) | about 940K tokens | 1/1 correct (depth 0.5) | 204.8 s | 0.890 s | `api/recommended_vllm_serve_needle_940k.md` |
| Recommended, 32K-context variant of Step 4 | about 30K tokens | 3/3 correct | 3.26 s | 0.408 s | `api/recommended_32k_needle_30k.md` |
| Config A | about 30K tokens | 3/3 correct | 7.98–8.00 s | — | `api/config_a_needle_30k.md` |
| Config A | about 940K tokens | 1/1 correct, **depth 0.5 only** | 323.6 s | 1.407 s | `api/config_a_needle_940k_depth50.md` |
| Without the video bucket | about 30K tokens | 3/3 correct | 7.96–7.97 s | 0.996 s | `api/no_video_buckets_needle_30k.md` |
| Without the video bucket | about 940K tokens | 3/3 correct | 323.2–323.3 s | 1.405 s | `api/no_video_buckets_needle_940k.md` |

- Prefix-cache hit (same prompt sent again, recommended): 940K 204.2 → 2.89 s, 30K 3.88 → 0.25 s, answer identical to
  the first request (`api/recommended_prefix_cache_{940k,30k}.json`). An earlier build measured 120K 31.3 → 0.58 s and
  470K 137.9 → 1.63 s (before the expert-kernel speed-up; raw files not shipped).
  Exactly the Step 4 command (re-run): 940K 204.5 → 2.82 s, 30K 3.87 → 0.26 s, answers identical
  (`api/recommended_vllm_serve_prefix_cache_{940k,30k}.json`).
- Decode at about 30K context: 47.5 ms/token on the recommended command (`api/recommended_decode_30k.json`), 48.5
  ms/token on Config A, 48.0 ms/token without the video bucket (raw files of the last two not shipped).
- Before the speed-ups (segment 512, torch attention, no decode buckets) the same 940K prompt took 1065 s
  (multimodal config) or 1279 s (text-only config); see [design-notes.md](../docs/design-notes.md#kv-write) and
  [V stored with 128 dims](../docs/design-notes.md#v-no-pad).

## 3. Image encoding (without the video bucket, earlier version)

Conditions: `scripts/api_tests/mm_encode_timing.py --sizes 448,896 --n 5`, median of 5 requests, end-to-end
request latency.

| Image | `MIMO_VIT_DP_GROUP=0` | `MIMO_VIT_DP_GROUP=world` (now the default) | Files |
|---|---|---|---|
| 448 × 448 | 1.587 s | 0.633 s (encoder part 0.331 s) | `bench/vit_timing_dp_default.json`, `bench/vit_timing_dp_world.json` |
| 896 × 896 | 2.319 s | 1.369 s | same |

## 3b. Long video and large images (Config A)

Conditions: thinking off, greedy, `max_tokens` 1, so the request time is about the TTFT. Files under
`results/multimodal/`. The authors used a 3-minute 720p recording and two photos (2048 × 1365 and a smaller one);
[docs/development.md](../docs/development.md#larger-test-media) generates synthetic media of the same sizes, which the
authors did not run.

On the recommended command the authors also sent 720p videos of 2, 3 and 4 minutes (cut from the same recording,
2 fps): each was 14,211 input tokens (the frame count is fixed, so the token count follows the resolution, not the
duration), all were answered correctly, first token 3.3–5.2 s with thinking off (raw output not shipped).

| Measurement | Result | File |
|---|---|---|
| 3-minute 720p video (14,080 vision tokens, prompt 14,210 tokens), first request after start-up | 9.95 s | `multimodal/vision_limits_check.json` (`video1`) |
| Same video again (encoder already run once) | 8.86 / 8.85 s | `multimodal/long_video_timing.json` (its `cold` entry ran after the check above, so it is not a cold request) |
| Text-only request of the same length | 3.97 s | same |
| Same video in `ask.py` (sent as base64), time to first token | 9.13 s | `multimodal/ask/vid_space_3min.txt` |
| 1024 × 1024 / 2048 × 2048 noise image (both served at ≤ 1024 × 1024), median of 3 | 1.39 s / 1.50 s | `multimodal/large_image_timing.json` |
| 20-token / 1033-token text-only request | 0.31 s / 0.99 s | `multimodal/large_image_timing.json`, `multimodal/long_video_timing.json` |
| 2048 × 1365 photo in `ask.py`, time to first token | 1.05 s | `multimodal/ask/img_beach_dog.txt` |

- Estimates from differences (not timed separately): for the video, media decoding + preprocessing in the API process
  about 4.9 s (8.86 − 3.97; OpenCV picks 32 of 5334 frames, unrelated to Neuron) and ViT encoding about 1.1 s (9.95 −
  8.86, assuming the second request reuses the encoder cache). For the large images the vision part (PNG decoding,
  resizing and ViT) is about 0.40–0.51 s against the 1033-token text request.

## 4. Start-up time

From the authors' record of validated server commands (`ready_minutes` = launch until `/v1/models` answers,
including weight loading and compilation):

| Config | Ready (minutes) | Note |
|---|---|---|
| Recommended, Step 4 command | 16.0 | compile cache hit for every graph; first compile about 1 hour |
| Recommended (`serve.sh` + environment variables) | 17.9 | compile cache hit for all but the two text prefill graphs (allow about 20 minutes) |
| Config A | 16.3 | compile cache hit |
| 1M + multimodal without the video bucket | 15.8 | compile cache hit |
| Native FP8, 4096, CPU sampling (`MIMO_SAMPLING=cpu MIMO_GENCFG=auto`) | 14.9 / 15.6 | the device-sampling variant has no recorded ready time |
| BF16, 4096, API (`MIMO_API=1`) | 4.2–8.8 | needs the BF16 directory |

## 5. Device memory (per NeuronCore, limit 24 GiB)

Recommended: compiler peak 17.35 GiB for the two prefill graphs (17.23 GiB without the video bucket), decode graph
about 11.1 GiB ([design-notes.md](../docs/design-notes.md#hbm-budget)). At run time neuron-monitor showed **20.13 GiB**
on the busiest core (tensors 13.17, shared scratchpad 4.5, model code 2.26, constants 0.19 GiB;
`multimodal/recommended_hbm_runtime_summary.json`); Config A 20.25 GiB (`multimodal/hbm_runtime_summary.json`), 19.94
GiB without the video bucket; the gate was 23 GiB.
