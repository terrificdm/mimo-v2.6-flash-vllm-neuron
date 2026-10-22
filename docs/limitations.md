# Known limitations

Status of **this version** of the tutorial (the recommended command in README Step 4: 1M context + image / video /
audio + prefix caching, Neuron SDK 2.32.0, vllm-neuron 0.24.0.1.1.0, XLA backend). Several items are choices made in
this port rather than limits of the model; they are marked as such. Directions for lifting them are in
[roadmap.md](roadmap.md).

## Platform

- **Versions.** Validated only on the versions listed in the README "Prerequisites" section.
  `plugin/mimo_v2_neuron/runner_patches.py` patches vllm-neuron 0.24.0.1.1.0 at run time and checks short source
  excerpts of that release; other vllm-neuron versions are expected to need changes. Not validated on Neuron SDK 2.33 /
  vllm-neuron 0.29 or on the native PyTorch backend (`VLLM_NEURON_BACKEND=neuron_native`).
- **Hardware.** One trn2.48xlarge (64 logical NeuronCores, TP=64). Trn3 and multi-instance serving: not validated.
- **Not an official vllm-neuron model.** It is an out-of-tree vLLM general plugin (`mimo_v2_neuron`).
- **No authentication.** The recommended command binds `127.0.0.1:8000` and starts no auth layer. Expose it only
  through an authenticated proxy or an SSH tunnel.

## Start-up

- First compile of the recommended 1M + multimodal configuration takes about 1 hour; with a warm compile cache the
  server is ready in about 16–18 minutes (16.0 minutes recorded with the Step 4 command and every graph cached;
  17.9 minutes for a start that had to recompile the two text prefill graphs).
- The compile cache (`$VLLM_CACHE_ROOT`) and the weights live on instance-store NVMe. Stopping or terminating the
  instance loses them; the RAID is not in `/etc/fstab` and must be re-assembled after a reboot.
- Compiling un-partitioned graphs is memory-hungry on the host (up to 135 GB for one compiler process), which is why
  only decode graphs are compiled without partitions and `scripts/mem_guard.sh` should run during compilation.

## Context length and concurrency

- **1M context admits one request at a time** by default: vllm-neuron's scheduler caps running requests at
  `KV tokens // max_model_len` = 36864 × 32 / 1,048,576 = 1. Other requests queue.
- `MIMO_KV_CONCURRENCY=4` lifts the cap (c4 throughput 20.4 → 50.8 tok/s, measured with an earlier version) but there
  is no admission check or preemption on the real KV usage: when the running requests together exceed the KV pool
  (about 1.18 M tokens) the server stops making progress (tested: 3 × 300K answered correctly; 2 × 600K stalled). Not
  a default.
- At 4096 context, raising `--max-num-seqs` to 8 gives little: prefill cannot share a batch with decode in vllm-neuron.
- Decode is compiled for batch 1 and 4 only (`[1, 4]`), so 2 or 3 running requests are padded to 4.
- **Prefix caching** (`--enable-prefix-caching`) works up to 940K tokens: the second request with the same prompt
  reaches its first token in 2.89 s at 940K and 0.25 s at 30K, with the same answer. On a *partial* hit (for example
  an image + audio request that only shares the prefix of an earlier image request) the prompt is split into segments
  differently from a cold request, so greedy output may differ in wording (content was correct in the tests). An
  earlier version hung on long cache hits; the cause and fix are in
  [design-notes.md](design-notes.md#prefix-cache-odd-tiles).
- Prefill graphs are still compiled with the runner's fixed modular-flow partitioning, so about 6 GiB of compiler
  scratch per core scales with the KV pool size.
- **Long prompts are slow to the first token:** about 204 s at 940K tokens (230 segments of 4096 tokens, about 0.89 s
  each). The largest part is now the GA (global attention) prefix attention over the earlier tokens, about 0.42 s per
  segment at 940K; six rewrites of that kernel gained at most 5 % and were not merged. The rest (about 0.47 s) was not
  broken down further. Short prompts and decode are not affected.

## Multimodal

- **Single images: still at most about 1024 × 1024 pixels / 1024 vision tokens.** Any resolution is accepted, but an
  image above 1,048,576 pixels is downscaled (pixel cap written into the `-mm` directory's config.json; a 2048 × 1365
  photo becomes 1248 × 832, 1014 tokens), so fine detail of large images is lost. At most 2 images per request. This is
  a choice of this port, not a model limit: the model's own processor allows about 8.4 million pixels per image, which
  needs a banded windowed ViT attention with an NKI flash kernel (not done, about 1–2 days; see
  [design-notes.md](design-notes.md#vision-bucket) and [roadmap.md](roadmap.md)). The encoder cache has 65 blocks
  (`encoder_cache_num_blocks`).
- **Video: 1 per request, at most 16,384 vision tokens** (`num_vision_tokens_buckets [4096, 65536]`; with `[4096]`
  only, at most 1024). vLLM samples a fixed number of frames from the whole video and each frame is capped at
  1,048,576 pixels, so the limit is set by **resolution, not duration**: 720p videos of 2, 3 and 4 minutes were all
  14,211 input tokens and were answered correctly (first token 3.3–5.2 s, thinking off). Longer videos are sampled more
  sparsely (a 4-minute video gets about one frame every 7.5 s) and may miss details. 1080p and above were not tested.
  Videos above 16,384 tokens (for example with more frames through `--media-io-kwargs`) are rejected
  ([design-notes.md](design-notes.md#vision-bucket)).
- **At most 16 encoder blocks per request** (1 per image, 1 per 1,024-token time group of a video and up to 16 per
  video, 1 per audio clip). A request above that gets HTTP 400 with the block counts; the server is not affected. So a
  full 16-block video (such as a 720p video of a few minutes) cannot be sent together with an image or an audio clip;
  send them in separate requests or lower the video resolution
  ([design-notes.md](design-notes.md#vision-request-blocks)).
- **Mixed requests** — image + audio, two images + audio, video + audio, and image + video + audio + text in one
  request — work when the total is at most 16 blocks. With several modalities in one prompt the model sometimes mixes
  up which content came from which input.
- Video frames are chosen by vLLM's video loader (default 32 frames; change with
  `--media-io-kwargs '{"video": {"num_frames": N}}'`). vLLM treats frames as 1 fps, so for a 2 fps video the timestamp
  tokens are twice the real time (upstream vLLM behavior).
- **Audio: WAV, FLAC, MP3 and Ogg**, at most 1 clip per request and at most 32 seconds (longer requests get HTTP 400).
  FLAC / MP3 / Ogg need soundfile 0.14.0 installed outside the venv and on the server's `PYTHONPATH` (README Step 1);
  without it only PCM WAV is read and other formats fail with `file does not start with RIFF id`. The directory has to
  be reinstalled after changing the SDK or the venv. Why soundfile is not installed into the venv:
  [design-notes.md](design-notes.md#audio-input).
- `file://` media URLs must point inside `--allowed-local-media-path`; otherwise send base64 data URLs.
- The video bucket makes text decode about 0.3–0.7 ms per token slower (TPOT +0.6–1.6 % against a server compiled
  without it, measured on the same day). Within the 5 % gate; not changed.

## API behavior

- **With thinking on, the model sometimes does not emit the end-of-thinking marker `</think>`.** vLLM's `mimo`
  reasoning parser then puts the whole reply into `reasoning` and `content` is empty (seen when transcribing jfk.wav
  with thinking on, temperature 0; the transcription in `reasoning` was correct). This is the behavior of the official
  chat template and parser, not of this port, but it was not compared on a GPU. Workarounds: read `reasoning` when
  `content` is empty, or turn thinking off (`"chat_template_kwargs": {"enable_thinking": false}`, `ask.py --no-think`).
- The `--generation-config` parameter case times out on the 1M configuration because the test sends no `max_tokens`
  and ignores EOS, so it generates towards 1M tokens; it passes at 4096. In practice: always set `max_tokens`.
- `/v1/completions` without `max_tokens` returns 16 tokens (vLLM / OpenAI default).
- With thinking on, long reasoning can use up `max_tokens` (`finish_reason: length`) before any answer appears.
- `tool_choice: none` still puts the tools into the prompt (vLLM default, same as on GPU), so the model may write
  `<tool_call>` text into `content`.
- `seed`, `presence_penalty` and `frequency_penalty` are not supported by vllm-neuron's on-device sampler; the plugin
  re-samples such batches on CPU (`MIMO_ODS_CPU_RESAMPLE=1`, default).

## Validation scope of the recommended configuration

The recommended command was validated with the full regression on the device: short-request benchmark, 30K and 940K
needles, prefix-cache hits at 30K and 940K, the multimodal suite (8/8), the long-video and mixed-request checks, smoke
test, decode at 30K, and the L3 logit validation. Measured with earlier versions of the code (the text path is the
same; the numbers are kept in [results/bench_summary.md](../results/bench_summary.md) with their configuration):

- Tool calling, the parameter suite and the reasoning suite on the 1M configuration were run as `--quick` subsets
  (2, 3 and 2 cases); the full suites passed on earlier versions.
- The c4 benchmark, GSM8K / MMLU, the 448 / 896 image timings, and the 32K / 128K / 512K / text-only / BF16
  configurations.
- The plugin switches now default to the validated values. Older configurations (BF16, 4096 context with the original
  kernels) were not re-run under the new defaults; to reproduce one exactly, set the old values listed in
  [configuration.md](configuration.md#plugin-switches).

## Not supported

- Speculative decoding: the model ships MTP (3 layers) and DFlash (5 layers) drafts, but vllm-neuron 0.24 supports
  only EAGLE3. Drafts are not loaded. vllm-neuron 0.29 contains a DFlash framework ([roadmap.md](roadmap.md)).
- FP8 KV cache (`MIMO_KV_DTYPE=fp8`): not validated; the recommended configuration uses BF16 KV like the GPU deployment.
- DP attention (KV heads are replicated 16× / 8× at TP=64).
- Device profiling of the full 1M configuration (the profiler's HBM pool does not fit); use system traces or the
  3-layer model.
