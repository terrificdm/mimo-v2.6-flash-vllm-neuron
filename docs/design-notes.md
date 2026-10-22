# Design notes

The decisions behind the plugin, the scripts and the default switches, written so that each entry stands on its own:
what was observed, why it happened, what the plugin does and the data that backed the choice. Code comments and the
other documents link here with `docs/design-notes.md#<anchor>`. Numbers are from the authors' validation runs on one
trn2.48xlarge with the versions in the [README](../README.md#requirements); result files that ship with this repository
are linked, everything else is summarized here.

Contents: [Plugin switches](#plugin-switches) · [API](#api) · [Long context and KV cache](#long-context-and-kv-cache) · [FP8](#fp8) ·
[Vision](#vision) · [Audio](#audio) · [Compilation and host memory](#compilation-and-host-memory) ·
[Validation method](#validation-method)

Entries are not numbered: each one is identified by its anchor (the `<a id="…">` line above its heading), and code
comments and documents refer to it as `docs/design-notes.md#<anchor>`. (Test-case IDs such as `T1`–`T10` or `I1`–`I4`
in the result files belong to the API test suites; see [results/README.md](../results/README.md).)

## Plugin switches

<a id="plugin-defaults"></a>
### A plain `vllm serve` is the validated configuration

- **Why:** the validated 1M + multimodal service used to export about a dozen `MIMO_*` variables. Each one is an
  optimisation the deployment needs, so the plugin now takes the validated value when a variable is unset
  (`plugin/mimo_v2_neuron/switches.py`): `MIMO_FP8_IMPL=nki`, `MIMO_FP8_EXPERT_IMPL=row`, `MIMO_MOE_CTE_IMPL=hybrid`,
  `MIMO_KV_WRITE=nki`, `MIMO_KV_VNOPAD=1`, `MIMO_PREFILL_ATTN=nki`, `MIMO_PREFILL_KIDX=arith`,
  `MIMO_VIT_DP_GROUP=world`, `MIMO_CC_MODULAR_MAC=100000000000000` with `MIMO_CC_MODULAR_MAC_MAXTOK=64` (decode
  graphs only), `MIMO_PREFILL_BUCKETS=auto` (512 + segment), `MIMO_DECODE_MOE_IMPL=tkg`.
- **Auto rules:** a default is only used where it is safe, otherwise the old implementation runs and a warning is
  logged: the FP8 switches only for FP8 checkpoints; `hybrid` only for FP8 `row` experts; the NKI prefill kernel
  only for segmented prefill and layers whose shapes fit; the 512 bucket only when the segment is > 512 and shorter
  than `--max-model-len`; vision switches only with a vision tower; the compiler option only for decode graphs.
  The vision buckets come from `--additional-config` (`vision_neuron_config.num_vision_tokens_buckets`). The
  plugin also sets three vLLM timeouts when they are unset (engine ready 6 h, execute model 2 h, compilation 2 h),
  because a cached 1M start takes about 18 minutes.
- **Troubleshooting:** setting a variable always wins, so each old implementation is reachable, for example
  `MIMO_FP8_IMPL=emul`, `MIMO_KV_WRITE=index_put`, `MIMO_PREFILL_BUCKETS=off`, `MIMO_CC_MODULAR_MAC=off`. CPU mode
  (`VLLM_NEURON_CPU_MODE=1`, the unit tests) keeps the old defaults. One startup log line
  `MiMoV2 Neuron: switches: NAME=value(source) …` shows every value and whether it came from the environment or the
  default.

## API

<a id="reasoning-field"></a>
### Reasoning output is returned as `reasoning`

- **Observed:** vLLM 0.24 returns the reasoning text in the field `reasoning`; the older `reasoning_content` is
  deprecated and renamed when it appears in a request.
- **Choice:** accept `reasoning` as the field. The test helpers read `reasoning` and fall back to `reasoning_content`
  only if that is the one present. No proxy or patch renames the field.
- **Why:** this is vLLM behavior and is the same on GPU; it is unrelated to Neuron.

<a id="local-media-path"></a>
### Local media must be under the real allowed path

- **Observed:** vLLM resolves the requested file path but not `--allowed-local-media-path`, so a symlinked allowed path
  rejects every `file://` request.
- **Choice:** `serve.sh` passes the resolved media directory (`$MIMO_MM_ASSETS`), and `mm_qa.py` builds `file://` URLs
  from the same directory. One case (`B1`) sends the image as a base64 data URL instead, which needs no local path.

<a id="sampling-ties"></a>
### `top_k` / `top_p` "partial" results are exact logit ties

- **Observed:** with CPU sampling (`MIMO_SAMPLING=cpu`) and native FP8, 15 of the 17 parameter cases pass and two
  (`temperature_top_p` with top_p=0.001, `top_k` with top_k=1, both compared against greedy) are "partial".
- **Why:** every divergence happens at a position where the two best tokens have exactly the same logprob
  (for example ' tape' and ' palette' both −0.89985). Logits are BF16, so ties are common; vLLM's top-k / top-p masks
  keep both tied tokens and pick one at random, while greedy takes the argmax.
- **Choice:** these cases count as passes. `seed`, `presence_penalty`, `frequency_penalty`, `json_schema` and
  `--generation-config` pass outright.

<a id="cpu-resample"></a>
### `seed` and penalties with on-device sampling

- **Observed:** vllm-neuron's on-device sampler takes only top_k, top_p and temperature per request. It has no random
  generator input and no presence / frequency / repetition penalty, so those parameters were silently ignored.
- **Choice:** when a batch contains a request with `seed` or a penalty, the runner patch re-samples that batch on the
  CPU with vLLM's own `Sampler`, using the full-vocabulary logits the model already outputs. Other batches keep the
  device tokens; no graph is recompiled. `MIMO_ODS_CPU_RESAMPLE=0` turns this off.
- **Data:** the parameter suite passes in device-sampling mode (16 pass + 1 info on the 1M configuration, see
  [results/api/no_video_buckets_params_mode_a.md](../results/api/no_video_buckets_params_mode_a.md)). The extra cost is one CPU sampling
  step over a 152K vocabulary, only for those requests.

## Long context and KV cache

<a id="segmented-prefill"></a>
### Segmented prefill: segment length and prior-key block size

- **Choice:** long contexts use vllm-neuron's segmented prefill. The prior-key block size grows with the context tier
  so that each layer unrolls at most 32 key blocks and one score block stays ≤ 512 MB per core: 32K → 8192,
  128K → 16384, 512K / 1M → 32768 (`config.py`, overridable by environment variables).
- **Observed:** the attention scratch of a prefill graph grows linearly with `max_model_len` (32K 0.42 GiB,
  128K 2.05 GiB, 512K 9.19 GiB on the 3-layer model): neuronx-cc gives every key block its own FP32
  `[heads, segment, key_block]` buffer, so the scratch is about segment × context × 4 bytes, whatever the blocking.
  Forcing the blocks to run one after another (a data dependency between blocks) did not change it.
- **Consequence:** the segment length is the effective knob. At 1M context a 512-token segment kept the prefill graph
  under the HBM budget with the PyTorch prefix attention; the final configuration uses 4096-token segments with the
  NKI flash kernel for the prefix part (`MIMO_PREFILL_ATTN=nki`), which keeps only one tile alive.

<a id="kv-write"></a>
### KV-cache writes and partition copies

- **Observed:** decode time grew with the size of the KV pool, not with the attention width: on the 4096 tier the
  same command went from TPOT 43.1 ms with 4096 blocks to 210.5 ms with 36,864 blocks. The functional `index_put_`
  that wrote K/V made the compiler copy the whole cache tensor several times per decode graph.
- **Choice 1 (`MIMO_KV_WRITE=nki`):** an NKI kernel writes the new rows with an indirect (scatter) DMA and returns the
  cache tensor it was given, so the cache becomes an input/output alias and the update is in place. `kv_gather` reads
  blocks the same way and also returns the cache alias, so reads and writes form one chain.
- **Observed next:** the remaining copies came from the compiler's modular flow. vllm-neuron passes
  `--modular-flow-mac-threshold=10`, the compiler splits each graph into partitions, and the KV cache is copied whole
  at every partition boundary (10 partitions, 4 × 453 MB of intermediates on the 3-layer model).
- **Choice 2 (`MIMO_CC_MODULAR_MAC`, `MIMO_CC_MODULAR_MAC_MAXTOK=64`):** drop the partitioning only for decode graphs
  (≤ 64 tokens). Without partitions a 4096-token prefill graph compiled for 59 minutes and its loading exhausted
  host memory (64 workers × about 33 GB), so prefill graphs keep the runner's default.
- **Data:** with both choices, TPOT on the 4096 tier with 36,864 blocks went from 133.2 ms to 38.6 ms (4096-block
  baseline 43.1 ms), with correct output.

<a id="v-no-pad"></a>
### V stored with 128 dims (`MIMO_KV_VNOPAD=1`)

- **Why:** the GPU deployment selects vLLM's DIFFKV attention backend because `v_head_dim` (128) differs from the QK
  head_dim (192), and stores K and V separately without padding.
- **How:** the vllm-neuron runner allocates one `(2, blocks, kv_heads, 32, head_size)` tensor per layer with a single
  head_size. The plugin declares head_size = (192 + 128) / 2 = 160 and views the same memory as K `[…, 192]` followed by
  V `[…, 128]` in `bind_kv_cache`; no copy and no change to installed files.
- **Data:** at 1M + multimodal the compiler peak dropped from 22.64 to 20.09 GiB per core, the 940K-token TTFT from
  1279 s to 1065 s; L3 logits identical to the padded layout.

<a id="hbm-budget"></a>
### HBM budget at 1M context

- Each NeuronCore has 24 GiB of HBM; the authors kept compiled graphs under 22.5 GiB.
- FP8 weights save about 4.5 GiB per core compared with BF16, which is why BF16 KV (as on GPU) fits at 1M and no FP8
  KV cache is needed. KV pool: 36,864 blocks of 32 tokens; each additional 1000 blocks costs about 0.41 GiB per core.
- Recommended configuration: compiler peak 17.35 GiB per core for the 4096 and 512 prefill graphs (6.2 GiB scratch;
  17.23 GiB on an earlier build without the larger vision buckets), about 11.1 GiB for the un-partitioned decode
  graphs. At run time neuron-monitor showed 20.13 GiB on the busiest core with the recommended command (an earlier
  version without the larger vision buckets: 19.94 GiB). The device profiler needs a further
  ~3.4 GB pool plus 128 MB per core, so it cannot start on the full 1M configuration.

<a id="concurrency"></a>
### One running request on the 1M configuration

- **Observed:** vllm-neuron's scheduler admits `KV tokens // max_model_len` requests = 36,864 × 32 // 1,048,576 = 1.
- **Choice:** keep the default. `MIMO_KV_CONCURRENCY=4` lifts the cap (c4 50.8 tok/s) but stalls when the running
  requests together exceed the KV pool, so it is only for workloads whose total length fits.

<a id="prefix-cache-odd-tiles"></a>
### Prefix-cache hits with an odd number of 512-key tiles

- **Observed:** with `--enable-prefix-caching`, a repeated long prompt (470K, 940K tokens) hung the service on the
  second request, and so did a 597-token prompt on a 3-layer model: the hang depends on the cached length, not on
  the prompt length.
- **Cause:** the GA prior-key NKI kernel runs as two programs (LNC=2) that split the 512-key tiles; the loop count
  is `cached_seq_len // 512`. Cold requests start every segment on a 4096 boundary (even tile count), but a cache
  hit starts on any 32-token boundary; with an odd tile count the two programs loop a different number of times and
  the device execution stops. The NKI simulator does not model the two physical cores, so unit tests missed it.
- **Choice:** the caller rounds the tile count down to an even number and sends the remaining < 1024 keys through
  one masked torch block (`attention_torch._nki_whole_tiles`); the kernel is unchanged and cold requests give the
  same logits as before. Prefix caching is on in the recommended command: the second 940K request takes about
  2.9 s instead of minutes. A partial hit splits the prompt differently from a cold request, so greedy output can
  differ in wording (content the same). `MIMO_DEBUG_METADATA=hit` logs the hit lengths (diagnostics only).

<a id="moe-hybrid"></a>
### Long prompts: experts loop over non-empty blocks only

- **Observed:** a device profile of a 940K prompt (3-layer model, 1M configuration) showed the prefill expert kernel
  (`moe_cte shard_on_i`) looping statically over all 67 blocks of 256 tokens, while with EP=64 each rank had only
  4–6 non-empty blocks: about 90 % of the expert matmuls were on empty blocks. The GA prior-key kernel's key-row
  table was also built with a 1M-element `index_select` (about 9 ms of indirect DMA per GA layer).
- **Choice:** `MIMO_MOE_CTE_IMPL=hybrid` (nkilib's own `shard_on_i_hybrid`: the same per-block code, but the loop
  runs over the number of non-empty blocks; experts stay FP8 codes with BF16 activations) and
  `MIMO_PREFILL_KIDX=arith` (the same key rows by broadcast arithmetic, bit-identical). Both are plugin defaults.
  940K time to first token 324.7 → 204.2 s, 30K 8.01 → 3.9 s, short request 310 → 198 ms; TPOT unchanged.
- **Left:** the GA prior-key kernel is now the largest part (about 0.42 s of 0.89 s per 4096-token segment at
  940K); six scheduling rewrites gave at most 5 %, so a structural rewrite is listed in the roadmap.

## FP8

<a id="fp8-alignment"></a>
### FP8 weights aligned with the GPU deployment

Full description: [gpu-alignment.md](gpu-alignment.md). The decisions in short:

- **E4M3 on trn2 tops out at 240.** Checkpoint E4M3FN codes copied to the device and cast to BF16 give NaN for 448 and
  288 and Inf for 256; every value ≤ 240 is correct. A BF16 → FP8 → BF16 round trip inside one graph looked lossless
  only because the compiler folded it; the stored FP8 bits were checked instead. So the weights are converted offline.
- **Conversion format.** Linear-layer scales are expanded to one scale per (row, 128 columns) so any row split needs no
  re-quantization; groups whose largest code exceeds 240 are halved with the scale doubled (7.2×10⁻⁵ of codes
  change). MXFP4 experts become exact FP8 codes with one scale per output channel: all 302,795,194,368 elements are
  bit-exact.
- **Linear layers compute like the GPU (W8A8).** `torch.matmul` on float8 inputs returns float8 and `torch._scaled_mm`
  aborts, so the plugin first emulated the GPU algorithm in BF16 (`fp8_linear.py`) and then wrote its own NKI kernel
  (`nki_fp8.w8a8_block_linear`, `MIMO_FP8_IMPL=nki`): per-token per-128-channel dynamic activation quantization
  (scale = amax/448), FP8 × FP8 on the tensor engine, FP32 accumulation. Three device workarounds were needed:
  reciprocal-multiply instead of divide, BF16 copies for the transpose, and `abs` + `max` instead of `abs_max`.
- **Experts compute like the GPU (W4A16).** nkilib's MoE kernels in ROW mode read FP8 codes with a per-channel scale
  and keep BF16 activations (`MIMO_FP8_EXPERT_IMPL=row`). The `shard_on_block` prefill variant ignores the FP8 scale,
  so `shard_on_i` is used.
- **KV layout like the GPU:** see [V stored with 128 dims](#v-no-pad).

## Vision

<a id="omni-architecture"></a>
### Separate multimodal architecture and directory

- **Choice:** the multimodal model is registered as a second class, `MiMoV2OmniForCausalLM`, served from its own
  directory (`tools/make_mm_checkpoint.py`: symlinks to the text shards plus the vision / audio shards). vLLM's
  `MiMoV2OmniMultiModalProcessor` is reused.
- **Why:** the text-only path stays unchanged; vLLM's processor matches the model authors' preprocessing; the text
  decoder uses 1-D positions, so no M-RoPE is needed (see [positions](#positions)).

<a id="positions"></a>
### Positions are one-dimensional

HF `MiMoV2Model.forward` uses `cache_position` (1-D) and vLLM's `MiMoV2OmniForCausalLM` does not implement
`SupportsMRoPE`. A tiny multimodal CPU run with `MIMO_DEBUG_METADATA=1` confirmed that image tokens get consecutive
1-D positions starting at 0, the same as HF; no M-RoPE is implemented.

<a id="vit-sinks"></a>
### ViT attention sinks follow HF

HF adds a sink to the first key of every frame in the 24 windowed ViT layers; vLLM's GPU implementation ignores it.
The plugin follows HF (the reference the encoder is compared against); `MIMO_VIT_SINKS=0` drops the sinks for
comparisons. For multi-frame inputs the sink index is set per row to the start of its time group.

<a id="vision-bucket"></a>
### Vision parallelism and vision buckets

- **Choice:** vision TP 8 × DP 8 (8 KV heads → one per core), PyTorch attention with explicit boundary masks and sink
  bias, blocks of 4096 patches (`vision_attention_block_size` 4096 = 1024 merged tokens = one encoder-cache block).
  Items never share a block; an image takes one block, a video is cut into blocks at time-group boundaries.
- **Buckets:** the default `num_vision_tokens_buckets` is `[4096]`: one block per image or video, so at most 1024
  vision tokens after the 2 × 2 merge. The recommended command sets `[4096, 65536]`, which adds a 16-block graph:
  a video can then use up to 16 blocks = 16,384 vision tokens. With vLLM's default of 32 sampled frames a 3-minute
  720p video becomes a 16 × 44 × 80 grid = 14,080 tokens (16 blocks of 880 tokens). Because the frame count is fixed,
  the token count depends on the resolution, not the duration: 720p videos of 2, 3 and 4 minutes were all 14,211 input
  tokens in a later test (longer videos are sampled more sparsely). Single images are unchanged (at
  most 1024 tokens, see the [pixel cap](#pixel-cap)).
- **Why the largest bucket matters:** vllm-neuron derives three limits from the largest bucket divided by the merge
  factor 4: the per-item validation limit, the scheduler's encoder budget per step and the number of encoder-cache
  blocks the text prefill graph takes per request. With `[4096]` all three are 1024 tokens / 1 block. The checks the
  plugin adds for larger buckets are described in [the next entry](#vision-request-blocks).
- **Cost (first launch with the new buckets):** the 16-block vision graph compiles in 8 DP variants of
  about 215 s each (vision graph compiler peak 3.58 GiB per core); the 512 and 4096 text prefill graphs are recompiled (78 s and
  381 s, compiler peak 17.35 GiB per core instead of 17.23); host compiler peak 19 GB. The decode and audio graphs
  reuse the compile cache. Highest per-core HBM use at run time: 20.25 of 24 GiB (19.94 with the default bucket).
- **ViT accuracy on real inputs** ([results/multimodal/vit_compare.json](../results/multimodal/vit_compare.json)):
  for the 3-minute video cos(device, HF BF16) 0.99834 and cos(device, HF FP32) 0.99837 ≥ cos(HF BF16, HF FP32)
  0.99831 − 0.002, so it passes the fallback criterion of [the ViT entry](#vit-accuracy); a 2048 × 1365 photo also
  passes the fallback criterion (margin 0.0004). The main criterion (0.999 against HF BF16) was set on small synthetic
  images; on real photos and long videos HF's own BF16 run reaches only 0.995–0.998 against HF FP32.
- **Why single images were not raised:** reaching the model's default of about 8.4 million pixels per image needs
  blocks of 16,384–32,768 patches. The ViT attention here is plain PyTorch, so each score tensor grows with the
  square of the block (4 GiB per tensor at 16,384), and the server already uses 20.25 of 24 GiB per core. It needs a
  banded windowed attention with an NKI flash kernel ([roadmap.md](roadmap.md)).
- **Test video:** vLLM's processor resamples to 2 fps with at least 8 frames; at 512 × 512 a 4-second video lands
  exactly on 4096 patches, so the test video uses 384 × 384 (2304 patches) and the size is asserted on the CPU before
  compiling.

<a id="vision-request-blocks"></a>
### Per-request block check, video budget and encoder-step split for larger buckets

- **Observed (code reading):** the text prefill graph takes at most `max_vision_blocks_per_request = ceil(largest
  bucket / merge / cache block)` encoder-cache blocks per request; a request whose images, videos and audio need more
  fails inside the worker with `Vision block truncation`. With the default bucket that limit is 1, so two images, or an
  image and an audio clip, landing in the same prefill segment could hit it (not reproduced on the device).
- **Choice 1, check in the API process (when the largest bucket is above 4096):** the plugin reads the buckets from
  `--additional-config` (`vision_neuron_config.num_vision_tokens_buckets`) into `MIMO_MM_VISION_BUCKETS` unless that
  variable is set; its multimodal processor computes the block plan of every item (1 per image, 1 per time group of a video up to 16, 1 per audio clip) and
  rejects a request that needs more than largest bucket ÷ 4096 blocks (16 with the recommended buckets) with HTTP 400; the message names
  the block counts and the server keeps serving. Items that come back from the processor cache without their grid use
  the count remembered for their hash, or a lower bound from their embedding count.
- **Choice 2, video encoder budget:** vLLM estimates the largest video from 14 frames of 1024 × 1024 (about 14,336
  tokens) and sizes the encoder budget and cache from that; a video between that estimate and the 16,384-token limit
  would pass validation but never be scheduled for encoding. The plugin reports the video maximum as at least the
  largest bucket ÷ 4.
- **Choice 3, encoder-step split:** the scheduler's budget is in tokens, so one encoder step can schedule several items
  whose blocks together exceed the largest vision graph; `embed_multimodal` then encodes them in groups of whole items
  that fit.
- **Consistency:** the workers check that `MIMO_MM_VISION_BUCKETS` equals
  `vision_neuron_config.num_vision_tokens_buckets` and that the block size is 4096. Leave the variable unset so the
  plugin derives it from `--additional-config` (the optional `scripts/serve.sh` writes both).
- **Data (recommended buckets):** the 3-minute 720p video (16 blocks) is answered, first request 9.95 s with `max_tokens` 1; a
  1024 × 1024, 64-frame video (16,384 tokens, the limit) is answered; two images in one request are answered; the
  video plus two images (18 blocks) and the video plus one image (17 blocks) return HTTP 400 and the next request is
  served normally ([results/multimodal/vision_limits_check.json](../results/multimodal/vision_limits_check.json),
  [video_16384_check.json](../results/multimodal/video_16384_check.json),
  [ask/reject_video_img.txt](../results/multimodal/ask/reject_video_img.txt)). Not tested: an image and an
  audio clip in the same prefill segment.

<a id="pixel-cap"></a>
### Pixel cap written into the directory's config.json

- **Observed:** `--mm-processor-kwargs '{"max_pixels": 1048576}'` has no effect: `MiMoOmniProcessor.__call__` ignores
  it (a 2048 × 2048 image still became 4096 vision tokens) and `process_video` does not resample by fps.
- **Choice:** `tools/make_mm_checkpoint.py --max-pixels 1048576` writes `processor_config.image_max_pixels` and
  `video_max_pixels` into the multimodal directory's config.json. Oversized images are then downscaled instead of
  rejected; the original and text directories stay untouched.

<a id="vit-accuracy"></a>
### ViT accuracy criterion and rounding

- **Observed:** the first implementation reached cosine 0.9973–0.9982 against HF BF16 on 3 of 4 images, below the
  0.999 target. The error accumulates over the 28 layers and the last two layers amplify it (activations up to
  3.7e5). HF's own BF16 run reaches only 0.9976–0.9992 against HF FP32.
- **Choice:** remove the extra BF16 roundings so they match HF: biases folded into the matmul (`F.linear`), SiLU in
  FP32 with one rounding, row-parallel projections with FP32 partial sums and FP32 all-reduce, then one rounding.
  Result: all inputs ≥ 0.999 against HF BF16 (0.99922–0.99974).
- **Fallback criterion (written before the device run):** cos(plugin, HF FP32) ≥ cos(HF BF16, HF FP32) − 0.002 plus the
  default three-way thresholds. Device result: 0.998–0.999.

<a id="mm-reference-inputs"></a>
### References are built from the processor the server uses

The ViT and multimodal goldens are computed from the output of vLLM's MiMo-Omni processor (the tensors the server
actually feeds the model), after checking that the HF and vLLM processors agree (2.4e-7 on 448 × 448).

<a id="encoder-cache-blocks"></a>
### Encoder cache blocks are set explicitly

- **Observed:** the runner derives the number of encoder-cache blocks from the scheduler's token budget, but every
  image takes at least one whole block. The scheduler then admits images the worker has no block for, and the worker
  fails with `Encoder cache full` (in the tiny test on the 6th request; with the automatic value on the full model,
  at the 5th distinct image). The failure appears only at run time.
- **Choice:** `serve.sh` sets `vision_neuron_config.encoder_cache_num_blocks` (`MIMO_MM_CACHE_BLOCKS`, default 65;
  8 MiB per block, about 520 MiB per core). `run_mm_validation.py --vision-cache-blocks` does the same for validation.

<a id="vision-dp-group"></a>
### Vision data parallelism over the whole TP world

- **Observed:** the vision DP all-gather over the platform's column groups (ranks 0, 8, …, 56 for TP 8 × DP 8) fails
  at warm-up: the runtime has no collective algorithm for that strided replica group on trn2.
- **Choices:** `MIMO_VIT_DP_GROUP=0` (the older implementation, CPU-mode default): each vision TP group of 8
  consecutive ranks encodes every block, so no DP collective is needed. `MIMO_VIT_DP_GROUP=world` (default): blocks are split over all 64 ranks and gathered with
  one all-gather over the whole TP world, which is a supported contiguous group; image encoding is 2–4× faster
  (448 × 448: 0.633 s instead of 1.587 s, [results/bench/vit_timing_dp_world.json](../results/bench/vit_timing_dp_world.json)).
  `1` keeps the column groups for CPU tests only.

## Audio

<a id="audio-input"></a>
### Audio decoding: WAV built in, MP3 / FLAC through soundfile outside the venv

- **Observed:** vLLM reads audio files with soundfile or PyAV and resamples with PyAV; the DLAMI vLLM venv has none of
  soundfile, librosa, PyAV or torchcodec.
- **Choice:** `mm_patches.patch_audio_loading` decodes with soundfile when it can be imported and the file header is
  RIFF / RIFX / RF64, fLaC, OggS, ID3 or an MPEG frame sync, and resamples with SciPy. When soundfile is missing, the
  container is not one of these, or decoding fails, it reads PCM WAV with Python's `wave` module; the 32-second limit
  applies on every path. soundfile 0.14.0 (it bundles libsndfile 1.2.2) is installed into a directory outside the venv,
  `~/mimo-work/pydeps` in the README, which is put on the server's `PYTHONPATH` (the optional `scripts/serve.sh` appends
  `MIMO_PYDEPS`, default `$MIMO_WORK/pydeps`, when it exists; `MIMO_PYDEPS=off` disables it). Without that directory
  the server still starts and reads PCM WAV only.
- **Install** (README Step 1): `pip install --no-deps --target ~/mimo-work/pydeps soundfile==0.14.0` in the venv,
  or download the wheel, check its SHA-256 and install it with `--no-index`.
- **Why outside the venv:** the venv is the DLAMI's, and this tutorial keeps its installed packages unchanged so the
  validated environment can be reproduced exactly. `--no-deps` matters: soundfile's dependencies (cffi, numpy) are
  already in the venv, and because `PYTHONPATH` entries come before the venv's site-packages on `sys.path`, a second
  numpy in `$MIMO_PYDEPS` would shadow the venv's. After the install the directory holds only soundfile.py,
  _soundfile.py, _soundfile_data/, licensing/ and soundfile-0.14.0.dist-info/. Reinstall it after changing
  the SDK or the venv.
- **Security:** soundfile is BSD-3-Clause; libsndfile 1.2.2 (LGPL-2.1) has a published CVE in its IRCAM-header reader
  (CVE-2025-52194). The plugin hands it only the container types above and never encodes. Install it only after your
  own review.
- **Data:** with the server's `PYTHONPATH` and the WAV reader replaced by a stub that fails, jfk.wav, a speech WAV, an
  MP3 and a FLAC file were all decoded by soundfile, and 7 WAV files gave the same samples as the WAV reader
  ([results/multimodal/audio_offline.txt](../results/multimodal/audio_offline.txt)). On the server an
  MP3 and the same clip as WAV were both identified as breaking glass, and the transcription of a FLAC file matched
  that of the same speech as WAV word for word ([results/multimodal/ask/](../results/multimodal/ask/)). Ogg Vorbis and Ogg
  Opus versions of jfk.wav gave the same transcription as the WAV word for word, and an Ogg of the breaking-glass clip
  was identified correctly.

<a id="audio-criteria"></a>
### Audio accuracy criteria

- **Observed:** the RVQ codes are discrete and HF's own BF16 run agrees with HF FP32 on only 57–83 % of them, with an
  embedding cosine of 0.973–0.984, so a 0.999 end-to-end criterion is unreachable.
- **Criteria:** features before RVQ vs HF BF16 ≥ 0.999; the embedding tower fed the same codes vs HF BF16 ≥ 0.999;
  end to end vs HF FP32 ≥ cos(HF BF16, HF FP32) − 0.01; code agreement no lower than HF BF16's − 0.05.
  For the logit check the audio path is split: the encoder is judged by these criteria, and the decoder is judged
  with HF's BF16 audio rows injected (`MIMO_AUDIO_EMBED_OVERRIDE`) by the usual logit criteria.
- **GPU alignment:** vLLM on GPU casts the tokenizer to BF16, so the codebooks are BF16-rounded although the checkpoint
  stores F32; the plugin does the same (`MIMO_AUDIO_CODEBOOK=fp32` keeps the F32 values). GELU is computed in FP32
  with one rounding. Padded mel frames are zeroed after conv1, as HF convolves only the real length.
- **Data:** device vs HF FP32 cosine jfk 0.9782 (HF BF16 0.9794), beeps 0.9871 (0.9880); RVQ codes 90.4 % equal to the
  CPU plugin ([results/accuracy_summary.md](../results/accuracy_summary.md)).

<a id="audio-on-device"></a>
### Audio towers on the device: the RVQ top-k

- **Observed:** the compiled audio graph failed to load with `Failed to load collectives`, which looked like a
  collective-group problem. The first error in the runtime log was actually an out-of-bounds DMA
  (`mem_ref_to_addr invalid offset in reshape.69 …, 6400 < (6400 + 8)`) in a 1600-element FP32 intermediate, produced
  by the compiler for `topk` inside the RVQ. Compiling each part of the towers separately showed that only the RVQ part
  failed.
- **Interim choice:** the towers ran on each worker's CPU (`MIMO_AUDIO_DEVICE=cpu`); an 11-second clip took about
  10 s per request on the 3-layer model.
- **Final choice (`MIMO_AUDIO_RVQ=first_gather`):** select the nearest codeword with compare + minimum index (the first
  maximum, same as top-1) and look it up with `index_select`. It loads, gives 100 % the same codes as the CPU, and the
  towers run on the device (`MIMO_AUDIO_DEVICE=neuron`, default). Audio graphs are warmed up after the vision warm-up
  inside the runner's own compile flow.

## Compilation and host memory

<a id="host-oom"></a>
### Host out-of-memory during compilation

- **Observed:** the compiler back end (`walrus_driver`) reached 135 GB RSS while compiling the vision graph without
  partitions, the host ran out of memory and the kernel also killed tmux, the server and the background jobs. The
  "≤ 64 tokens" test for decode graphs had matched the vision graph, whose first input has 8 rows.
- **Fix:** only graphs whose first input is a 1-D integer tensor (token ids) with ≤ `MIMO_CC_MODULAR_MAC_MAXTOK` rows
  lose the partitioning; vision, audio and prefill graphs keep the default.
- **Guards:** `scripts/check_bg.sh` marks a job `FAILED rc=lost` when its status says RUNNING but the process is gone;
  `scripts/mem_guard.sh` logs free memory every 30 s and kills only compiler processes when free memory drops below
  200 GB. Peaks seen: 135 GB (vision graph un-partitioned), 107 GB (4096-token prefill un-partitioned), about 26 GB
  (1M decode graph un-partitioned).

## Validation method

<a id="validation-rules"></a>
### Logit-validation rules fixed in advance

- **Main criterion:** the vllm-neuron three-way comparison on a 3-layer (L3) model: device vs reference, with the BF16
  reference error as the yardstick (sigma ratio ≤ 1 passes).
- **Order when it fails:** an extended fallback criterion (mechanical, `scripts/accuracy/fallback_check.py`), then for
  multimodal only one relaxation (top-5 rtol 0.011 → 0.02, `run_logit_validation.py --top5-rtol-override 0.02`, top-1
  agreement ≥ 95 % unchanged), allowed only after the ViT checks pass. In question-answer tests an image that HF itself
  answers wrong is not scored.
- `--max-model-len` of the validation scripts defaults to 1024, the value the first-phase L3 graphs were compiled with,
  so earlier results stay comparable.
- **Why:** runs were unattended, so the criteria had to be mechanical and written down before the results.

<a id="l3-segment-accuracy"></a>
### Slightly higher L3 error with 4096-token segments

- **Observed:** the final configuration (segment 4096, NKI prefix attention, extra 512-token prefill bucket) has an L3
  sigma ratio of 0.907 against the GPU-algorithm reference; the earlier 512-segment configuration had 0.664. Both pass.
- **Control run:** segment 4096 with PyTorch attention and no 512 bucket gave 0.887 and the same fallback results
  (3/4), with failing steps at nearly the same positions.
- **Conclusion:** the difference comes from the 4096-token prefill graph itself (graph shape and compiler blocking;
  all test prompts fit in one segment), not from the new kernel or the small bucket. The final configuration is kept.
  Files: [results/accuracy/](../results/accuracy/).

<a id="task-eval"></a>
### Task-level evaluation

GSM8K and MMLU, 200 questions each (fixed random sample, ids in
[results/accuracy/task_eval_item_ids.json](../results/accuracy/task_eval_item_ids.json)), 0-shot, thinking off, greedy: 98.0 % and
80.5 % on the final FP8 configuration; same answer as the BF16-weight server on 199/200 and 197/200. One MMLU reply
could not be parsed and counts as wrong.
