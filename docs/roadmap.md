# Roadmap

> [!IMPORTANT]
> **Current version (the only one this tutorial has been validated on):**
> Neuron SDK 2.32.0 · vllm-neuron 0.24.0.1.1.0 · vLLM 0.24.0 · torch 2.11 / torch-xla 2.11 ·
> libtorch-neuronx-lite 2.11.0.1.0.1284 · neuronx-cc 2.27.5334.0 · NKI 0.6.0 · driver aws-neuronx-dkms 2.30.2.0 ·
> runtime aws-neuronx-runtime-lib 2.34.10.0.
> This tutorial has only been validated on these versions.

> [!NOTE]
> **Backend in use: vllm-neuron's default XLA backend** (torch.compile backend `neuron_libtorch`, built on torch-xla).
> The PyTorch native backend (TorchNeuron, `VLLM_NEURON_BACKEND=neuron_native`) has **not** been used.
> According to the Neuron documentation as summarized by the authors (not re-checked against the original pages here),
> from SDK 2.32 the XLA-based torch-neuronx is in maintenance and the native backend is recommended for new work; vllm-neuron 0.29.0.1.2.0 (Neuron SDK 2.33) makes native torch.compile the default.

> [!IMPORTANT]
> **First priority: migrate to Neuron SDK 2.33 + vllm-neuron 0.29 and switch to the native backend, then re-check
> whether each device-side workaround (`torch.split`, `torch.topk`, the RVQ out-of-bounds DMA, …) is still needed.**
> Estimated effort: 3–5 days to validate the plugin on the native backend (L3 accuracy, full regression, performance
> comparison); about 1 week for the whole SDK 2.33 / vllm-neuron 0.29 migration as its own project (rough
> estimates from the authors' migration planning, not measured).

Everything below is the authors' assessment of the code in this repository and their migration planning notes. A migration was
started but stopped before any SDK component was upgraded, so **nothing below has been tried on SDK 2.33 / 0.29**.

## 1. Migration to SDK 2.33 + vllm-neuron 0.29 + native backend

### 1.1 Why

- Follows the officially recommended path; the XLA route only receives maintenance.
- Release-note highlights for vllm-neuron 0.29 as summarized by the authors (not re-checked here): native
  torch.compile is the default compile path; cold start about 2× and warm start about 6× faster; up to about 2.5×
  throughput for GPT-OSS-120B on Trn3; up to 38 % higher Qwen3-VL throughput; Qwen3-ASR-1.7B added (a possible
  reference for the audio encoder). Model Runner V2 supports dense models only; MoE and multimodal models stay on V1.
- Several device bugs worked around here may be fixed on the native path, letting the plugin drop code.

### 1.2 Differences 0.24 → 0.29 already assessed (code diff, not run)

| Area | Change in 0.29 | Impact on this plugin |
|---|---|---|
| `vllm_neuron/vllm/worker/neuron_model_runner.py` | rewritten (about 3,900 changed lines); new `neuron_input_build`, `request_state`, `slot_mapping`, `forward_adapter` | `plugin/mimo_v2_neuron/runner_patches.py` patches this runner at run time (5 engine-crash fixes, CPU re-sampling for `seed` / penalties, compiler-flag tuning, prefill buckets, concurrency cap) and must be re-targeted |
| `vllm_neuron/parallel/neuron_parallel_state.py` | about 740 changed lines | TP / EP group handling and the vision `world` all-gather need re-checking |
| `vllm_neuron/model/kv_cache.py`, `vllm_neuron/vllm/worker/encoder_cache_blocks.py` | changed | KV layout (`MIMO_KV_VNOPAD`), in-place KV write kernel, encoder cache blocks; the per-request block check of `MIMO_MM_VISION_BUCKETS` assumes `max_vision_blocks_per_request = ceil(largest bucket / merge / cache block)` |
| `vllm_neuron/utils/vision_utils.py` | **deleted** | `plugin/mimo_v2_neuron/mm_patches.py` wraps it; needs a new hook |
| `vllm_neuron/functional/` | about 10,000 changed lines, including `vllm_neuron/functional/topk.py` | the chunked top-k workaround may be obsolete |
| contrib directory (unconfirmed) | listed in an earlier note as a new directory for community models; not present in the v0.29.0.1.2.0 tag | possible home for this model upstream, if it exists |
| Built-in attention kernels | head_dim limit is still 128 | MiMo's QK head_dim 192 still needs this plugin's own NKI kernels |
| NKI kernel integration | — | the plugin embeds its kernels with `libtorch_neuronx_lite.nki.nki_hop.wrap_nki`; the native path uses torch-neuronx APIs (`wrap_nki` / `nki_op`) and may need changes |

### 1.3 Workarounds to re-validate one by one after the migration

| Workaround in this version | Original problem on SDK 2.32 / XLA |
|---|---|
| `torch.tensor_split` instead of `torch.split` | wrong results from `torch.split` on the device (`tools/repro_split.py`) |
| chunked top-k for MoE routing | `torch.topk` over 256 columns returns wrong indices on the device (`tools/repro_topk.py`) |
| RVQ code selection by compare + min index (`MIMO_AUDIO_RVQ=first_gather`) | `topk` in the audio tokenizer compiled to an out-of-bounds DMA; the NEFF failed to load |
| NKI in-place KV write (`MIMO_KV_WRITE=nki`) | `index_put_` is a functional scatter in the compiled graph; the compiler copies the whole KV pool each step |
| un-partitioned decode graphs (`MIMO_CC_MODULAR_MAC=1e14`, `MIMO_CC_MODULAR_MAC_MAXTOK=64`) | the runner's fixed `--modular-flow-mac-threshold=10` splits graphs, and KV is copied whole at every partition boundary |
| vision DP over the whole TP group (`MIMO_VIT_DP_GROUP=world`) | non-contiguous replica groups (`[0,8,…,56]`) fail to load on trn2 |
| own NKI kernels for head_dim 192 attention (`MIMO_PREFILL_ATTN=nki`) | built-in kernels support head_dim ≤ 128 (still true in 0.29) |
| runner runtime patches (`runner_patches.py`) | 5 cases where one valid request crashed the engine; on-device sampler ignores `seed` and penalties |
| API-side encoder-block check, video budget override and encoder-step split for `MIMO_MM_VISION_BUCKETS` (`factory_mm.py`, `model_mm.py`) | 0.24's encoder-cache conventions: the largest bucket ÷ 4 sets the request limit, the encoder budget and the blocks per prefill request; an over-full request fails in the worker with `Vision block truncation` |
| soundfile 0.14.0 outside the venv (on `PYTHONPATH`) for MP3 / FLAC / Ogg | the 2.32 DLAMI venv has no audio decoder; check whether the new venv ships one |

### 1.4 Acceptance for the migration

Same gates as this version: L3 logit validation against the GPU-algorithm reference (sigma ratio ≤ 1 or the other
default criteria), full API regression (`scripts/api_tests/run_suite.sh` smoke / reasoning / tools / params / mm /
needle 30K and 940K), GSM8K / MMLU agreement, and `scripts/bench.sh` c1 / c4 compared with
[results/bench_summary.md](../results/bench_summary.md).

### 1.5 Known preparation steps

- On the validated instance the Neuron apt packages are on hold (`apt-mark unhold` is needed before upgrading).
- The NVMe RAID is not in `/etc/fstab`; re-assemble it after the reboot that the driver upgrade needs.
- Compile caches are tied to the SDK version and compiler flags; expect a full recompile.

## 2. Other directions (after the migration)

In the authors' priority order. Effort figures are rough estimates, not measurements.

1. **Let the compiler pass KV across partitions by alias.** Prefill graphs are still split by the runner's fixed
   `--modular-flow-mac-threshold=10`, and KV is copied whole at every partition boundary: about 6 GiB of compiler
   scratch per core that scales with the KV pool, and about 16 ms of short-request TTFT at 1M. Needs a neuronx-cc
   change; on the plugin side half a day of regression.
2. **Prefill expert layers — mostly done.** The expert prefill kernel used to loop statically over all 67 blocks of
   256 tokens although only 4–6 are non-empty per rank at EP=64; the plugin now uses nkilib's `shard_on_i_hybrid`
   (`MIMO_MOE_CTE_IMPL=hybrid`, default), which loops over the non-empty blocks only (940K first token 324.7 →
   204.2 s). The experts are now about 4 % of a segment; little is left here.
3. **Speculative decoding (MTP / DFlash).** The checkpoint ships a 3-layer MTP draft and a 5-layer DFlash draft (Qwen3
   structure, block size 8). vllm-neuron 0.24 supports only EAGLE3. vllm-neuron 0.29 already contains a DFlash
   framework (a Llama-3.1-8B example); MTP is not there. On GPU, the vLLM recipe for MiMo-V2.6-Flash shows DFlash with
   `num_speculative_tokens` 7 (vLLM ≥ 0.31). Recommended route: migrate to 0.29 first, then reuse its DFlash framework;
   the target side still needs auxiliary hidden states, a decode graph that verifies k+1 tokens (in-place KV write,
   hybrid KV, decode buckets) and a check of the draft's 1024 sliding window. If 2–3 tokens are accepted per step,
   TPOT could drop to 1/2–1/2.5 (not measured). About 2–4 days after the migration; 4–6 days from scratch on 0.24.
4. **DP attention.** At TP=64 each KV head is replicated 16× (GA) / 8× (SWA). Attention DP=16 × TP=4 would hold about
   13–14× more KV tokens in the same HBM. 1–2 weeks; replica groups must be contiguous on trn2.
5. **Admission / preemption when the KV pool is full.** Makes `MIMO_KV_CONCURRENCY` safe (measured c4 on 1M:
   20.4 → 50.8 tok/s). 1–2 days; re-check that the 940K needle does not regress.
6. **Prefix cache: handle odd tile counts inside the kernel.** The hang on long cache hits is fixed by rounding the
   tile count down to even in the caller and handling the remaining keys in a masked torch block
   ([design-notes.md](design-notes.md#prefix-cache-odd-tiles)). Handling it in the kernel would remove that block, and
   the "two programs with different loop counts hang" behavior is worth reporting upstream. About half a day.
7. **Faster NKI flash attention kernel (now the largest part of a long segment).** At 940K each 0.89 s segment spends
   about 0.42 s in the GA prefix kernel. Its engines are partly idle (DMA about 8 % busy), so the cost is the
   dependency chain QK → row max → exp → P transpose → PV and instruction overhead; six scheduling-only rewrites gained
   at most 5 %. Structural options: no P transpose, tile the K=64 QK part on the PE array, move in-segment blocks into
   the kernel. A 2× faster kernel would bring 940K from about 204 s to about 155 s. 2–3 days, decode 2 more.
8. **Validate on the PyTorch native backend** (`VLLM_NEURON_BACKEND=neuron_native`) — part of the migration in
   section 1. 3–5 days.
9. **Migrate to Neuron SDK 2.33 + vllm-neuron 0.29** — section 1. About 1 week.
10. **MiMo-V2.6-Pro-MOPD on Trn3** (same architecture family, 70 layers, hidden 6144, 384 experts, about 1T
    parameters). Rough estimate 1.5–3 weeks after the 0.29 migration; sharding and HBM planning must be redone,
    likely with context parallelism. Not measured.
11. **Compile cache on S3.** Saves the first-launch compile (about 1 hour) on new instances. Half a day; the cache is
    invalidated by SDK or flag changes.
12. **Report the 18 upstream issues** (with minimal reproductions) found during the port. Half a day.
13. **Contribute the plugin upstream** (`vllm_neuron/model/mimo_v2/`), after aligning with the maintainers and the
    0.29 APIs; the runtime patches would become regular upstream changes.
14. **Larger single images and longer videos.** Videos already go up to 16,384 vision tokens, but one image is still
    capped at about 1024 × 1024 pixels / 1024 tokens, and a 16-block video cannot share a request with an image or
    audio. Reaching the model's default of about 8.4 million pixels needs blocks of 16,384–32,768 patches, and the plain
    PyTorch ViT attention cannot hold them (4 GiB per score tensor at 16,384; the server already uses about 20.1 of
    24 GiB per core). Rewrite the ViT attention as banded windowed attention with an NKI flash kernel that supports
    the time-group masks, then re-check ViT accuracy and run the full regression. 1–2 days.
15. **Thinking end marker: GPU comparison and upstream report.** With thinking on the model sometimes omits
    `</think>` and the whole reply lands in `reasoning` ([limitations.md](limitations.md)). Run the same request at
    temperature 0 on the official GPU vLLM deployment (about half a day) and, if it behaves the same, report it to vLLM
    (half a day). Changing the parser default would affect every model that uses the Qwen3 parser.
16. **Load the long-video vision graphs on demand.** The video bucket adds about 0.3–0.7 ms to TPOT (the server loads
    16 more vision graphs and two larger prefill graphs, 0.45–0.64 GiB more HBM per core; the decode graph is the
    same). Removing the bucket would reject long videos; loading graphs on demand needs vllm-neuron support. 2–3 days
    (upstream change).
