# FP8 weights + BF16 KV: alignment with the GPU deployment

The goal of the FP8 path is to match, tensor by tensor, the precision of the official GPU deployment, not to invent a
Trainium-specific quantization. The GPU reference is the vLLM command on the model card (`vllm serve
XiaomiMiMo/MiMo-V2.6-Flash-MOPD --tensor-parallel-size 4 ...`, no `--kv-cache-dtype`, i.e. `auto`): FP8 checkpoint
weights, dynamic activation quantization, BF16 KV cache, evaluated for H100.

Decision summaries: [design-notes.md](design-notes.md#fp8-alignment), [V stored with 128 dims](design-notes.md#v-no-pad).

## 1. What the checkpoint contains

`quantization_config`: `quant_method=fp8, fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128,128],
store_dtype=mxfp4, mxfp4_block_size=32`.

- q/k/v and the layer-0 dense MLP: FP8 **E4M3FN** codes with one F32 scale per 128 × 128 block. The fused QKV tensor is
  pre-split for TP=4 and interleaved (`[Q0|K0|V0|…|Q3|K3|V3]`), so it cannot be loaded as is with TP=64.
- 256 routed experts × 47 layers: **MXFP4** (packed E2M1 values plus one E8M0 scale per 32 elements).
- o_proj (`ignored_layers`), embeddings, lm_head, norms, router weights, sinks: BF16. Router
  `e_score_correction_bias`: F32.

## 2. E4M3 (max 240) vs E4M3FN (max 448)

The two 8-bit formats share the same bit layout (1 sign, 4 exponent, 3 mantissa bits) but treat the top exponent
differently:

| | E4M3FN (checkpoint, NVIDIA GPUs) | E4M3 on trn2 (IEEE-like) |
|---|---|---|
| Largest finite value | 448 | 240 |
| Top exponent | holds finite values (only `S.1111.111` is NaN) | reserved for Inf / NaN |
| Effect on trn2 | codes for 256–448 are read as Inf / NaN | — |

Measured on the device (single-core probe `tools/probe_fp8.py`):

- neuronx-cc rejects F8E4M3FN tensors on trn2 (`NCC_EVRF051`) unless `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`
  is passed; the vllm-neuron runner passes it by default, and then E4M3FN bit patterns are simply read as E4M3.
- Host E4M3FN codes copied to the device and cast to BF16: 448 / 288 → NaN, 256 → Inf, every value ≤ 240 correct.
- MXFP4 / MX matmul instructions exist in NKI 0.6.0 but are documented for NeuronCore-v4 (trn3) and later only;
  trn2 is NeuronCore-v3.

Feeding the checkpoint codes unchanged would therefore corrupt every weight above 240, and the MXFP4 experts need a
different storage format on trn2.

## 3. The offline conversion (`tools/convert_checkpoint.py --fp8`)

- **Linear layers.** Scales are expanded from 128 × 128 blocks to one scale per (row, 128 columns) group, so any row
  split (q/k/v split, TP=64 sharding) needs no re-quantization. In each group whose largest code exceeds 240, all codes
  are halved and the scale is doubled. Halving is exact except for odd codes in the sub-normal range: 222,672 of
  3,073,376,256 elements change (7.2×10⁻⁵).
- **Experts.** MXFP4 values (E2M1 × 2^k) are written as exact FP8 codes and the 32-element E8M0 scales are merged into
  one F32 scale per output channel. All 302,795,194,368 elements and all 98,566,144 row scales are bit-exact.
- `quantization_config` is replaced by a `neuron_fp8_config` block so vLLM's GPU FP8 path does not try to parse it.
- Check: `tools/check_fp8_converted.py` (result shipped as `results/conversion/fp8_convert_check_fulldir.json`).

## 4. How each tensor is computed

| Tensor | GPU (H100, model card command) | Trainium (recommended command) | Difference |
|---|---|---|---|
| q/k/v, layer-0 dense MLP | W8A8 block GEMM; activations quantized per token per 128 channels | own NKI kernel `nki_fp8.w8a8_block_linear`: per-token per-128-channel dynamic activation quantization inside the kernel (scale = amax/448, stored as halved code + doubled scale), FP8 × FP8 on the tensor engine, FP32 accumulation in PSUM, BF16 output (`MIMO_FP8_IMPL=nki`) | activation codes can differ only where \|x/scale\| < 2⁻⁵ (trn2 sub-normals); reciprocal multiply can move single codes by one step; summation order |
| 256 experts | Marlin W4A16 (BF16 activations) | nkilib MoE kernels in ROW mode read the FP8 codes + per-channel scale directly, BF16 activations (`moe_cte` prefill, `moe_tkg` decode; `MIMO_FP8_EXPERT_IMPL=row`) | summation order only (numerically W4A16, same as H100). B200's MXFP4 × MXFP8 path is not emulated |
| o_proj, embeddings, lm_head | BF16 | BF16 | none |
| Norms, router | FP32 norm / FP32 sigmoid + `noaux_tc` | same | none |
| `e_score_correction_bias` | F32 | F32 end to end (asserted at load) | none |
| `attention_sink_bias` | BF16, used in FP32 softmax | converted to F32 at load (same values) | none |
| KV cache | BF16, K 192-dim and V 128-dim stored separately (DIFFKV backend) | BF16, K 192 / V 128 stored separately (`MIMO_KV_VNOPAD=1`) | blocking and summation order of attention |
| Vision encoder | BF16 | BF16 on Trainium (TP 8 × DP 8) | none |
| Audio towers, audio tokenizer | BF16; vLLM rounds the F32 codebooks to BF16 | BF16; codebooks rounded to BF16 like vLLM (`MIMO_AUDIO_CODEBOOK=bf16`); RVQ code choice via compare + min index | code index can differ only on exact ties |
| MTP / DFlash draft layers | optional | not loaded | no speculative decoding |

Conclusion: apart from two differences forced by trn2 hardware — MXFP4 experts stored as FP8
codes (lossless) and FP8 linear-layer codes rescaled to fit 240 (about 7×10⁻⁵ of codes) — every tensor class has
the same precision and the same kind of computation as the H100 deployment. FP8 KV cache was not used: with FP8
weights saving about 4.5 GiB per core, BF16 KV fits at 1M context ([HBM budget](design-notes.md#hbm-budget)).

## 5. The BF16-emulation path (plugin default)

Without `MIMO_FP8_IMPL=nki` / `MIMO_FP8_EXPERT_IMPL=row` the plugin uses an emulation path: linear layers quantize
activations in the graph with the GPU algorithm (`MIMO_FP8_ACT_MAX=448`) and feed exact FP8 values as BF16 to the
tensor engine; experts are expanded to BF16 every step. It is numerically close but slow (4096 context, c1 TPOT
89.5 ms vs 43.1 ms native). Use it only for comparisons. Related switches: `MIMO_FP8_ACT`, `MIMO_FP8_ACT_MAX`,
`MIMO_FP8_MM` (see [configuration.md](configuration.md)).

## 6. How alignment was checked

- L3 logit validation against an FP32 emulation of the GPU algorithm (`scripts/accuracy/gen_gpu_sim_goldens.py`):
  final switches sigma ratio 0.907, pass. The emulation itself agrees with the BF16-weight reference: same FP32
  greedy tokens, top-1 agreement 1.000, mean top-5 relative error 0.0022 (FP32) / 0.0051 (BF16), max 0.021.
  Not emulated: GPU kernel summation order, K/V re-quantization when TP ≠ 4, B200 MXFP8 activations.
- GSM8K / MMLU, FP8 final config vs BF16 weights: same answer 199/200 and 197/200.

Numbers and conditions: [results/accuracy_summary.md](../results/accuracy_summary.md).
