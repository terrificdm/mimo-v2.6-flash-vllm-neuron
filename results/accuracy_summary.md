# Accuracy summary

"Recommended" is the README Step 4 command with this version of the code. The L3 logit validation (section 1) and the
multimodal Q&A were repeated on it. The task evaluation (section 2), the first multimodal rows and sections 3b / 3c
were measured with earlier versions of the code (same text path; "without the video bucket" = `num_vision_tokens_buckets
[4096]` only and WAV-only audio, "Config A" = an earlier version of the recommended command); the precision of weights,
activations and KV is the same in all of them. Files shipped here are listed in [README.md](README.md). Method details: [docs/gpu-alignment.md](../docs/gpu-alignment.md) and
[docs/design-notes.md](../docs/design-notes.md#validation-method).

## 1. Logit validation on a 3-layer model (L3)

Model: the converted FP8 checkpoint cut to original layers 0 (global attention + dense MLP), 1 (sliding-window
attention + sink + MoE) and 5 (global attention + MoE), with the real embedding, norm and lm_head. Run on the device
with TP=64 / EP=64 and the same sharding code as the full model. B=1: 4 prompts (13–375 tokens) × 16 greedy steps.

Reference ("L3_gpusim"): the HF model with the GPU algorithm emulated in FP32 (W8A8 block-FP8 linear layers with
per-token-group activation quantization to E4M3FN, W4A16 MXFP4 experts) and a BF16 run of the same model as baseline
(`scripts/accuracy/gen_gpu_sim_goldens.py`).

Criterion: vllm-neuron's `multi_prompt_logit_validation` defaults. A run passes if every step is within the per-token
tolerances (top-5 rtol 0.011, top-50 0.02, top-1000 0.03, full vocab 0.05), or the sigma ratio is ≤ 1, or every step's
Bhattacharyya coefficient is > 0.99. Sigma ratio = RMS(device − FP32 reference) / RMS(BF16 baseline − FP32 reference)
over all prompts and steps.

| Config | Sigma ratio | `pp_static_0.05` | Result | File |
|---|---|---|---|---|
| Recommended (plugin defaults: segment 4096, NKI prefill attention, 512 bucket, hybrid expert prefill kernel) | 0.907 | 3/4 | pass (sigma ratio ≤ 1) | `accuracy/l3_fp8_recommended_b1_vs_gpusim.json` |
| Earlier version (same without the hybrid expert kernel and arithmetic key indices) | 0.907 | 3/4 | pass | `accuracy/l3_fp8_final_b1_vs_gpusim.json` |
| Control: segment 4096, torch attention, no 512 bucket | 0.887 | 3/4 | pass | `accuracy/l3_fp8_seg4k_torch_b1_vs_gpusim.json` |
| Native FP8, segment 512, torch attention | 0.664 | 4/4 | pass | not shipped |
| Native FP8 + decode fixes, segment 512 | 0.715 | 4/4 | pass | not shipped |

The first two rows agree to six digits (sigma ratio 0.9070886 / 0.9070888), so the expert-kernel change does not
alter the values measurably. The earlier-version row and the control row are almost equal, so the rise from about 0.66 to about 0.9 comes from the 4096-token prefill
segment path itself (different compiled graphs and blocking), not from the NKI kernel or the 512 bucket
([design-notes.md](../docs/design-notes.md#l3-segment-accuracy)).

Phase 1 (BF16 weights, BF16 reference): B=1 63/64 steps within tolerance, sigma ratio 1.315 (main criterion not met);
the single failing step is a near-tie in MoE routing (8th vs 9th expert score gap 1.5×10⁻⁴). It passed the extended
fallback criterion approved by the user after the run. The 16 generated tokens equal the HF FP32 golden.

## 2. Task accuracy (GSM8K, MMLU)

Conditions: `scripts/api_tests/task_eval.py`, 200 questions per task sampled from the official test sets with
`random.Random(0)`, 0-shot chat, thinking off (`enable_thinking: false`), greedy (temperature 0), 4 concurrent
requests; GSM8K `max_tokens` 1024 (score = last number), MMLU `max_tokens` 16 (score = first A–D letter).

| Task | FP8, 1M without the video bucket | BF16 weights, 4096 server | Same answer | Same correctness | File |
|---|---|---|---|---|---|
| GSM8K | 196/200 = 98.0 % | 98.0 % | 199/200 | 200/200 | `accuracy/task_eval_gsm8k_fp8_vs_bf16.json` |
| MMLU | 161/200 = 80.5 % | 79.0 % | 197/200 | 197/200 (3 only FP8 right, 0 only BF16 right) | `accuracy/task_eval_mmlu_fp8_vs_bf16.json` |

- GSM8K replies are word-for-word identical in only 29.5 % of questions, but final answers match 199/200.
- One MMLU reply could not be parsed (counted wrong).
- Public reference values (different checkpoint and protocol, magnitude only): MiMo-V2-Flash Base, GSM8K 8-shot 92.3,
  MMLU 5-shot 86.7 (arXiv 2601.02780). Xiaomi has not published these two scores for V2.6-Flash-MOPD.
- The question files are not redistributed. `accuracy/task_eval_item_ids.json` lists the `idx` field of each item; which
  split ordering `idx` refers to was not recorded (unverified).

## 3. Multimodal encoders

| Check | Result | Source |
|---|---|---|
| ViT on device vs HF | cosine 0.998–0.999 | [design-notes.md](../docs/design-notes.md#vit-accuracy) |
| Image / video / audio Q&A, without the video bucket | I1–I4, base64, V1, U1, U2 pass; A1, U3 descriptive (not scored) | `api/no_video_buckets_mm.md` |
| Image / video / audio Q&A, Config A | same: 8 pass, 2 descriptive | `api/config_a_mm.md` |
| Image / video / audio Q&A, recommended | same: 8 pass, 2 descriptive | `api/recommended_mm.md` |
| Audio encoder on device vs HF FP32 | jfk 0.9782 (HF BF16 0.9794), beeps 0.9871 (0.9880) | [design-notes.md](../docs/design-notes.md#audio-criteria) |
| Audio RVQ codes, device vs CPU plugin BF16 | 90.4 % equal (HF BF16 vs FP32 itself: 57–83 %) | [design-notes.md](../docs/design-notes.md#audio-on-device) |

### 3b. ViT on long video and real photos (Config A)

Device ViT rows of a running Config A server compared with HF ViT references built from the same processor output
(`scripts/accuracy/gen_vit_reference_files.py`, `scripts/accuracy/compare_vit_dumps.py`;
`multimodal/vit_compare.json`):

| Input | cos(device, HF BF16) | cos(device, HF FP32) | cos(HF BF16, HF FP32) | Result |
|---|---|---|---|---|
| 3-minute 720p video (14,080 tokens) | 0.99834 | 0.99837 | 0.99831 | pass (fallback criterion) |
| 2048 × 1365 photo | 0.99493 | 0.99382 | 0.99544 | pass (fallback criterion, margin 0.0004) |

The main criterion (cos(device, HF BF16) ≥ 0.999) was set on small synthetic images; on these real inputs HF's own
BF16 run reaches only 0.995–0.998 against HF FP32, so both pass through the fallback criterion
(cos(device, HF FP32) ≥ cos(HF BF16, HF FP32) − 0.002, [design-notes.md](../docs/design-notes.md#vit-accuracy)).

### 3c. Audio decoding

MP3 and FLAC are decoded by soundfile 0.14.0 outside the venv; 7 WAV files decode to the same samples as the built-in
WAV reader (`multimodal/audio_offline.txt`). On the server an MP3 and the same clip as WAV were both answered as
breaking glass, and a FLAC file gave the same transcription as the same speech in WAV (`multimodal/ask/` on Config A,
`multimodal/ask_recommended/` on the recommended command). Ogg Vorbis and Ogg Opus versions of jfk.wav gave the same
transcription as the WAV word for word (authors' check; raw output not shipped).

## 4. FP8 conversion check

`tools/check_fp8_converted.py` on the full FP8 directory (`conversion/fp8_convert_check_fulldir.json`):

- MoE experts, MXFP4 → FP8 codes: 302,795,194,368 elements, 0 changed (lossless); per-output-channel scale merge
  lossless for all 98,566,144 rows.
- Linear layers, E4M3FN (max 448) → trn2 E4M3 (max 240): 222,672 of 3,073,376,256 elements changed
  (fraction 7.2×10⁻⁵, sub-normal range only).
- Index: 49 shards, 72,766 tensors, `total_size` 349,634,457,472 bytes; `pass: true`.
