# Tutorial: Deploy MiMo-V2.6-Flash-MOPD with vLLM Neuron

[中文版](README.zh-CN.md)

This tutorial deploys [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) on one
AWS **trn2.48xlarge** with vLLM and the vllm-neuron plugin, serving a **1M-token context together with image, video
and audio input** from a single `vllm serve` command. The model support is a vLLM general plugin in [plugin/](plugin/)
(`mimo_v2_neuron`); the weights are converted to a trn2-compatible FP8 layout on the instance.

Flow: launch the instance → install the plugin → download the weights → convert them → `vllm serve` → send requests.

## Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1. Install the plugin](#step-1-install-the-plugin)
- [Step 2. Download the weights](#step-2-download-the-weights)
- [Step 3. Convert the weights](#step-3-convert-the-weights)
- [Step 4. Launch the server](#step-4-launch-the-server)
- [Step 5. Send requests](#step-5-send-requests)
- [Step 6. Benchmark](#step-6-benchmark)
- [Step 7. Accuracy (optional)](#step-7-accuracy-optional)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Related resources](#related-resources)
- [License](#license)

## Overview

What works with the recommended command (validated by the authors with the code in this repository):

- **1M context** (1,048,576 tokens), FP8 weights (linear layers FP8 weight × FP8 activation; experts FP8 codes × BF16
  activation, numerically equal to the GPU's MXFP4 × BF16), BF16 KV cache — the same precision as the official H100
  vLLM deployment.
- **Image, video and audio** in the same server; the vision encoder and both audio towers run on Trainium.
- **Prefix caching**: a repeated or extended long prompt reaches its first token in seconds.
- **Reasoning and tool calling** with vLLM's `mimo` parsers; sampling parameters including `seed` and the penalties.

Key numbers (one trn2.48xlarge, details in [Step 6](#step-6-benchmark)):

| Measurement | Result |
|---|---|
| Short request (512 in / 128 out), concurrency 1 | TTFT 198.3 ms, TPOT 47.15 ms |
| First token at 940K / 30K tokens (first request) | 204.2 s / 3.83–3.88 s |
| Same prompt again (prefix cache hit), 940K / 30K | 2.89 s / 0.25 s |
| Needle in a haystack, 30K / 940K | 3/3 / all depths tested correct |
| GSM8K / MMLU (200 questions each, 0-shot, greedy) | 98.0 % / 80.5 % |

## Prerequisites

**Instance**

- One **trn2.48xlarge** (16 Trainium2 chips, 64 logical NeuronCores at LNC=2, 24 GiB HBM per core), from an EC2
  Capacity Block for ML or on-demand capacity.
- Inbound **SSH only** in the security group. The server listens on `127.0.0.1:8000`; reach it with
  `ssh -L 8000:127.0.0.1:8000 ...`.
- A 500 GiB gp3 root volume was used; the model files go on the local NVMe drives (below).

**AMI and software** — the Neuron multi-framework DLAMI for **Neuron SDK 2.32.0** (Ubuntu 24.04). Find its ID in your
Region through the SSM parameters in the
[Neuron multi-framework DLAMI guide (2.32.0)](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/setup/multiframework-dlami.html).
It ships the venv `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0`.

> [!IMPORTANT]
> Validated only on these versions:
>
> | Component | Version |
> |---|---|
> | Neuron SDK | 2.32.0 |
> | vllm-neuron | 0.24.0.1.1.0 |
> | vLLM | 0.24.0 |
> | torch / torch-xla | 2.11 / 2.11 |
> | neuronx-cc | 2.27.5334.0 |
> | Backend | vllm-neuron's default XLA backend (`neuron_libtorch`); the PyTorch native backend (`VLLM_NEURON_BACKEND=neuron_native`) is not used |
>
> Do not upgrade or add packages inside the venv (the plugin is installed with `--no-deps`).

**Disk** — original checkpoint 166 GB + FP8 directory 349.6 GB (326 GiB) + multimodal directory about 2.6 GB
(vision 1.46 GB + audio 1.16 GB; text shards are symlinks) + compile cache (tens of GB): plan for at least 1 TB.
Use the four instance-store NVMe drives in RAID 0 (about 6.9 TiB; check the device names with `lsblk`):

```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo mkfs.xfs /dev/md0
sudo mkdir -p /mnt/nvme && sudo mount -o noatime /dev/md0 /mnt/nvme && sudo chown "$USER:$USER" /mnt/nvme
mkdir -p /mnt/nvme/mimo-work /mnt/nvme/models /mnt/nvme/vllm-cache /mnt/nvme/hf-cache
ln -s /mnt/nvme/mimo-work ~/mimo-work
```

> [!WARNING]
> Instance-store data is lost when the instance stops. After a reboot:
> `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`.

**Base environment**

```bash
neuron-ls                                    # 16 devices, logical-neuroncore-config 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
command -v hf                                # Hugging Face CLI (huggingface_hub) from the venv
export HF_TOKEN=<your token>                 # only if your account needs one
```

## Step 1. Install the plugin

About 1 minute.

```bash
cd ~/mimo-work
git clone https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron.git
cd mimo-v2.6-flash-vllm-neuron
pip install --no-deps -e plugin
python -c "import mimo_v2_neuron; print('plugin ok')"
```

The plugin registers through vLLM's `vllm.general_plugins` entry point. Its switches default to the validated
configuration, so `vllm serve` needs no extra environment variables
([design note](docs/design-notes.md#plugin-defaults)).

Optional — **MP3 / FLAC / Ogg audio** (WAV works without it): install soundfile into a directory outside the venv and
put that directory on `PYTHONPATH` when you start the server.

```bash
pip install --no-deps --target ~/mimo-work/pydeps soundfile==0.14.0
PYTHONPATH=~/mimo-work/pydeps python -c 'import soundfile; print(soundfile.__version__, soundfile.__libsndfile_version__)'
# expected: 0.14.0 1.2.2
```

soundfile bundles libsndfile 1.2.2, which has a published CVE in its IRCAM reader (CVE-2025-52194); the plugin only
passes WAV / FLAC / Ogg / MP3 to it. Review it before installing
([audio design note](docs/design-notes.md#audio-input)).

## Step 2. Download the weights

```bash
export MIMO_MODELS=/mnt/nvme/models
tmux new -d -s download "bash scripts/download_weights.sh > ~/mimo-work/download.log 2>&1"
tail -f ~/mimo-work/download.log             # ends with "DONE <time> <size>"; exit code 1 after 5 failed tries
bash scripts/prepare_hf_reference.sh         # patched copy of the model's remote code, used by the converter
```

166 GB (129 shards, revision `b12f5573350bb1704513fcf119fca48e498b0423`) to `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD`; the
time depends on bandwidth (not recorded). `hf download` resumes, so re-run the same command after a failure.

## Step 3. Convert the weights

**Why.** The checkpoint stores linear layers in FP8 **E4M3FN** (max 448), but trn2's FP8 is **E4M3** (max 240), and the
experts are **MXFP4** (4-bit values with a shared exponent per 32 elements), which the Neuron MoE kernels do not read.
The converter rescales the linear layers into E4M3 and re-encodes the experts as FP8 with one scale per output channel.
Expert values are reproduced exactly; linear layers change in about 7.2×10⁻⁵ of the elements (rounding at the new
scale). Details: [docs/gpu-alignment.md](docs/gpu-alignment.md).

```bash
python tools/convert_checkpoint.py \
  --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --workers 16 --fp8 \
  --modeling-file reference/hf_patched/modeling_mimo_v2.py \
  --configuration-file reference/hf_patched/configuration_mimo_v2.py
```

About **5 minutes** (321 s with 16 workers); 49 shards, 72,766 tensors, 349.6 GB. Resumable.

Verify every element against the original (about **40 minutes**):

```bash
python tools/check_fp8_converted.py --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD \
  --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --workers 12 --out ~/mimo-work/fp8_convert_check.json
```

Expected: the last terminal line is `experts lossless=True; linear changed 222672 / 3073376256 (7.25e-05); errors 0; -> ...`.
The matching fields in the output JSON are `"pass": true`, `experts_mxfp4_to_fp8.changed: 0` and
`linear_e4m3fn_to_trn_e4m3.changed_fraction` about 7.2×10⁻⁵
(reference: [results/conversion/fp8_convert_check_fulldir.json](results/conversion/fp8_convert_check_fulldir.json)).

Add the vision encoder and audio towers (seconds; they stay BF16 and the text shards are symlinked, so keep the
`-fp8` directory):

```bash
OMP_NUM_THREADS=16 python tools/make_mm_checkpoint.py --audio --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD \
  --text $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8-mm-audio
```

Copy the test media to the directory the server may read local files from:

```bash
mkdir -p /mnt/nvme/mimo-work/mm_assets && cp -r assets/. /mnt/nvme/mimo-work/mm_assets/
(cd /mnt/nvme/mimo-work/mm_assets && sha256sum -c SHA256SUMS)
```

## Step 4. Launch the server

Run `scripts/mem_guard.sh` during the first compile: when available host memory drops below 200 GB it kills only the
compiler processes, so the machine does not run out of memory ([why](docs/design-notes.md#host-oom)). Start it in its
own tmux session from the repository root **before** you start the server:

```bash
tmux new -d -s memguard "bash scripts/mem_guard.sh > ~/mimo-work/mem_guard.log 2>&1"
tail -f ~/mimo-work/mem_guard.log            # every 30 s: avail=<free memory> walrus_max=<compiler peak>; Ctrl-C only stops viewing
```

If the log shows `killing compiler processes`, that start fails; start the server again and the graphs that finished
compiling are reused from the cache. Once the server is ready you can `tmux kill-session -t memguard` (later starts
load from the cache and do not compile).

Recommended configuration — **1M context + image / video / audio + prefix caching**:

```bash
tmux new -s serve                                       # detach with Ctrl-b d
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
export VLLM_CACHE_ROOT=/mnt/nvme/vllm-cache             # compile cache on the NVMe RAID
export PYTHONPATH=$HOME/mimo-work/pydeps                # only for MP3 / FLAC / Ogg (Step 1, optional)
vllm serve /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio \
  --served-model-name mimo-v2.6-flash --trust-remote-code \
  --tensor-parallel-size 64 --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 4096 --max-num-seqs 4 --block-size 32 \
  --num-gpu-blocks-override 36864 --enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [4096], "num_seqs_buckets": [1, 4],
      "kv_segment_size_buckets": [4096], "enable_structured_outputs": true,
      "decode_context_length_buckets": [4096, 32768, 131072, 524288]},
    "vision_neuron_config": {"tp_size": 8, "dp_size": 8, "num_vision_tokens_buckets": [4096, 65536],
      "vision_attention_block_size": 4096, "encoder_cache_num_blocks": 65}}' \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --no-async-scheduling \
  --generation-config vllm --limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 1}' \
  --allowed-local-media-path /mnt/nvme/mimo-work/mm_assets --mm-processor-kwargs '{"max_pixels": 1048576}' \
  --host 127.0.0.1 --port 8000
```

**Time:** the first launch compiles for about **1 hour**; later launches reuse the compile cache and are ready in
about **18 minutes** (loading 350 GB onto 64 cores). Ready when `curl -sf http://127.0.0.1:8000/v1/models` lists
`mimo-v2.6-flash`. The startup log prints one line per worker,
`MiMoV2 Neuron: switches: MIMO_FP8_IMPL=nki(default) ...`, with the value and source of every plugin switch.

| Parameter | Why |
|---|---|
| `--tensor-parallel-size 64 --enable-expert-parallel` | all 64 cores; experts split 64 ways |
| `--max-model-len 1048576` | 1M context |
| `--max-num-batched-tokens 4096`, `kv_segment_size_buckets [4096]` | long prompts are prefilled in 4096-token segments; the plugin adds a 512-token bucket for short prompts |
| `--num-gpu-blocks-override 36864` | KV pool of 36,864 × 32 tokens; required, otherwise the pool is sized from free HBM and the decode graph does not fit |
| `decode_context_length_buckets` | decode attention compiled for 4K / 32K / 128K / 512K / 1M instead of always 1M |
| `--enable-prefix-caching` | repeated long prefixes are read from the KV cache |
| `num_vision_tokens_buckets [4096, 65536]`, `encoder_cache_num_blocks 65` | videos up to 16,384 vision tokens; at most 16 encoder blocks per request (HTTP 400 above) |
| `--limit-mm-per-prompt` | up to 2 images, 1 video and 1 audio clip per request |
| `--mm-processor-kwargs max_pixels 1048576` | images above about 1024 × 1024 are downscaled (1,024 vision tokens) |
| `enable_structured_outputs` (in `neuron_config`), `--no-async-scheduling` | structured output (JSON schema, `tool_choice` with a named function) on vllm-neuron |
| `--generation-config vllm` | the model's generation_config.json does not override request defaults |

**Variant — 32K context** (same multimodal features, faster to start): change `--max-model-len 32768`,
`--num-gpu-blocks-override 8192` and remove `"decode_context_length_buckets": [...]`. Validated with the same
zero-variable setup (smoke, multimodal 8/8, 30K needle 3/3 at 3.26 s, benchmark). Other contexts (128K, 512K, text
only, BF16) were measured with earlier versions of the code: [docs/configuration.md](docs/configuration.md).

To switch a plugin optimization back to its older implementation for troubleshooting, set its variable, for example
`MIMO_FP8_IMPL=emul` ([docs/configuration.md](docs/configuration.md#plugin-switches)).

> [!CAUTION]
> The server has **no authentication**. Keep `--host 127.0.0.1`; put an authenticating proxy in front of it if other
> machines must reach it.

## Step 5. Send requests

Model name `mimo-v2.6-flash`; thinking is on by default (deltas first in `reasoning`, then in `content`).

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95,
  "messages": [{"role": "user", "content": "Introduce the Great Wall in three sentences."}]}'
```

Thinking off: add `"chat_template_kwargs": {"enable_thinking": false}`. Streaming: add `"stream": true`.

Image / video / audio by `file://` URL (the file must be under `--allowed-local-media-path`):

```bash
A=/mnt/nvme/mimo-work/mm_assets
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "file://'"$A"'/number_739.png"}},
    {"type": "text", "text": "What number is written in the image? Answer with the number only."}]}]}'
# video: {"type": "video_url", "video_url": {"url": "file://'"$A"'/digits_1234.mp4"}}   -> "1, 2, 3, 4"
# audio: {"type": "audio_url", "audio_url": {"url": "file://'"$A"'/audio/jfk.wav"}}     -> JFK quote
```

Tool calling (OpenAI format, `"tools": [...]`, `"tool_choice": "auto"`) and the OpenAI Python SDK work as with any
vLLM server. More examples (base64 audio, tools, SDK): [docs/development.md](docs/development.md#request-examples).

`scripts/api_tests/ask.py` is a small streaming client that sends local files as base64 (standard library only):

```bash
python3 scripts/api_tests/ask.py "What is 17×23?"
python3 scripts/api_tests/ask.py --no-think --image assets/number_739.png "What number is in the image?"
python3 scripts/api_tests/ask.py --no-think --video assets/digits_1234.mp4 "Which digits appear, in order?"
python3 scripts/api_tests/ask.py --no-think --audio assets/audio/jfk.wav "Transcribe this audio exactly."
```

It prints `思考` (reasoning), `回答` (answer), `[结束原因]` (finish reason), `[用量]` (usage), `[首字时间]` (time to
first token).

Quick check: `bash scripts/smoke_test.sh` sends 10 requests and expects `10/10 HTTP 200` (use
`env SMOKE_MAX_TOKENS=128` on the 1M server).

## Step 6. Benchmark

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model mimo-v2.6-flash \
  --tokenizer /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio --trust-remote-code \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --max-concurrency 1 --num-prompts 8
```

`scripts/bench.sh c1` wraps the same command with 2 warm-up requests. Results of the recommended configuration:

| Measurement | Result | Conditions |
|---|---|---|
| Short request, concurrency 1 | TTFT 198.3 ms, TPOT 47.15 ms | 512 in / 128 out, 8 requests |
| Long prompt, first request | 940K: 204.2 s (0.89 s per 4096-token segment); 30K: 3.83–3.88 s | needle prompts, one request at a time |
| Long prompt, prefix-cache hit | 940K: 2.89 s; 30K: 0.25 s | same prompt sent again; answer identical |
| Decode at about 30K context | 47.5 ms/token | — |
| 720p video, 2 / 3 / 4 minutes | 14,211 input tokens each, first token 3.3–5.2 s | thinking off |
| HBM per core at run time (busiest core) | 20.13 / 24 GiB | neuron-monitor |

These numbers were measured with the same compiled graphs started through `scripts/serve.sh` with the plugin switches
as environment variables (the code just before the plugin defaults were changed to these values). The exact Step 4
command, re-run afterwards: TTFT 198.8 ms (median), TPOT 47.46 ms, 940K first token 204.5–204.8 s, prefix-cache hit
2.82 s, 30K 3.87 s, HBM 20.13 GiB, image / video / audio regression 8/8.
The 32K-context variant (32K): TTFT 125.9 ms (median), TPOT 38.35 ms, 30K needle 3/3 with first token 3.26 s,
regression 8/8. Files: [results/README.md](results/README.md).

The 1M configuration admits one running request at a time (the KV pool holds one 1M sequence), so higher concurrency
queues. All numbers, earlier configurations and raw files: [results/bench_summary.md](results/bench_summary.md).

## Step 7. Accuracy (optional)

- **Logit validation** on a 3-layer cut of the FP8 model (original layers 0, 1, 5) on the device with the same
  sharding and kernels, against an FP32 emulation of the GPU algorithm, using vllm-neuron's
  `multi_prompt_logit_validation` criteria: sigma ratio 0.907 (≤ 1 passes) — **pass**.
- **GSM8K 98.0 %, MMLU 80.5 %** (200 questions each, 0-shot, thinking off, greedy). Against BF16 weights: same answer
  on 199/200 and 197/200 questions.
- Vision encoder vs Hugging Face: cosine 0.998–0.999; audio encoder: 0.978 (HF's own BF16: 0.979).

Rerun the task evaluation with `scripts/api_tests/task_eval.py`; method, conditions and all numbers:
[results/accuracy_summary.md](results/accuracy_summary.md), [docs/development.md](docs/development.md#accuracy).

## Limitations

Details and causes: [docs/limitations.md](docs/limitations.md).

- Validated only on Neuron SDK 2.32.0 / vllm-neuron 0.24.0.1.1.0 with the XLA backend.
- One running request at a time on the 1M configuration.
- Long first prompts are slow: about 204 s at 940K tokens (prefix-cache hits take seconds).
- Single images up to about 1024 × 1024 (1,024 vision tokens; larger ones are downscaled).
- Video limit is **16,384 vision tokens**, set by resolution, not duration: 720p videos of 2–4 minutes all fit; long
  videos are sampled more sparsely (a 4-minute video gets about one frame every 7.5 s). 1080p and above not tested.
- At most **16 encoder blocks per request** (each image or 1,024-token video group = 1 block, each audio clip = 1 block):
  a long 720p video (16 blocks) cannot share a request with an image or audio (HTTP 400). Image + video + audio +
  text in one request works when the total fits; the model may mix up content between modalities.
- Audio: WAV, and FLAC / MP3 / Ogg with soundfile; at most 32 s, one clip per request.
- With thinking on, the model sometimes omits the end-of-thinking marker and the whole reply lands in `reasoning`.
- No speculative decoding, FP8 KV cache or DP attention.
- First compile about 1 hour; weights and compile cache live on instance storage.

## Roadmap

Details and estimates: [docs/roadmap.md](docs/roadmap.md).

1. **Migrate to Neuron SDK 2.33 + vllm-neuron 0.29** and the PyTorch native backend (about 1 week).
2. **Speculative decoding**: the checkpoint ships a 3-layer MTP draft and a 5-layer DFlash draft. vllm-neuron 0.29
   already contains a DFlash framework; MiMo needs target-side support (auxiliary hidden states, verifying k+1 tokens
   in the decode graph). Estimated 2–4 days after the migration, not measured.
3. Faster long prefill (the global-attention prefix kernel is now the largest part), DP attention for concurrency,
   KV-aware admission, larger single images, compile cache on S3, upstreaming the plugin.

## Troubleshooting

More cases: [docs/troubleshooting.md](docs/troubleshooting.md).

- **Not ready after 20 minutes** — the first launch compiles for about 1 hour; watch the log in the `serve` tmux.
- **Server or compiler disappears during compilation** — host out of memory; run `scripts/mem_guard.sh`.
- **`NCC_EOOM002` / decode graph above 24 GB HBM** — `--num-gpu-blocks-override` is missing.
- **HTTP 400 `this request needs N vision/audio encoder blocks ...`** — more than 16 blocks in one request; send the
  items separately or lower the video resolution.
- **`file does not start with RIFF id`** — soundfile is not on the server's `PYTHONPATH` (MP3 / FLAC / Ogg).
- **`file://` media refused** — put the file under `--allowed-local-media-path` or send base64.
- **Only reasoning, no answer** — raise `max_tokens` or turn thinking off.
- **`/mnt/nvme` empty after reboot** — re-assemble the RAID (data is gone after stop / terminate).

## Related resources

- [Tutorial: Deploy Llama 3.3 70B (FP8) with vLLM Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-llama3-70b.html) (the format this tutorial follows)
- [XiaomiMiMo/MiMo-V2.6-Flash-MOPD on Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [Neuron release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
- [Port GPU-compatible models to vLLM Neuron with the Autoport skill](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-agentic-development/developer_guides/neuron-framework-autoport-vllm-neuron.html)
  (not used here; it covers roughly the scope of a first BF16 port)
- In this repository: [design notes](docs/design-notes.md), [configuration](docs/configuration.md),
  [GPU alignment](docs/gpu-alignment.md), [development and test tools](docs/development.md),
  [results](results/README.md), [pre-release checks](docs/release-checklist.md).

## License

Apache-2.0, see [LICENSE](LICENSE). The model weights are MIT-licensed by Xiaomi and are not included. Third-party
sources are listed in [NOTICE](NOTICE).
