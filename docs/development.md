# Development and test tools

The README deploys the model with a plain `vllm serve`. This page collects the helper scripts the authors used to run
long jobs, validate the server, measure performance and accuracy, and check the repository before a release. None of
them is needed for the deployment itself.

## Background jobs

Long jobs run in tmux through `scripts/start_bg.sh <name> <command…>`; the log goes to `$MIMO_WORK/logs/<name>.log`
(default `MIMO_WORK=~/mimo-work`) and the status to `$MIMO_WORK/status/<name>.status`.

```bash
bash scripts/start_bg.sh download bash scripts/download_weights.sh
bash scripts/check_bg.sh download            # RUNNING / DONE / FAILED rc=<n> / FAILED rc=lost
bash scripts/stop_bg.sh download
```

tmux does not inherit `export`ed variables: put every variable a job needs after `env` in the command, for example
`bash scripts/start_bg.sh download env HF_TOKEN=<token> bash scripts/download_weights.sh`.
`scripts/venv_exec.sh <command…>` activates the Neuron venv and runs the command.

Host-memory guard while compiling (kills only compiler processes when available memory drops below 200 GB):

```bash
bash scripts/start_bg.sh mem_guard bash scripts/mem_guard.sh
```

## `scripts/serve.sh`

An optional wrapper that assembles the `vllm serve` command from `MIMO_*` switches (all listed in
[configuration.md](configuration.md#2-scriptsservesh-switches)). `SERVE_DRY_RUN=1` prints the command without
starting it. The README Step 4 command corresponds to:

```bash
bash scripts/start_bg.sh serve env -u VLLM_NEURON_CPU_MODE -u NKI_SIMULATOR -u VLLM_NEURON_DISABLE_NKI_KERNELS \
  MODEL_DIR=/mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio MIMO_API=1 MIMO_CTX=1048576 MIMO_MM=1 MIMO_AUDIO=1 \
  MIMO_DECODE_CTX_BUCKETS=4096,32768,131072,524288 MIMO_KV_SEG=4096 \
  MIMO_MM_VISION_BUCKETS=4096,65536 MIMO_PREFIX_CACHE=1 bash scripts/serve.sh
```

The plugin optimizations are defaults, so no plugin switch is needed. `serve.sh` also appends `$MIMO_WORK/pydeps` to
`PYTHONPATH` when the directory exists (`MIMO_PYDEPS=off` disables it) and sets `VLLM_CACHE_ROOT=/mnt/nvme/vllm-cache`.

Older measured configurations (earlier code; not re-run with the current defaults), all with `MIMO_API=1`:

| Context | `serve.sh` switches | Ready (min) |
|---|---|---|
| 4096, text only, FP8 | `MODEL_DIR=...-fp8` | not recorded |
| 512K + image / video / audio, FP8 | `MODEL_DIR=...-fp8-mm-audio MIMO_CTX=524288 MIMO_KV_SEG=1024 MIMO_MM=1 MIMO_AUDIO=1` | 12.6 |
| 1M text only, FP8 | `MODEL_DIR=...-fp8 MIMO_CTX=1048576 MIMO_KV_SEG=512` | 12.7 |
| 32K / 128K / 512K, BF16 | `MODEL_DIR=...-bf16` with `MIMO_CTX=32768` / `131072` / `524288 MIMO_KV_SEG=1024` | 8.0 / 9.3 / 7.6 |

The BF16 directory is built with `bash scripts/start_bg.sh convert_bf16 bash scripts/venv_exec.sh bash scripts/convert_full.sh`
(157 s, 617.6 GB); it is only needed for the BF16 comparisons.

## Validation suites

```bash
bash scripts/smoke_test.sh                                   # 10 requests, "10/10 HTTP 200"
bash scripts/api_tests/run_suite.sh mm --video --audio --tag mine
bash scripts/api_tests/run_suite.sh reasoning                # also: tools, params --mode A, needle ...
python3 scripts/api_tests/vision_limits_check.py --out ~/mimo-work/results/vision_limits_check
```

Expected multimodal result: I1–I4, B1, V1, U1, U2 pass; A1 and U3 are descriptive (A1 reads
`/mnt/nvme/models/MiMo-V2.6-Flash-MOPD/assets/architecture.png` from the download). `vision_limits_check.py` needs
the larger test media below; its 18-block request must return HTTP 400 and the text request after it 200.
Needle tests: `scripts/api_tests/needle.py` (options in the header); prefix-cache test:
`scripts/api_tests/prefix_cache_test.py`; decode timing at a given context: `scripts/api_tests/decode_timing.py`
(with prefix caching on, run it twice and take the second run).

<a id="larger-test-media"></a>
### Larger test media

Synthetic files with the resolutions used for the limits (the authors validated with their own recordings, which are
not redistributed):

```bash
A=${MIMO_MM_ASSETS:-$HOME/mimo-work/mm_assets}; mkdir -p "$A/large" "$A/audio"
bash scripts/venv_exec.sh python - "$A/large" <<'PY'
import sys, cv2, numpy as np
out = sys.argv[1]
def video(path, w, h, frames, fps):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), (i * 7) % 256, np.uint8)
        cv2.putText(img, f"{i // fps:03d}", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
        vw.write(img)
    vw.release()
video(f"{out}/long_720p_180s.mp4", 1280, 720, 180 * 30, 30)   # 32 sampled frames -> 14,080 vision tokens (16 blocks)
video(f"{out}/square_1024_64f.mp4", 1024, 1024, 64, 8)        # 16,384 vision tokens, the per-video limit
rng = np.random.default_rng(0)
cv2.imwrite(f"{out}/photo_2048x1365.jpg", rng.integers(0, 256, (1365, 2048, 3), np.uint8))   # served as 1248 x 832
cv2.imwrite(f"{out}/photo_small.jpg", rng.integers(0, 256, (600, 800, 3), np.uint8))
PY
PYTHONPATH="${MIMO_PYDEPS:-$HOME/mimo-work/pydeps}" bash scripts/venv_exec.sh python -c 'import sys, soundfile as sf
x, sr = sf.read("assets/audio/jfk.wav"); sf.write(sys.argv[1] + "/jfk.flac", x, sr); sf.write(sys.argv[1] + "/jfk.mp3", x, sr)' "$A/audio"
```

Vision tokens depend on the resolution and the number of sampled frames, not on the duration: the processor samples
a fixed number of frames, so longer videos are sampled more sparsely.

<a id="request-examples"></a>
## Request examples

Completion (without `max_tokens`, `/v1/completions` returns only 16 tokens):

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash",
  "prompt": "Question: What is the capital of France? Answer in one sentence.\nAnswer:",
  "max_tokens": 32, "temperature": 0}'
```

Tool call:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is the weather in Beijing right now?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "description": "Get the current weather of a city",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
      "required": ["city"]}}}],
  "tool_choice": "auto"}'
```

(The suite `scripts/api_tests/test_tools.py` asks the Chinese question `北京现在天气怎么样？` with three tools and gets a
`get_weather` call with `{"city": "北京"}`; the English wording above was not part of the suite.)

Audio as an OpenAI `input_audio` part (base64 WAV):

```bash
B64=$(base64 -w0 /mnt/nvme/mimo-work/mm_assets/audio/beeps_3.wav)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "'"$B64"'", "format": "wav"}},
    {"type": "text", "text": "How many beeps are there in this audio? Answer with a number."}]}]}'
```

OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
r = client.chat.completions.create(model="mimo-v2.6-flash", max_tokens=1024,
                                   messages=[{"role": "user", "content": "What is 17×23?"}])
print(r.choices[0].message.content)
```

`ask.py` options: `--image` (up to 2), `--video`, `--audio`, `--no-think`, `--max-tokens` (default 4096),
`--temperature` (1.0), `--top-p` (0.95). `[结束原因] length` means `max_tokens` was used up. If only `思考` appears with
`stop`, the model ended without the end-of-thinking marker and that text is the answer.

## Benchmark

```bash
bash scripts/start_bg.sh bench_c1 env MODEL_DIR=/mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio \
  BENCH_OUT=~/mimo-work/results/bench bash scripts/bench.sh c1
```

`bench.sh` runs `vllm bench serve` (random 512 input / 128 output tokens, `--ignore-eos`, 2 warm-up requests; c1 = 8
requests at concurrency 1, c4 = 16 requests at concurrency 4). Results: [results/bench_summary.md](../results/bench_summary.md).

<a id="accuracy"></a>
## Accuracy

- Task evaluation: place the question files at `results/task_eval_data/{gsm8k,mmlu}_test_200_seed0.jsonl` (not
  shipped; the item ids are in [results/accuracy/task_eval_item_ids.json](../results/accuracy/task_eval_item_ids.json))
  and run `python3 scripts/api_tests/task_eval.py --task gsm8k --n 200 --out <prefix>`.
- Logit validation on the 3-layer model: `scripts/accuracy/run_logit_validation.py` with goldens from
  `scripts/accuracy/gen_gpu_sim_goldens.py`; multimodal encoders: `scripts/accuracy/run_mm_validation.py`.
- Method and numbers: [results/accuracy_summary.md](../results/accuracy_summary.md),
  [gpu-alignment.md](gpu-alignment.md), [design-notes.md](design-notes.md#validation-rules).

## CPU tests

```bash
bash scripts/prepare_hf_reference.sh          # once
bash scripts/start_bg.sh tests bash scripts/venv_exec.sh bash scripts/run_tests.sh
```

The CPU tests run with `VLLM_NEURON_CPU_MODE=1`, where the plugin switches keep their older defaults. Some cases need
files that are not shipped and are skipped ([tests/README.md](../tests/README.md)); the soundfile case runs only with
`PYTHONPATH=$MIMO_PYDEPS`.

## Repository layout

```text
plugin/            vLLM general plugin mimo_v2_neuron (model, multimodal encoders, NKI kernels, runtime patches)
tools/             checkpoint conversion and checks, device-bug reproducers, helper tools
scripts/           serve.sh, bench.sh, smoke_test.sh, background-job helpers, weight download
scripts/api_tests/ ask.py and the API test suites (reasoning, tools, params, mm, needle, task_eval)
scripts/accuracy/  logit-validation and multimodal accuracy tooling (goldens, L3 validation)
scripts/dev/       pre-release checks (documentation paths, sensitive patterns, self-containment)
tests/             CPU / NKI-simulator tests
patches/           patches to the model's remote code for transformers 5.15
assets/            test images, video and WAV files
docs/              design notes, configuration, limitations, roadmap, troubleshooting, GPU alignment, this page
results/           curated benchmark / accuracy / API results with provenance
```

## Pre-release checks

Commands, pass criteria and the latest results: [release-checklist.md](release-checklist.md) (in Chinese).
