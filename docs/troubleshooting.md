# Troubleshooting

Problems a deployer is likely to hit, in the format "symptom → cause → what to do". With the README command the log
is the `serve` tmux window (redirect it with `2>&1 | tee ~/mimo-work/serve.log` if you want a file). With the optional
helper scripts ([development.md](development.md)), background jobs write `$MIMO_WORK/logs/<name>.log` and
`$MIMO_WORK/status/<name>.status` (`scripts/start_bg.sh`).

## Start-up and compilation

### The server is still not ready after 20 minutes

- First launch of a configuration compiles every graph. The 1M + multimodal configuration needs about 1 hour the
  first time; later launches with the same command reuse `$VLLM_CACHE_ROOT` and take about 18 minutes.
- vLLM's default timeouts (300 s / 600 s) are far too short for this; the plugin raises them when they are not set
  (`VLLM_NEURON_COMPILATION_TIMEOUT=7200`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`,
  `VLLM_ENGINE_READY_TIMEOUT_S=21600`). If you set them yourself, do not set lower values.
- With `scripts/start_bg.sh`: `bash scripts/check_bg.sh serve` and `tail -f $MIMO_WORK/logs/serve.log`.

### The server or compiler dies without an error; `check_bg.sh` reports `FAILED rc=lost`

- Usually host out-of-memory during compilation. Un-partitioned graphs need a lot of host memory (vision encoder
  135 GB for one compiler process; a 4096-token prefill graph 107 GB at compile time and about 33 GB per worker when
  64 workers load it).
- Run `bash scripts/mem_guard.sh` in another tmux window before compiling: below 200 GB available it kills only the
  compiler processes, so the compile fails cleanly instead of the whole machine OOM-killing tmux.
- Leave `MIMO_CC_MODULAR_MAC_MAXTOK` at its default (`64`): it limits the un-partitioned compilation to the decode
  graphs.
- `check_bg.sh` rewrites a stale `RUNNING` status whose process is gone to `FAILED rc=lost`.

### Compile error `NCC_EOOM002` / HBM above 24 GB for the decode graph

- The KV pool was sized from free HBM. `--num-gpu-blocks-override` is required (36864 at 1M, 8192 at 32K, as in the
  README).

### `serve.sh: ... must be one of ...` and exit code 2

- Only with the optional `scripts/serve.sh`: a switch has an invalid value, or a combination is not allowed (for
  example `MIMO_AUDIO=1` without `MIMO_MM=1`, `MIMO_PREFIX_CACHE=1` with `MIMO_CTX` < 32768, `MIMO_KV_SEG` with
  `MIMO_CTX` < 32768, `MIMO_MM_VISION_BUCKETS` without `MIMO_MM=1`, or with values that are not increasing multiples
  of 4096 or exceed 65536). Run with `SERVE_DRY_RUN=1` to print the effective command.

### Workers fail with `MIMO_MM_VISION_BUCKETS [...] != vision_neuron_config.num_vision_tokens_buckets [...]`

- `MIMO_MM_VISION_BUCKETS` is set in the environment and differs from `num_vision_tokens_buckets` in
  `--additional-config`, so the API-side request check would not match the compiled graphs. Unset the variable: the
  plugin then takes the buckets from `--additional-config`.

### A switch seems to have no effect

- Variables exported in your shell are not seen by jobs started in another tmux session. Export them in the same
  shell that runs `vllm serve`. Each worker logs one `MiMoV2 Neuron: switches: NAME=value(source) ...` line with the
  effective value and whether it came from the environment or the default.

### Errors about missing `reference/hf_patched/...`

- Run `bash scripts/prepare_hf_reference.sh` after downloading the checkpoint. It copies the upstream remote code of
  revision `b12f557`, checks SHA-256 and applies `patches/*.patch`.

### The download ends with exit code 1

- `scripts/download_weights.sh` failed 5 times in a row (network, Hugging Face access or disk space; the reason is in
  `~/mimo-work/download.log`). `hf download` resumes, so run the same command again; it continues where it stopped.

## Storage

### `/mnt/nvme` is empty after a reboot

- The RAID 0 array is not in `/etc/fstab` (and `mdadm.conf` has no `ARRAY` line in the validated setup). Re-assemble and
  mount: `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`.
- A stop / terminate (as opposed to a reboot) erases instance-store data: weights, converted directories and the compile
  cache must be recreated.

## Requests

### Only "thinking" output, `finish_reason` is `length` or `stop`, no answer

- `length`: the reasoning used up `max_tokens`. Raise `max_tokens` (`ask.py --max-tokens`) or turn thinking off
  (`"chat_template_kwargs": {"enable_thinking": false}`, `ask.py --no-think`).
- `stop` with only reasoning text: the model ended without the end-of-thinking marker, so vLLM's `mimo` reasoning parser
  put the whole reply into `reasoning`. That text is the answer; use `--no-think` to get it in `content`. This is the
  behavior of the model + parser, not of the port.

### Request rejected with HTTP 400 `this request needs N vision/audio encoder blocks ...`

- The request needs more than 16 encoder blocks: for example a full 16-block video (such as a 720p video of a few
  minutes) together with an image or an audio clip, or a video above 16,384 vision tokens. Send the items in separate
  requests, or use fewer frames or a lower resolution. The server keeps serving. See
  [limitations.md](limitations.md#multimodal).
- With `num_vision_tokens_buckets [4096]` only (no `65536`), a video may have at most 1024 vision tokens; use
  `[4096, 65536]` as in the README for longer videos.

### Audio request fails with `file does not start with RIFF id`

- soundfile is not on the server's `PYTHONPATH`, so only PCM WAV can be read: the optional part of README Step 1 was not
  done, `PYTHONPATH` was not exported in the server's shell, or the directory was moved or was installed for another
  venv. Install soundfile (Step 1) and restart with `PYTHONPATH=$HOME/mimo-work/pydeps`, or convert the file to WAV.
  Audio longer than 32 seconds is rejected with HTTP 400.

### `file://` image / video / audio is refused

- The file must be inside `--allowed-local-media-path` (`/mnt/nvme/mimo-work/mm_assets` in the README), or send a
  base64 data URL (`ask.py` does this).

### "Encoder cache full" on the 5th different image

- Happens when `encoder_cache_num_blocks` is derived automatically. Keep `"encoder_cache_num_blocks": 65` in
  `vision_neuron_config`.

### Second request with a long prompt returns an empty answer, then HTTP 500, then nothing responds

- Seen with an earlier version of the plugin when a long prompt hit the prefix cache (the cached length gave an odd
  number of 512-key tiles; [design-notes.md](design-notes.md#prefix-cache-odd-tiles)). Fixed in this version and
  validated up to 940K. If you see it, check that the installed plugin is this repository's version, and restart the
  server; removing `--enable-prefix-caching` avoids the cache path.

### Requests queue on the 1M server

- Expected: the 1M configuration admits one running request. `MIMO_KV_CONCURRENCY` raises the cap but the server stalls
  if the running requests together exceed about 1.18 M KV tokens. Use it only when you know the total length fits.

### `/v1/completions` returns only a few tokens

- Without `max_tokens` the completions endpoint returns 16 tokens (vLLM default). Set `max_tokens`.

## Getting help from logs

- The first error in the runtime log matters more than the last one. For example, the audio encoder once failed with
  `Failed to load collectives`, while the first error was an out-of-bounds DMA generated by the compiler for `topk`.
- 18 upstream issues were found during the port; the ones that affect this deployment and their workarounds are
  described in [design-notes.md](design-notes.md) and on this page. Reporting them upstream is a
  [roadmap](roadmap.md) item; the device-bug reproducers `tools/repro_split.py` and `tools/repro_topk.py` are
  included.
