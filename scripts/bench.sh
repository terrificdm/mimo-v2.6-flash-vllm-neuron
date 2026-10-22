#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 简单性能测试：vllm bench serve，random 512 输入 / 128 输出。
# 用法：start_bg.sh bench_c1 bash scripts/bench.sh c1   （c1：并发 1 / 8 条；c4：并发 4 / 16 条）
# 正式跑之前先发 2 个预热请求。环境变量 MODEL_DIR、PORT 可覆盖（默认全量 BF16 目录 / 8000）；
# BENCH_OUT 改结果目录（默认 $MIMO_WORK/results/bench；不同配置用不同目录，避免互相覆盖）。
# Path env: MIMO_MODELS (default /mnt/nvme/models), MIMO_WORK (default ~/mimo-work), MIMO_VENV (see serve.sh).
set -u
mode=${1:-}
case "$mode" in
  c1) CONC=1; N=8 ;;
  c2) CONC=2; N=8 ;;     # 并发评估（docs/design-notes.md#concurrency）
  c4) CONC=4; N=16 ;;
  c8) CONC=8; N=32 ;;    # 并发评估；服务要 MIMO_MAX_SEQS>=8，否则多出的请求排队
  *) echo "usage: bench.sh c1|c2|c4|c8" >&2; exit 2 ;;
esac
M=${MODEL_DIR:-${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD-bf16}
PORT=${PORT:-8000}
OUT=${BENCH_OUT:-${MIMO_WORK:-$HOME/mimo-work}/results/bench}
source "${MIMO_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}/bin/activate" || exit 10
mkdir -p "$OUT"

echo "bench.sh: $(date -u +%FT%TZ) mode=$mode concurrency=$CONC num_prompts=$N model=$M port=$PORT"
for i in 1 2; do
  curl -sf -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model": "mimo-v2.6-flash", "prompt": "Hello, my name is", "max_tokens": 16, "temperature": 0}' \
    > /dev/null || { echo "bench.sh: warmup request $i failed" >&2; exit 3; }
  echo "bench.sh: warmup $i ok"
done

vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
  --model "$M" --served-model-name mimo-v2.6-flash --tokenizer "$M" \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --max-concurrency "$CONC" --num-prompts "$N" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --result-dir "$OUT" --result-filename "$mode.json"
