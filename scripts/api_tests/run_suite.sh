#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: run_suite.sh <smoke|reasoning|tools|params|needle|mm> [--tag T] [--quick] [--mode A|B] [其他参数...]
# （实例上）先激活 vLLM venv，再跑对应脚本；结果写
#   $MIMO_RESULTS/api/<suite>[_<mode>][_<tag>].{json,md}（MIMO_RESULTS 默认 $MIMO_WORK/results）。
# smoke = env SMOKE_OUT=<上面前缀>/ bash scripts/smoke_test.sh，再 smoke_gate.py 机械判定（391×2、20×2）。
# 其他参数原样传给测试脚本（--quick、--mode 也会传）。
# 退出码：0 全部通过；1 有用例未通过；2 参数错误 / 脚本不存在 / 服务连不上。
# 环境变量：VENV（默认 DLAMI 的 vLLM venv）、RUN_SUITE_NO_VENV=1（只给单元测试用，跳过 venv 激活）、PORT。
# Path env: MIMO_WORK (default ~/mimo-work), MIMO_VENV (fallback for VENV, default DLAMI vLLM venv).
set -u
usage() { echo "usage: run_suite.sh <smoke|reasoning|tools|params|needle|mm> [--tag T] [--quick] [--mode A|B] [args...]" >&2; exit 2; }
[ $# -ge 1 ] || usage
suite=$1; shift
case $suite in
  smoke) script=smoke_gate.py ;;
  reasoning) script=test_reasoning.py ;;
  tools) script=test_tools.py ;;
  params) script=test_params.py ;;
  needle) script=needle.py ;;
  mm) script=mm_qa.py ;;
  *) echo "run_suite: unknown suite '$suite'" >&2; usage ;;
esac
tag="" mode="" fwd=()
while [ $# -gt 0 ]; do
  case $1 in
    --tag) [ $# -ge 2 ] && [ -n "$2" ] || { echo "run_suite: --tag needs a value" >&2; usage; }; tag=$2; shift 2 ;;
    --mode) [ $# -ge 2 ] || { echo "run_suite: --mode needs a value" >&2; usage; }
            case $2 in A|B) mode=$2 ;; *) echo "run_suite: --mode must be A or B" >&2; usage ;; esac
            fwd+=(--mode "$2"); shift 2 ;;
    *) fwd+=("$1"); shift ;;
  esac
done
case $tag in */*|.*) echo "run_suite: bad tag '$tag'" >&2; usage ;; esac
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
[ -f "$here/$script" ] || { echo "run_suite: $here/$script not found (suite '$suite' not implemented yet)" >&2; exit 2; }
R2=${MIMO_RESULTS:-${MIMO_WORK:-$HOME/mimo-work}/results}
out="$R2/api/$suite${mode:+_$mode}${tag:+_$tag}"
mkdir -p "$R2/api"
if [ "${RUN_SUITE_NO_VENV:-0}" != 1 ]; then
  VENV=${VENV:-${MIMO_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}}
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || { echo "run_suite: cannot activate $VENV" >&2; exit 2; }
fi
echo "run_suite: suite=$suite tag=${tag:-<none>} mode=${mode:-<none>} out=$out utc=$(date -u +%FT%TZ)"
if [ "$suite" = smoke ]; then
  if ! curl -sf "http://127.0.0.1:${PORT:-8000}/v1/models" >/dev/null; then
    echo "run_suite: server not reachable on port ${PORT:-8000}" >&2; exit 2
  fi
  env SMOKE_OUT="$out" bash "$repo/scripts/smoke_test.sh"; rc_s=$?
  # smoke 没有子用例，--quick 等参数忽略
  python3 "$here/smoke_gate.py" "$out" --out "$out"; rc_g=$?
  echo "run_suite: smoke_test rc=$rc_s smoke_gate rc=$rc_g"
  [ $rc_g = 2 ] && exit 2
  [ $rc_s = 0 ] && [ $rc_g = 0 ] && exit 0
  exit 1
fi
python3 "$here/$script" --out "$out" ${tag:+--tag "$tag"} "${fwd[@]+"${fwd[@]}"}"
rc=$?
echo "run_suite: $script rc=$rc"
exit $rc
