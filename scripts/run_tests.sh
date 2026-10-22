#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 全部 CPU 回归测试（实例上，经 venv_exec.sh 在 venv 里运行）：
#   ① CPU 模式（不设 NKI_SIMULATOR）：pytest tests -m "not nkisim"（单元、模块、tiny e2e）
#   ② CPU 模式 + NKI 模拟器：只跑模拟器测试 moe_tkg / FP8 NKI 核（NKI_SIMULATOR 只在这一段设置）
# 任一段失败则非零退出。用法：start_bg.sh tests bash scripts/venv_exec.sh bash scripts/run_tests.sh [pytest 额外参数]
set -u
cd "$(dirname "$0")/.."
rc=0
echo "== [1/2] CPU mode: pytest tests -m 'not nkisim' =="
env -u NKI_SIMULATOR VLLM_NEURON_CPU_MODE=1 python -m pytest tests -m "not nkisim" -v -s -p no:cacheprovider "$@" || rc=1
echo "== [2/2] CPU mode + NKI simulator: tests/test_moe_tkg_sim.py tests/test_fp8_nki_sim.py =="
env VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest tests/test_moe_tkg_sim.py tests/test_fp8_nki_sim.py tests/test_kv_write_nki_sim.py tests/test_prefill_attn_nki_sim.py -v -s -p no:cacheprovider || rc=1
echo "run_tests: rc=$rc"
exit $rc
