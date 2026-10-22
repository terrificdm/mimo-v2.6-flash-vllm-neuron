#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: venv_exec.sh <command...>   激活 Neuron vLLM venv 后 exec 命令（保证 neuronx-cc/vllm 在 PATH 上）
VENV=${VENV:-${MIMO_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}}
# shellcheck disable=SC1091
source "$VENV/bin/activate" || exit 10
exec "$@"
