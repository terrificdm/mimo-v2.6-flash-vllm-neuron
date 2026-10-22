#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 全量 BF16 转换（经 start_bg.sh convert_full 启动；需在 venv 中运行）
set -u
cd "$(dirname "$0")/.." || exit 10
SRC=${SRC:-${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD}
DST=${DST:-${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD-bf16}
WORKERS=${WORKERS:-32}
EXTRA=()
if [ -f reference/hf_patched/modeling_mimo_v2.py ]; then EXTRA=(--modeling-file reference/hf_patched/modeling_mimo_v2.py); fi
exec python tools/convert_checkpoint.py --src "$SRC" --dst "$DST" --workers "$WORKERS" "${EXTRA[@]}"
