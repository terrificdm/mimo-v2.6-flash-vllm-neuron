#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: stop_bg.sh <name> [wait_seconds=60]
# 只停止本项目自己经 start_bg.sh 启动的 tmux 任务：先发 C-c，等待退出，超时后 kill 会话。
set -u
name=${1:?usage: stop_bg.sh <name> [wait_seconds]}
wait_s=${2:-60}
if ! tmux has-session -t "=$name" 2>/dev/null; then echo "stop_bg: no session '$name'"; exit 0; fi
tmux send-keys -t "=$name:" C-c
for _ in $(seq "$wait_s"); do
  tmux has-session -t "=$name" 2>/dev/null || { echo "stop_bg: '$name' exited"; exit 0; }
  sleep 1
done
tmux kill-session -t "=$name" 2>/dev/null
echo "FAILED rc=killed $(date -u +%FT%TZ)" > "${MIMO_WORK:-$HOME/mimo-work}/status/$name.status"
echo "stop_bg: '$name' killed after ${wait_s}s"
