#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: start_bg.sh mem_guard bash scripts/mem_guard.sh   （实例上；主机内存防护，docs/design-notes.md#host-oom）
# 每 MEM_GUARD_INTERVAL 秒（默认 30）记录一次可用内存和 RSS 最大的 5 个进程，并跟踪 walrus_driver（编译器后端）
# 的单进程峰值 RSS。可用内存低于 MEM_GUARD_MIN_GB（默认 200）时，先杀掉编译器进程（walrus_driver、neuronx-cc），
# 让这次编译失败退出，避免整机 OOM 连带杀掉 tmux 和其他任务；只杀编译器，不碰 vLLM worker。
# 输出写到 stdout（start_bg.sh 收进 $MIMO_WORK/logs/mem_guard.log）；峰值另写 $MIMO_WORK/status/mem_guard.peak。
set -u
interval=${MEM_GUARD_INTERVAL:-30}
min_gb=${MEM_GUARD_MIN_GB:-200}
peak_kb=0
peak_file=${MIMO_WORK:-$HOME/mimo-work}/status/mem_guard.peak
mkdir -p "$(dirname "$peak_file")"
echo "mem_guard: start $(date -u +%FT%TZ) interval=${interval}s min_available=${min_gb}GB"
while true; do
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  top=$(ps -eo rss=,comm= --sort=-rss | head -5 | awk '{printf "%s:%dGB ", $2, $1 / 1048576}')
  w_kb=$(ps -C walrus_driver -o rss= 2>/dev/null | sort -n | tail -1 | tr -d ' ')
  w_kb=${w_kb:-0}
  if [ "$w_kb" -gt "$peak_kb" ]; then
    peak_kb=$w_kb
    echo "walrus_driver peak RSS $((peak_kb / 1048576)) GB at $(date -u +%FT%TZ)" > "$peak_file"
  fi
  echo "$(date -u +%FT%TZ) avail=${avail_gb}GB walrus_max=$((w_kb / 1048576))GB top: $top"
  if [ "$avail_gb" -lt "$min_gb" ]; then
    echo "mem_guard: available ${avail_gb}GB < ${min_gb}GB, killing compiler processes $(date -u +%FT%TZ)"
    ps -eo pid=,rss=,args= | grep -E 'walrus_driver|neuronx-cc' | grep -v grep | head -20
    pkill -9 -x walrus_driver
    pkill -9 -f 'bin/neuronx-cc compile'
  fi
  sleep "$interval"
done
