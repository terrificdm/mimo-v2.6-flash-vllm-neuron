#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: run_bg.sh <name> <command...>   由 tmux 会话 <name> 调用（经 start_bg.sh）
# 状态文件只由本脚本写：RUNNING / DONE / FAILED rc=<rc>
set -uo pipefail                      # 不用 -e，保证命令失败后仍能写出最终状态
name=$1; shift
W=${MIMO_WORK:-$HOME/mimo-work}       # work dir for logs / status (env MIMO_WORK)
mkdir -p "$W/logs" "$W/status"
st=$W/status/$name.status; log=$W/logs/$name.log
echo "RUNNING $(date -u +%FT%TZ) pid=$$" > "$st"
export PYTHONUNBUFFERED=1             # Python 输出经管道到 tee 时不缓冲，日志实时可见
echo "===== [$name] start $(date -u +%FT%TZ) cmd: $*" >> "$log"
printf '%q ' "$@" > "$W/status/$name".cmd   # 记下带引号的完整命令（status/<name>.cmd），方便原样重跑
# stop_bg.sh 发 C-c 时：命令自己处理 SIGINT 退出；tee 忽略 SIGINT 以保住日志；本脚本继续写最终状态
trap 'echo "===== [$name] SIGINT received $(date -u +%FT%TZ)" >> "$log"' INT
"$@" 2>&1 | (trap '' INT; exec tee -a "$log")
rc=${PIPESTATUS[0]}
echo "===== [$name] end $(date -u +%FT%TZ) rc=$rc" >> "$log"
if [ "$rc" -eq 0 ]; then echo "DONE $(date -u +%FT%TZ)" > "$st"; else echo "FAILED rc=$rc $(date -u +%FT%TZ)" > "$st"; fi
exit "$rc"
