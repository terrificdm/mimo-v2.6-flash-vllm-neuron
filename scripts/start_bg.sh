#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: start_bg.sh <name> <command...>
# 在 tmux 会话 <name> 里经 run_bg.sh 启动后台任务，工作目录为仓库根。
# 环境变量必须用 env 前缀写进 <command...>（tmux 不继承调用方 shell 的 export）。
# 同名任务仍在运行（状态 RUNNING 且 tmux 会话存活）时拒绝启动，退出码 3。
set -u
if [ $# -lt 2 ]; then echo "usage: start_bg.sh <name> <command...>" >&2; exit 2; fi
name=$1; shift
REPO=${MIMO_REPO:-$(cd "$(dirname "$0")/.." && pwd)}   # default: the repo this script lives in
W=${MIMO_WORK:-$HOME/mimo-work}
st=$W/status/$name.status
mkdir -p "$W/logs" "$W/status"
if tmux has-session -t "=$name" 2>/dev/null; then
  if [ -f "$st" ] && grep -q '^RUNNING' "$st"; then
    echo "start_bg: task '$name' is still running (tmux session alive, status: $(cat "$st"))" >&2
    exit 3
  fi
  # 旧会话残留但任务已结束（DONE/FAILED），清掉再起
  tmux kill-session -t "=$name" 2>/dev/null
fi
cmd="bash $(printf %q "$REPO/scripts/run_bg.sh") $(printf %q "$name")"
for a in "$@"; do cmd+=" $(printf %q "$a")"; done
tmux new-session -d -s "$name" -c "$REPO" "$cmd"
echo "start_bg: started '$name' (log $W/logs/$name.log, status $W/status/$name.status)"
