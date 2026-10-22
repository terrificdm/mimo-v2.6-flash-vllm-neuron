#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 用法: check_bg.sh <name>...   （实例上）后台任务（start_bg.sh / run_bg.sh）的真实状态，每个任务打印一行：
#   <name> RUNNING|DONE|FAILED|MISSING <status 原文>
# status 写着 RUNNING 但记录的 pid 已不在（被 OOM 或其他原因杀掉，run_bg.sh 来不及写结果）时，
# 把 status 改成 "FAILED rc=lost ..." 并按 FAILED 报告（docs/design-notes.md#host-oom）。退出码：全部 DONE 为 0，有 RUNNING 为 3，
# 有 FAILED/MISSING 为 1。
set -u
[ $# -ge 1 ] || { echo "usage: check_bg.sh <name>..." >&2; exit 2; }
rc=0
for name in "$@"; do
  st=${MIMO_WORK:-$HOME/mimo-work}/status/$name.status
  if [ ! -f "$st" ]; then echo "$name MISSING"; rc=1; continue; fi
  line=$(cat "$st")
  case "$line" in
    RUNNING*)
      pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<< "$line")
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$name RUNNING $line"; [ $rc -eq 0 ] && rc=3
      else
        new="FAILED rc=lost $(date -u +%FT%TZ) (was: $line; pid ${pid:-?} gone)"
        echo "$new" > "$st"
        echo "$name FAILED $new"; rc=1
      fi ;;
    DONE*) echo "$name DONE $line" ;;
    *) echo "$name FAILED $line"; rc=1 ;;
  esac
done
exit $rc
