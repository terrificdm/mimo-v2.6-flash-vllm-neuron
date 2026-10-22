#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Self-containment check: copy this repository alone (without .git) into a temporary directory and verify that it
# does not depend on the authors' development repository.
#   1. no references to development-repository documents, internal decision numbers, task labels or internal
#      operation paths
#      (pattern list below; this script itself is excluded because it contains the patterns)
#   1b. no git commit hashes (7-40 hex digits) and no development-stage terms; SHA-256 values, the SHA256SUMS file and
#      the pinned Hugging Face model revision (an upstream dependency the reader needs) are allowed
#   1c. no internal configuration version labels (v1-v9 as a standalone word, the ctx1m-v<N> tags): the results are
#      called "recommended configuration" plus their test conditions. Allowed: real software versions (a dot and a
#      digit follow, e.g. v0.24), URL paths such as /v1/models, the hardware name NeuronCore-v3 / -v4, the package
#      name mimo-v2-neuron and Python code (local variable names); file names are checked too
#   2. documentation paths resolve (scripts/dev/check_doc_paths.py)
#   3. every .py passes python3 -m py_compile
#   4. every .sh passes bash -n
#   5. generic sensitive-pattern scan is clean (scripts/dev/scan_sensitive.sh)
# Usage: bash scripts/dev/check_independence.sh        Exit code: 0 all pass, 1 otherwise.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SELF=scripts/dev/$(basename "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir "$TMP/repo"
(cd "$ROOT" && tar --exclude=./.git --exclude='*/__pycache__' -cf - .) | (cd "$TMP/repo" && tar -xf -)
cd "$TMP/repo" || exit 1
echo "isolated copy: $(find . -type f | wc -l | tr -d ' ') files"

# Development-repository report / plan / decision-log names, internal decision numbers (P-xx, D2-xx), task labels
# (the letter T plus two digits and an optional letter, except the tool suite's test-case ID T10; review-finding and
# design-risk labels; the test-case IDs T1-T10 stay allowed), internal operation paths and the development
# repository's name.
PATTERN='REPORT_PHASE2|REPORT\.md|REPRODUCE\.md|\.agents/|mimo2\.6-adaption|trn-study|source project|source repo|源项目|源仓库|P-[0-9]{2,3}|D2-[0-9]{2}|mimo-work/status|phase2-design|phase2-plan|design\.md|logs/instance'
PATTERN+='|(^|[^A-Za-z0-9_])T(0[0-9]|1[1-9]|[2-9][0-9])[a-z]?([^A-Za-z0-9_]|$)|R3-#[0-9]|design R[0-9]|R[5]/R6'

fail=0
echo "== 1. development-repository references"
hits=$(grep -rnIE --exclude-dir=.git -- "$PATTERN" . | grep -v "^\./$SELF:" || true)
n=$(printf '%s' "$hits" | grep -c . || true)
[ "$n" -gt 0 ] && { printf '%s\n' "$hits"; fail=1; }
echo "hits=$n"

# 1b. Commit hashes: a token of 7-40 hex digits with at least one letter a-f. Not counted: lines that carry a SHA-256
# value (the word sha256 / SHA-256 on the line, or the SHA256SUMS file), the mantissa of a float such as 1.234e-05,
# and the pinned model revision. Development-stage terms are matched case-insensitively.
HF_REV='b12f5573350bb1704513fcf119fca48e498b0423'
STAGE='last_validated|restore_serve|phase ?2|阶段|编排者|workflow|ctx1m-v[0-9]'
echo "== 1b. commit hashes and development-stage terms"
hex=$(grep -rnIoE --exclude-dir=.git --exclude=SHA256SUMS -- '\b[0-9a-f]{7,40}\b' . | grep -v "^\./$SELF:" \
  | awk -F: -v rev="$HF_REV" '{t=$NF} t ~ /[a-f]/ && t !~ /^[0-9]+e$/ && index(rev, t) != 1 {print $1 ":" $2 ":" t}' \
  | while IFS=: read -r f l t; do sed -n "${l}p" "$f" | grep -qiE 'sha-?256' || echo "$f:$l: $t"; done)
nh=$(printf '%s' "$hex" | grep -c . || true)
[ "$nh" -gt 0 ] && { printf '%s\n' "$hex"; fail=1; }
echo "hash_hits=$nh"
stage=$(grep -rnIiE --exclude-dir=.git -- "$STAGE" . | grep -v "^\./$SELF:" || true)
nst=$(printf '%s' "$stage" | grep -c . || true)
[ "$nst" -gt 0 ] && { printf '%s\n' "$stage"; fail=1; }
echo "stage_hits=$nst"

# 1c. Configuration version labels. A preceding "/" (API path /v1/...) or a following ".<digit>" (software version)
# is not a label.
VER='(^|[^A-Za-z0-9_/])v[0-9]([^A-Za-z0-9_.]|[.]([^0-9]|$)|$)|ctx1m-v'
echo "== 1c. configuration version labels"
ver=$( { grep -rnIE --exclude-dir=.git --exclude='*.py' -- "$VER" . | grep -v "^\./$SELF:" \
  | sed -E 's/NeuronCore-v[0-9]//g; s/mimo-v2-neuron//g' | grep -E -- "$VER"; find . -path ./.git -prune -o -type f -print | grep -E -- "$VER"; } || true)
nv=$(printf '%s' "$ver" | grep -c . || true)
[ "$nv" -gt 0 ] && { printf '%s\n' "$ver"; fail=1; }
echo "version_label_hits=$nv"

echo "== 2. documentation paths"
python3 scripts/dev/check_doc_paths.py || fail=1

echo "== 3. py_compile"
np=0; fp=0
while IFS= read -r -d '' f; do
  np=$((np + 1)); python3 -m py_compile "$f" 2>/dev/null || { fp=$((fp + 1)); echo "FAIL $f"; }
done < <(find . -name '*.py' -type f -print0)
echo "py files=$np fail=$fp"; [ "$fp" -eq 0 ] || fail=1

echo "== 4. bash -n"
ns=0; fs=0
while IFS= read -r -d '' f; do
  ns=$((ns + 1)); bash -n "$f" 2>/dev/null || { fs=$((fs + 1)); echo "FAIL $f"; }
done < <(find . -name '*.sh' -type f -print0)
echo "sh files=$ns fail=$fs"; [ "$fs" -eq 0 ] || fail=1

echo "== 5. sensitive patterns"
find . -name __pycache__ -type d -prune -exec rm -rf {} +
bash scripts/dev/scan_sensitive.sh . || fail=1

echo "== result: $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
exit "$fail"
