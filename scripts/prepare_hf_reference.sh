#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Populate reference/hf/ and reference/hf_patched/ from a downloaded copy of the Hugging Face model repo
# XiaomiMiMo/MiMo-V2.6-Flash-MOPD (revision b12f5573350bb1704513fcf119fca48e498b0423).
#
#   reference/hf/          unmodified upstream modeling_mimo_v2.py, configuration_mimo_v2.py, config.json
#   reference/hf_patched/  the two .py files with patches/*.patch applied (transformers 5.15 API fixes)
#
# The upstream remote code is NOT redistributed in this repo (see NOTICE); only the patches are.
# Used by: scripts/convert_full.sh (--modeling-file), tools/check_converted.py --patched-dir,
# tests/hf_ref.py, tests/test_config.py, tests/test_attention_long.py, tests/test_vision_cpu.py.
#
# Usage: bash scripts/prepare_hf_reference.sh [SRC_DIR]
#   SRC_DIR defaults to $MIMO_SRC_DIR, else ${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD
#   (the directory scripts/download_weights.sh writes). Only the small files are needed, e.g.:
#     hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD --revision b12f5573350bb1704513fcf119fca48e498b0423 \
#       --include '*.py' '*.json' --local-dir <dir>
# Exit codes: 0 ok, 1 missing input / checksum mismatch / patch failure.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-${MIMO_SRC_DIR:-${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD}}

# sha256 of the upstream files at the pinned revision, and of the expected patched results
UPSTREAM_SHA256="a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad  modeling_mimo_v2.py
773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e  configuration_mimo_v2.py
61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621  config.json"
PATCHED_SHA256="975fda0dbc6681cadc01567e57da28126566f52f6cf4c46617f64161a031615e  modeling_mimo_v2.py
b085f681722af2253844bc876dfa60e1defcb07da2d5adbeb77c3387c5d0b0d1  configuration_mimo_v2.py"

die() { echo "prepare_hf_reference: $*" >&2; exit 1; }
sha256_check() {  # sha256_check <dir> <"sum  name" lines>
  if command -v sha256sum >/dev/null; then (cd "$1" && sha256sum -c --quiet - <<< "$2")
  else (cd "$1" && shasum -a 256 -c --quiet - <<< "$2"); fi
}

for f in modeling_mimo_v2.py configuration_mimo_v2.py config.json; do
  [ -f "$SRC/$f" ] || die "$SRC/$f not found (set MIMO_SRC_DIR or pass the downloaded model dir)"
done
sha256_check "$SRC" "$UPSTREAM_SHA256" \
  || die "upstream files in $SRC do not match revision b12f557 (re-download with --revision b12f5573350bb1704513fcf119fca48e498b0423)"

mkdir -p "$ROOT/reference/hf" "$ROOT/reference/hf_patched"
cp "$SRC/modeling_mimo_v2.py" "$SRC/configuration_mimo_v2.py" "$SRC/config.json" "$ROOT/reference/hf/"
cp "$SRC/modeling_mimo_v2.py" "$SRC/configuration_mimo_v2.py" "$ROOT/reference/hf_patched/"
for f in modeling_mimo_v2 configuration_mimo_v2; do
  p="$ROOT/patches/$f.py.patch"
  if command -v patch >/dev/null; then
    patch -s -p1 -d "$ROOT/reference/hf_patched" < "$p" || die "patch failed: $p"
  else
    (cd "$ROOT/reference/hf_patched" && git apply -p1 --whitespace=nowarn "$p") || die "git apply failed: $p"
  fi
done
sha256_check "$ROOT/reference/hf_patched" "$PATCHED_SHA256" || die "patched files do not match the expected checksums"
echo "prepare_hf_reference: wrote $ROOT/reference/hf and $ROOT/reference/hf_patched from $SRC"
