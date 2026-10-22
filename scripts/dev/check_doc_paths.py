#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that file paths referenced in this repository's Markdown files exist.

Usage (from any directory):

    python3 scripts/dev/check_doc_paths.py            # list missing references only
    python3 scripts/dev/check_doc_paths.py --verbose  # also list every checked / ignored reference

What is scanned (every *.md in the repository, excluding .git/):
  * inline code spans (`...`) outside fenced code blocks (``` blocks hold commands and are not scanned);
  * Markdown link targets [text](target).

A token is "path-like" when, after trimming punctuation, it contains `/` or ends with a common extension (EXTS).

Resolution rules:
  1. Inline-code paths are resolved against the repository root; link targets against the directory of the
     Markdown file that contains them. If that fails, the other base directory is tried as well.
  2. A trailing `:line`, `:line-line`, `::test_name` or `#anchor` is stripped; a leading `!` (.gitignore
     negation) is stripped; a bare extension such as `.pt` is not a path.
  3. `{a,b}` is brace-expanded and every alternative must exist; tokens with `*`, `?` or `<placeholder>`
     (`<...>` is treated as `*`) are globbed and must match at least once.
  4. If a relative path still does not resolve, it is looked up as a path suffix in the repository file
     list (e.g. a table that lists `common.py` under a `scripts/api_tests/` heading). A suffix hit counts as
     present and is labelled "suffix" with --verbose. The suffix index skips .git/ and __pycache__/.
  5. Absolute local paths are checked as-is.

Ignore rules (not counted as checked; the reason is printed with --verbose):
  * URLs: contain `://`, or start with `www.` / `mailto:`.
  * Command-line fragments: contain whitespace, start with `-` or `$`, or contain `=`, `|`, `;`, `&&`, `>`,
    `(` or `)`.
  * Obvious placeholders: the whole token is `<...>`, or it contains `…`, `...`, `XXX` or `xxx`.
  * Paths on the Trainium instance: start with /home/, /opt/, /mnt/, /tmp/, /var/, /etc/, /usr/, /dev/,
    /proc/, /sys/, /root/ or /run/.
  * Local caches / home-directory paths outside the repository: start with `~`.
  * Paths inside the installed vllm-neuron package or its upstream source tree: start with one of
    PACKAGE_PREFIXES (vllm_neuron/, examples/vllm_neuron/, test/vllm_neuron/).
  * HTTP routes: single-segment absolute paths (e.g. `/health`, `/start_profile`) or paths under /v1/.
  * "org/name" identifiers that are not files: the HF model / GitHub repository IDs in NON_PATH_IDS.
  * Files created locally by the setup steps and git-ignored, listed one by one in GENERATED.

Output: the missing references (file:line: token) and totals. Exit code 0 when nothing is missing, else 1.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXTS = (
    ".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".toml", ".yaml", ".yml", ".pt", ".patch", ".log",
    ".wav", ".mp3", ".flac", ".png", ".jpg", ".jpeg", ".mp4", ".html", ".tsv", ".csv", ".safetensors",
    ".cfg", ".ini", ".parquet", ".neff", ".ntff", ".diff", ".ipynb",
)

DEVICE_PREFIXES = (
    "/home/", "/opt/", "/mnt/", "/tmp/", "/var/", "/etc/", "/usr/", "/dev/", "/proc/", "/sys/", "/root/", "/run/",
)

# Installed vllm-neuron package / upstream source tree (on the instance or upstream, not in this repo).
PACKAGE_PREFIXES = ("vllm_neuron/", "examples/vllm_neuron/", "test/vllm_neuron/")

# "org/name" identifiers that are not file paths.
NON_PATH_IDS = {
    "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",           # HF model ID
    "XiaomiMiMo/MiMo-V2.6-Pro",                   # HF model ID (same family)
    "aws-neuron/neuron-agentic-development",      # GitHub repository ID
    "aws-neuron/vllm-neuron",                     # GitHub repository ID
    "vllm-project/vllm",                          # GitHub repository ID
    "terrificdm/mimo-v2.6-flash-vllm-neuron",     # GitHub repository ID (this repo)
}

# Created locally by the tutorial's setup steps (git-ignored, never shipped).
GENERATED = {
    "reference/",                      # scripts/prepare_hf_reference.sh
    "reference/hf/",                   # scripts/prepare_hf_reference.sh
    "reference/hf/config.json",        # scripts/prepare_hf_reference.sh
    "reference/hf_patched/",           # scripts/prepare_hf_reference.sh
    # question files the user places locally for scripts/api_tests/task_eval.py (README "Accuracy")
    "results/task_eval_data/{gsm8k,mmlu}_test_200_seed0.jsonl",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache"}

_TICK = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_LINK = re.compile(r"\[(?:[^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^\s*(```|~~~)")


def md_files() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        out.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".md"))
    return out


def iter_refs(path: str):
    """Yield (line_no, kind, token) for inline code spans and link targets outside fenced blocks."""
    in_fence = False
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if _FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for m in _LINK.finditer(line):
                yield no, "link", m.group(1)
            for m in _TICK.finditer(line):
                yield no, "tick", m.group(1)


def ignore_reason(tok: str) -> str | None:
    if "://" in tok or tok.startswith(("www.", "mailto:")):
        return "url"
    if re.search(r"\s", tok) or tok.startswith(("-", "$")) or re.search(r"=|\||;|&&|>|\(|\)", tok):
        return "command"
    if (tok.startswith("<") and tok.endswith(">")) or any(p in tok for p in ("…", "...", "XXX", "xxx")):
        return "placeholder"
    if tok.startswith(DEVICE_PREFIXES):
        return "device-path"
    if tok.startswith("~"):
        return "home-cache"
    if tok.startswith(PACKAGE_PREFIXES):
        return "package-path"
    if tok.startswith("/v1/") or re.fullmatch(r"/[A-Za-z0-9_.-]+/?", tok):
        return "http-route"
    if tok in NON_PATH_IDS:
        return "non-path-id"
    if tok in GENERATED:
        return "generated"
    return None


def looks_like_path(tok: str) -> bool:
    if re.fullmatch(r"\.[A-Za-z0-9]+", tok):  # bare extension
        return False
    return "/" in tok or tok.lower().endswith(EXTS)


def normalize(tok: str) -> str:
    tok = tok.strip().strip("“”‘’\"'，。；：,;")
    tok = tok.split("#", 1)[0]
    tok = re.sub(r"::[\w\[\]-]+$", "", tok)
    tok = re.sub(r":\d+(-\d+)?$", "", tok)
    return tok.lstrip("!")


def expand_braces(s: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", s)
    if not m:
        return [s]
    out: list[str] = []
    for part in m.group(1).split(","):
        out.extend(expand_braces(s[:m.start()] + part + s[m.end():]))
    return out


def build_index(root: str) -> list[str]:
    """All repo-relative files and directories (posix), excluding SKIP_DIRS."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel != ".":
            out.append(rel + "/")
        for f in filenames:
            out.append(f if rel == "." else f"{rel}/{f}")
    return out


def exists_at(base: str, p: str) -> bool:
    full = p if os.path.isabs(p) else os.path.join(base, p)
    if any(ch in p for ch in "*?["):
        return bool(glob.glob(full, recursive=True))
    return os.path.exists(full)


def suffix_hit(p: str, index: list[str]) -> bool:
    p = p.lstrip("./")
    if any(ch in p for ch in "*?["):
        # fnmatch's * also matches /, which is fine for this lenient fallback
        rx = re.compile("(^|/)" + fnmatch.translate(p.rstrip("/")))
        return any(rx.search(e.rstrip("/")) for e in index)
    want = p.rstrip("/")
    return any(e.rstrip("/") == want or e.rstrip("/").endswith("/" + want) for e in index)


def check(tok: str, kind: str, md_dir: str, index: list[str]) -> str | None:
    """Return how the path resolved ('root' / 'file-dir' / 'suffix' / 'abs'), or None when missing."""
    bases = [md_dir, ROOT] if kind == "link" else [ROOT, md_dir]
    pats = [re.sub(r"<[^>]*>", "*", p) for p in expand_braces(tok)]
    how = None
    for p in pats:
        if os.path.isabs(p):
            if not exists_at("/", p):
                return None
            how = "abs"
            continue
        for base in bases:
            if exists_at(base, p):
                how = how or ("root" if base == ROOT else "file-dir")
                break
        else:
            if not suffix_hit(p, index):
                return None
            how = "suffix"
    return how


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="list every checked / ignored reference")
    args = ap.parse_args()

    index = build_index(ROOT)
    files = md_files()
    checked = 0
    ignored = 0
    missing: list[str] = []
    for md in files:
        rel_md = os.path.relpath(md, ROOT)
        for no, kind, raw in iter_refs(md):
            tok = normalize(raw)
            if not tok or not looks_like_path(tok):
                continue
            reason = ignore_reason(tok)
            if reason:
                ignored += 1
                if args.verbose:
                    print(f"IGNORE {reason:12s} {rel_md}:{no}: {raw}")
                continue
            checked += 1
            how = check(tok, kind, os.path.dirname(md), index)
            if how is None:
                missing.append(f"{rel_md}:{no}: {raw}")
            elif args.verbose:
                print(f"OK     {how:12s} {rel_md}:{no}: {raw}")

    for m in missing:
        print(f"MISSING {m}")
    print(f"checked={checked} ignored={ignored} missing={len(missing)} files={len(files)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
