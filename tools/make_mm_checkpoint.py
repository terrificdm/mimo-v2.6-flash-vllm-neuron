#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a multimodal (image / video) MiMo-V2 checkpoint directory from a converted text checkpoint.

See docs/design-notes.md#omni-architecture. The new directory

  * symlinks every text shard of ``--text`` (no copy of the 618 GB BF16 weights);
  * adds ``model-visual.safetensors`` with the 364 ``visual.*`` tensors of the original checkpoint (``--src``,
    already BF16, copied unchanged);
  * writes a merged ``model.safetensors.index.json`` (``total_size`` recomputed from the tensor headers);
  * writes ``config.json`` = text config + the original ``vision_config``, ``audio_config`` left ``{}``,
    ``architectures = ["MiMoV2OmniForCausalLM"]`` and ``processor_config.image_max_pixels / video_max_pixels``
    capped at ``--max-pixels`` (the vLLM processor ignores ``--mm-processor-kwargs max_pixels``, docs/design-notes.md#pixel-cap);
  * copies ``preprocessor_config.json`` and the remaining small files of ``--text`` (tokenizer, chat template,
    patched remote code).

``--audio`` (docs/design-notes.md#audio-criteria) also adds ``model-audio.safetensors`` with the audio towers unchanged -- ``audio_encoder.*`` (75)
and ``speech_embeddings.*`` (20) from the original shard, and the tokenizer encoder from
``audio_tokenizer/model.safetensors`` under the prefix ``audio_tokenizer.`` (BF16 weights, F32 codebooks; the unused
decoder and the EMA buffers ``cluster_size`` / ``embed_avg`` / ``inited`` are left out) --, restores ``audio_config``
and copies ``audio_tokenizer/config.json``. Without ``--audio`` the output is unchanged.

It writes ``<dst>.tmp`` first and renames it to ``<dst>`` only on success; an existing ``<dst>`` is refused
unless ``--force``.

Usage: make_mm_checkpoint.py --src <original dir> --text <bf16 dir> --dst <new dir> [--max-pixels 1048576]
Exit codes: 0 ok, 1 missing ``visual.*`` keys / bad source, 2 bad args (e.g. dst exists).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

VISUAL_SHARD = "model-visual.safetensors"
AUDIO_SHARD = "model-audio.safetensors"
N_AUDIO = 75 + 20
AUDIO_TOK_SKIP = ("_codebook.cluster_size", "_codebook.embed_avg", "_codebook.inited")
N_VISUAL = 364
BIAS_FREE = ("visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias", "visual.merger.ln_q.bias")


def header_bytes(path: str) -> dict:
    """{tensor name: data bytes} from a safetensors header (no tensor data read)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in hdr.items() if k != "__metadata__"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original HF checkpoint (has visual.* and vision_config)")
    ap.add_argument("--text", required=True, help="converted BF16 text checkpoint")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--audio", action="store_true", help="also add the audio towers")
    a = ap.parse_args(argv)
    dst = os.path.abspath(a.dst.rstrip("/"))
    tmp = dst + ".tmp"
    if os.path.exists(dst) and not a.force:
        print(f"ERROR: {dst} exists (use --force to replace)", file=sys.stderr)
        return 2
    for d in (a.src, a.text):
        if not os.path.isfile(os.path.join(d, "config.json")):
            print(f"ERROR: {d} has no config.json", file=sys.stderr)
            return 2

    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    vis_keys = sorted(k for k in src_idx if k.startswith("visual."))
    if len(vis_keys) != N_VISUAL:
        print(f"ERROR: expected {N_VISUAL} visual.* keys in {a.src}, found {len(vis_keys)}", file=sys.stderr)
        return 1
    bad = [k for k in BIAS_FREE if k in src_idx]
    if bad:   # the merger has no biases in the checkpoint; the plugin and the HF reference both rely on that
        print(f"ERROR: unexpected merger bias keys in source: {bad}", file=sys.stderr)
        return 1
    src_cfg = json.load(open(os.path.join(a.src, "config.json")))
    if not src_cfg.get("vision_config"):
        print(f"ERROR: {a.src}/config.json has no vision_config", file=sys.stderr)
        return 1

    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    # 1) visual tensors, unchanged
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for k in vis_keys:
        by_shard.setdefault(src_idx[k], []).append(k)
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(a.src, shard), framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k).contiguous()
    dtypes = sorted({str(t.dtype) for t in tensors.values()})
    save_file(tensors, os.path.join(tmp, VISUAL_SHARD), metadata={"format": "pt"})
    del tensors

    audio_keys = []
    if a.audio:   # 1b) audio towers, unchanged dtypes
        audio_keys = sorted(k for k in src_idx if k.startswith(("audio_encoder.", "speech_embeddings.")))
        if len(audio_keys) != N_AUDIO or not src_cfg.get("audio_config"):
            print(f"ERROR: expected {N_AUDIO} audio_encoder./speech_embeddings. keys and an audio_config in {a.src}, "
                  f"found {len(audio_keys)}", file=sys.stderr)
            shutil.rmtree(tmp)
            return 1
        tok_file = os.path.join(a.src, "audio_tokenizer", "model.safetensors")
        tensors = {}
        by_shard = {}
        for k in audio_keys:
            by_shard.setdefault(src_idx[k], []).append(k)
        for shard, keys in by_shard.items():
            with safe_open(os.path.join(a.src, shard), framework="pt") as f:
                for k in keys:
                    tensors[k] = f.get_tensor(k).contiguous()
        with safe_open(tok_file, framework="pt") as f:
            for k in f.keys():
                if k.startswith("encoder.") and not k.endswith(AUDIO_TOK_SKIP):
                    tensors["audio_tokenizer." + k] = f.get_tensor(k).contiguous()
        audio_keys = sorted(tensors)
        dtypes = sorted(set(dtypes) | {str(t.dtype) for t in tensors.values()})
        save_file(tensors, os.path.join(tmp, AUDIO_SHARD), metadata={"format": "pt"})
        del tensors
        os.makedirs(os.path.join(tmp, "audio_tokenizer"))
        shutil.copy(os.path.join(a.src, "audio_tokenizer", "config.json"), os.path.join(tmp, "audio_tokenizer"))

    # 2) text shards: symlinks; small files: copies
    text_idx = json.load(open(os.path.join(a.text, "model.safetensors.index.json")))["weight_map"]
    clash = [k for k in text_idx if k.startswith("visual.")]
    if clash:
        print(f"ERROR: text checkpoint already has visual.* keys: {clash[:3]}", file=sys.stderr)
        shutil.rmtree(tmp)
        return 1
    text_dir = os.path.abspath(a.text)
    for fn in sorted(os.listdir(text_dir)):
        s = os.path.join(text_dir, fn)
        if fn.endswith(".safetensors"):
            os.symlink(s, os.path.join(tmp, fn))
        elif os.path.isfile(s) and fn not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(s, os.path.join(tmp, fn))
    if os.path.isfile(os.path.join(a.src, "preprocessor_config.json")):
        shutil.copy(os.path.join(a.src, "preprocessor_config.json"), os.path.join(tmp, "preprocessor_config.json"))

    # 3) merged index, total_size recomputed from headers
    wm = dict(text_idx)
    wm.update({k: VISUAL_SHARD for k in vis_keys})
    wm.update({k: AUDIO_SHARD for k in audio_keys})
    total = 0
    for shard in sorted(set(wm.values())):
        sizes = header_bytes(os.path.join(tmp, shard))
        missing = [k for k, s in wm.items() if s == shard and k not in sizes]
        if missing:
            print(f"ERROR: {shard} lacks {missing[:3]}", file=sys.stderr)
            shutil.rmtree(tmp)
            return 1
        total += sum(v for k, v in sizes.items() if wm.get(k) == shard)
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=1)

    # 4) config
    cfg = json.load(open(os.path.join(text_dir, "config.json")))
    cfg["vision_config"] = src_cfg["vision_config"]
    cfg["audio_config"] = src_cfg["audio_config"] if a.audio else {}
    cfg["architectures"] = ["MiMoV2OmniForCausalLM"]
    pc = dict(cfg.get("processor_config") or src_cfg.get("processor_config") or {})
    capped = {}
    for key in ("image_max_pixels", "video_max_pixels"):
        old = pc.get(key)
        if old is None or old > a.max_pixels:
            pc[key] = a.max_pixels
            capped[key] = (old, a.max_pixels)
    cfg["processor_config"] = pc
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if os.path.exists(dst):   # --force
        shutil.rmtree(dst)
    os.rename(tmp, dst)
    vis_bytes = header_bytes(os.path.join(dst, VISUAL_SHARD))
    print(json.dumps({"dst": dst, "visual_tensors": len(vis_keys), "visual_dtypes": dtypes,
                      "visual_bytes": sum(vis_bytes.values()), "text_tensors": len(text_idx),
                      "audio_tensors": len(audio_keys), "total_size": total, "processor_config_capped": capped},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
