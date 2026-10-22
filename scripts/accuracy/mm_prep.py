#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multimodal preparation checks on the instance CPU (docs/design-notes.md#pixel-cap, docs/design-notes.md#mm-reference-inputs).

Checks (all CPU, no NeuronCores):
  import       `import vllm.model_executor.models.mimo_v2_omni` works on the Neuron platform
  parity       HF Qwen2VLImageProcessor (preprocessor_config.json) vs the processor vLLM serves with
               (MiMoOmniProcessor inside MiMoV2OmniMultiModalProcessor): grid_thw equal, pixel_values atol 1e-5;
               plus the tensors vLLM actually hands to the model (dtype / equality after the cast)
  video        the mp4 decoded by vLLM's own VideoMediaIO (default loader kwargs), processed by the vLLM processor:
               frames, grid_thw, patches (must be <= --patch-budget), embeds, and the ProcessingInfo estimate
               (get_num_video_tokens) next to the real count
  max_pixels   whether --mm-processor-kwargs max_pixels / hf_overrides processor_config.image_max_pixels change
               the real number of image tokens of a 2048x2048 image
The model dir's config.json must have the original vision_config; the original FP8 dir is accepted (its
quantization_config is dropped through hf_overrides, architectures forced to MiMoV2OmniForCausalLM, nothing on disk
changes).

  python scripts/accuracy/mm_prep.py --model-dir $MIMO_MODELS/MiMo-V2.6-Flash-MOPD \
      --assets $MIMO_WORK/mm_assets --out $MIMO_WORK/results/mm/prep.json
Exit codes: 0 all checks pass, 1 a check failed (details in the JSON), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback

IMAGES = ["red_square.png", "number_739.png", "three_circles.png", "word_trainium.png"]
VIDEO = "digits_1234.mp4"
IMG_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>Describe the image."
VID_PROMPT = "<|vision_start|><|video_pad|><|vision_end|>Describe the video."


def build_model_config(model_dir: str, processor_overrides: dict | None = None):
    from vllm.config import ModelConfig

    def ov(cfg):
        if hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")
        cfg.architectures = ["MiMoV2OmniForCausalLM"]
        if processor_overrides:
            pc = copy.deepcopy(getattr(cfg, "processor_config", {}) or {})
            pc.update(processor_overrides)
            cfg.processor_config = pc
        return cfg

    return ModelConfig(model=model_dir, tokenizer=model_dir, trust_remote_code=True, hf_overrides=ov,
                       max_model_len=8192)


def vllm_processor(model_dir: str, processor_overrides: dict | None = None):
    from vllm.multimodal import MULTIMODAL_REGISTRY

    return MULTIMODAL_REGISTRY.create_processor(build_model_config(model_dir, processor_overrides))


def run_vllm(proc, prompt: str, mm_data: dict, mm_kwargs: dict | None = None) -> dict:
    items = proc.info.get_data_parser().parse_mm_data(mm_data)
    return proc(prompt, items, hf_processor_mm_kwargs=mm_kwargs or {})


def item_field(out: dict, modality: str, idx: int, key: str):
    return out["mm_kwargs"][modality][idx][key].data


def hf_image_processor(model_dir: str):
    from transformers import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor.from_pretrained(model_dir)


def load_video(path: str):
    """Decode like the API server does: VideoMediaIO with its default kwargs (num_frames=32, opencv backend)."""
    from pathlib import Path

    from vllm.multimodal.media import ImageMediaIO, VideoMediaIO

    return VideoMediaIO(ImageMediaIO()).load_file(Path(path))


def image_parity(proc, hf_ip, assets: str, atol: float) -> dict:
    import torch
    from PIL import Image

    res = {"atol": atol, "hf_class": type(hf_ip).__name__, "items": {}, "ok": True}
    hf_proc = proc.info.get_hf_processor()
    for name in IMAGES:
        img = Image.open(os.path.join(assets, name)).convert("RGB")
        ref = hf_ip(images=[img], return_tensors="pt")
        raw = hf_proc(text=IMG_PROMPT, images=[img], return_tensors="pt")  # MiMoOmniProcessor, before vLLM's cast
        out = run_vllm(proc, IMG_PROMPT, {"image": [img]})
        pv_served = item_field(out, "image", 0, "pixel_values")
        thw_served = item_field(out, "image", 0, "image_grid_thw")
        ref_pv, raw_pv = ref["pixel_values"].float(), raw["pixel_values"].float()
        same_shape = tuple(ref_pv.shape) == tuple(raw_pv.shape)
        max_abs = float((ref_pv - raw_pv).abs().max()) if same_shape else None
        rec = {
            "hf_grid_thw": ref["image_grid_thw"].tolist(), "vllm_grid_thw": raw["image_grid_thw"].tolist(),
            "served_grid_thw": thw_served.tolist(), "pixel_values_shape": list(raw_pv.shape),
            "hf_vs_vllm_max_abs": max_abs, "served_dtype": str(pv_served.dtype), "raw_dtype": str(raw_pv.dtype),
            "served_equals_raw_cast": bool(torch.equal(pv_served, raw_pv.to(pv_served.dtype))),
            "served_vs_raw_max_abs": float((pv_served.float() - raw_pv).abs().max()),
            "num_embeds": int(out["mm_placeholders"]["image"][0].get_num_embeds()),
        }
        rec["ok"] = bool(ref["image_grid_thw"].tolist() == raw["image_grid_thw"].tolist() == [thw_served.tolist()]
                         and same_shape and max_abs is not None and max_abs <= atol and rec["served_equals_raw_cast"])
        res["items"][name] = rec
        res["ok"] = res["ok"] and rec["ok"]
    return res


def video_check(proc, model_dir: str, assets: str, budget: int) -> dict:
    frames, meta = load_video(os.path.join(assets, VIDEO))
    out = run_vllm(proc, VID_PROMPT, {"video": [(frames, meta)]})
    thw = item_field(out, "video", 0, "video_grid_thw").tolist()
    pv = item_field(out, "video", 0, "pixel_values_videos")
    spt = float(item_field(out, "video", 0, "second_per_grid_ts"))
    ph = out["mm_placeholders"]["video"][0]
    patches = int(thw[0] * thw[1] * thw[2])
    info = proc.info
    est = info.get_num_video_tokens(image_width=int(frames.shape[2]), image_height=int(frames.shape[1]),
                                    num_frames=int(frames.shape[0]), image_processor=info.get_image_processor(),
                                    mm_kwargs={})
    tok = info.get_tokenizer()
    ids = out["prompt_token_ids"]
    text = tok.decode(ids[ph.offset: ph.offset + ph.length])
    import re

    timestamps = re.findall(r"\d\d:\d\d", text)
    rec = {
        "decoded_frames": list(frames.shape), "loader_metadata": {k: (v if isinstance(v, (int, float, str, bool))
                                                                      else str(v)) for k, v in meta.items()
                                                                  if k != "frames_indices"},
        "video_grid_thw": thw, "pixel_values_videos_shape": list(pv.shape), "patches": patches,
        "patch_budget": budget, "num_embeds": int(ph.get_num_embeds()), "placeholder_len": int(ph.length),
        "second_per_grid_ts": spt, "timestamp_tokens": timestamps,
        "processing_info_estimate_tokens": int(est), "processing_info_estimate_patches": int(est) * 4,
    }
    rec["ok"] = bool(patches <= budget and pv.shape[0] == patches)
    # Independent HF reference for the video patches (Qwen2VL video processor, no frame sampling).
    try:
        import torch
        from transformers import Qwen2VLVideoProcessor

        vp = Qwen2VLVideoProcessor.from_pretrained(model_dir)
        hv = vp(videos=[torch.from_numpy(frames).permute(0, 3, 1, 2)], do_sample_frames=False, return_tensors="pt")
        hpv = hv["pixel_values_videos"].float()
        rec["hf_video_grid_thw"] = hv["video_grid_thw"].tolist()
        rec["hf_vs_vllm_max_abs"] = (float((hpv - item_field(out, "video", 0, "pixel_values_videos").float())
                                           .abs().max()) if hpv.shape == pv.shape else None)
        rec["hf_vs_vllm_note"] = "served tensor (after vLLM cast) vs HF fp32"
    except Exception as e:  # noqa: BLE001
        rec["hf_video_ref_error"] = repr(e)[:400]
    return rec


def max_pixels_check(proc, model_dir: str) -> dict:
    from PIL import Image

    big = Image.new("RGB", (2048, 2048), (0, 128, 255))
    res = {"image": "2048x2048 solid colour", "target_max_pixels": 1048576}
    out = run_vllm(proc, IMG_PROMPT, {"image": [big]})
    res["default_embeds"] = int(out["mm_placeholders"]["image"][0].get_num_embeds())
    try:
        out = run_vllm(proc, IMG_PROMPT, {"image": [big]}, {"max_pixels": 1048576})
        res["mm_processor_kwargs_max_pixels_embeds"] = int(out["mm_placeholders"]["image"][0].get_num_embeds())
    except Exception as e:  # noqa: BLE001
        res["mm_processor_kwargs_max_pixels_error"] = repr(e)[:400]
    try:
        p2 = vllm_processor(model_dir, {"image_max_pixels": 1048576, "video_max_pixels": 1048576})
        out = run_vllm(p2, IMG_PROMPT, {"image": [big]})
        res["hf_overrides_processor_config_embeds"] = int(out["mm_placeholders"]["image"][0].get_num_embeds())
        res["hf_overrides_max_image_tokens_estimate"] = int(p2.info.get_max_image_tokens())
    except Exception as e:  # noqa: BLE001
        res["hf_overrides_error"] = repr(e)[:400]
    res["default_max_image_tokens_estimate"] = int(proc.info.get_max_image_tokens())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
    ap.add_argument("--assets", default=os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--patch-budget", type=int, default=4096)
    a = ap.parse_args()
    if not os.path.isdir(a.model_dir) or not os.path.isdir(a.assets):
        print("mm_prep: --model-dir / --assets not found", file=sys.stderr)
        return 2
    res: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_dir": a.model_dir,
                 "assets": a.assets}
    ok = True
    try:
        import vllm.model_executor.models.mimo_v2_omni as m

        res["import"] = {"ok": True, "file": m.__file__}
    except Exception as e:  # noqa: BLE001
        res["import"] = {"ok": False, "error": repr(e)[:400]}
        ok = False
    steps = []
    if ok:
        proc = vllm_processor(a.model_dir)
        res["processor_class"] = type(proc).__name__
        steps = [("parity", lambda: image_parity(proc, hf_image_processor(a.model_dir), a.assets, a.atol)),
                 ("video", lambda: video_check(proc, a.model_dir, a.assets, a.patch_budget)),
                 ("max_pixels", lambda: max_pixels_check(proc, a.model_dir))]
    for name, fn in steps:
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001
            res[name] = {"ok": False, "error": traceback.format_exc()[-1500:]}
        if name != "max_pixels":
            ok = ok and bool(res[name].get("ok"))
    res["ok"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
