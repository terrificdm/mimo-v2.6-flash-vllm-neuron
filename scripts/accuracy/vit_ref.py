# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the MiMo ViT reference and tests (docs/design-notes.md#vit-accuracy).

  hf_vit(src, dtype)          HF ``MiMoVisionTransformer`` (patched HF remote code, ``tests/hf_ref.py``) with the
                              ``visual.*`` weights of the original checkpoint; merger / ln_q biases forced to 0
                              after asserting the checkpoint has none
  hf_run(vit, pv, grid)       forward, optionally recording every block's input and output
  served_inputs(src, assets)  the tensors vLLM actually hands to the model (vLLM MiMo-Omni processor, BF16 cast):
                              4 test images + the test video decoded by VideoMediaIO (as served)
  our_vit(src, tp_group, rank) the plugin ``MiMoVisionModel`` for one TP rank, weights through its loaders
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "plugin"), os.path.join(ROOT, "scripts", "accuracy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MERGER_BIASES = ("merger.mlp.0.bias", "merger.mlp.2.bias", "merger.ln_q.bias")


def vision_config(src: str) -> dict:
    with open(os.path.join(src, "config.json")) as f:
        vc = json.load(f).get("vision_config") or {}
    if not vc:
        raise ValueError(f"{src}/config.json has no vision_config")
    return vc


class VisualCkpt:
    """``visual.*`` tensors of a checkpoint directory (lazy safetensors handles)."""

    def __init__(self, src: str):
        from safetensors import safe_open

        with open(os.path.join(src, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        self.src = src
        self.wm = {k: v for k, v in wm.items() if k.startswith("visual.")}
        if not self.wm:
            raise ValueError(f"{src}: no visual.* tensors in model.safetensors.index.json")
        self._open, self._files = safe_open, {}

    def _f(self, key):
        fn = self.wm[key]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.src, fn), "pt")
        return self._files[fn]

    def keys(self):
        return sorted(self.wm)

    def slice(self, key):
        return self._f(key).get_slice(key)

    def tensor(self, key):
        return self._f(key).get_tensor(key)


def hf_vit(src: str, dtype: torch.dtype, ckpt: VisualCkpt | None = None):
    from tests.hf_ref import load as load_hf

    mdl, _ = load_hf()
    ckpt = ckpt or VisualCkpt(src)
    present = [k for k in ckpt.keys() if k.endswith(MERGER_BIASES)]
    if present:
        raise ValueError(f"checkpoint unexpectedly has merger biases {present} (assumption of the plugin and the reference broken)")
    vit = mdl.MiMoVisionTransformer(SimpleNamespace(**vision_config(src)))
    sd = {k[len("visual."):]: ckpt.tensor(k) for k in ckpt.keys()}
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    if unexpected or sorted(missing) != sorted(MERGER_BIASES):
        raise ValueError(f"HF ViT state dict mismatch: missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        for name in MERGER_BIASES:
            mod = vit.get_submodule(name.rsplit(".", 1)[0])
            mod.bias.zero_()
    vit = vit.to(dtype).eval()
    # .to(dtype) would also cast the non-persistent inv_freq buffer; from_pretrained(torch_dtype=bf16) keeps it FP32
    # (created with an explicit dtype=torch.float), so restore it to match a real HF BF16 load.
    rot = vit.rotary_pos_emb
    rot.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot.inv_freq.numel() * 2, 2, dtype=torch.float)
                                      / (rot.inv_freq.numel() * 2)))
    return vit


@torch.no_grad()
def hf_run(vit, pixel_values: torch.Tensor, grid_thw: torch.Tensor, record: bool = False):
    """-> (merged [tokens, out_hidden] fp32, block_inputs list, block_outputs list) (lists empty unless record)."""
    ins, outs, hooks = [], [], []
    if record:
        for blk in vit.blocks:
            hooks.append(blk.register_forward_pre_hook(lambda m, a, kw: ins.append(a[0].detach().clone()),
                                                       with_kwargs=True))
            hooks.append(blk.register_forward_hook(lambda m, a, o: outs.append(o.detach().clone())))
    try:
        y = vit(pixel_values, grid_thw)
    finally:
        for h in hooks:
            h.remove()
    return y.float(), ins, outs


def served_inputs(src: str, assets: str) -> list:
    """[{name, kind, pixel_values (BF16, as served), grid_thw [n, 3] int64}] for the 4 images and the video."""
    import mm_prep
    from PIL import Image

    proc = mm_prep.vllm_processor(src)
    items = []
    for name in mm_prep.IMAGES:
        img = Image.open(os.path.join(assets, name)).convert("RGB")
        out = mm_prep.run_vllm(proc, mm_prep.IMG_PROMPT, {"image": [img]})
        items.append({"name": name, "kind": "image",
                      "pixel_values": mm_prep.item_field(out, "image", 0, "pixel_values"),
                      "grid_thw": mm_prep.item_field(out, "image", 0, "image_grid_thw").reshape(1, 3).long()})
    frames, meta = mm_prep.load_video(os.path.join(assets, mm_prep.VIDEO))
    out = mm_prep.run_vllm(proc, mm_prep.VID_PROMPT, {"video": [(frames, meta)]})
    items.append({"name": mm_prep.VIDEO, "kind": "video",
                  "pixel_values": mm_prep.item_field(out, "video", 0, "pixel_values_videos"),
                  "grid_thw": mm_prep.item_field(out, "video", 0, "video_grid_thw").reshape(1, 3).long()})
    return items


def our_vit(src: str, tp_group, ckpt: VisualCkpt | None = None):
    """Plugin ``MiMoVisionModel`` for ``tp_group.rank_in_group`` (no DP group), weights via the plugin loaders."""
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import MiMoVisionConfig, MiMoVisionModel
    from vllm_neuron.utils.weight_loader import get_weight_loader

    ckpt = ckpt or VisualCkpt(src)
    cfg = MiMoVisionConfig.from_hf(vision_config(src))
    m = MiMoVisionModel(cfg, torch.bfloat16, tp_group=tp_group, dp_group=None, use_dp_group=False)
    MiMoVisionModel.check_no_merger_bias(ckpt.keys())
    params = dict(m.named_parameters())
    mapping = m.mappings()
    if set(mapping) != set(params):
        raise ValueError(f"unmapped params: {sorted(set(params) - set(mapping))[:5]}")
    for pname, key in mapping.items():
        t = get_weight_loader(params[pname]).load([ckpt.slice(key)], tp_group.rank_in_group)
        if tuple(t.shape) != tuple(params[pname].shape):
            raise ValueError(f"{pname}: loaded {tuple(t.shape)} != param {tuple(params[pname].shape)}")
        params[pname].data = t.to(params[pname].dtype)
    return m.eval(), cfg
