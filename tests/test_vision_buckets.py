# SPDX-License-Identifier: Apache-2.0
"""Larger vision buckets (docs/design-notes.md#vision-request-blocks; VLLM_NEURON_CPU_MODE=1, no checkpoint needed).

  * MIMO_MM_VISION_BUCKETS parsing; unset = the default single bucket (no override, no check);
  * block plan of a 3-minute 720p video (vLLM default 32 frames -> grid 16 x 44 x 80): 16 blocks, 14080 tokens;
  * per-request block check (API process): 16 blocks pass, 17 / two images + a 16-block video are rejected, items
    returned as None by the sender cache use the remembered count or the embeds lower bound;
  * the video budget override (>= max bucket / 4) and the worker consistency check;
  * embed_multimodal splits an encoder step whose items need more blocks than the largest vision graph.
"""

from types import SimpleNamespace

import pytest
import torch

from mimo_v2_neuron.model.mimo_v2 import factory_mm as F
from mimo_v2_neuron.model.mimo_v2.model_mm import MiMoV2OmniForCausalLMNeuron, vision_block_plan
from mimo_v2_neuron.model.mimo_v2.vision_encoder import MiMoVisionConfig

GRID_720P = (16, 44, 80)    # 32 frames / temporal patch 2; 704 x 1280 after smart_resize, 16 px patches


def test_buckets_env(monkeypatch):
    monkeypatch.delenv("MIMO_MM_VISION_BUCKETS", raising=False)
    assert F.vision_buckets_env() is None
    monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", "4096,65536")
    assert F.vision_buckets_env() == [4096, 65536] and F.max_request_blocks([4096, 65536]) == 16
    assert F.max_request_blocks([4096]) == 1 and F.max_request_blocks([4096, 32768]) == 8
    for bad in ("4096,6000", "65536,4096", "a", "4096,4096"):
        monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", bad)
        with pytest.raises(ValueError):
            F.vision_buckets_env()


def test_720p_video_plan():
    plan = vision_block_plan([GRID_720P], 4096, 4)
    assert len(plan[0]) == 16 and sum(plan[0]) == 14080 and set(plan[0]) == {880}
    assert F.item_blocks("video", torch.tensor([GRID_720P])) == 16
    assert F.item_blocks("image", torch.tensor([1, 64, 64])) == 1      # 1024 x 1024 image = one full block
    assert F.item_blocks("audio", None) == 1
    with pytest.raises(ValueError):                                     # one group of 4100 patches
        F.item_blocks("image", torch.tensor([1, 2, 2050]))


def test_request_check_limits():
    F.check_request_blocks([("video", 16)], 16)
    F.check_request_blocks([("image", 1), ("image", 1), ("audio", 1)], 16)
    with pytest.raises(ValueError, match="17"):
        F.check_request_blocks([("video", 17)], 16)
    with pytest.raises(ValueError, match="18"):
        F.check_request_blocks([("image", 1), ("image", 1), ("video", 16)], 16)


class _Item:
    def __init__(self, **data):
        self.data = {k: torch.tensor(v) for k, v in data.items()}

    def get_data(self):
        return self.data


class _Pos:
    def __init__(self, n):
        self.n = n

    def get_num_embeds(self):
        return self.n


def _processor():
    return F.MiMoV2NeuronMultiModalProcessor.__new__(F.MiMoV2NeuronMultiModalProcessor)


def test_request_blocks_and_cache_items():
    p = _processor()
    out = {"mm_kwargs": {"image": [_Item(image_grid_thw=[1, 64, 64]), None],
                         "video": [_Item(video_grid_thw=list(GRID_720P))], "audio": [None]},
           "mm_hashes": {"image": ["i0", "i1"], "video": ["v0"], "audio": ["a0"]},
           "mm_placeholders": {"image": [_Pos(1024), _Pos(1500)], "video": [_Pos(14080)], "audio": [_Pos(800)]}}
    # i1 not seen before: lower bound ceil(1500 / 1024) = 2
    assert p._request_blocks(out) == [("image", 1), ("image", 2), ("video", 16), ("audio", 1)]
    again = {"mm_kwargs": {"video": [None]}, "mm_hashes": {"video": ["v0"]}, "mm_placeholders": {"video": [_Pos(1)]}}
    assert p._request_blocks(again) == [("video", 16)]      # remembered from the first request


def test_apply_checks_only_with_env(monkeypatch):
    big = {"mm_kwargs": {"video": [_Item(video_grid_thw=list(GRID_720P))],
                         "image": [_Item(image_grid_thw=[1, 64, 64])]},
           "mm_hashes": {"video": ["v"], "image": ["i"]}, "mm_placeholders": {}}
    monkeypatch.setattr(F.MiMoV2OmniMultiModalProcessor, "apply", lambda self, *a, **k: big)
    p = _processor()
    monkeypatch.delenv("MIMO_MM_VISION_BUCKETS", raising=False)
    assert p.apply() is big                                   # unset: no request check
    monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", "4096,65536")
    with pytest.raises(ValueError, match="17 vision/audio encoder blocks"):
        p.apply()
    big["mm_kwargs"].pop("image")
    assert p.apply() is big


def test_video_budget_override(monkeypatch):
    info = F.MiMoV2NeuronProcessingInfo.__new__(F.MiMoV2NeuronProcessingInfo)
    hf = SimpleNamespace(vision_config={"spatial_merge_size": 2, "patch_size": 16}, audio_config=None)
    monkeypatch.setattr(F.MiMoV2NeuronProcessingInfo, "get_hf_config", lambda self: hf)
    monkeypatch.setattr(F.MiMoV2OmniProcessingInfo, "get_mm_max_tokens_per_item",
                        lambda self, s, c: {"image": 8192, "video": 14336})
    monkeypatch.delenv("MIMO_MM_VISION_BUCKETS", raising=False)
    assert info.get_mm_max_tokens_per_item(1, {}) == {"image": 8192, "video": 14336}
    monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", "4096,65536")
    assert info.get_mm_max_tokens_per_item(1, {}) == {"image": 8192, "video": 16384}
    monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", "4096")
    assert info.get_mm_max_tokens_per_item(1, {})["video"] == 14336


def _vnc(buckets, bs=4096):
    return SimpleNamespace(tp_size=8, dp_size=8, vision_attention_block_size=bs, num_vision_tokens_buckets=buckets,
                           quantization=None)


def test_worker_consistency(monkeypatch):
    vcfg, cfg = MiMoVisionConfig(), SimpleNamespace(hidden_size=4096)
    monkeypatch.delenv("MIMO_MM_VISION_BUCKETS", raising=False)
    F.MiMoV2OmniForCausalLM._validate_vision(vcfg, _vnc([4096]), cfg)
    monkeypatch.setenv("MIMO_MM_VISION_BUCKETS", "4096,65536")
    F.MiMoV2OmniForCausalLM._validate_vision(vcfg, _vnc([4096, 65536]), cfg)
    with pytest.raises(ValueError, match="MIMO_MM_VISION_BUCKETS"):
        F.MiMoV2OmniForCausalLM._validate_vision(vcfg, _vnc([4096]), cfg)
    with pytest.raises(ValueError, match="block_size"):
        F.MiMoV2OmniForCausalLM._validate_vision(vcfg, _vnc([4096, 65536], bs=8192), cfg)


def test_embed_splits_encoder_step():
    calls = []
    grids = [GRID_720P, (1, 64, 64), GRID_720P]          # 16 + 1 + 16 blocks > 16 (largest graph)
    rows = [t * h * w for t, h, w in grids]
    pv = torch.arange(sum(rows), dtype=torch.float32).reshape(-1, 1)
    me = SimpleNamespace(vision_cfg=MiMoVisionConfig(), vision_neuron_config=_vnc([4096, 65536]),
                         embed_multimodal=lambda **k: calls.append(k))
    MiMoV2OmniForCausalLMNeuron.embed_multimodal(me, pixel_values_videos=pv, video_grid_thw=torch.tensor(grids),
                                                 encoder_cache=None, mm_hashes=["a", "b", "c"])
    assert [c["mm_hashes"] for c in calls] == [["a"], ["b"], ["c"]]
    starts = [0, rows[0], rows[0] + rows[1]]
    for c, s, r, g in zip(calls, starts, rows, grids):
        assert c["image_grid_thw"].tolist() == [list(g)]
        assert c["pixel_values"].shape[0] == r and float(c["pixel_values"][0, 0]) == s
    calls.clear()
    small = [(1, 64, 64)] * 3                               # 3 blocks fit one graph: no split (normal path)
    me.embed_multimodal = lambda **k: calls.append(k)
    with pytest.raises(AttributeError):                      # reaches the encoder (stub has no encoder cache)
        MiMoV2OmniForCausalLMNeuron.embed_multimodal(me, pixel_values=torch.zeros(3 * 4096, 1),
                                                     image_grid_thw=torch.tensor(small), encoder_cache=None,
                                                     mm_hashes=["x", "y", "z"])
    assert calls == []
