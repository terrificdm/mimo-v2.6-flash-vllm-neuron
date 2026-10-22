# SPDX-License-Identifier: Apache-2.0
"""HF Qwen2VLImageProcessor vs the vLLM MiMo-Omni processor (docs/design-notes.md#mm-reference-inputs).

Needs the original model dir (config with vision_config, preprocessor_config.json, tokenizer) and the assets from
scripts/api_tests/make_mm_assets.py; skipped when either is missing (e.g. on the Mac).
Override paths with MIMO_MM_MODEL_DIR / MIMO_MM_ASSETS.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "accuracy"))

MODEL_DIR = os.environ.get("MIMO_MM_MODEL_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
ASSETS = os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(MODEL_DIR, "preprocessor_config.json"))
         and os.path.isfile(os.path.join(ASSETS, "red_square.png"))),
    reason="model dir or mm assets not present",
)


@pytest.fixture(scope="module")
def proc():
    import mm_prep

    return mm_prep.vllm_processor(MODEL_DIR)


def test_image_parity(proc):
    import mm_prep

    res = mm_prep.image_parity(proc, mm_prep.hf_image_processor(MODEL_DIR), ASSETS, 1e-5)
    for name, rec in res["items"].items():
        assert rec["hf_grid_thw"] == rec["vllm_grid_thw"] == [rec["served_grid_thw"]], name
        assert rec["hf_vs_vllm_max_abs"] is not None and rec["hf_vs_vllm_max_abs"] <= 1e-5, (name, rec)
        assert rec["served_equals_raw_cast"], (name, rec)
        assert rec["num_embeds"] == 196, (name, rec)  # 448x448 -> 28x28 patches -> 196 merged tokens


def test_video_budget(proc):
    import mm_prep

    rec = mm_prep.video_check(proc, MODEL_DIR, ASSETS, 4096)
    assert rec["video_grid_thw"] == [4, 24, 24], rec
    assert rec["patches"] == 2304 <= 4096, rec
    assert rec["num_embeds"] == 576, rec
