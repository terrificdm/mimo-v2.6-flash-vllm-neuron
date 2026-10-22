# SPDX-License-Identifier: Apache-2.0
"""Tiny random MiMo-V2 Omni (text + ViT) end to end in vllm-neuron CPU mode (docs/design-notes.md#omni-architecture).

Checkpoint: ``tools/make_tiny_checkpoint.py --with-vision --out $MIMO_WORK/tiny/tiny3_mm`` (the tiny3
text model + a 4-block ViT, ``architectures = ["MiMoV2OmniForCausalLM"]``). Each case runs
``tests/e2e_runner.py`` with two image prompts (red_square / three_circles from ``make_mm_assets.py``) through
``LLM.generate`` with ``multi_modal_data`` and compares the 16 greedy steps with HF (``MiMoV2ForCausalLM`` with
``visual``, teacher-forced on the generated tokens, pixel values from vLLM's own MiMo-Omni processor) using the
same checks as ``test_tiny_e2e.py`` (top-1 at clear steps + ``assert_close_three_way``). Because the image
embeddings enter the prefill, a wrong ViT, encoder-cache write or embedding merge fails the logit check.

Cases:
  tp2_mm_vtp2      text TP 2, vision TP 2 / DP 1, text prompts with the placeholder (processor expands it);
                   MIMO_DEBUG_METADATA=1 records the prefill positions (one-dimensional, 0..n-1; docs/design-notes.md#positions)
  tp2_mm_vdp2_ids  text TP 2, vision TP 1 / DP 2 (block split over DP ranks, MIMO_VIT_DP_GROUP=1), prompts given as
                   token ids with one image placeholder token + image (vLLM expands it; the path
                   ``run_mm_validation.py`` uses)
  tp2_mm_vdp2_nodpgroup  same, default MIMO_VIT_DP_GROUP=0 (each vision TP group encodes every block; docs/design-notes.md#vision-dp-group)
  tp2_mm_vdp2_world  same, MIMO_VIT_DP_GROUP=world (block split, one all-gather over the TP world)
Logs $MIMO_WORK/logs/e2e/<case>.log; summaries $MIMO_WORK/results/e2e/<case>.json.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from tests.test_tiny_e2e import LOG_DIR, ROOT, STEPS, TIMEOUT_S, _check_against_hf, _metadata, _save, _tail

TINY_MM_DIR = os.environ.get("MIMO_TINY_MM_DIR", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "tiny/tiny3_mm"))
ASSETS = os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
IMAGES = ["red_square.png", "three_circles.png"]
QUESTIONS = ["What color is the square?", "How many circles are there?"]
BLOCK = 1024

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(TINY_MM_DIR, "model.safetensors.index.json"))
         and all(os.path.isfile(os.path.join(ASSETS, i)) for i in IMAGES)),
    reason="tiny multimodal checkpoint or mm assets missing (make_tiny_checkpoint.py --with-vision, make_mm_assets.py)")


def _text_prompt(q):
    return (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n"
            f"<|im_start|>assistant\n")


CASES = {
    "tp2_mm_vtp2": dict(tp=2, vision=dict(block=BLOCK, tp=2, dp=1), ids=False, debug_metadata=True, eager=True),
    "tp2_mm_vdp2_ids": dict(tp=2, vision=dict(block=BLOCK, tp=1, dp=2), ids=True, env={"MIMO_VIT_DP_GROUP": "1"}),
    # MIMO_VIT_DP_GROUP=0 default: DP 2 block counts, but each vision TP group encodes every block (no DP all-gather)
    "tp2_mm_vdp2_nodpgroup": dict(tp=2, vision=dict(block=BLOCK, tp=1, dp=2), ids=True),
    # docs/design-notes.md#vision-dp-group: DP 2 split with one all-gather over the TP world (MIMO_VIT_DP_GROUP=world)
    "tp2_mm_vdp2_world": dict(tp=2, vision=dict(block=BLOCK, tp=1, dp=2), ids=True,
                              env={"MIMO_VIT_DP_GROUP": "world"}),
}

_HF = {}
_PROC = {}


def _processor():
    if "p" not in _PROC:
        sys.path.insert(0, os.path.join(ROOT, "scripts", "accuracy"))
        import mm_prep

        _PROC["p"] = (mm_prep, mm_prep.vllm_processor(TINY_MM_DIR))
    return _PROC["p"]


def _served_inputs(i):
    """(prompt_token_ids, pixel_values fp32, image_grid_thw) exactly as vLLM's processor builds them."""
    from PIL import Image

    mm_prep, proc = _processor()
    img = Image.open(os.path.join(ASSETS, IMAGES[i])).convert("RGB")
    out = mm_prep.run_vllm(proc, _text_prompt(QUESTIONS[i]), {"image": [img]})
    pv = mm_prep.item_field(out, "image", 0, "pixel_values")
    thw = mm_prep.item_field(out, "image", 0, "image_grid_thw")
    return list(out["prompt_token_ids"]), pv.float(), thw.reshape(1, 3)


def _hf_models():
    if not _HF:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        sys.path.insert(0, os.path.join(ROOT, "scripts", "accuracy"))
        import gen_mm_goldens

        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            m = AutoModelForCausalLM.from_pretrained(TINY_MM_DIR, trust_remote_code=True, dtype=dt,
                                                     attn_implementation="eager").eval()
            assert hasattr(m, "visual"), "HF tiny model has no visual tower (vision_config missing?)"
            with torch.no_grad():   # merger / ln_q biases are not in the checkpoint: zero
                for mod in (m.visual.merger.mlp[0], m.visual.merger.mlp[2], m.visual.merger.ln_q):
                    mod.bias.zero_()
            gen_mm_goldens.restore_vit_inv_freq(m)   # from_pretrained leaves it uninitialised (NaN logits)
            with safe_open(os.path.join(TINY_MM_DIR, "model.safetensors"), "pt") as f:
                for i, layer in enumerate(m.model.layers):
                    gate = getattr(layer.mlp, "gate", None)
                    if gate is not None:
                        gate.e_score_correction_bias.data = f.get_tensor(
                            f"model.layers.{i}.mlp.gate.e_score_correction_bias")
            _HF[name] = m
    return _HF


def _hf_fn_for(served):
    """hf_fn(prompt_ids, tokens) for _check_against_hf: teacher-forced HF logits with the prompt's image."""
    by_prompt = {tuple(ids): (pv, thw) for ids, pv, thw in served}

    @torch.inference_mode()
    def fn(prompt, tokens):
        pv, thw = by_prompt[tuple(prompt)]
        ids = torch.tensor([prompt + tokens[:-1]])
        p = len(prompt)
        res = {}
        for name, m in _hf_models().items():
            dt = torch.float32 if name == "fp32" else torch.bfloat16
            lg = m(input_ids=ids, pixel_values=pv.to(dt), image_grid_thw=thw, use_cache=False,
                   return_dict=True).logits[0].float()
            res[name] = lg[p - 1:p - 1 + len(tokens)]
            assert torch.isfinite(res[name]).all(), f"HF {name} reference logits are not finite"
        return res

    return fn


def collapse_image_tokens(ids, image_token=151655):
    """Expanded prompt ids -> one placeholder token per image run (vLLM re-expands token prompts)."""
    out = []
    for t in ids:
        if not (t == image_token and out and out[-1] == image_token):
            out.append(t)
    return out


def _run(case, c, tmp_path, served):
    os.makedirs(LOG_DIR, exist_ok=True)
    out = str(tmp_path / f"{case}.pt")
    mm = []
    for i, (ids, _, _) in enumerate(served):
        item = {"image": os.path.join(ASSETS, IMAGES[i])}
        item.update({"prompt_token_ids": collapse_image_tokens(ids)} if c["ids"]
                    else {"prompt": _text_prompt(QUESTIONS[i])})
        mm.append(item)
    spec = dict(model=TINY_MM_DIR, arch="MiMoV2OmniForCausalLM", tp=c["tp"], max_num_seqs=2, mm_prompts=mm,
                vision=c["vision"], max_tokens=STEPS, enforce_eager=c.get("eager", False), out=out,
                max_model_len=1024)
    spec_path = tmp_path / f"{case}.json"
    spec_path.write_text(json.dumps(spec))
    env = {k: v for k, v in os.environ.items()
           if k not in ("NKI_SIMULATOR", "MIMO_DECODE_MOE_IMPL", "MIMO_DEBUG_METADATA",
                        "MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK", "MIMO_VIT_DP_GROUP")}
    env.update(VLLM_NEURON_CPU_MODE="1", MIMO_DECODE_MOE_IMPL="tkg", PYTHONUNBUFFERED="1")
    if c.get("debug_metadata"):
        env["MIMO_DEBUG_METADATA"] = "1"
    env.update(c.get("env", {}))
    log_path = os.path.join(LOG_DIR, f"{case}.log")
    with open(log_path, "w") as log:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tests/e2e_runner.py"), "--spec", str(spec_path)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
    return p.returncode, open(log_path, errors="replace").read(), out


@pytest.mark.parametrize("case", list(CASES))
def test_tiny_mm_e2e(case, tmp_path):
    c = CASES[case]
    served = [_served_inputs(i) for i in range(len(IMAGES))]
    rc, text, out = _run(case, c, tmp_path, served)
    assert rc == 0, f"{case}: runner exit code {rc}\n{_tail(text)}"
    result = torch.load(out)
    assert result["resolved_class"] == "mimo_v2_neuron.model.mimo_v2.factory_mm.MiMoV2OmniForCausalLM", \
        result["resolved_class"]
    assert "Multimodal on-device encoder cache path enabled" in text, f"{case}: runner did not enable the mm path"
    # the engine saw the same expanded prompt as vLLM's processor (image placeholder -> 196 image tokens)
    for i, (ids, _, thw) in enumerate(served):
        assert result["prompt_token_ids"][i] == ids, f"{case}[{i}]: prompt ids differ from the processor output"
    image_token = 151655
    n_img = [ids.count(image_token) for ids, _, _ in served]
    assert all(n == int(thw.prod()) // 4 for n, (_, _, thw) in zip(n_img, served)), n_img
    summary = _check_against_hf(case, result, [s[0] for s in served], hf_fn=_hf_fn_for(served))
    # the two images give different continuations' logits (the image actually reaches the decoder)
    d = (result["logits"][0][0] - result["logits"][1][0]).abs().max().item()
    summary.update(resolved_class=result["resolved_class"], elapsed_s=result["elapsed_s"], tp=c["tp"],
                   vision=c["vision"], prompt_ids_mode=c["ids"], image_tokens=n_img,
                   step0_logit_maxabs_between_images=d)
    if c.get("debug_metadata"):
        meta = _metadata(text)
        pre = [dd for ph, dd in meta if ph == "prefill" and isinstance(dd, dict)]
        assert pre, f"{case}: no prefill metadata"
        pos = [dd["positions"] for dd in pre]
        summary["prefill_positions"] = pos
        for pp in pos:   # plain 1-D positions starting at 0 (no M-RoPE rows)
            assert len(pp["shape"]) == 1, pp
            head = pp.get("head") or pp.get("values", [])[:4]
            assert head[:4] == [0, 1, 2, 3], pp
    _save(case, summary)
