# SPDX-License-Identifier: Apache-2.0
"""Tiny random MiMo-V2 end to end in vllm-neuron CPU mode (VLLM_NEURON_CPU_MODE=1, no NKI_SIMULATOR).

Every case runs ``tests/e2e_runner.py`` in its own process (fresh vLLM engine; environment passed
explicitly through the subprocess env dict) and compares the result with the HF reference of the same
tiny checkpoint (``tools/make_tiny_checkpoint.py``):

* logits: three-way ``assert_close_three_way`` (baseline HF FP32, expected HF BF16, actual vllm-neuron),
  default thresholds. HF runs teacher-forced on the tokens vllm-neuron generated, so a near-tie that
  flips one greedy token does not make the remaining steps incomparable.
* tokens: each generated token is the argmax of the logits returned with it, and equals the HF FP32
  argmax at every "clear" step (FP32 top1-top2 margin > 2x the HF BF16-vs-FP32 max error at that step).
* routing flips: isolated single-step error spikes that only this implementation shows (a near-tie in the
  random tiny router resolved differently under BF16 noise) are excluded from the three-way check, at
  most 1 per prompt and 3 per case (see ``_check_against_hf``).

Cases (tiny-model drill): TP=1/2/4 with MIMO_DECODE_MOE_IMPL=tkg (PyTorch experts on CPU), TP=2 with 4 sequences
of different lengths for tkg and cte, TP=2 torch, and an invalid value that must fail at start-up.
The 4-sequence tkg case also records the runner metadata (MIMO_DEBUG_METADATA=1, design V3/V4).
Long context: ``tp2_tkg_seg`` runs segmented prefill (max_model_len 2048, segment 512, prompts
of 1300 and 300 tokens in one batch, MIMO_ATTN_{PREFILL,DECODE}_KEY_BLOCK=512 so both blocked attention
paths run with several key blocks) and checks from the metadata that every prefill step holds exactly one
request and that cached_seq_len runs 0/512/1024 for the long prompt and 0 for the short one.
Logs: $MIMO_WORK/logs/e2e/<case>.log; summaries: $MIMO_WORK/results/e2e/<case>.json.
"""

import ast
import json
import os
import re
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_DIR = os.environ.get("MIMO_TINY_DIR", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "tiny/tiny3"))
LOG_DIR = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "logs/e2e")
RES_DIR = os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "results/e2e")
STEPS = 16
TIMEOUT_S = 3600
FLIP_FACTOR = 5.0      # a step error > 5x the median HF BF16 step error counts as a routing-flip spike
MAX_FLIP_STEPS = 3     # implementation-only spike steps allowed per case (and at most 1 per prompt)

needs_tiny = pytest.mark.skipif(not os.path.isfile(os.path.join(TINY_DIR, "model.safetensors.index.json")),
                                reason="tiny checkpoint not available (run tools/make_tiny_checkpoint.py)")
pytestmark = needs_tiny


def _prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 150000, (n,), generator=g).tolist()


# lengths > 128 exercise the SWA window and the sink; 4 lengths for the multi-sequence cases
PROMPTS_1 = [_prompt(20, 1), _prompt(300, 2)]
PROMPTS_4 = [_prompt(20, 11), _prompt(150, 12), _prompt(300, 13), _prompt(500, 14)]
# segmented prefill: 1300 tokens = segments at 0/512/1024 (last one padded); the 300-token prompt shares the
# batch, so a wrong cached_seq_len (runner takes the max over input_batch) would show up in either prefill
PROMPTS_SEG = [_prompt(1300, 21), _prompt(300, 22)]
SEG_KEY_BLOCK_ENV = {"MIMO_ATTN_PREFILL_KEY_BLOCK": "512", "MIMO_ATTN_DECODE_KEY_BLOCK": "512"}

CASES = {
    "tp1_tkg": dict(tp=1, impl="tkg", seqs=1, prompts=PROMPTS_1),
    "tp2_tkg": dict(tp=2, impl="tkg", seqs=1, prompts=PROMPTS_1),
    "tp4_tkg": dict(tp=4, impl="tkg", seqs=1, prompts=PROMPTS_1),
    # enforce_eager: real requests are the first forwards (no warmup), so the metadata dump shows them
    "tp2_tkg_b4": dict(tp=2, impl="tkg", seqs=4, prompts=PROMPTS_4, debug_metadata=True, eager=True),
    "tp2_cte_b4": dict(tp=2, impl="cte", seqs=4, prompts=PROMPTS_4),   # cte with 4 sequences of different lengths
    "tp2_torch": dict(tp=2, impl="torch", seqs=1, prompts=PROMPTS_1),
    "tp2_tkg_seg": dict(tp=2, impl="tkg", seqs=2, prompts=PROMPTS_SEG, debug_metadata=True, eager=True,
                        max_model_len=2048, kv_segment_size=512, env=SEG_KEY_BLOCK_ENV,
                        expect_cached=[[0, 512, 1024], [0]]),
}


def _run(case, tp, impl, seqs, prompts, debug_metadata=False, eager=False, tmp_path=None, max_model_len=1024,
         kv_segment_size=0, extra_env=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    out = str(tmp_path / f"{case}.pt")
    spec = dict(model=TINY_DIR, tp=tp, max_num_seqs=seqs, prompts=prompts, max_tokens=STEPS,
                enforce_eager=eager, out=out, max_model_len=max_model_len, kv_segment_size=kv_segment_size)
    spec_path = tmp_path / f"{case}.json"
    spec_path.write_text(json.dumps(spec))
    env = {k: v for k, v in os.environ.items()
           if k not in ("NKI_SIMULATOR", "MIMO_DECODE_MOE_IMPL", "MIMO_DEBUG_METADATA",
                        "MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_ATTN_DECODE_KEY_BLOCK")}
    env.update(VLLM_NEURON_CPU_MODE="1", MIMO_DECODE_MOE_IMPL=impl, PYTHONUNBUFFERED="1")
    env.update(extra_env or {})
    if debug_metadata:
        env["MIMO_DEBUG_METADATA"] = "1"
    log_path = os.path.join(LOG_DIR, f"{case}.log")
    with open(log_path, "w") as log:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tests/e2e_runner.py"), "--spec", str(spec_path)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
    text = open(log_path, errors="replace").read()
    return p.returncode, text, out


def _tail(text, n=60):
    return "\n".join(text.splitlines()[-n:])


# ── HF reference (teacher-forced on the generated tokens) ────────────────────
_HF = {}


def _hf_models():
    if not _HF:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            m = AutoModelForCausalLM.from_pretrained(TINY_DIR, trust_remote_code=True, dtype=dt,
                                                     attn_implementation="eager").eval()
            with safe_open(os.path.join(TINY_DIR, "model.safetensors"), "pt") as f:
                for i, layer in enumerate(m.model.layers):
                    gate = getattr(layer.mlp, "gate", None)
                    if gate is not None:
                        t = f.get_tensor(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
                        assert t.dtype == torch.float32
                        gate.e_score_correction_bias.data = t
            _HF[name] = m
    return _HF


@torch.inference_mode()
def _hf_logits(prompt, tokens):
    """Logits [STEPS, V] (fp32 view) predicting tokens[s] from prompt + tokens[:s], for both HF dtypes."""
    ids = torch.tensor([prompt + tokens[:-1]])
    p = len(prompt)
    res = {}
    for name, m in _hf_models().items():
        lg = m(ids, use_cache=False, return_dict=True).logits[0].float()
        res[name] = lg[p - 1:p - 1 + len(tokens)]
    return res


def _check_against_hf(case, result, prompts, hf_fn=None):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    per_prompt = []
    summary = {"case": case, "prompt_lengths": [len(p) for p in prompts], "prompts": []}
    for i, (prompt, toks, lg) in enumerate(zip(prompts, result["tokens"], result["logits"])):
        assert len(toks) == STEPS and lg.shape[0] == STEPS
        assert torch.isfinite(lg).all(), f"{case}[{i}]: non-finite logits"
        # token = argmax of the logits returned with it (ties allowed)
        own = lg.gather(1, torch.tensor(toks).unsqueeze(1)).squeeze(1)
        assert torch.equal(own, lg.max(-1).values), f"{case}[{i}]: sampled token is not the argmax of its logits"
        hf = (hf_fn or _hf_logits)(prompt, toks)
        f32, b16 = hf["fp32"], hf["bf16"]
        top2 = f32.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        dtype_err = (b16 - f32).abs().max(-1).values
        clear = margin > 2 * dtype_err
        hf_top1 = f32.argmax(-1)
        agree = hf_top1 == torch.tensor(toks)
        bad = (clear & ~agree).nonzero().flatten().tolist()
        step_ours = ((lg - f32).norm(dim=-1) / f32.norm(dim=-1))
        step_hf = ((b16 - f32).norm(dim=-1) / f32.norm(dim=-1))
        summary["prompts"].append({
            "prompt_tokens": len(prompt), "tokens": toks,
            "top1_agreement_vs_hf_fp32": agree.float().mean().item(),
            "clear_steps": int(clear.sum()), "clear_step_mismatches": bad,
            "relL2_vs_fp32": ((lg - f32).norm() / f32.norm()).item(),
            "relL2_hf_bf16_vs_fp32": ((b16 - f32).norm() / f32.norm()).item(),
            "step_relL2_vs_fp32": [round(x, 5) for x in step_ours.tolist()],
            "step_relL2_hf_bf16_vs_fp32": [round(x, 5) for x in step_hf.tolist()],
        })
        assert not bad, f"{case}[{i}]: token != HF FP32 argmax at clear steps {bad}"
        per_prompt.append((f32, b16, lg, step_ours, step_hf))

    # Routing flips: the tiny random router has near-ties, so BF16 noise occasionally flips one top-4
    # choice at a single step (HF BF16 itself does this, e.g. relL2 0.16-0.23 vs ~0.007 elsewhere).
    # A flip is an isolated step whose error is > FLIP_FACTOR x the median HF BF16 step error. Steps where
    # only this implementation flips are excluded from the three-way check, at most 1 per prompt and
    # MAX_FLIP_STEPS per case; implementation bugs show up as errors on many steps and still fail.
    noise = torch.cat([s_hf for *_, s_hf in per_prompt]).median().item()
    base, exp, act, flips = [], [], [], []
    for i, (f32, b16, lg, s_ours, s_hf) in enumerate(per_prompt):
        only_ours = ((s_ours > FLIP_FACTOR * noise) & (s_hf <= FLIP_FACTOR * noise)).nonzero().flatten().tolist()
        flips.append(only_ours)
        assert len(only_ours) <= 1, f"{case}[{i}]: implementation-only error spikes at steps {only_ours} " \
                                    f"(step relL2 {[round(x, 4) for x in s_ours.tolist()]})"
        keep = torch.tensor([s not in only_ours for s in range(STEPS)])
        base.append(f32[keep])
        exp.append(b16[keep])
        act.append(lg[keep])
    n_flips = sum(len(f) for f in flips)
    summary["routing_flip_handling"] = {"hf_bf16_median_step_relL2": noise, "flip_factor": FLIP_FACTOR,
                                        "implementation_only_flip_steps": flips, "excluded_steps": n_flips}
    assert n_flips <= MAX_FLIP_STEPS, f"{case}: {n_flips} implementation-only spike steps > {MAX_FLIP_STEPS}"
    r = assert_close_three_way(base, exp, act, name=case)
    summary["three_way"] = {"passed": bool(r.passed), "sigma_ratio": float(r.sigma_ratio), "bc": float(r.bc),
                            "worst_linf_ratio": float(r.linf_ratio), "worst_l2_ratio": float(r.l2_ratio)}
    print(r.summary())
    return summary


_META_RE = re.compile(r"MIMO_DEBUG_METADATA (prefill|decode) (\{.*\})\s*$")


def _metadata(text):
    """Parsed MIMO_DEBUG_METADATA dumps (one per rank per call) -> list of (phase, dict | raw str)."""
    out = []
    for line in text.splitlines():
        m = _META_RE.search(line)
        if m:
            try:
                out.append((m.group(1), ast.literal_eval(m.group(2))))
            except (ValueError, SyntaxError):
                out.append((m.group(1), m.group(2)))
    return out


def _save(case, summary):
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f"{case}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


@pytest.mark.parametrize("case", list(CASES))
def test_tiny_e2e(case, tmp_path):
    c = CASES[case]
    rc, text, out = _run(case, c["tp"], c["impl"], c["seqs"], c["prompts"], c.get("debug_metadata", False),
                         c.get("eager", False), tmp_path, c.get("max_model_len", 1024), c.get("kv_segment_size", 0),
                         c.get("env"))
    assert rc == 0, f"{case}: runner exit code {rc}\n{_tail(text)}"
    result = torch.load(out)
    # V1: the architecture resolves to this plugin, not to vLLM's own (GPU) MiMoV2 implementation
    assert result["resolved_class"].startswith("mimo_v2_neuron."), result["resolved_class"]
    summary = _check_against_hf(case, result, c["prompts"])
    summary.update(resolved_class=result["resolved_class"], elapsed_s=result["elapsed_s"], tp=c["tp"],
                   decode_moe_impl=c["impl"], max_num_seqs=c["seqs"],
                   impl_log=[l.strip() for l in text.splitlines() if "decode MoE implementation" in l][:2])
    assert any(f"decode MoE implementation = {c['impl']}" in l for l in summary["impl_log"]), \
        f"{case}: decode MoE implementation log line not found"
    if c.get("debug_metadata"):
        meta = _metadata(text)
        assert meta, f"{case}: no MIMO_DEBUG_METADATA lines in the log"
        summary["metadata"] = meta
        pre = [d for ph, d in meta if ph == "prefill" and isinstance(d, dict)]
        dec = [d for ph, d in meta if ph == "decode" and isinstance(d, dict)]
        assert pre and dec, f"{case}: need parsed prefill and decode metadata, got {len(pre)} / {len(dec)}"
        swa = "layers.1.self_attn"
        if c.get("kv_segment_size"):
            summary["segmented_prefill"] = _check_segments(case, pre, c)
        v3 = []
        for d in pre:
            g = d["layers.0.self_attn"]
            v3.append({"kv_segment_size": g.get("kv_segment_size"), "cached_seq_len": g.get("cached_seq_len"),
                       "block_table": g.get("block_table_tensor")})
            if c.get("kv_segment_size"):
                continue
            # V3: single-shot prefill of one request with nothing cached
            assert not g.get("kv_segment_size"), g
            csl = g.get("cached_seq_len")
            if isinstance(csl, dict) and "values" in csl:
                assert all(v == 0 for v in torch.tensor(csl["values"]).flatten().tolist()), csl
        v4 = []
        for d in dec:
            s = d[swa]
            v4.append({"block_table": s.get("block_table_tensor"), "swa_kv_pos_offset": s.get("swa_kv_pos_offset"),
                       "ga_block_table": d["layers.0.self_attn"].get("block_table_tensor")})
            # V4: the runner always provides swa_kv_pos_offset for SWA decode (attention uses it)
            assert "swa_kv_pos_offset" in s, s.keys()
        summary["V3_prefill"] = v3
        summary["V4_decode"] = v4
    _save(case, summary)


def _check_segments(case, pre, c):
    """Segmented prefill: one request per prefill step, segment size as configured, cached_seq_len per prompt.

    Every rank logs every call, so each expected call appears ``tp`` times. A prefill step belongs to the
    prompt whose block-table row it shows (the GA row of a request is fixed for its lifetime).
    """
    from collections import Counter

    seg, tp = c["kv_segment_size"], c["tp"]
    calls = []
    for d in pre:
        g = d["layers.0.self_attn"]
        bt = g["block_table_tensor"]
        assert bt["shape"][0] == 1, f"{case}: prefill step with {bt['shape'][0]} block-table rows (expected 1)"
        assert g.get("kv_segment_size") == seg, f"{case}: kv_segment_size {g.get('kv_segment_size')} != {seg}"
        assert list(d["positions"]["shape"]) == [seg], f"{case}: prefill positions {d['positions']['shape']} != [{seg}]"
        csl = g["cached_seq_len"]
        assert list(csl["shape"]) == [1, 1], csl
        first_block = bt["values"][0][0]
        calls.append((first_block, csl["values"][0][0]))
    per_req = {}
    for blk, v in calls:
        per_req.setdefault(blk, []).append(v)
    got = sorted(tuple(sorted(Counter(v).items())) for v in per_req.values())
    want = sorted(tuple(sorted(Counter({x: tp for x in e}).items())) for e in c["expect_cached"])
    assert got == want, f"{case}: cached_seq_len per request {per_req} != expected {c['expect_cached']} x tp={tp}"
    return {"prefill_calls": [{"ga_first_block": b, "cached_seq_len": v} for b, v in calls],
            "per_request_cached_seq_len": {str(k): v for k, v in per_req.items()}}


def test_tiny_e2e_invalid_impl(tmp_path):
    rc, text, _ = _run("tp1_bad_impl", 1, "foo", 1, PROMPTS_1[:1], tmp_path=tmp_path)
    assert rc != 0, "MIMO_DECODE_MOE_IMPL=foo must fail at start-up"
    assert "MIMO_DECODE_MOE_IMPL must be one of tkg/cte/torch" in text, _tail(text)
    _save("tp1_bad_impl", {"returncode": rc,
                           "error_lines": [l for l in text.splitlines() if "MIMO_DECODE_MOE_IMPL must be" in l][:3]})
