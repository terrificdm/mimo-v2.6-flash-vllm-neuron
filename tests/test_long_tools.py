# SPDX-License-Identifier: Apache-2.0
"""Long-context tools (HBM report parser, needle prompt builder / judge, probe helpers).

CPU only, no model and no device. The HBM sample is an NCC_EOOM002 log of the BF16 4096 decode compile
(a full neuronx-cc log of a failed decode-graph compile, not shipped; a trimmed copy made by the tool itself is tests/data/
neuroncc_eoom_sample.log, which is shipped)."""

import importlib.util
import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_SAMPLE = os.path.join(ROOT, "logs", "instance", "logit_b1_try1_neuroncc_decode.log")
TRIM_SAMPLE = os.path.join(ROOT, "tests", "data", "neuroncc_eoom_sample.log")

from tools import neuroncc_hbm_summary as H  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


N = _load(os.path.join(ROOT, "scripts", "api_tests", "needle.py"), "_needle")

SUCCESS_LOG = """2026-10-19T11:11:52Z INFO 1 [root]: neuronx-cc compile graph.hlo
2026-10-19T11:12:16Z INFO 1663572 (nc00/sgLnk) [HBMUsage]: 
┌─────────────────────┬──────────┐
│ DRAM Memory Usage   │ Size     │
├─────────────────────┼──────────┤
│ Total:              │ 9.59GB   │
│ Model Code          │ 0B       │
│ Model Constants     │ 328.00KB │
│ Unallocated Tensors │ 9.39GB   │
│ Allocated Tensors   │ 120.12MB │
│ DMA Ring IO         │ 12.53MB  │
│ DMA Ring Spill      │ 533.56KB │
└─────────────────────┴──────────┘

2026-10-19T11:12:16Z INFO 1663572 (nc01/sgLnk) [HBMUsage]: 
┌─────────────────────┬──────────┐
│ DRAM Memory Usage   │ Size     │
├─────────────────────┼──────────┤
│ Total:              │ 9.46GB   │
│ Unallocated Tensors │ 9.39GB   │
│ Allocated Tensors   │ 60.05MB  │
└─────────────────────┴──────────┘

2026-10-19T11:12:16Z INFO 1663572 [HBMUsage]: Total estimated HBM usage is: 9.59GB
2026-10-19T11:12:17Z INFO 1663572 [BackendPassManager]: done
"""


# ------------------------------------------------------------------ HBM summary
@pytest.mark.parametrize("path", [FULL_SAMPLE, TRIM_SAMPLE])
def test_hbm_parse_eoom_sample(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present (the full compiler log is not shipped)")
    rec, keep = H.parse_log(open(path, errors="replace").read().splitlines())
    assert rec["eoom"] and rec["eoom_code"] == "NCC_EOOM002"
    assert rec["io_gib"] == pytest.approx(14.12, abs=1e-6)
    assert rec["internal_gib"] == pytest.approx(26.50, abs=1e-6)
    assert rec["unallocated_gib"] == pytest.approx(7.06, abs=1e-6)
    assert rec["peak_gib"] == pytest.approx(33.56, abs=1e-6)
    assert rec["limit_gib"] == pytest.approx(24.0)
    assert rec["allocated_gib_per_core"] == {"nc00": pytest.approx(19.87), "nc01": pytest.approx(6.62)}
    # scratch = max(sum Allocated 26.49, peak - I/O 19.44)
    assert rec["scratch_gib"] == pytest.approx(26.49, abs=1e-6)
    # the compiler's EOOM I/O figure is the sum of the per physical core Unallocated tensors
    assert rec["sum_unallocated_gib"] == pytest.approx(rec["io_gib"], abs=1e-6)
    assert keep


def test_hbm_parse_success_log_and_units():
    rec, _ = H.parse_log(SUCCESS_LOG.splitlines())
    assert not rec["eoom"]
    assert rec["peak_gib"] == pytest.approx(9.59)
    assert rec["io_gib"] == pytest.approx(18.78)          # 2 x 9.39
    assert rec["allocated_gib_per_core"]["nc00"] == pytest.approx(120.12 / 1024, abs=1e-4)
    assert rec["scratch_gib"] == pytest.approx((120.12 + 60.05) / 1024, abs=1e-4)
    assert rec["compile_s"] == 25.0
    assert H.to_gib("512", "MB") == 0.5 and H.to_gib("0", "B") == 0.0


def test_hbm_summarise_window_and_trimmed_copy(tmp_path):
    cache, logs = tmp_path / "cache", tmp_path / "logs"
    inside, outside = cache / "aaaa", cache / "bbbb" / "dev0_63.rank0"
    for d in (inside, outside):
        d.mkdir(parents=True)
        (d / "log-neuron-cc.txt").write_text(SUCCESS_LOG)
    (inside / "graph_aaaa.neff").write_bytes(b"x" * 10)
    (inside / ".compilation_complete").write_text("")
    t_in, t_out = time.time() - 100, time.time() - 10000
    os.utime(inside / "log-neuron-cc.txt", (t_in, t_in))
    os.utime(outside / "log-neuron-cc.txt", (t_out, t_out))
    doc = H.summarise(str(cache), t_in - 50, t_in + 50, "t", str(logs))
    assert doc["n_neff"] == 1 and doc["neffs"][0]["hash"] == "aaaa"
    assert doc["neffs"][0]["neff_bytes"] == 10 and doc["neffs"][0]["compilation_complete"]
    trimmed = logs / "t_neuroncc_aaaa.log"
    assert trimmed.exists() and "Total estimated HBM usage" in trimmed.read_text()
    # --hash picks a directory outside the window (plan alternative: locate by NEFF hash)
    doc2 = H.summarise(str(cache), None, None, "t", "", hashes=["bbbb"])
    assert [r["hash"] for r in doc2["neffs"]] == ["bbbb"]


def test_hbm_cli_errors(tmp_path):
    with pytest.raises(SystemExit) as e:
        H.main(["--cache", str(tmp_path), "--task", "x", "--out", str(tmp_path / "o.json")])
    assert e.value.code == 2
    rc = H.main(["--cache", str(tmp_path), "--since", "2026-10-19T00:00:00Z", "--until", "2026-10-19T00:01:00Z",
                 "--task", "x", "--out", str(tmp_path / "o.json"), "--log-dir", ""])
    assert rc == 1 and json.load(open(tmp_path / "o.json"))["n_neff"] == 0


# ------------------------------------------------------------------ needle
def fake_count(s):
    """~1 token per CJK character or ASCII word."""
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff") + len([w for w in s.split() if w.isascii()])


FAKE_COUNTER = (fake_count, lambda s: [ord(c) for c in s[:50]], lambda c: fake_count(c) + 7)


def test_needle_haystack_deterministic_and_bounded():
    a = N.build_content(fake_count, 3000, 0.5, "N1", seed=0)
    b = N.build_content(fake_count, 3000, 0.5, "N1", seed=0)
    assert a == b
    text, n, k, ns = a
    assert n <= 3000 and n > 2900
    assert text.endswith(N.NEEDLES["N1"]["question"])
    lines = text.split("\n")
    assert lines[k] == N.NEEDLES["N1"]["text"] and k == round(0.5 * ns)
    assert all(ln in N.SENTENCES for i, ln in enumerate(lines[:ns + 1]) if i != k)
    c = N.build_content(fake_count, 3000, 0.5, "N1", seed=1)
    assert c[0] != text
    d = N.build_content(fake_count, 3000, 0.1, "N1", seed=0)
    assert d[2] == round(0.1 * d[3])


def test_needle_haystack_has_no_answers():
    for s in N.SENTENCES:
        for nd in N.NEEDLES.values():
            assert N.normalise(nd["answer"]) not in N.normalise(s)


def test_needle_prompts_and_depth_assignment():
    ps = N.build_prompts(FAKE_COUNTER, 8192, [3000], [0.1, 0.5, 0.9])
    assert [p["needle"] for p in ps] == ["N1", "N2", "N3"]
    assert [p["id"] for p in ps] == ["N1@3000@10", "N2@3000@50", "N3@3000@90"]
    assert ps[0]["chat_prompt_tokens"] == ps[0]["content_tokens"] + 7
    assert ps == N.build_prompts(FAKE_COUNTER, 8192, [3000], [0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        N.build_prompts(FAKE_COUNTER, 3050, [3000], [0.5])


@pytest.mark.parametrize("content,answer,ok", [
    ("7492-ALPHA", "7492-ALPHA", True),
    ("通行码是 ７４９２－ＡＬＰＨＡ。", "7492-ALPHA", True),
    ("7492 – alpha", "7492-ALPHA", True),
    ("biscuit-42", "Biscuit-42", True),
    ("第 18 号码头", "18", True),
    ("7492-BETA", "7492-ALPHA", False),
    ("", "18", False),
    (None, "18", False),
])
def test_needle_answer_normalisation(content, answer, ok):
    assert N.is_correct(content, answer) is ok


def _case(kind, length, depth, correct):
    return {"kind": kind, "correct": correct, "records": [{"prompt": {"length": length, "depth": depth}}]}


def test_needle_tier_verdict():
    assert N.tier_verdict([_case("new", 1, 0.1, True), _case("reuse", 2, 0.5, True)]) == "pass"
    assert N.tier_verdict([_case("new", 1, 0.1, False), _case("backup", 1, 0.1, True)]) == "pass"
    assert N.tier_verdict([_case("new", 1, 0.1, False), _case("backup", 1, 0.1, False),
                           _case("reuse", 2, 0.5, True)]) == "model_limited"
    assert N.tier_verdict([_case("new", 1, 0.1, True), _case("reuse", 2, 0.5, False)]) == "impl_error"


# ------------------------------------------------------------------ probe helpers (import needs torch)
def test_probe_fp8_classify_and_conclusions():
    pytest.importorskip("torch")
    P = _load(os.path.join(ROOT, "tools", "probe_fp8.py"), "_probe_fp8")
    assert P.classify(float("nan")) == "nan" and P.classify(240.0) == "saturate_240"
    assert P.classify(-448.0) == "keep_448" and P.classify(float("inf")) == "inf"
    assert set(P.VARIANTS) == {"a_default", "b_runner_fp8", "c_neuron_cc_flags"}
    assert P.UNSAFE in P.compiler_args(P.VARIANTS["b_runner_fp8"]["hlo"])[3]
    assert P.UNSAFE not in P.compiler_args(P.VARIANTS["a_default"]["hlo"])[3]
    dq = {"ok": True, "device_bf16": [float("nan"), 240.0], "host_value": [448.0, 240.0], "matches_cpu": [False, True]}
    c = P.conclusions({"item2_cast": {"variants": {"b_runner_fp8": {"ok": True, "value_448": "saturate_240",
                                                                    "fp8_bits": [0x78, 0, 0x78, 0, 0, 0x77],
                                                                    "host_fp8_to_bf16": dq}}},
                       "item4_mx": {"isa_docs": {"nc_matmul_mx": {"requires_neuroncore_v4": True}}}})
    assert c["checkpoint_fp8_direct_use"].startswith("no") and "not available" in c["mxfp4_matmul_on_trn2"]
    assert c["device_fp8_encoding"].startswith("E4M3 ") and c["host_e4m3fn_dequant_le240_match_cpu"] is True
    assert P.conclusions({})["checkpoint_fp8_direct_use"].startswith("unknown")


def test_l3_compile_probe_args(monkeypatch):
    pytest.importorskip("torch")
    P = _load(os.path.join(ROOT, "tools", "l3_compile_probe.py"), "_l3_probe")
    a = P.parse_args(["--max-model-len", "32768", "--kv-segment-size", "4096", "--num-gpu-blocks-override", "8192",
                      "--decode-key-block", "4096", "--out", "/tmp/x.json"])
    assert a.max_num_seqs == 4 and a.prefill_key_block is None
    for k in ("MIMO_ATTN_DECODE_KEY_BLOCK", "MIMO_ATTN_PREFILL_KEY_BLOCK", "MIMO_DECODE_MOE_IMPL"):
        monkeypatch.delenv(k, raising=False)   # restored after the test (apply_env writes os.environ)
    env = P.apply_env(a)
    assert env["MIMO_ATTN_DECODE_KEY_BLOCK"] == "4096" and env["MIMO_DECODE_MOE_IMPL"] == "tkg"
    for bad in (["--kv-segment-size", "4096", "--max-model-len", "4096"], ["--decode-key-block", "100"]):
        args = ["--max-model-len", "32768", "--num-gpu-blocks-override", "1", "--out", "x"] + bad
        with pytest.raises(SystemExit):
            P.parse_args(args)
