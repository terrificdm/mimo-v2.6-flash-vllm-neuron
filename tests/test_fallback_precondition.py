# SPDX-License-Identifier: Apache-2.0
"""User-approved extended fallback precondition (2026-10-19): Neuron-side routing near-tie."""

import importlib.util
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("_fb", os.path.join(ROOT, "scripts", "accuracy", "fallback_check.py"))
FB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FB)

P = 4
HF = [torch.arange(8).repeat(P + 16, 1), torch.arange(8, 16).repeat(P + 16, 1)]   # layers 1, 2
GOLD = list(range(100, 116))


def _np(routing, tokens=None):
    return {"neuron_tokens": tokens or GOLD, "fp32_golden_tokens": GOLD, "routing": routing}


def _same(q):
    return {"1": {"idx": list(range(8)), "gap89": 0.5}, "2": {"idx": list(range(8, 16)), "gap89": 0.5}}


def test_near_tie_found():
    r = {str(q): _same(q) for q in range(P - 1, P + 16)}
    r["5"] = {"1": {"idx": [0, 1, 2, 3, 4, 5, 6, 99], "gap89": 1.5e-4}, "2": _same(5)["2"]}
    assert FB.neuron_near_tie(P, 6, _np(r), HF, [1, 2]) == {"position": 5, "layer": 1, "gap89": 1.5e-4}
    assert FB.neuron_near_tie(P, 4, _np(r), HF, [1, 2]) is None          # flip after p


def test_wide_gap_or_same_set_not_covered():
    r = {str(q): _same(q) for q in range(P - 1, P + 16)}
    r["5"] = {"1": {"idx": [0, 1, 2, 3, 4, 5, 6, 99], "gap89": 2e-3}, "2": _same(5)["2"]}
    assert FB.neuron_near_tie(P, 10, _np(r), HF, [1, 2]) is None
    r["5"] = {"1": {"idx": list(range(8)), "gap89": 1e-5}, "2": _same(5)["2"]}
    assert FB.neuron_near_tie(P, 10, _np(r), HF, [1, 2]) is None


def test_diverged_context_not_used():
    r = {str(q): _same(q) for q in range(P - 1, P + 16)}
    r["6"] = {"1": {"idx": [0, 1, 2, 3, 4, 5, 6, 99], "gap89": 1e-4}, "2": _same(6)["2"]}
    tokens = list(GOLD)
    tokens[1] = 7                                                        # token at position P+1 differs
    assert FB.neuron_near_tie(P, 8, _np(r, tokens), HF, [1, 2]) is None
    assert FB.neuron_near_tie(P, 8, _np(r), HF, [1, 2])["position"] == 6
