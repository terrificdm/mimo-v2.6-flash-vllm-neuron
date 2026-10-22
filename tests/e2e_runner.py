#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One vllm-neuron CPU-mode ``LLM.generate`` run for the tiny e2e tests (one process per test case).

The caller (tests/test_tiny_e2e.py) passes every environment variable explicitly through the
subprocess env dict (VLLM_NEURON_CPU_MODE=1, MIMO_DECODE_MOE_IMPL, ...).

Usage: e2e_runner.py --spec spec.json
  spec: {"model": dir, "tp": int, "max_num_seqs": int, "prompts": [[token ids], ...],
         "max_tokens": 16, "enforce_eager": false, "out": "result.pt",
         optional: "max_model_len": 1024, "kv_segment_size": 0 (segmented prefill segment, 0 = off),
         "arch": "MiMoV2ForCausalLM", multimodal: "mm_prompts": [{"prompt": str with the image
         placeholder, "image": png path} or {"prompt_token_ids": [already expanded ids], "image": ...}, ...] (replaces
         "prompts"), "vision": {"block": 1024, "tp": 1, "dp": 2}}
Output (torch.save): {"tokens": [[16 ids] per prompt], "logits": [FloatTensor[16, V] per prompt],
                      "prompt_token_ids": [[ids] per prompt], "resolved_class": "<module>.<qualname>",
                      "elapsed_s": float}
Prints "E2E_RESOLVED_CLASS <module>.<qualname>" and "E2E_DONE" on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    with open(a.spec) as f:
        spec = json.load(f)

    # NOTE: import vllm before transformers (vllm-neuron example convention)
    from vllm import LLM, ModelRegistry, SamplingParams
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    cls = ModelRegistry._try_load_model_cls(spec.get("arch", "MiMoV2ForCausalLM"))
    resolved = f"{cls.__module__}.{cls.__qualname__}" if cls is not None else "None"
    print("E2E_RESOLVED_CLASS", resolved, flush=True)

    from transformers import AutoConfig

    vocab = AutoConfig.from_pretrained(spec["model"], trust_remote_code=True).vocab_size
    n_seqs = int(spec["max_num_seqs"])
    max_model_len = int(spec.get("max_model_len", 1024))
    seg = int(spec.get("kv_segment_size", 0))
    nbt = seg or max_model_len
    seg_cfg = {"kv_segment_size_buckets": [seg]} if seg else {}
    mm = spec.get("mm_prompts")
    mm_args = {}
    if mm:
        v = spec["vision"]
        mm_args = dict(limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0})
        vision_cfg = {"vision_neuron_config": {"num_vision_tokens_buckets": [int(v["block"])],
                                               "vision_attention_block_size": int(v["block"]),
                                               "tp_size": int(v["tp"]), "dp_size": int(v["dp"])}}
    else:
        vision_cfg = {}
    t0 = time.time()
    llm = LLM(
        model=spec["model"],
        trust_remote_code=True,
        tensor_parallel_size=int(spec["tp"]),
        # EP=TP; vllm-neuron only builds the EP groups for ep_degree > 1 (GPT-OSS behaves the same), so
        # TP=1 runs without the flag (all experts on the single rank).
        enable_expert_parallel=int(spec["tp"]) > 1,
        max_model_len=max_model_len,
        max_num_batched_tokens=nbt,
        max_num_seqs=n_seqs,
        block_size=32,
        enable_prefix_caching=False,
        # CPU mode sizes the KV cache from host memory (~90M tokens, >500 GB); a fixed small pool is plenty
        num_gpu_blocks_override=512,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        async_scheduling=False,
        enforce_eager=bool(spec.get("enforce_eager", False)),
        additional_config={"neuron_config": {
            "on_device_sampling_config": {},
            "num_batched_tokens_buckets": [nbt],
            "num_seqs_buckets": [n_seqs],
            **seg_cfg,
        }, **vision_cfg},
        **mm_args,
    )
    max_tokens = int(spec.get("max_tokens", 16))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=vocab, detokenize=False)
    if mm:
        from PIL import Image

        inputs = [{**({"prompt_token_ids": m["prompt_token_ids"]} if "prompt_token_ids" in m else {"prompt": m["prompt"]}),
                   "multi_modal_data": {"image": Image.open(m["image"]).convert("RGB")}} for m in mm]
    else:
        inputs = [{"prompt_token_ids": ids} for ids in spec["prompts"]]
    outs = llm.generate(inputs, sp)

    tokens, logits, prompt_ids = [], [], []
    for o in outs:
        prompt_ids.append(list(o.prompt_token_ids))
        c = o.outputs[0]
        if len(c.token_ids) != max_tokens or not c.logprobs or len(c.logprobs) != max_tokens:
            print(f"ERROR: expected {max_tokens} tokens with logprobs, got {len(c.token_ids)} / "
                  f"{len(c.logprobs or [])}", flush=True)
            return 3
        steps = []
        for d in c.logprobs:
            if len(d) < vocab:
                print(f"ERROR: logprob entries {len(d)} < vocab {vocab}", flush=True)
                return 3
            row = torch.full((vocab,), -float("inf"))
            ids = torch.tensor(list(d.keys()), dtype=torch.long)
            vals = torch.tensor([lp.logprob for lp in d.values()], dtype=torch.float32)
            row[ids] = vals
            steps.append(row)
        tokens.append(list(c.token_ids))
        logits.append(torch.stack(steps, 0))
    elapsed = time.time() - t0
    torch.save({"tokens": tokens, "logits": logits, "prompt_token_ids": prompt_ids, "resolved_class": resolved,
                "elapsed_s": elapsed}, spec["out"])
    print(f"E2E_TOKENS {json.dumps(tokens)}", flush=True)
    print(f"E2E_DONE {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    import os
    import traceback

    try:
        rc = main()
    except BaseException:  # noqa: BLE001 - report and exit; vLLM background threads can block a normal exit
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit: after an engine start-up failure the vLLM client keeps non-daemon threads alive and the
    # interpreter never exits; the results (if any) are already saved. Engine workers exit with the parent.
    os._exit(rc)
