#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""T2b: the 'mimo' tool parser (vllm Qwen3EngineToolParser) can return several calls.

Runs on the instance CPU inside the vLLM venv (no server, no NeuronCores). Builds the parser with the model's
tokenizer, parses a fixed text with two <tool_call> blocks in the MiMo chat-template format and checks that both
calls come back with the right names and JSON arguments.
Usage: offline_parser_check.py [--model-dir DIR] [--out PREFIX]   exit 0 pass, 1 fail, 2 setup error.
"""

import argparse
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from test_tools import TOOLS  # noqa: E402

TEXT = ("我来分别查询两个城市的天气。"
        "<tool_call><function=get_weather><parameter=city>北京</parameter><parameter=unit>celsius</parameter>"
        "</function></tool_call>"
        "<tool_call><function=get_weather><parameter=city>上海</parameter><parameter=unit>celsius</parameter>"
        "</function></tool_call>")
EXPECTED = [("get_weather", {"city": "北京", "unit": "celsius"}), ("get_weather", {"city": "上海", "unit": "celsius"})]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        from transformers import AutoTokenizer
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers import ToolParserManager
    except Exception:  # noqa: BLE001
        try:
            from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # older layout
            from vllm.tool_parsers import ToolParserManager
            from transformers import AutoTokenizer
        except Exception as e:  # noqa: BLE001
            print(f"offline_parser_check: import failed: {e!r}", file=sys.stderr)
            return 2
    cls = ToolParserManager.get_tool_parser("mimo")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    req = ChatCompletionRequest(model=C.MODEL, messages=[{"role": "user", "content": "北京和上海的天气"}], tools=TOOLS)
    params = inspect.signature(cls.__init__).parameters
    parser = cls(tok, req.tools) if "tools" in params else cls(tok)
    info = parser.extract_tool_calls(TEXT, req)
    calls = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (info.tool_calls or [])]
    problems = []
    if not info.tools_called:
        problems.append("tools_called is False")
    if len(calls) != len(EXPECTED):
        problems.append(f"{len(calls)} calls parsed, want {len(EXPECTED)}")
    for i, (c, (name, a)) in enumerate(zip(calls, EXPECTED)):
        try:
            got = json.loads(c["arguments"])
        except ValueError:
            problems.append(f"call {i}: arguments not JSON: {c['arguments']!r}")
            continue
        if c["name"] != name or got != a:
            problems.append(f"call {i}: got {c['name']} {got}, want {name} {a}")
    res = {"parser_class": f"{cls.__module__}.{cls.__name__}", "text": TEXT, "tools_called": info.tools_called,
           "calls": calls, "content": info.content, "problems": problems, "pass": not problems}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out + ".json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
