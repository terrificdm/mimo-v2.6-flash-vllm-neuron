#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# OpenAI 接口功能测试：temperature=0；
#   单并发：/v1/completions 与 /v1/chat/completions 各发中文、英文、数学题；
#   4 并发：同时发 4 个 chat 请求（中文、英文、两道数学题）。
# 原始请求/响应写 $SMOKE_OUT/{single,concurrent4}.json，并打印“问题 / 回答 / 是否含期望答案”摘要。
# 用法：start_bg.sh smoke bash scripts/smoke_test.sh
# 环境变量：PORT（默认 8000）、SMOKE_OUT（默认 $MIMO_WORK/results/smoke）、
#           SMOKE_MAX_TOKENS（覆盖 completions 128 / chat 512，tiny 演练用）。
# 退出码：任一请求不是 HTTP 200 → 1；否则 0（回答是否正确由人工按摘要判断）。
set -u
export PORT=${PORT:-8000}
export SMOKE_OUT=${SMOKE_OUT:-${MIMO_WORK:-$HOME/mimo-work}/results/smoke}
export SMOKE_MAX_TOKENS=${SMOKE_MAX_TOKENS:-}
mkdir -p "$SMOKE_OUT"
exec python3 - <<'PY'
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = f"http://127.0.0.1:{os.environ['PORT']}"
OUT = os.environ["SMOKE_OUT"]
MT = os.environ.get("SMOKE_MAX_TOKENS") or ""
MODEL = "mimo-v2.6-flash"
NO_THINK = {"enable_thinking": False}

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status, resp = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, resp = e.code, {"error": e.read().decode(errors="replace")}
    except Exception as e:  # noqa: BLE001
        status, resp = -1, {"error": repr(e)}
    return {"endpoint": path, "request": body, "http_status": status, "elapsed_s": round(time.time() - t, 2), "response": resp}

def completion(prompt):
    return ("/v1/completions", {"model": MODEL, "prompt": prompt, "temperature": 0, "max_tokens": int(MT or 128)})

def chat(content):
    return ("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0,
                                     "max_tokens": int(MT or 512), "chat_template_kwargs": NO_THINK})

def answer(r):
    try:
        c = r["response"]["choices"][0]
        return c["text"] if "text" in c else (c["message"].get("content") or "")
    except Exception:  # noqa: BLE001
        return ""

def summarize(title, items):
    print(f"===== {title} =====")
    for (q, expect), r in items:
        a = answer(r)
        hit = "n/a" if expect is None else ("YES" if expect in a else "NO")
        print(f"[{r['endpoint']}] HTTP {r['http_status']} {r['elapsed_s']}s  expect={expect} contains={hit}")
        print(f"  Q: {q}")
        print("  A: " + a.strip().replace("\n", "\\n")[:600])
    sys.stdout.flush()

single_cases = [
    (completion("问题：中国的首都是哪座城市？请用一句话介绍它。\n回答："), "中国的首都是哪座城市？", None),
    (completion("Question: What is the capital of France? Answer in one sentence.\nAnswer:"), "capital of France", None),
    (completion("问题：17×23=?\n回答："), "17×23=?", "391"),
    (chat("请用两三句话介绍一下长城。"), "请用两三句话介绍一下长城。", None),
    (chat("Explain in two or three sentences why the sky is blue."), "why the sky is blue", None),
    (chat("一个数加上它的一半等于 30，这个数是多少？"), "一个数加上它的一半等于 30，这个数是多少？", "20"),
]
single = []
for (path, body), q, expect in single_cases:
    single.append(((q, expect), post(path, body)))
json.dump([r | {"question": q, "expect": e} for (q, e), r in single], open(f"{OUT}/single.json", "w"), ensure_ascii=False, indent=1)
summarize("single (concurrency 1)", single)

conc_cases = [
    (chat("请用两三句话介绍一下长城。"), "请用两三句话介绍一下长城。", None),
    (chat("Explain in two or three sentences why the sky is blue."), "why the sky is blue", None),
    (chat("17×23 等于多少？"), "17×23 等于多少？", "391"),
    (chat("一个数加上它的一半等于 30，这个数是多少？"), "一个数加上它的一半等于 30，这个数是多少？", "20"),
]
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(post, path, body) for (path, body), _, _ in conc_cases]
    conc = [((q, e), f.result()) for (_, q, e), f in zip(conc_cases, futs)]
json.dump([r | {"question": q, "expect": e} for (q, e), r in conc], open(f"{OUT}/concurrent4.json", "w"), ensure_ascii=False, indent=1)
summarize(f"concurrent4 (wall {time.time() - t0:.1f}s)", conc)

bad = [r for _, r in single + conc if r["http_status"] != 200]
print(f"smoke_test: {len(single) + len(conc) - len(bad)}/{len(single) + len(conc)} HTTP 200; results in {OUT}")
sys.exit(1 if bad else 0)
PY
