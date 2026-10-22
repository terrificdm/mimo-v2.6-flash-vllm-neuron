#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 4: needle-in-a-haystack test for one context tier.

Haystack: sentences drawn with a fixed seed from 200 neutral Chinese / English sentences (weather, traffic,
everyday scenes; none mentions a needle). One needle per request; the question comes last. Prompts are built
deterministically from (length, depth, needle id, --seed) with the model tokenizer, so the content has at most
the target number of tokens. Each prompt's messages text, the content token ids and the chat-templated prompt
length are saved to --prompts-out; --reuse <prompts.json> re-sends saved prompts verbatim (optionally only the
--reuse-depths) and checks usage.prompt_tokens against the saved length.

Needles (new length: needle k goes to depth k of --depths, default 10%/50%/90%):
  N1 蓝色仓库的通行码是 7492-ALPHA。   N2 The name of the lighthouse keeper's cat is Biscuit-42.
  N3 会议改在第 18 号码头举行。          backup N4 紫色信封里的数字是 5813。 (--retry-backup: a failed new-length
  case is retried once with N4 at the same length and depth)
Request: chat, enable_thinking false, temperature 0, max_tokens 32, streamed with include_usage (TTFT is the time
to the first content / reasoning delta; t_seg = TTFT / ceil(prompt_tokens / --segment-size)).
Correct = the normalised answer (NFKC, case-folded, whitespace removed, dash variants -> '-') is a substring of
the normalised content.
Tier verdict: pass = every reuse case and every new length correct; model_limited = every reuse case
correct, some new length still wrong after the backup retry; impl_error = a reuse case is wrong.
Exit codes: 0 pass, 1 impl_error, 5 model_limited, 2 bad arguments / server unreachable.

  run_suite.sh needle --tag ctx32k --ctx 32768 --lengths 30000 [--reuse needle_ctx32k_prompts.json --reuse-depths 0.5]
  needle.py --dry-run --ctx 32768 --lengths 30000 --out /tmp/n   (only build and save the prompts)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DEFAULT_TOKENIZER = os.environ.get("MODEL_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD-bf16"))
MAX_TOKENS = 32
TEMPLATE_RESERVE = 64      # chat template tokens + safety margin, checked against --ctx

NEEDLES = {
    "N1": {"text": "蓝色仓库的通行码是 7492-ALPHA。", "question": "根据上文，蓝色仓库的通行码是什么？只回答通行码。",
           "answer": "7492-ALPHA"},
    "N2": {"text": "The name of the lighthouse keeper's cat is Biscuit-42.",
           "question": "According to the text above, what is the name of the lighthouse keeper's cat? "
                       "Answer with the name only.", "answer": "Biscuit-42"},
    "N3": {"text": "会议改在第 18 号码头举行。", "question": "根据上文，会议改在第几号码头举行？只回答数字。",
           "answer": "18"},
    "N4": {"text": "紫色信封里的数字是 5813。", "question": "根据上文，紫色信封里的数字是多少？只回答数字。",
           "answer": "5813"},
}
NEW_ORDER = ("N1", "N2", "N3")

_ZH_SUBJ = ["清晨的公园里", "城市的主干道上", "小镇的集市旁", "学校的操场边", "河边的步道上",
            "地铁站的出口处", "图书馆的阅览室里", "社区的菜市场里", "郊外的田野上", "商场的停车场里"]
_ZH_PRED = ["几位老人正在慢慢散步。", "微风吹过，树叶轻轻摇动。", "公交车按时进站，乘客有序上车。",
            "天空有些多云，气温比昨天略低。", "孩子们在草地上放风筝。", "路边的店铺陆续开门营业。",
            "一只小狗安静地趴在台阶上。", "人们撑着伞走过湿润的街道。", "远处传来自行车的铃声。",
            "下午的阳光照在白色的墙上。"]
_EN_SUBJ = ["In the morning market,", "On the quiet side street,", "At the edge of the old harbor,",
            "Near the train station,", "Along the river path,", "Inside the small bakery,",
            "On the school playground,", "At the busy crossroads,", "Under the tall oak trees,",
            "Beside the public library,"]
_EN_PRED = ["a few people waited for the bus.", "the light rain stopped before noon.",
            "the traffic moved slowly but steadily.", "a cyclist rang a bell and rode past.",
            "the shop owners arranged fresh fruit.", "the wind carried the smell of bread.",
            "children played a simple ball game.", "the clouds drifted toward the hills.",
            "an old man read a newspaper on a bench.", "the street lamps turned on one by one."]
SENTENCES = ([s + p for s in _ZH_SUBJ for p in _ZH_PRED] + [s + " " + p for s in _EN_SUBJ for p in _EN_PRED])
assert len(SENTENCES) == 200 and len(set(SENTENCES)) == 200


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = re.sub(r"[\u2010-\u2015\u2212\uff0d]", "-", s)
    return re.sub(r"\s+", "", s)


def is_correct(content: str, answer: str) -> bool:
    return normalise(answer) in normalise(content)


def _rng(seed, length, depth, nid):
    return random.Random(f"needle|{seed}|{length}|{depth:.4f}|{nid}")


def build_content(count, length: int, depth: float, nid: str, seed: int = 0):
    """Haystack + needle + question with count(content) <= length (count: str -> number of tokens).
    Returns (content, n_tokens, needle_sentence_index, n_sentences)."""
    nd = NEEDLES[nid]
    rng = _rng(seed, length, depth, nid)
    per = {s: count(s + "\n") for s in SENTENCES}
    budget = length - count(nd["text"] + "\n") - count("\n" + nd["question"])
    hay, n = [], 0
    while True:
        s = SENTENCES[rng.randrange(len(SENTENCES))]
        if n + per[s] > budget:
            break
        hay.append(s)
        n += per[s]

    def render(h):
        k = round(depth * len(h))
        parts = h[:k] + [nd["text"]] + h[k:]
        return "\n".join(parts) + "\n\n" + nd["question"], k

    text, k = render(hay)
    m = count(text)
    while m > length and hay:          # boundary merges can add a few tokens: drop sentences until it fits
        hay.pop()
        text, k = render(hay)
        m = count(text)
    return text, m, k, len(hay)


def make_counter(tokenizer_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    def count(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def ids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    def chat_len(content):
        try:
            r = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
            if isinstance(r, dict) or hasattr(r, "keys"):
                r = r["input_ids"]
            return len(r)
        except Exception:  # noqa: BLE001
            return None
    return count, ids, chat_len


def build_prompts(counter, ctx, lengths, depths, seed=0, nids=None):
    count, ids, chat_len = counter
    out = []
    for L in lengths:
        if L + TEMPLATE_RESERVE + MAX_TOKENS > ctx:
            raise ValueError(f"length {L} + {TEMPLATE_RESERVE} + {MAX_TOKENS} > --ctx {ctx}")
        for i, d in enumerate(depths):
            nid = (nids or NEW_ORDER)[i % len(nids or NEW_ORDER)]
            out.append(make_prompt(counter, L, d, nid, seed))
    return out


def make_prompt(counter, length, depth, nid, seed=0):
    count, ids, chat_len = counter
    content, n, k, ns = build_content(count, length, depth, nid, seed)
    return {"id": f"{nid}@{length}@{int(round(depth * 100))}", "needle": nid, "length": length, "depth": depth,
            "seed": seed, "answer": NEEDLES[nid]["answer"], "content_tokens": n, "needle_sentence": k,
            "n_sentences": ns, "chat_prompt_tokens": chat_len(content),
            "messages": [{"role": "user", "content": content}], "content_token_ids": ids(content)}


def chat_stream(messages, timeout):
    body = C.chat_body(messages, max_tokens=MAX_TOKENS, stream=True, stream_options={"include_usage": True},
                       chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(C.base_url() + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    rec = {"http_status": None, "ttft_s": None, "content": "", "reasoning": "", "usage": None, "finish": None,
           "error": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["http_status"] = r.status
            buf = []
            for raw in iter(r.readline, b""):
                ln = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if ln.startswith("data:"):
                    buf.append(ln[5:].lstrip())
                    continue
                if ln != "" or not buf:
                    continue
                data, buf = "\n".join(buf), []
                if data.strip() == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except ValueError:
                    continue
                if c.get("usage"):
                    rec["usage"] = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or ""
                    rpiece = d.get("reasoning") or d.get("reasoning_content") or ""
                    if (piece or rpiece) and rec["ttft_s"] is None:
                        rec["ttft_s"] = round(time.time() - t0, 3)
                    rec["content"] += piece
                    rec["reasoning"] += rpiece
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        rec["http_status"], rec["error"] = e.code, e.read().decode(errors="replace")[:2000]
    except Exception as e:  # noqa: BLE001
        rec["http_status"] = rec["http_status"] or -1
        rec["error"] = repr(e)
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def run_case(suite, p, kind, timeout, seg, desc_extra=""):
    rec = chat_stream(p["messages"], timeout)
    ok = rec["http_status"] == 200 and rec["error"] is None and is_correct(rec["content"], p["answer"])
    pt = (rec["usage"] or {}).get("prompt_tokens")
    probs = []
    if pt is not None and p.get("chat_prompt_tokens") is not None and pt != p["chat_prompt_tokens"]:
        probs.append(f"usage.prompt_tokens {pt} != saved {p['chat_prompt_tokens']}")
    tseg = None
    if rec["ttft_s"] is not None and pt:
        tseg = round(rec["ttft_s"] / math.ceil(pt / seg), 4)
    reason = (f"{'correct' if ok else 'WRONG'}: content={rec['content'][:80]!r} answer={p['answer']!r} "
              f"prompt_tokens={pt} ttft={rec['ttft_s']}s t_seg={tseg}s" + (f" error={rec['error'][:200]}" if rec['error'] else ""))
    if probs:
        reason += " | " + "; ".join(probs)
    summary = {k: v for k, v in p.items() if k not in ("messages", "content_token_ids")}
    return suite.add(f"{kind}:{p['id']}", f"{kind} {p['needle']} len {p['length']} depth {p['depth']}{desc_extra}",
                     "pass" if ok else "fail", reason, records=[dict(rec, prompt=summary)], kind=kind,
                     correct=ok, prompt_tokens=pt, ttft_s=rec["ttft_s"], t_seg_s=tseg, length_mismatch=bool(probs))


def tier_verdict(cases):
    reuse = [c for c in cases if c["kind"] == "reuse"]
    new = {}
    for c in cases:
        if c["kind"] in ("new", "backup"):
            key = (c["records"][0]["prompt"]["length"], c["records"][0]["prompt"]["depth"])
            new[key] = new.get(key, False) or c["correct"]
    if any(not c["correct"] for c in reuse):
        return "impl_error"
    if all(new.values()):
        return "pass"
    return "model_limited"


def save_prompts(path, prompts, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(meta, prompts=prompts), f, ensure_ascii=False)
    os.replace(tmp, path)


def main(argv=None):
    ap = C.base_argparser("needle-in-a-haystack")
    ap.add_argument("--ctx", type=int, required=True, help="tier max_model_len")
    ap.add_argument("--lengths", default="", help="comma-separated content token counts for new needles")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="append", default=[], help="saved prompts.json of an earlier tier (repeatable)")
    ap.add_argument("--reuse-depths", default=None, help="only reuse prompts at these depths, e.g. 0.5")
    ap.add_argument("--retry-backup", action="store_true", help="retry a failed new-length case with needle N4")
    ap.add_argument("--segment-size", type=int, default=4096)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompts-out", default=None, help="default <out>_prompts.json")
    ap.add_argument("--dry-run", action="store_true", help="build and save prompts only")
    a = ap.parse_args(argv)
    try:
        lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
        depths = [float(x) for x in a.depths.split(",") if x.strip()]
        rdepths = None if a.reuse_depths is None else {round(float(x), 4) for x in a.reuse_depths.split(",")}
    except ValueError as e:
        ap.error(f"bad list: {e}")
    if not lengths and not a.reuse:
        ap.error("give --lengths and/or --reuse")
    if any(not 0 <= d <= 1 for d in depths):
        ap.error("--depths must be in [0, 1]")
    if not a.out:
        ap.error("--out is required")
    counter = make_counter(a.tokenizer)
    try:
        new = build_prompts(counter, a.ctx, lengths, depths, a.seed)
    except ValueError as e:
        ap.error(str(e))
    reused = []
    for path in a.reuse:
        doc = json.load(open(path))
        for p in doc["prompts"]:
            if rdepths is not None and round(p["depth"], 4) not in rdepths:
                continue
            if p["chat_prompt_tokens"] and p["chat_prompt_tokens"] + MAX_TOKENS > a.ctx:
                ap.error(f"reused prompt {p['id']} ({p['chat_prompt_tokens']} tokens) does not fit --ctx {a.ctx}")
            reused.append(dict(p, source=path))
    pout = a.prompts_out or a.out + "_prompts.json"
    meta = {"ctx": a.ctx, "seed": a.seed, "tokenizer": a.tokenizer, "created_utc": C.now_utc()}
    save_prompts(pout, new, meta)
    print(f"needle: {len(new)} new prompts saved to {pout}; {len(reused)} reused", flush=True)
    for p in new:
        print(f"  {p['id']}: content {p['content_tokens']} tok, chat {p['chat_prompt_tokens']} tok", flush=True)
    if a.dry_run:
        return 0
    C.require_server()
    suite = C.Suite("needle", a.out, a.tag, meta=dict(meta, prompts_file=pout, reuse=a.reuse))
    for p in reused:
        run_case(suite, p, "reuse", a.timeout, a.segment_size, f" (from {os.path.basename(p['source'])})")
    for p in new:
        c = run_case(suite, p, "new", a.timeout, a.segment_size)
        if not c["correct"] and a.retry_backup:
            b = make_prompt(counter, p["length"], p["depth"], "N4", a.seed)
            run_case(suite, b, "backup", a.timeout, a.segment_size, f" (retry of {p['id']})")
    v = tier_verdict(suite.cases)
    suite.meta["tier_verdict"] = v
    suite.meta["new_correct"] = f"{sum(c['correct'] for c in suite.cases if c['kind'] == 'new')}/{len(new)}"
    suite.meta["reuse_correct"] = f"{sum(c['correct'] for c in suite.cases if c['kind'] == 'reuse')}/{len(reused)}"
    ttfts = [c["t_seg_s"] for c in suite.cases if c.get("t_seg_s") is not None]
    suite.meta["t_seg_mean_s"] = round(sum(ttfts) / len(ttfts), 4) if ttfts else None
    suite.write()
    print(f"needle: tier verdict {v}; new {suite.meta['new_correct']}, reuse {suite.meta['reuse_correct']}", flush=True)
    return {"pass": 0, "impl_error": 1, "model_limited": 5}[v]


if __name__ == "__main__":
    sys.exit(main())
