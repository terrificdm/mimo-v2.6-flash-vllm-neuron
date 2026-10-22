#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 5: image / video question answering over the OpenAI chat API.
Thinking off, temperature 0. Run through run_suite.sh (venv).

I1-I4  the 4 synthetic 448x448 images of make_mm_assets.py as file:// URLs under --assets (the server's
       --allowed-local-media-path); question and accepted answers from <assets>/answers.json
B1     number_739.png sent as a base64 data URL (docs/design-notes.md#local-media-path)
A1     the model's own assets/architecture.png (--architecture) as a base64 data URL, qualitative description:
       verdict 'info', not scored (docs/design-notes.md#validation-rules)
V1     (--video) digits_1234.mp4 as a file:// video_url; pass = the digits 1, 2, 3, 4 appear in this order
U1-U3  (--audio, needs MIMO_AUDIO=1) the WAV files under <assets>/audio (answers in <assets>/audio/answers.json):
       U1 jfk.wav as a file:// audio_url, transcription; pass = every keyword appears (case / punctuation ignored)
       U2 beeps_3.wav as an OpenAI input_audio part (base64 wav); pass = "3" or "three" in the answer
       U3 tone_440.wav as a file:// audio_url, qualitative description: verdict 'info', not scored
       Audio requests also need prompt_tokens above the same question sent without audio (the text-only count is
       measured with one extra max_tokens=1 request per case).
Scored cases pass when the lower-cased answer contains one of the accepted strings and there is no reasoning /
literal <think>. Every response also records usage.prompt_tokens (an image adds 196 tokens at 448x448).
--plumbing-only (tiny-model drill): HTTP 200, non-empty content and prompt_tokens above the text-only count are
enough (a random tiny model cannot answer); verdict 'pass' then means "the request went through".
--quick = I1 + B1.
Usage: mm_qa.py --out PREFIX [--tag T] [--quick] [--cases I1,B1] [--video] [--audio] [--assets DIR]
                [--architecture PNG] [--max-tokens N] [--plumbing-only] [--timeout S]
"""

import base64
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OFF = {"enable_thinking": False}
IMAGES = [("I1", "red_square.png"), ("I2", "number_739.png"), ("I3", "three_circles.png"),
          ("I4", "word_trainium.png")]
BASE64_IMAGE = "number_739.png"
VIDEO = "digits_1234.mp4"
ARCH_Q = "Describe this figure briefly: what does it show?"
AUDIO = [("U1", "jfk.wav"), ("U2", "beeps_3.wav"), ("U3", "tone_440.wav")]
AUDIO_CASES = [c for c, _ in AUDIO]
ALL = ["I1", "I2", "I3", "I4", "B1", "A1", "V1"] + AUDIO_CASES
TEXT_ONLY_TOKENS = 64   # a question without media stays well below this; 196 more tokens per 448x448 image


def data_url(path, mime="image/png"):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def body(question, media, max_tokens):
    content = [media, {"type": "text", "text": question}]
    return C.chat_body([{"role": "user", "content": content}], max_tokens=max_tokens, chat_template_kwargs=OFF)


def image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def video_part(url):
    return {"type": "video_url", "video_url": {"url": url}}


def audio_url_part(url):
    return {"type": "audio_url", "audio_url": {"url": url}}


def input_audio_part(path):
    with open(path, "rb") as f:
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(f.read()).decode(), "format": "wav"}}


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


def judge(content, ans):
    """-> list of problems (empty = correct)."""
    text = (content or "").lower()
    if "keywords" in ans:
        miss = [k for k in ans["keywords"] if _norm(k) not in _norm(content)]
        return [f"keywords {miss} missing in {content!r}"] if miss else []
    if "match_order" in ans:
        pos, at = [], 0
        for d in ans["match_order"]:
            i = text.find(d.lower(), at)
            if i < 0:
                return [f"{ans['match_order']} not found in order in {content!r}"]
            pos.append(i)
            at = i + len(d)
        return []
    if not any(m.lower() in text for m in ans["match"]):
        return [f"none of {ans['match']} in {content!r}"]
    return []


class Runner:
    def __init__(self, suite, a, answers):
        self.s, self.a, self.answers = suite, a, answers

    def _case(self, cid, desc, question, media, ans=None, scored=True, min_tokens=TEXT_ONLY_TOKENS):
        rec = C.post("/v1/chat/completions", body(question, media, self.a.max_tokens), self.a.timeout)
        for part in rec.get("request", {}).get("messages", [{}])[0].get("content", []):
            u = (part.get("image_url") or {}).get("url", "")
            if u.startswith("data:"):   # keep the result json small
                part["image_url"]["url"] = u[:48] + f"...<{len(u)} chars>"
            d = (part.get("input_audio") or {}).get("data", "")
            if d:
                part["input_audio"]["data"] = d[:48] + f"...<{len(d)} chars>"
        msg = C.message(rec)
        content, reasoning = msg.get("content"), C.reasoning_of(msg)
        usage = (rec.get("response") or {}).get("usage") or {}
        f = {"content": content, "reasoning": reasoning, "finish": C.choice0(rec).get("finish_reason"),
             "prompt_tokens": usage.get("prompt_tokens")}
        if rec["http_status"] != 200:
            return self.s.add(cid, desc, "fail", f"HTTP {rec['http_status']} {rec.get('error') or ''}"[:300],
                              [rec], fields=f)
        p = []
        if not (content or "").strip():
            p.append("empty content")
        if (f["prompt_tokens"] or 0) <= min_tokens:
            p.append(f"prompt_tokens {f['prompt_tokens']} <= {min_tokens}: media not expanded?")
        if self.a.plumbing_only or not scored:
            verdict = "fail" if p else ("pass" if scored else "info")
            return self.s.add(cid, desc, verdict, "; ".join(p) or f"answer {content!r} "
                              f"(prompt_tokens {f['prompt_tokens']}, not scored)", [rec], fields=f)
        if (reasoning or "").strip() or "<think>" in (content or ""):
            p.append("reasoning present with thinking off")
        p += judge(content, ans)
        return self.s.add(cid, desc, "fail" if p else "pass",
                          "; ".join(p) or f"answer {content!r} contains {ans['answer']!r} "
                          f"(prompt_tokens {f['prompt_tokens']})", [rec], fields=f)

    def image(self, cid, fn):
        ans = self.answers[fn]
        url = "file://" + os.path.join(os.path.realpath(self.a.assets), fn)
        return self._case(cid, f"{fn} (file:// URL)", ans["question"], image_part(url), ans)

    def I1(self):
        return self.image("I1", IMAGES[0][1])

    def I2(self):
        return self.image("I2", IMAGES[1][1])

    def I3(self):
        return self.image("I3", IMAGES[2][1])

    def I4(self):
        return self.image("I4", IMAGES[3][1])

    def B1(self):
        ans = self.answers[BASE64_IMAGE]
        url = data_url(os.path.join(self.a.assets, BASE64_IMAGE))
        return self._case("B1", f"{BASE64_IMAGE} (base64 data URL)", ans["question"], image_part(url), ans)

    def A1(self):
        if not os.path.isfile(self.a.architecture):
            return self.s.add("A1", "architecture.png qualitative", "info", f"{self.a.architecture} not found")
        return self._case("A1", "architecture.png (base64, qualitative, not scored)", ARCH_Q,
                          image_part(data_url(self.a.architecture)), scored=False)

    def V1(self):
        ans = self.answers[VIDEO]
        url = "file://" + os.path.join(os.path.realpath(self.a.assets), VIDEO)
        return self._case("V1", f"{VIDEO} (file:// video_url)", ans["question"], video_part(url), ans)

    def _text_only_tokens(self, question):
        b = C.chat_body([{"role": "user", "content": [{"type": "text", "text": question}]}], max_tokens=1,
                        chat_template_kwargs=OFF)
        rec = C.post("/v1/chat/completions", b, self.a.timeout)
        return ((rec.get("response") or {}).get("usage") or {}).get("prompt_tokens") or TEXT_ONLY_TOKENS

    def audio(self, cid):
        fn = dict(AUDIO)[cid]
        adir = os.path.join(os.path.realpath(self.a.assets), "audio")
        ans = dict(self.audio_answers[fn])
        base = self._text_only_tokens(ans["question"])
        path = os.path.join(adir, fn)
        if cid == "U2":
            ans.setdefault("match", [ans["answer"], "three"])
            return self._case(cid, f"{fn} (input_audio base64 wav)", ans["question"], input_audio_part(path), ans,
                              min_tokens=base)
        part = audio_url_part("file://" + path)
        if cid == "U3":
            return self._case(cid, f"{fn} (file:// audio_url, qualitative, not scored)", ans["question"], part,
                              scored=False, min_tokens=base)
        return self._case(cid, f"{fn} (file:// audio_url, transcription)", ans["question"], part, ans,
                          min_tokens=base)

    def U1(self):
        return self.audio("U1")

    def U2(self):
        return self.audio("U2")

    def U3(self):
        return self.audio("U3")


def main(argv=None):
    ap = C.base_argparser(__doc__)
    ap.add_argument("--assets", default=os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets")))
    ap.add_argument("--architecture", default=os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD/assets/architecture.png"))
    ap.add_argument("--video", action="store_true", help="add V1 (video)")
    ap.add_argument("--audio", action="store_true", help="add U1-U3 (audio, server started with MIMO_AUDIO=1)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--plumbing-only", action="store_true", help="tiny drill: judge HTTP / token counts only")
    a = ap.parse_args(argv)
    cases = ["I1", "B1"] if a.quick else [c for c in ALL if (c != "V1" or a.video)
                                          and (c not in AUDIO_CASES or a.audio)]
    if a.cases:
        cases = [c.strip().upper() for c in a.cases.split(",") if c.strip()]
        bad = [c for c in cases if c not in ALL]
        if bad:
            print(f"unknown cases {bad}; valid {ALL}", file=sys.stderr)
            return 2
    ap_path = os.path.join(a.assets, "answers.json")
    if not os.path.isfile(ap_path):
        print(f"{ap_path} not found (run make_mm_assets.py)", file=sys.stderr)
        return 2
    answers = json.load(open(ap_path))
    audio_answers = {}
    if any(c in AUDIO_CASES for c in cases):
        aa_path = os.path.join(a.assets, "audio", "answers.json")
        if not os.path.isfile(aa_path):
            print(f"{aa_path} not found", file=sys.stderr)
            return 2
        audio_answers = json.load(open(aa_path))
    C.require_server()
    suite = C.Suite("mm", a.out, a.tag, meta={"cases_requested": cases, "max_tokens": a.max_tokens,
                                             "assets": a.assets, "plumbing_only": a.plumbing_only})
    r = Runner(suite, a, answers)
    r.audio_answers = audio_answers
    for c in cases:
        getattr(r, c)()
    print(f"mm: {suite.counts()}")
    return suite.exit_code()


if __name__ == "__main__":
    sys.exit(main())
