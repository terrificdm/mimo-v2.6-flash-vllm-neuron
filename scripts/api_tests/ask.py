#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send a chat request (text / image / video / audio) to the local MiMo vLLM server
and print the streamed reply. Standard library only.

Examples:
  python3 ask.py "用三句话介绍一下长城"
  python3 ask.py --image cat.jpeg "图里有什么？"
  python3 ask.py --video digits_1234.mp4 "视频里依次出现了哪些数字？"
  python3 ask.py --audio jfk.wav "把这段音频转写出来"
  python3 ask.py --no-think --image beach_dog.jpeg "描述这张图"
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
}


def data_url(path):
    """Return a base64 data URL for a local file (sent in the request body, not argv)."""
    ext = os.path.splitext(path)[1].lower()
    mime = MIME.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def build_messages(args):
    content = []
    for p in args.image:
        content.append({"type": "image_url", "image_url": {"url": data_url(p)}})
    for p in args.video:
        content.append({"type": "video_url", "video_url": {"url": data_url(p)}})
    for p in args.audio:
        content.append({"type": "audio_url", "audio_url": {"url": data_url(p)}})
    content.append({"type": "text", "text": args.question})
    if len(content) == 1:
        return [{"role": "user", "content": args.question}]
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--no-think", action="store_true", help="disable thinking (enable_thinking=false)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    args = ap.parse_args()

    body = {
        "model": args.model,
        "messages": build_messages(args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    mode = None
    got = {"思考": 0, "回答": 0}
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("error"):
            err = d["error"]
            print(f"\n[服务端错误] {err.get('message', err) if isinstance(err, dict) else err}", file=sys.stderr)
            continue
        if d.get("usage"):
            u = d["usage"]
            print(f"\n\n[用量] 输入 {u['prompt_tokens']} token，输出 {u['completion_tokens']} token")
        for c in d.get("choices", []):
            delta = c.get("delta") or {}
            for kind, piece in (("思考", delta.get("reasoning") or delta.get("reasoning_content")),
                                ("回答", delta.get("content"))):
                if not piece:
                    continue
                if first is None:
                    first = time.time() - t0
                got[kind] += len(piece)
                if mode != kind:
                    print(f"\n\n========== {kind} ==========")
                    mode = kind
                print(piece, end="", flush=True)
            if c.get("finish_reason"):
                fr = c["finish_reason"]
                print(f"\n\n[结束原因] {fr}" + ("（达到 --max-tokens 上限，被截断，可以调大 --max-tokens 或加 --no-think）" if fr == "length" else ""))
                if got["回答"] == 0 and got["思考"] > 0:
                    print("[提示] 模型没有输出思考结束标记，vLLM 的 mimo 推理解析器把全部输出都归到了“思考”里，"
                          "上面“思考”部分就是它的回答。加 --no-think 可以直接得到“回答”。")
    total = time.time() - t0
    print(f"[首字时间] {first:.2f} 秒  [总耗时] {total:.2f} 秒" if first is not None else f"[总耗时] {total:.2f} 秒")


if __name__ == "__main__":
    main()
