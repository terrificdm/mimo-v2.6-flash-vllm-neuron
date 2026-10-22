#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic multimodal test assets.

Writes into --out (default $MIMO_MM_ASSETS, else $MIMO_WORK/mm_assets):
  red_square.png      448x448, white background, one red square           Q: square colour   -> red
  number_739.png      448x448, the digits "739" in large black text       Q: the number      -> 739
  three_circles.png   448x448, three blue filled circles                  Q: how many circles -> 3
  word_trainium.png   448x448, the word "TRAINIUM" in black text          Q: the word        -> TRAINIUM
  digits_1234.mp4     384x384, 4 s at 2 fps (8 frames, mp4v); seconds 1..4 show the digits 1..4 (each digit on
                      2 consecutive frames, so with temporal_patch_size 2 each digit is one temporal group)
  answers.json        question / expected answer per asset
  SHA256SUMS          sha256 of every file above

Only OpenCV Hershey fonts and numpy drawing are used (no system font files), so the output is byte-identical
across runs on the same OpenCV build. Run twice and compare SHA256SUMS (asset determinism check).
  python scripts/api_tests/make_mm_assets.py --out $MIMO_WORK/mm_assets
Exit codes: 0 ok, 1 the mp4 could not be written or read back with 8 frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np

IMG = 448
VID = 384
VID_FPS = 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

ANSWERS = {
    "red_square.png": {"question": "What color is the square in the image? Answer with one word.",
                       "answer": "red", "match": ["red"]},
    "number_739.png": {"question": "What number is written in the image? Answer with the number only.",
                       "answer": "739", "match": ["739"]},
    "three_circles.png": {"question": "How many circles are in the image? Answer with a number.",
                          "answer": "3", "match": ["3", "three"]},
    "word_trainium.png": {"question": "What word is written in the image? Answer with the word only.",
                          "answer": "TRAINIUM", "match": ["trainium"]},
    "digits_1234.mp4": {"question": "Which digits appear in the video, in order? Answer with the digits separated "
                                    "by commas.", "answer": "1, 2, 3, 4", "match_order": ["1", "2", "3", "4"]},
}


def canvas(size: int) -> np.ndarray:
    return np.full((size, size, 3), 255, dtype=np.uint8)  # BGR (OpenCV order)


def centered_text(img: np.ndarray, text: str, scale: float, thickness: int) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - w) // 2
    y = (img.shape[0] + h) // 2
    cv2.putText(img, text, (x, y), font, scale, BLACK, thickness, lineType=cv2.LINE_8)
    return img


def red_square() -> np.ndarray:
    img = canvas(IMG)
    img[124:324, 124:324] = (0, 0, 255)  # BGR red, 200x200 centred
    return img


def number_739() -> np.ndarray:
    return centered_text(canvas(IMG), "739", 5.0, 14)


def three_circles() -> np.ndarray:
    img = canvas(IMG)
    for cx in (84, 224, 364):
        cv2.circle(img, (cx, 224), 56, (255, 0, 0), -1, lineType=cv2.LINE_8)  # BGR blue
    return img


def word_trainium() -> np.ndarray:
    return centered_text(canvas(IMG), "TRAINIUM", 2.2, 6)


def video_frames() -> list[np.ndarray]:
    frames = []
    for d in (1, 2, 3, 4):
        f = centered_text(canvas(VID), str(d), 8.0, 20)
        frames += [f, f.copy()]
    return frames


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("red_square.png", red_square), ("number_739.png", number_739),
                     ("three_circles.png", three_circles), ("word_trainium.png", word_trainium)):
        ok = cv2.imwrite(os.path.join(a.out, name), fn(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            print(f"make_mm_assets: failed to write {name}", file=sys.stderr)
            return 1
    vpath = os.path.join(a.out, "digits_1234.mp4")
    wr = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), VID_FPS, (VID, VID))
    if not wr.isOpened():
        print("make_mm_assets: cv2.VideoWriter(mp4v) could not be opened", file=sys.stderr)
        return 1
    for f in video_frames():
        wr.write(f)
    wr.release()
    cap = cv2.VideoCapture(vpath)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if n != 8:
        print(f"make_mm_assets: mp4 read back {n} frames (expected 8)", file=sys.stderr)
        return 1
    with open(os.path.join(a.out, "answers.json"), "w") as fh:
        json.dump(ANSWERS, fh, indent=1, sort_keys=True)
        fh.write("\n")
    names = sorted(ANSWERS) + ["answers.json"]
    sums = "".join(f"{sha256(os.path.join(a.out, n_))}  {n_}\n" for n_ in names)
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write(sums)
    print(sums, end="")
    print(f"make_mm_assets: done ({a.out}; video {n} frames @ {fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
