#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audio test assets with known answers. Stdlib + numpy only (no soundfile / av in the venv).

  python scripts/api_tests/make_audio_assets.py --out $MIMO_WORK/mm_assets/audio [--copy-to assets/audio]

Writes 16-bit PCM mono WAV files at 24 kHz (the MiMo audio tokenizer rate, so no resampling is needed):
  * ``beeps_3.wav``     three 0.25 s 880 Hz beeps separated by 0.5 s silence (Q: how many beeps -> 3);
  * ``tone_440.wav``    2 s steady 440 Hz sine (Q: describe the sound -> a steady tone / beep, no speech);
  * ``jfk.wav``         J. F. Kennedy, 1961 inaugural address excerpt (US federal government work, public domain),
                        fetched from the Hugging Face ``Xenova/transformers.js-docs`` dataset (16 kHz), resampled to
                        24 kHz with ``scipy.signal.resample_poly(3, 2)``; transcript in ``answers.json``.
The synthetic files are deterministic; ``answers.json`` records the sha256 of every file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import wave

import numpy as np

SR = 24000
JFK_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
JFK_TEXT = ("And so, my fellow Americans, ask not what your country can do for you, "
            "ask what you can do for your country.")


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got sample width {sw}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def tone(freq: float, seconds: float, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    ramp = min(len(x) // 10, int(0.01 * SR))           # 10 ms fade in / out (no clicks)
    env = np.ones_like(x)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return x * env


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-to", default=None)
    ap.add_argument("--no-download", action="store_true", help="skip jfk.wav")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    gap = np.zeros(int(0.5 * SR))
    beeps = np.concatenate([np.zeros(int(0.25 * SR))] + [np.concatenate([tone(880, 0.25), gap]) for _ in range(3)])
    write_wav(os.path.join(a.out, "beeps_3.wav"), beeps)
    write_wav(os.path.join(a.out, "tone_440.wav"), tone(440, 2.0))
    answers = {
        "beeps_3.wav": {"question": "How many beeps are there in this audio? Answer with a number.", "answer": "3",
                        "seconds": len(beeps) / SR},
        "tone_440.wav": {"question": "Describe this sound in one sentence. Is there any speech?",
                         "answer": "a steady tone / beep, no speech (qualitative)", "seconds": 2.0},
    }
    if not a.no_download:
        with urllib.request.urlopen(JFK_URL, timeout=60) as r:
            raw = r.read()
        x, sr = read_wav(raw)
        if sr != SR:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        write_wav(os.path.join(a.out, "jfk.wav"), x)
        answers["jfk.wav"] = {"question": "Transcribe this audio exactly.", "answer": JFK_TEXT,
                              "keywords": ["fellow Americans", "ask not", "your country"], "seconds": len(x) / SR,
                              "source": JFK_URL, "source_sha256": hashlib.sha256(raw).hexdigest(),
                              "source_sample_rate": sr, "license": "US federal government work (public domain)"}
    for k in answers:
        answers[k]["sha256"] = sha256(os.path.join(a.out, k))
        answers[k]["sample_rate"] = SR
    with open(os.path.join(a.out, "answers.json"), "w") as f:
        json.dump(answers, f, indent=1, ensure_ascii=False)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        for k in list(answers) + ["answers.json"]:
            shutil.copy2(os.path.join(a.out, k), os.path.join(a.copy_to, k))
    print(json.dumps({k: {"seconds": v["seconds"], "sha256": v["sha256"][:12]} for k, v in answers.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
