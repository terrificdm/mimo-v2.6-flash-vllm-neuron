# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the plugin MiMo audio towers against HF (docs/design-notes.md#audio-input, docs/design-notes.md#audio-criteria).

Inputs are the mels vLLM's MiMo-Omni processor produces for the WAV assets (scripts/api_tests/make_audio_assets.py).
Tensor parallel 8 is simulated with one thread per rank (as tests/test_vision_cpu.py).

Criteria: the RVQ codes are discrete and HF BF16 itself agrees with HF FP32 on only 57-83 % of them
(measured on the three WAV assets), so
  * pre-VQ features: cos(ours, HF BF16) >= 0.999;
  * code embedding tower fed the same codes: cos(ours, HF BF16) >= 0.999 (no discrete step);
  * end to end: cos(ours, HF FP32) >= cos(HF BF16, HF FP32) - 0.01, and the code agreement with HF FP32 is not more
    than 0.05 below HF BF16's.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "accuracy"))

from tests.test_vision_cpu import _run_tp, _ThreadTP  # noqa: E402

SRC = os.environ.get("MIMO_SRC_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
ASSETS = os.environ.get("MIMO_AUDIO_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets/audio"))
needs = pytest.mark.skipif(not (os.path.isfile(os.path.join(SRC, "audio_tokenizer", "model.safetensors"))
                                and os.path.isdir(ASSETS)), reason="original checkpoint / audio assets not available")
TP = 8
_C = {}


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _mels():
    if "mels" not in _C:
        import audio_ref
        _C["mels"] = audio_ref.served_mels(SRC, ASSETS)
    return _C["mels"]


def _hf(name):
    import audio_ref
    if "hf" not in _C:
        _C["hf"] = audio_ref.hf_towers(SRC)
        _C["hfo"] = {}
    if name not in _C["hfo"]:
        M, se, enc, tok = _C["hf"]
        mel = _mels()[name][0]
        _C["hfo"][name] = {dt: audio_ref.hf_run(M, se, enc, tok, mel, dt) for dt in (torch.float32, torch.bfloat16)}
    return _C["hfo"][name]


def _ours():
    import audio_ref
    if "ours" not in _C:
        grp = _ThreadTP(TP)
        ck = audio_ref.AudioCkpt(SRC)
        mods = [audio_ref.our_audio(SRC, grp.member(r), ck)[0] for r in range(TP)]
        _C["ours"] = (grp, mods, audio_ref.audio_config(SRC))
    return _C["ours"]


def test_lengths_match_processor_formula():
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import MiMoAudioConfig, mel_to_lengths

    cfg = MiMoAudioConfig()
    for L in (1, 2, 3, 201, 251, 1101, 3200):
        n = L + 3 - 3
        n = (n + 2 - 3) // 2 + 1
        n = n // 2 + int(n % 2 != 0)
        assert mel_to_lengths(L, cfg)[2] == -(-n // 4)          # vLLM processor preprocess_audio:497-501
    assert (cfg.t1, cfg.t2, cfg.g) == (1600, 800, 200)


@needs
def test_config_and_mapping():
    import audio_ref
    cfg = audio_ref.audio_config(SRC)
    assert cfg.channels == 20 and cfg.codebook_sizes[:3] == (1024, 1024, 256) and cfg.local_causal is False
    assert [cfg.window_layer(i) for i in range(4)] == [True, False, True, False]
    ck = audio_ref.AudioCkpt(SRC)
    grp = _ThreadTP(1)
    m, _ = audio_ref.our_audio(SRC, grp.member(0), ck)
    assert m.codebooks[0].dtype == torch.float32 and m.conv1_w.dtype == torch.bfloat16


@needs
def test_processor_token_counts():
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import mel_to_lengths
    import audio_ref
    cfg = audio_ref.audio_config(SRC)
    for name, (mel, tl) in _mels().items():
        assert mel_to_lengths(mel.shape[0], cfg)[2] == tl, name


@needs
@pytest.mark.parametrize("name", ["beeps_3.wav", "jfk.wav", "tone_440.wav"])
def test_audio_towers_vs_hf(name):
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import prepare_inputs

    grp, mods, cfg = _ours()
    mel = _mels()[name][0]
    hf = _hf(name)
    inp = prepare_inputs(mel, cfg)
    _, t1, t2, g = inp.pop("lengths")
    feats = _run_tp(grp, mods, lambda m: m.tokenize(inp["mel"], inp["mel_valid"], inp["tok_cos"], inp["tok_sin"], inp["mask_full"],
                                                    inp["mask_window"], inp["t1_valid"])[1][:t2])
    codes = _run_tp(grp, mods, lambda m: m.tokenize(inp["mel"], inp["mel_valid"], inp["tok_cos"], inp["tok_sin"], inp["mask_full"],
                                                    inp["mask_window"], inp["t1_valid"])[0][:t2])
    # the code-embedding tower alone, fed HF BF16's codes (padded to the bucket)
    hc = torch.zeros(cfg.t2, cfg.channels, dtype=torch.int64)
    hc[:t2] = hf[torch.bfloat16]["codes"]
    emb_same_codes = _run_tp(grp, mods, lambda m: m.embed_codes(hc, inp["code_index"], inp["loc_cos"],
                                                                inp["loc_sin"])[:g])
    emb = _run_tp(grp, mods, lambda m: m(**inp)[0][:g])
    r = {"feat_cos_bf16": _cos(feats, hf[torch.bfloat16]["features"]),
         "feat_cos_fp32": _cos(feats, hf[torch.float32]["features"]),
         "codes_eq_hf_bf16": float((codes == hf[torch.bfloat16]["codes"]).float().mean()),
         "codes_eq_hf_fp32": float((codes == hf[torch.float32]["codes"]).float().mean()),
         "hf_bf16_codes_eq_fp32": float((hf[torch.bfloat16]["codes"] == hf[torch.float32]["codes"]).float().mean()),
         "emb_same_codes_cos_bf16": _cos(emb_same_codes, hf[torch.bfloat16]["embeds"]),
         "emb_cos_fp32": _cos(emb, hf[torch.float32]["embeds"]),
         "hf_bf16_emb_cos_fp32": _cos(hf[torch.bfloat16]["embeds"], hf[torch.float32]["embeds"])}
    print(name, {k: round(v, 5) for k, v in r.items()})
    assert emb.shape == hf[torch.float32]["embeds"].shape
    assert r["feat_cos_bf16"] >= 0.999
    assert r["emb_same_codes_cos_bf16"] >= 0.999
    assert r["emb_cos_fp32"] >= r["hf_bf16_emb_cos_fp32"] - 0.01
    assert r["codes_eq_hf_fp32"] >= r["hf_bf16_codes_eq_fp32"] - 0.05


@needs
@pytest.mark.parametrize("on_cpu", [False, True])
def test_embed_audio_writes_encoder_cache(on_cpu):
    """model_mm._embed_audio (docs/design-notes.md#audio-on-device): one cache block per item, rows = the towers' output; both the
    graph path (in-graph index_put) and the worker-CPU path (MIMO_AUDIO_DEVICE=cpu) write the same rows."""
    import types

    import audio_ref
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import prepare_inputs
    from mimo_v2_neuron.model.mimo_v2.model_mm import MiMoV2OmniForCausalLMNeuron as M

    grp = _ThreadTP(1)
    model, cfg = audio_ref.our_audio(SRC, grp.member(0))

    class Cache:
        block_size, num = 1024, 4

        def __init__(self):
            self.buffer = torch.full((self.num, self.block_size, 4096), 7.0, dtype=torch.bfloat16)
            self.alloc = {}

        def allocate(self, h, spans):
            self.alloc[h] = (len(self.alloc), spans)
            return [len(self.alloc) - 1]

    me = types.SimpleNamespace(audio=model, audio_cfg=cfg, audio_compile=None, _audio_fn=None,
                               audio_on_cpu=on_cpu, _audio_cpu=None, rank=0)
    for fn in ("_audio_graph", "_cpu_audio_model", "_run_audio_cpu", "_audio_override"):
        setattr(me, fn, types.MethodType(getattr(M, fn), me))
    cache = Cache()
    mels = [_mels()["beeps_3.wav"][0], _mels()["tone_440.wav"][0]]
    lens = torch.tensor([_mels()["beeps_3.wav"][1], _mels()["tone_440.wav"][1]])
    with torch.no_grad():
        M._embed_audio(me, mels, lens, cache, ["a", "b"])
        for blk, (mel, n) in enumerate(zip(mels, lens.tolist())):
            icfg = cfg
            if on_cpu:   # the CPU path pads only to a multiple of 16 mel frames (model_mm._embed_audio)
                import dataclasses
                q = cfg.stride * cfg.avg_pooler * cfg.group
                icfg = dataclasses.replace(cfg, mel_bucket=min(cfg.mel_bucket, -(-int(mel.shape[0]) // q) * q))
            inp = prepare_inputs(mel, icfg)
            inp.pop("lengths")
            # CPU path runs with MIMO_AUDIO_CPU_THREADS threads: the reference must use the same thread count
            ref = (me._run_audio_cpu(inp) if on_cpu else model(**inp)[0])[:n]
            assert cache.alloc["ab"[blk]][1] == [n]
            assert torch.equal(cache.buffer[blk, :n], ref)
    with pytest.raises(ValueError):
        M._embed_audio(me, mels[:1], torch.tensor([lens[0] + 1]), Cache(), ["c"])


def test_wav_reader_and_load_audio_patch(tmp_path):
    import wave

    import numpy as np

    from mimo_v2_neuron import mm_patches

    x = (np.sin(np.arange(4800) / 10) * 0.5).astype(np.float32)
    p = tmp_path / "t.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.repeat(np.round(x * 32767).astype("<i2"), 2).tobytes())
    y, sr = mm_patches.read_wav(p.read_bytes())
    assert sr == 16000 and len(y) == 4800 and np.abs(y - x).max() < 1e-4
    mm_patches.patch_audio_loading()
    import vllm.multimodal.media.audio as A
    z, sr2 = A.load_audio(str(p), sr=24000)
    assert sr2 == 24000 and abs(len(z) - 7200) <= 1


def test_load_audio_soundfile_matches_read_wav(tmp_path):
    """With soundfile (``$MIMO_WORK/pydeps``) the patched loader decodes WAV identically to read_wav,
    decodes FLAC, and sends non-allowlisted containers (e.g. IRCAM) to the read_wav fallback."""
    import io
    import wave

    import numpy as np

    sf = pytest.importorskip("soundfile")
    from mimo_v2_neuron import mm_patches

    mm_patches.patch_audio_loading()
    import vllm.multimodal.media.audio as A

    rng = np.random.default_rng(0)
    for ch, sw, dt in ((1, 2, "<i2"), (2, 2, "<i2"), (2, 4, "<i4")):
        q = rng.integers(-(2 ** (8 * sw - 1)), 2 ** (8 * sw - 1), size=(3000, ch)).astype(dt)
        p = tmp_path / f"t{ch}{sw}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(16000)
            w.writeframes(q.tobytes())
        ref, rsr = mm_patches.read_wav(p.read_bytes())
        for src in (str(p), io.BytesIO(p.read_bytes())):
            y, ysr = A.load_audio(src, sr=None)
            assert ysr == rsr and y.dtype == np.float32 and np.array_equal(y, ref), (ch, sw)
    x = (np.sin(np.arange(4800) / 10) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="FLAC", subtype="PCM_16")
    y, ysr = A.load_audio(io.BytesIO(buf.getvalue()), sr=None)
    assert ysr == 16000 and len(y) == 4800 and np.abs(y - x).max() < 1e-4
    assert not mm_patches.sf_allowed(b"\x64\xa3\x01\x00")   # IRCAM magic
    with pytest.raises(ValueError, match="not handed to soundfile"):
        A.load_audio(io.BytesIO(b"\x64\xa3\x01\x00" + bytes(1020)), sr=None)


def test_processing_info_audio_budget():
    from mimo_v2_neuron.model.mimo_v2 import factory_mm

    assert factory_mm.MiMoV2NeuronDummyInputsBuilder.AUDIO_DUMMY_SAMPLES == 720000
    from mimo_v2_neuron.model.mimo_v2.audio_encoder import MiMoAudioConfig, mel_to_lengths

    assert mel_to_lengths(1 + 720000 // 240, MiMoAudioConfig())[2] <= MiMoAudioConfig().g


@pytest.mark.parametrize("mode", ["gather", "onehot", "topk_onehot", "first_gather"])
def test_rvq_modes_same_codes(mode, monkeypatch):
    """Every MIMO_AUDIO_RVQ form gives the codes and residual updates of the original top-k + gather form
    (docs/design-notes.md#audio-on-device)."""
    from types import SimpleNamespace

    from mimo_v2_neuron.model.mimo_v2.audio_encoder import _rvq

    g = torch.Generator().manual_seed(0)
    m = SimpleNamespace(codebooks=[torch.randn(n, 64, generator=g) for n in (1024, 256, 128, 128)])
    h = torch.randn(40, 64, generator=g).to(torch.bfloat16)
    monkeypatch.setenv("MIMO_AUDIO_RVQ", "gather")
    ref = _rvq(m, h)
    monkeypatch.setenv("MIMO_AUDIO_RVQ", mode)
    got = _rvq(m, h)
    assert got.dtype == torch.int64 and torch.equal(got, ref)
