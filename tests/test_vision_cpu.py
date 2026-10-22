# SPDX-License-Identifier: Apache-2.0
"""MiMo ViT CPU tests against HF (VLLM_NEURON_CPU_MODE=1; criteria in docs/design-notes.md#vit-accuracy).

  * CPU preprocessing (2D RoPE row / column order, column permutation, time-group bounds, window bounds, per-row
    sink key) equals the HF functions element for element, incl. a grid_t = 4 video and two segments in one block;
  * single blocks (full attention, row-order window + sink, column-order window + sink, last full layer), the patch
    embedding and the merger: three-way comparison HF FP32 / HF BF16 / plugin BF16
    (``vllm_neuron.accuracy.testing.assert_close_three_way`` default thresholds);
  * the whole ViT on the served inputs of the 4 test images and the test video (grid_t = 4): three-way comparison,
    cos(plugin, HF BF16) >= 0.999 and cos(plugin, HF FP32) >= cos(HF BF16, HF FP32) - 0.002 (docs/design-notes.md#vit-accuracy); the
    video is also run split across two vision blocks.
Vision TP = 8 is simulated with 8 threads sharing an all-reduce (one plugin module per rank).

Needs the original checkpoint (``visual.*`` in shard 0) and the mm assets; HF goldens from
``scripts/accuracy/gen_vit_reference.py`` are used when present (``MIMO_VIT_GOLDENS``), else HF runs here.
"""

import os
import sys
import threading

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "accuracy"))

SRC = os.environ.get("MIMO_MM_MODEL_DIR", os.path.join(os.environ.get("MIMO_MODELS", "/mnt/nvme/models"), "MiMo-V2.6-Flash-MOPD"))
ASSETS = os.environ.get("MIMO_MM_ASSETS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "mm_assets"))
GOLDENS = os.environ.get("MIMO_VIT_GOLDENS", os.path.join(os.environ.get("MIMO_WORK", os.path.expanduser("~/mimo-work")), "goldens/mm/vit_goldens.pt"))
TP = int(os.environ.get("MIMO_VIT_TEST_TP", "8"))  # vision TP simulated with threads (the server uses 8)
needs_ckpt = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(SRC, "model.safetensors.index.json"))
         and os.path.isfile(os.path.join(ASSETS, "red_square.png"))),
    reason="original checkpoint or mm assets not present")


def _cfg():
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import MiMoVisionConfig

    import vit_ref

    if os.path.isfile(os.path.join(SRC, "config.json")):
        return MiMoVisionConfig.from_hf(vit_ref.vision_config(SRC))
    import json

    with open(os.path.join(ROOT, "reference/hf/config.json")) as f:
        return MiMoVisionConfig.from_hf(json.load(f)["vision_config"])


def _hf_module():
    from tests.hf_ref import load

    return load()[0]


def _hf_vit_shell(cfg_dict=None):
    """HF ViT with random weights: only its index / RoPE helper functions are used."""
    import json
    from types import SimpleNamespace

    with open(os.path.join(ROOT, "reference/hf/config.json")) as f:
        vc = json.load(f)["vision_config"]
    vc = dict(vc, depth=1, vit_window_attn_types=[-1], fullatt_block_indexes=[0])
    return _hf_module().MiMoVisionTransformer(SimpleNamespace(**vc))


# ── thread-simulated TP group ─────────────────────────────────────────────
class _ThreadTP:
    def __init__(self, n):
        self.n, self.bar, self.buf = n, threading.Barrier(n), [None] * n

    def member(self, rank):
        g = self

        class _M:
            world_size, rank_in_group = g.n, rank

            def all_reduce(self, x):
                g.buf[rank] = x
                g.bar.wait()
                s = g.buf[0].float()
                for t in g.buf[1:]:
                    s = s + t.float()
                g.bar.wait()
                return s.to(x.dtype)

        return _M()


def _run_tp(grp, objs, fn):
    """``fn(objs[r])`` on one thread per rank (``objs`` already wired to ``grp``); returns rank 0's result."""
    n = grp.n
    res, err = [None] * n, []

    def work(r):
        try:
            with torch.no_grad():
                res[r] = fn(objs[r])
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            grp.bar.abort()

    ths = [threading.Thread(target=work, args=(r,)) for r in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if err:
        raise err[0]
    for r in range(1, n):
        assert torch.equal(res[0], res[r]), f"rank {r} result differs from rank 0"
    return res[0]


def _three_way(base, exp, act, name):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    for b, e, a in zip(base, exp, act):
        assert b.shape == e.shape == a.shape, (name, b.shape, e.shape, a.shape)
        assert torch.isfinite(a.float()).all(), name
    r = assert_close_three_way(base, exp, act, name=name)
    cos = [_cos(a, e) for e, a in zip(exp, act)]
    rel = [float((a.float() - b).norm() / b.norm()) for b, a in zip(base, act)]
    rel_hf = [float((e.float() - b).norm() / b.norm()) for b, e in zip(base, exp)]
    print(f"[{name}] sigma_ratio={r.sigma_ratio:.4f} bc={r.bc:.4f} linf_ratio={r.linf_ratio:.3f} "
          f"l2_ratio={r.l2_ratio:.3f} cos(ours,hf_bf16)={min(cos):.6f} relL2(ours vs fp32)={max(rel):.5f} "
          f"relL2(hf_bf16 vs fp32)={max(rel_hf):.5f}")
    assert r.passed, r.summary()
    return r, min(cos)


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


# ── 1. CPU preprocessing == HF, element for element ──────────────────────
def _hf_reference_tables(vit, grids):
    g = torch.tensor(grids)
    rot = vit.rot_pos_emb(g)
    emb = torch.cat((rot, rot), dim=-1)
    widx = vit.get_window_index_1d(g, col=True)
    col_emb = vit.apply_index(emb, widx)
    patch_idx = vit.apply_index(torch.arange(emb.shape[0]), widx)
    cu = torch.nn.functional.pad(torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0]).cumsum(0), (1, 0))
    return emb, col_emb, patch_idx, cu


def _hf_masks(cu, window, total, sink=True):
    """Block-diagonal HF masks: visible[q, k] for window / full layers and the sink key per row."""
    win = torch.zeros(total, total, dtype=torch.bool)
    full = torch.zeros(total, total, dtype=torch.bool)
    sink_col = torch.full((total,), -1, dtype=torch.long)
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        n = b - a
        i = torch.arange(n)
        full[a:b, a:b] = True
        win[a:b, a:b] = ~((i.unsqueeze(1) - i.unsqueeze(0)).abs() > window)  # HF _build_window_mask
        sink_col[a:b] = a  # HF sink_bias[..., 0] of the chunk
    return win, full, sink_col


@pytest.mark.parametrize("grids,block", [([[1, 28, 28]], 1024), ([[4, 24, 24]], 4096), ([[1, 16, 40]], 640),
                                          ([[1, 8, 12], [2, 4, 6]], 256)])
def test_preprocess_matches_hf(grids, block):
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import block_inputs

    cfg = _cfg()
    vit = _hf_vit_shell()
    emb, col_emb, patch_idx, cu = _hf_reference_tables(vit, grids)
    n = emb.shape[0]
    d = block_inputs([tuple(x) for x in grids], block, cfg)
    assert torch.equal(d["cos_row"][:n], emb.cos()) and torch.equal(d["sin_row"][:n], emb.sin())
    assert torch.equal(d["cos_col"][:n], col_emb.cos()) and torch.equal(d["sin_col"][:n], col_emb.sin())
    assert torch.equal(d["col_index"][:n], patch_idx)
    assert torch.equal(d["row_index"][:n], torch.argsort(patch_idx))
    j = torch.arange(block)
    vis = lambda lo, hi: (j.view(1, -1) >= lo) & (j.view(1, -1) < hi)  # noqa: E731
    win, full, sink_col = _hf_masks(cu, cfg.window, n)
    assert torch.equal(vis(d["full_min"], d["full_max"])[:n, :n], full)
    assert torch.equal(vis(d["win_min"], d["win_max"])[:n, :n], win)
    assert not vis(d["full_min"], d["full_max"])[:n, n:].any()
    assert torch.equal(d["full_min"][:n, 0].long(), sink_col)
    # column order: the bounds are those of the permuted sequence -> identical (permutation stays in its group)
    cu_l = cu.tolist()
    for a, b in zip(cu_l[:-1], cu_l[1:]):
        assert ((patch_idx[a:b] >= a) & (patch_idx[a:b] < b)).all()
    # padding rows: only themselves, identity permutation
    if n < block:
        p = torch.arange(n, block)
        assert torch.equal(d["full_min"][n:, 0].long(), p) and torch.equal(d["full_max"][n:, 0].long(), p + 1)
        assert torch.equal(d["win_min"][n:, 0].long(), p) and torch.equal(d["col_index"][n:], p)


def test_pack_split_video_at_group_boundaries():
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import block_inputs, pack_items, unpack_merged

    cfg = _cfg()
    grid = torch.tensor([[4, 24, 24], [1, 16, 16]])
    total = int((grid.prod(1)).sum())
    pv = torch.arange(total, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    inp, layout = pack_items(pv, grid, 1152, cfg, num_blocks=4)
    assert layout == [[(0, 0, 1152), (1, 0, 1152)], [(2, 0, 256)]]
    assert inp["pixel_values"].shape == (4, 1152, 3)
    ref = block_inputs([(2, 24, 24)], 1152, cfg)
    for b in (0, 1):
        for k, v in ref.items():
            assert torch.equal(inp[k][b], v), (b, k)
    assert torch.equal(inp["pixel_values"][1, :, 0], torch.arange(1152, 2304, dtype=torch.float32))
    assert torch.equal(inp["full_max"][3, :, 0].long(), torch.arange(1, 1153))  # all-padding block
    merged = inp["pixel_values"][:, ::4, :1]
    items = unpack_merged(merged, layout, 4)
    assert torch.equal(items[0][:, 0], torch.arange(0, 2304, 4, dtype=torch.float32))
    assert torch.equal(items[1][:, 0], torch.arange(2304, 2560, 4, dtype=torch.float32))
    with pytest.raises(ValueError):
        pack_items(pv, grid, 1152, cfg, num_blocks=2)


# ── 2. modules: three-way vs HF ───────────────────────────────────────────
_STATE = {}


def _ckpt():
    import vit_ref

    if "ckpt" not in _STATE:
        _STATE["ckpt"] = vit_ref.VisualCkpt(SRC)
    return _STATE["ckpt"]


def _hf(dtype):
    import vit_ref

    key = ("hf", dtype)
    if key not in _STATE:
        _STATE[key] = vit_ref.hf_vit(SRC, dtype, _ckpt())
    return _STATE[key]


def _items():
    import vit_ref

    if "items" not in _STATE:
        _STATE["items"] = {it["name"]: it for it in vit_ref.served_inputs(SRC, ASSETS)}
    return _STATE["items"]


def _ours(rank_group):
    import vit_ref

    return vit_ref.our_vit(SRC, rank_group, _ckpt())[0]


def _ours_tp():
    """8 rank models (built once) with a fresh thread group per call."""
    if "ours" not in _STATE:
        grp = _ThreadTP(TP)
        _STATE["ours"] = [_ours(grp.member(r)) for r in range(TP)]
    return _STATE["ours"]


def _run_ours(fn):
    models = _ours_tp()
    grp = _ThreadTP(TP)
    for r, m in enumerate(models):
        mem = grp.member(r)
        for mod in m.modules():
            if hasattr(mod, "tp_group"):
                mod.tp_group = mem
    return _run_tp(grp, models, fn)


def _hf_layer_io():
    """HF FP32 per-layer (inputs, outputs) for red_square, BF16 (from goldens, else computed)."""
    import vit_ref

    if "layer_io" in _STATE:
        return _STATE["layer_io"]
    g = _goldens()
    rec = g["items"].get("red_square.png", {}) if g is not None else {}
    if "layers_in_fp32" in rec:
        ins, outs = rec["layers_in_fp32"], rec["layers_out_fp32"]
    else:
        it = _items()["red_square.png"]
        _, ins, outs = vit_ref.hf_run(_hf(torch.float32), it["pixel_values"], it["grid_thw"], record=True)
    _STATE["layer_io"] = ([x.to(torch.bfloat16) for x in ins], [x.to(torch.bfloat16) for x in outs])
    return _STATE["layer_io"]


def _goldens():
    if "goldens" not in _STATE:
        _STATE["goldens"] = torch.load(GOLDENS, weights_only=False) if os.path.isfile(GOLDENS) else None
    return _STATE["goldens"]


@needs_ckpt
@pytest.mark.parametrize("layer", [0, 1, 5, 8, 27])
def test_block_three_way(layer):
    """0 full, 1 row-order window + sink, 5 first column-order layer, 8 last column-order layer, 27 last full."""
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import block_inputs

    it = _items()["red_square.png"]
    grid = it["grid_thw"]
    x = _hf_layer_io()[0][layer]  # [784, 1280] BF16, in the order layer `layer` runs in
    n = x.shape[0]
    cfg = _cfg()
    vit32 = _hf(torch.float32)
    emb, col_emb, _, cu = _hf_reference_tables(vit32, grid.tolist())
    col = cfg.vit_window_attn_types[layer] == 1
    e = col_emb if col else emb
    full = layer in cfg.fullatt_block_indexes
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        blk = _hf(dt).blocks[layer]
        with torch.no_grad():
            outs[dt] = blk(x.to(dt), cu_seqlens=cu.to(torch.int32), position_embeddings=(e.cos(), e.sin()),
                           full_attn=full).float()
    d = block_inputs([tuple(grid[0].tolist())], 1024, cfg)
    xb = torch.zeros(1, 1024, x.shape[1], dtype=torch.bfloat16)
    xb[0, :n] = x
    cs = ("cos_col", "sin_col") if col else ("cos_row", "sin_row")
    lo, hi = ("full_min", "full_max") if full else ("win_min", "win_max")
    args = [d[k].unsqueeze(0) for k in (cs[0], cs[1], lo, hi, "full_min")]
    ours = _run_ours(lambda m: m.blocks[layer](xb, *args))[0, :n]
    assert (_ours_tp()[0].blocks[layer].attn.sinks is not None) == (not full)
    _three_way([outs[torch.float32]], [outs[torch.bfloat16]], [ours], f"vit.block{layer}.tp{TP}")


@needs_ckpt
def test_patch_embed_and_merger_three_way():
    it = _items()["red_square.png"]
    pv = it["pixel_values"]
    merger_in = _hf_layer_io()[1][-1]  # HF FP32 output of block 27 (row order), BF16
    ref_pe, ref_mg = {}, {}
    with torch.no_grad():
        for dt in (torch.float32, torch.bfloat16):
            ref_pe[dt] = _hf(dt).patch_embed(pv.to(dt)).float()
            ref_mg[dt] = _hf(dt).merger(merger_in.to(dt)).float()
    pe = _run_ours(lambda m: pv.unsqueeze(0).to(torch.bfloat16) @ m.patch_weight)[0]
    mg = _run_ours(lambda m: m.merger(merger_in.unsqueeze(0)))[0]
    _three_way([ref_pe[torch.float32]], [ref_pe[torch.bfloat16]], [pe], "vit.patch_embed")
    _three_way([ref_mg[torch.float32]], [ref_mg[torch.bfloat16]], [mg], f"vit.merger.tp{TP}")


# ── 3. whole ViT ──────────────────────────────────────────────────────────
def _hf_final(name):
    import vit_ref

    g = _goldens()
    if g is not None and name in g["items"]:
        r = g["items"][name]
        assert torch.equal(r["pixel_values"], _items()[name]["pixel_values"].to(torch.bfloat16)), \
            "goldens were made from different inputs"
        return r["hf_fp32"].float(), r["hf_bf16"].float()
    it = _items()[name]
    return tuple(vit_ref.hf_run(_hf(dt), it["pixel_values"], it["grid_thw"])[0]
                 for dt in (torch.float32, torch.bfloat16))


def _ours_final(name, block_size, num_blocks=None):
    from mimo_v2_neuron.model.mimo_v2.vision_encoder import pack_items, unpack_merged

    it = _items()[name]
    cfg = _cfg()
    inp, layout = pack_items(it["pixel_values"], it["grid_thw"], block_size, cfg, num_blocks)
    keys = ["pixel_values", "cos_row", "sin_row", "cos_col", "sin_col", "col_index", "row_index",
            "full_min", "full_max", "win_min", "win_max"]
    merged = _run_ours(lambda m: m(*[inp[k] for k in keys]))
    return unpack_merged(merged, layout, cfg.merge_unit)[0]


@needs_ckpt
@pytest.mark.parametrize("name", ["red_square.png", "number_739.png", "three_circles.png", "word_trainium.png",
                                  "digits_1234.mp4"])
def test_vit_end_to_end(name):
    fp32, bf16 = _hf_final(name)
    ours = _ours_final(name, 4096)
    _, cos = _three_way([fp32], [bf16], [ours], f"vit.e2e.{name}.tp{TP}")
    # Design gate: cos(ours, HF BF16) >= 0.999. This sits close to the BF16 noise floor of the 28-layer ViT
    # (~3e5 outlier activations in the last blocks; HF BF16 is cos 0.9976-0.9992 from HF FP32 and the same plugin
    # maths split over two blocks differs by cos ~0.9997), so also check the noise-robust form used as the ViT
    # fallback criterion (docs/design-notes.md#vit-accuracy): ours is no further from HF FP32 than HF BF16 is, within 0.002 in cosine.
    c_ours, c_hf = _cos(ours, fp32), _cos(bf16, fp32)
    print(f"[vit.e2e.{name}] cos(ours,fp32)={c_ours:.6f} cos(hf_bf16,fp32)={c_hf:.6f} cos(ours,hf_bf16)={cos:.6f}")
    assert cos >= 0.999, f"cos(ours, HF BF16) = {cos:.6f} < 0.999"
    assert c_ours >= c_hf - 0.002, f"cos(ours, FP32) {c_ours:.6f} < cos(HF BF16, FP32) {c_hf:.6f} - 0.002"
    if name.endswith(".mp4"):
        assert _items()[name]["grid_thw"][0, 0] == 4
        split = _ours_final(name, 1152, num_blocks=2)  # 2 time groups per block
        c = _cos(split, ours)
        print(f"[vit.e2e.video.split2] cos(split, single block)={c:.6f} max_abs={float((split - ours).abs().max()):.4g}")
        assert c >= 0.999  # same maths, different blocking: rounding noise only
        _three_way([fp32], [bf16], [split], f"vit.e2e.{name}.split2.tp{TP}")
