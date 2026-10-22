# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vllm-project/vllm-neuron (Apache-2.0): vllm_neuron/model/qwen3_vl/model_bf16.py
#   (embed_multimodal) and qwen3_vl/utils/merge_vision_embeds.py (embedding merge); modified. See NOTICE.
"""MiMo-V2 Omni (image / video) model for vllm-neuron: text model + MiMo ViT + encoder-cache plumbing.

See docs/design-notes.md#omni-architecture. The text decoder is :class:`MiMoV2ForCausalLMNeuron` unchanged; prefill
calls receive ``vision_embedding_blocks`` / ``vision_positions`` from the vllm-neuron runner
(``NeuronModelRunner._gather_mm_embeddings``) and :func:`merge_vision_embeddings` scatters the cached merger
outputs into the token embeddings (positions stay one-dimensional, like HF ``cache_position``; no M-RoPE).

``embed_multimodal`` follows ``vllm_neuron/model/qwen3_vl/model_bf16.py`` (allocate encoder-cache blocks,
pick the vision bucket, CPU pre-processing, run the separately compiled ``visual`` graph that scatter-writes
its output into the cache). Layout differences to Qwen3-VL: one item per vision block run (no FFD packing);
a video is split at time-group boundaries (2 frames per group, attention never crosses a group), so every
vision block maps to exactly one cache block of ``vision_attention_block_size / 4`` merged rows.

Audio (docs/design-notes.md#audio-input): with an ``audio_config`` the model also owns ``audio`` (:class:`~.audio_encoder.MiMoAudioModel`,
both towers in the vision TP group). The runner hands audio items to ``embed_multimodal`` as ``audio_features`` (the
processor's log-mel) and ``audio_token_lens``; each item runs one fixed-shape graph (mel bucket 3200 frames) that
writes its ``[G, 4096]`` rows into one encoder-cache block. The graph is compiled on first use with the runner's
compile options (``runner_patches`` hands them to the model); on CPU it runs eagerly.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from ... import switches
from .audio_encoder import MiMoAudioModel, prepare_inputs as audio_prepare_inputs
from .model import MiMoV2ForCausalLMNeuron
from .vision_encoder import INPUT_KEYS, MiMoVisionConfig, MiMoVisionModel, num_vision_blocks, pack_items

logger = logging.getLogger(__name__)
_P = "MiMoV2 Omni Neuron: "


def merge_vision_embeddings(hidden_states: torch.Tensor, vision_embedding_blocks, vision_positions: torch.Tensor,
                            rank) -> torch.Tensor:
    """Scatter encoder-cache rows into this SP rank's embeddings.

    ``hidden_states [local_len, H]`` (sequence-parallel slice of rank ``rank``); ``vision_embedding_blocks``
    tuple of ``[cache_block, H]`` views; ``vision_positions [num_blocks, cache_block]`` global batch positions
    (sentinel = total tokens). Same maths as ``qwen3_vl/utils/merge_vision_embeds.py`` (no deepstack), but
    ``rank`` may be the runner's rank tensor (``[1]``) so the traced graph does not depend on the process rank.
    """
    flat = torch.stack(vision_embedding_blocks).reshape(-1, vision_embedding_blocks[0].shape[-1])
    pos = vision_positions.reshape(-1)
    local_len = hidden_states.shape[0]
    if torch.is_tensor(rank):
        start = rank.reshape(-1)[:1].to(pos.dtype) * local_len
    else:
        start = int(rank) * local_len
    local = pos - start
    local = torch.where((local < 0) | (local >= local_len), torch.full_like(local, local_len), local)
    width = hidden_states.shape[-1]
    dummy = torch.zeros(1, width, dtype=hidden_states.dtype, device=hidden_states.device)
    out = torch.cat([hidden_states, dummy], dim=0)
    out.index_put_((local,), flat[:, :width].to(hidden_states.dtype))
    return out[:local_len]


def vision_block_plan(grid_rows, block_size: int, merge_unit: int):
    """Per item: the merged-token count of each vision block it occupies (same split as :func:`pack_items`)."""
    plan = []
    for t, h, w in grid_rows:
        g = h * w
        if g > block_size:
            raise ValueError(f"{_P}one time group ({h}x{w} = {g} patches) exceeds vision_attention_block_size "
                             f"{block_size}; lower processor_config.image_max_pixels / video_max_pixels")
        per = block_size // g
        spans, k = [], 0
        while k < t:
            n = min(per, t - k)
            spans.append(n * g // merge_unit)
            k += n
        plan.append(spans)
    return plan


class MiMoV2OmniForCausalLMNeuron(MiMoV2ForCausalLMNeuron):
    """Text model + ``visual`` (:class:`MiMoVisionModel`, vision TP / DP groups); BF16 only."""

    def __init__(self, config, vision_cfg: MiMoVisionConfig, vision_neuron_config, audio_cfg=None):
        super().__init__(config)
        self.vision_cfg = vision_cfg
        self.vision_neuron_config = vision_neuron_config
        # docs/design-notes.md#vision-dp-group: the vision DP all-gather over the platform's column group (ranks 0, 8, ..., 56 for TP 8 / DP 8) fails
        # to load on trn2 ("no_hier no_mesh replica-group"), so by default every vision TP group encodes all blocks
        # of the bucket itself (no DP collective; block counts still follow vision_neuron_config.dp_size).
        # MIMO_VIT_DP_GROUP=1 restores the DP split (CPU tests). MIMO_VIT_DP_GROUP=world (docs/design-notes.md#vision-dp-group): DP split with one
        # all-gather over the whole TP world instead of the column group (the default on the device; 0 in
        # CPU mode).
        _vdp = switches.get("MIMO_VIT_DP_GROUP")
        _world = None
        if _vdp == "world":
            from vllm.distributed import get_tp_group

            _world = get_tp_group()
        self.visual = MiMoVisionModel(vision_cfg, dtype=torch.bfloat16, use_dp_group=_vdp == "1", world_group=_world)
        logger.info(_P + "vision tower: depth %d, vision tp %d, dp %d, block %d, buckets %s",
                    vision_cfg.depth, self.visual.tp_group.world_size, self.visual.dp_size,
                    vision_neuron_config.vision_attention_block_size, vision_neuron_config.num_vision_tokens_buckets)
        self.audio_cfg = audio_cfg
        # an audio graph compiled outside the runner's capture flow with vision-TP collectives fails to load
        # on the device ("nrt_load_collectives: Failed to load collectives for model"), so by default every rank
        # runs both towers unsharded (no collective; ~1.1 GB of BF16 weights per core). MIMO_AUDIO_TP=vision
        # shards them over the vision TP group (CPU tests).
        self.audio = None
        if audio_cfg is not None:
            from types import SimpleNamespace

            grp = None if os.environ.get("MIMO_AUDIO_TP", "replicated") == "vision" else \
                SimpleNamespace(world_size=1, rank_in_group=0)
            self.audio = MiMoAudioModel(audio_cfg, torch.bfloat16, tp_group=grp)
        self.audio_compile = None          # set by runner_patches: callable(fn) -> compiled fn (Neuron only)
        self._audio_fn = None
        # docs/design-notes.md#audio-on-device: the compiled audio graph first failed to load on the device (NRT "Failed to load subgraph" /
        # "Failed to load collectives"); the real cause was the RVQ top-k (TDRV "invalid offset ...
        # create_data_refill_rings"), fixed in audio_encoder._rvq. So by default the towers run on the device
        # (one fixed-shape graph per item, compiled at the worker's vision warmup). MIMO_AUDIO_DEVICE=cpu runs them on
        # the worker CPU instead (same code, BF16) and only copies the result into the device encoder cache.
        self.audio_on_cpu = os.environ.get("MIMO_AUDIO_DEVICE", "neuron") == "cpu"
        self._audio_cpu = None
        if self.audio is not None:
            logger.info(_P + "audio towers: %d + %d layers, vision tp %d, mel bucket %d (%d tokens max)",
                        audio_cfg.tok_layers, audio_cfg.local_layers, self.audio.group.world_size,
                        audio_cfg.mel_bucket, audio_cfg.g)

    # ── weights ──────────────────────────────────────────────────────────
    def weight_mappings(self) -> dict:
        m = super().weight_mappings()
        m.update({f"visual.{k}": v for k, v in self.visual.mappings().items()})
        if self.audio is not None:
            m.update({f"audio.{k}": v for k, v in self.audio.checkpoint_keys().items()})
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        import json
        import os

        with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
            keys = json.load(f)["weight_map"].keys()
        MiMoVisionModel.check_no_merger_bias([k for k in keys if k.startswith("visual.")])
        super().load_weights(checkpoint_path, device, cache_dir)

    # ── vision encoder ───────────────────────────────────────────────────
    def _bucket_blocks(self, required_blocks: int) -> int:
        vnc = self.vision_neuron_config
        bs = vnc.vision_attention_block_size
        for bucket in sorted(vnc.num_vision_tokens_buckets):
            nb = num_vision_blocks(bucket, bs, vnc.dp_size)
            if nb >= required_blocks:
                return nb
        raise ValueError(f"{_P}request needs {required_blocks} vision blocks of {bs} patches; the largest bucket "
                         f"{max(vnc.num_vision_tokens_buckets)} has "
                         f"{num_vision_blocks(max(vnc.num_vision_tokens_buckets), bs, vnc.dp_size)}")

    def embed_multimodal(self, pixel_values=None, image_grid_thw=None, encoder_cache=None, mm_hashes=None,
                         pixel_values_videos=None, video_grid_thw=None, **kwargs) -> None:
        """Encode the images (or videos, or audios) of one modality group and write them into the on-device
        encoder cache."""
        if kwargs.get("audio_features") is not None:
            return self._embed_audio(kwargs["audio_features"], kwargs.get("audio_token_lens"), encoder_cache,
                                     mm_hashes)
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError(f"{_P}embed_multimodal got both images and videos; the runner groups by modality")
        if pixel_values is None:
            pixel_values, image_grid_thw = pixel_values_videos, video_grid_thw
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(f"{_P}embed_multimodal needs (pixel_values, image_grid_thw) or "
                             f"(pixel_values_videos, video_grid_thw); got keys {sorted(kwargs)}")
        t0 = time.perf_counter()
        pixel_values = pixel_values.to("cpu")   # CPU pre-processing (block packing) like qwen3_vl
        image_grid_thw = image_grid_thw.to("cpu")
        cfg, vnc = self.vision_cfg, self.vision_neuron_config
        bs = vnc.vision_attention_block_size
        grid_rows = [tuple(int(v) for v in r) for r in image_grid_thw.tolist()]
        if len(grid_rows) != len(mm_hashes):
            raise ValueError(f"{_P}{len(grid_rows)} grid rows for {len(mm_hashes)} mm_hashes")
        plan = vision_block_plan(grid_rows, bs, cfg.merge_unit)
        required = sum(len(p) for p in plan)
        max_nb = num_vision_blocks(max(vnc.num_vision_tokens_buckets), bs, vnc.dp_size)
        if required > max_nb and len(grid_rows) > 1:
            # With larger buckets one encoder step can schedule several items (budget in tokens) whose blocks exceed
            # the largest vision graph; encode them in item groups that fit (items never share a block;
            # docs/design-notes.md#vision-request-blocks).
            start, row0 = 0, 0
            while start < len(grid_rows):
                end, nb = start, 0
                while end < len(grid_rows) and (end == start or nb + len(plan[end]) <= max_nb):
                    nb += len(plan[end])
                    end += 1
                rows = sum(t * h * w for t, h, w in grid_rows[start:end])
                self.embed_multimodal(pixel_values=pixel_values[row0:row0 + rows],
                                      image_grid_thw=image_grid_thw[start:end], encoder_cache=encoder_cache,
                                      mm_hashes=list(mm_hashes)[start:end])
                start, row0 = end, row0 + rows
            return
        num_blocks = self._bucket_blocks(required)
        cache_blocks = [encoder_cache.allocate(h, spans) for h, spans in zip(mm_hashes, plan)]
        inputs, _layout = pack_items(pixel_values.to(torch.bfloat16), grid_rows, bs, cfg, num_blocks=num_blocks)
        flat = [b for blocks in cache_blocks for b in blocks]
        write_ids = torch.tensor(flat + [encoder_cache.scratch_block_id] * (num_blocks - len(flat)),
                                 dtype=torch.int64)
        t1 = time.perf_counter()
        device = next(self.visual.parameters()).device
        self.visual(**{k: v.to(device) for k, v in inputs.items()}, encoder_cache_buffer=encoder_cache.buffer,
                    write_block_ids=write_ids.to(device))
        dump = os.environ.get("MIMO_VIT_DUMP_DIR")
        if dump and self.rank == 0:   # device ViT check (docs/design-notes.md#vit-accuracy): merger output rows as written to the cache
            os.makedirs(dump, exist_ok=True)
            items, sums, row0 = [], [], 0
            for (t, h, w), blocks, spans in zip(grid_rows, cache_blocks, plan):
                items.append(torch.cat([encoder_cache.buffer[b, :n].to("cpu") for b, n in zip(blocks, spans)], 0))
                sums.append(float(pixel_values[row0:row0 + t * h * w].float().sum()))   # identifies the item
                row0 += t * h * w
            self._vit_dumps = getattr(self, "_vit_dumps", 0) + 1
            path = os.path.join(dump, f"vit_{self._vit_dumps:03d}.pt")
            torch.save({"grid_thw": grid_rows, "mm_hashes": list(mm_hashes), "embeds": items, "pv_sums": sums}, path)
            logger.warning(_P + "MIMO_VIT_DUMP_DIR: wrote %s (%d items)", path, len(items))
        logger.debug(_P + "embed_multimodal %d items, %d vision blocks (bucket %d): cpu %.1f ms, device %.1f ms",
                     len(grid_rows), required, num_blocks, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3)

    # ── audio encoder ────────────────────────────────────────────────────
    def _audio_graph(self, mel, mel_valid, tok_cos, tok_sin, loc_cos, loc_sin, mask_full, mask_window, t1_valid,
                     code_index, encoder_cache_buffer, write_block_id):
        emb, _ = self.audio(mel, mel_valid, tok_cos, tok_sin, loc_cos, loc_sin, mask_full, mask_window, t1_valid,
                            code_index)
        g = emb.shape[0]
        rows = write_block_id.reshape(1).expand(g)
        cols = torch.arange(g, device=emb.device)
        encoder_cache_buffer.index_put_((rows, cols), emb.to(encoder_cache_buffer.dtype)[:, :encoder_cache_buffer.shape[-1]])
        return encoder_cache_buffer

    def _embed_audio(self, audio_features, audio_token_lens, encoder_cache, mm_hashes) -> None:
        if self.audio is None:
            raise ValueError(f"{_P}got audio inputs but the model directory has no audio_config "
                             "(build it with tools/make_mm_checkpoint.py --audio)")
        mels = list(audio_features) if torch.is_tensor(audio_features) else list(audio_features)
        if len(mels) != len(mm_hashes):
            raise ValueError(f"{_P}{len(mels)} audio items for {len(mm_hashes)} mm_hashes")
        lens = None
        if audio_token_lens is not None:
            lens = [int(x) for x in (audio_token_lens.reshape(-1).tolist() if torch.is_tensor(audio_token_lens)
                                     else list(audio_token_lens))]
        cfg = self.audio_cfg
        if encoder_cache.block_size < cfg.g:
            raise ValueError(f"{_P}encoder cache block {encoder_cache.block_size} < audio bucket {cfg.g} tokens")
        device = next(self.audio.parameters()).device
        if self.audio_on_cpu:
            self._cpu_audio_model()
        elif self._audio_fn is None:
            self._audio_fn = self.audio_compile(self._audio_graph) if self.audio_compile else self._audio_graph
        for i, (h, mel) in enumerate(zip(mm_hashes, mels)):
            t0 = time.perf_counter()
            icfg = cfg
            if self.audio_on_cpu:   # no fixed shape needed on the CPU: pad only to a multiple of 16 mel frames
                import dataclasses
                q = cfg.stride * cfg.avg_pooler * cfg.group
                icfg = dataclasses.replace(cfg, mel_bucket=min(cfg.mel_bucket, -(-int(mel.shape[0]) // q) * q))
            inp = audio_prepare_inputs(mel.to("cpu"), icfg)
            _, _, _, g = inp.pop("lengths")
            if lens is not None and lens[i] != g:
                raise ValueError(f"{_P}audio item {i}: processor says {lens[i]} tokens, encoder gives {g}")
            block = encoder_cache.allocate(h, [g])
            if len(block) != 1:
                raise ValueError(f"{_P}audio item needs one encoder-cache block, got {block}")
            override = self._audio_override(mel, g)
            if override is not None or self.audio_on_cpu:
                emb = override if override is not None else self._run_audio_cpu(inp)[:g]
                buf = encoder_cache.buffer
                buf[block[0], :g] = emb.to(buf.dtype)[:, :buf.shape[-1]].to(buf.device)
            else:
                self._audio_fn(**{k: v.to(device) for k, v in inp.items()},
                               encoder_cache_buffer=encoder_cache.buffer,
                               write_block_id=torch.tensor([block[0]], dtype=torch.int64).to(device))
            dump = os.environ.get("MIMO_AUDIO_DUMP_DIR")
            if dump and self.rank == 0:   # device audio check (docs/design-notes.md#audio-on-device): rows as written to the encoder cache
                os.makedirs(dump, exist_ok=True)
                self._audio_dumps = getattr(self, "_audio_dumps", 0) + 1
                path = os.path.join(dump, f"audio_{self._audio_dumps:03d}.pt")
                torch.save({"mm_hash": h, "embeds": encoder_cache.buffer[block[0], :g].to("cpu"),
                            "mel_sum": float(mel.float().sum()), "mel_frames": int(mel.shape[0])}, path)
                logger.warning(_P + "MIMO_AUDIO_DUMP_DIR: wrote %s (%d tokens)", path, g)
            logger.debug(_P + "audio item %d: %d mel frames -> %d tokens in %.1f ms", i, mel.shape[0], g,
                         (time.perf_counter() - t0) * 1e3)

    def _audio_override(self, mel, g):
        """Debug / validation (docs/design-notes.md#audio-criteria): MIMO_AUDIO_EMBED_OVERRIDE=<goldens.pt> replaces the towers' output by
        the HF BF16 audio embeddings of the golden item with the same mel (sum match), so the text decoder with
        audio rows can be checked against HF without the RVQ code flips of the towers. Off by default."""
        path = os.environ.get("MIMO_AUDIO_EMBED_OVERRIDE")
        if not path:
            return None
        if getattr(self, "_audio_override_tab", None) is None:
            gd = torch.load(path, map_location="cpu", weights_only=False)
            self._audio_override_tab = [(float(e["mel"].float().sum()), e["audio_bf16"]) for e in gd["mm"].values()]
        want = float(mel.float().sum())
        err, emb = min(((abs(s - want), e) for s, e in self._audio_override_tab), key=lambda c: c[0])
        if err > 1e-3 * abs(want) + 1e-2 or emb.shape[0] != g:
            raise ValueError(f"{_P}MIMO_AUDIO_EMBED_OVERRIDE: no golden audio matches (mel sum {want})")
        logger.warning(_P + "MIMO_AUDIO_EMBED_OVERRIDE: using HF BF16 audio embeddings (%d rows)", g)
        return emb.to(torch.bfloat16)

    def _cpu_audio_model(self):
        """CPU copy of the audio towers (weights copied from the device once per worker)."""
        if self._audio_cpu is None:
            import copy

            t0 = time.perf_counter()
            m = copy.deepcopy(self.audio.to("cpu")) if next(self.audio.parameters()).device.type == "cpu" else None
            if m is None:
                sd = {k: v.detach().to("cpu") for k, v in self.audio.state_dict().items()}
                from types import SimpleNamespace

                m = MiMoAudioModel(self.audio_cfg, torch.bfloat16, tp_group=SimpleNamespace(world_size=1,
                                                                                               rank_in_group=0))
                m.load_state_dict(sd, assign=True)
            self._audio_cpu = m.eval()
            logger.info(_P + "audio towers on the worker CPU (weights copied in %.1f s)", time.perf_counter() - t0)
        return self._audio_cpu

    def _run_audio_cpu(self, inp: dict) -> torch.Tensor:
        n = int(os.environ.get("MIMO_AUDIO_CPU_THREADS", "2"))
        old = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            with torch.no_grad():
                return self._cpu_audio_model()(**inp)[0]
        finally:
            torch.set_num_threads(old)

    def warmup_audio(self, encoder_cache) -> None:
        """Prepare the audio path during the worker's vision warmup (runner_patches hook): CPU mode copies the
        weights to the CPU; device mode compiles + loads the graph with a silent full-bucket mel written to the
        scratch block, so the first request does not compile."""
        if self.audio is None:
            return
        if self.audio_on_cpu:
            self._cpu_audio_model()
            return
        t0 = time.perf_counter()
        device = next(self.audio.parameters()).device
        if self._audio_fn is None:
            self._audio_fn = self.audio_compile(self._audio_graph) if self.audio_compile else self._audio_graph
        inp = audio_prepare_inputs(torch.zeros(self.audio_cfg.mel_bucket, self.audio_cfg.n_mels), self.audio_cfg)
        inp.pop("lengths")
        self._audio_fn(**{k: v.to(device) for k, v in inp.items()}, encoder_cache_buffer=encoder_cache.buffer,
                       write_block_id=torch.tensor([encoder_cache.scratch_block_id], dtype=torch.int64).to(device))
        logger.info(_P + "audio warmup done in %.1f s", time.perf_counter() - t0)

    def build_vision_synthetic_inputs(self, bucket: int, vision_neuron_config, device) -> dict:
        """Shape-only inputs of ``visual`` for one bucket (the worker adds encoder_cache_buffer / write_block_ids)."""
        cfg = self.vision_cfg
        s = vision_neuron_config.vision_attention_block_size
        n = num_vision_blocks(bucket, s, vision_neuron_config.dp_size)
        z = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        out = {"pixel_values": z(n, s, cfg.patch_dim, dtype=torch.bfloat16)}
        for k in INPUT_KEYS:
            if k.startswith(("cos", "sin")):
                out[k] = z(n, s, cfg.head_dim, dtype=torch.float32)
            elif k.endswith("_index"):
                out[k] = z(n, s, dtype=torch.int64)
            else:
                out[k] = z(n, s, 1, dtype=torch.int32)
        return out
