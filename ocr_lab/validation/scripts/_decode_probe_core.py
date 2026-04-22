"""Shared core for the three B2 decode-divergence probes.

Factors out the (heavy) one-time setup so the three drivers can vary one
axis each (KV dtype, attention chunking strategy, prefill seq length) without
duplicating ~150 lines of boilerplate.

The structure mirrors ocr_lab/probe_tt_decode_divergence.py exactly so that
results are directly comparable to the existing baseline.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import setup_paths, PROJECT_ROOT  # noqa: E402

setup_paths()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))


# Imports require setup_paths() to have run first.
from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal  # noqa: E402

SD = str(patch_dots_mocr_config_for_tt_metal())

from ocr_lab.dots_model import ensure_dots_ocr_processor_compat  # noqa: E402

ensure_dots_ocr_processor_compat(SD)

from transformers import AutoProcessor, AutoModelForCausalLM  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from dots_tt_port.vision_tower_smoke import load_vision_model  # noqa: E402

import ttnn  # noqa: E402
from models.tt_transformers.tt.model_config import ModelArgs  # noqa: E402
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer  # noqa: E402
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs  # noqa: E402
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta  # noqa: E402
from models.demos.qwen25_vl.reference.functional import (  # noqa: E402
    qwen2_5_vision_transformer_preprocess,
)

from ocr_lab.ttnn_decoder_hybrid import (  # noqa: E402
    DIM,
    HEAD_DIM,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    decode_step_tt,
    init_tt_kv_cache,
    load_decoder_weights,
    precompute_attn_masks,
)


@dataclass
class ProbeConfig:
    """Configuration for a single probe run."""

    n_tt: int = 10
    max_seq: int = 8192
    max_new: int = 180
    native: bool = True
    img_w: int = 672
    img_h: int = 952
    truncate_prompt_to: Optional[int] = None  # if set, truncate the input_ids before prefill
    kv_dtype: Any = None  # ttnn.bfloat16 (default) or ttnn.bfloat8_b
    decode_step_hook: Optional[
        Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None  # post-process tt_logits before argmax (e.g. inject chunked CPU correction)
    label: str = ""


@dataclass
class ProbeResult:
    label: str = ""
    seq: int = 0
    n_tt: int = 0
    max_seq: int = 0
    kv_dtype: str = ""
    first_divergence_step: Optional[int] = None
    divergence_count: int = 0
    total_steps: int = 0
    cosine_at_steps: dict = field(default_factory=dict)
    argmax_log: list = field(default_factory=list)
    elapsed_s: float = 0.0
    error: Optional[str] = None


# ----------------------------------------------------------------------------
# One-time global setup (shared across probe runs)
# ----------------------------------------------------------------------------
_GLOBAL: dict[str, Any] = {}


def _global_setup(mesh_shape: tuple[int, int] = (1, 1)) -> dict[str, Any]:
    if _GLOBAL:
        return _GLOBAL
    print("[probe] global setup...", flush=True)
    proc = AutoProcessor.from_pretrained(SD, trust_remote_code=True)
    tokenizer = getattr(proc, "tokenizer", proc)
    cfg = json.loads((Path(SD) / "config.json").read_text())
    vocab_size = cfg.get("vocab_size", 151936)
    rope_theta = cfg.get("rope_theta", 1e6)

    cpu_vm, _ = load_vision_model(SD, "sdpa", limit_layers=0)
    cpu_vm = cpu_vm.eval().to(torch.float32)

    hf = (
        AutoModelForCausalLM.from_pretrained(
            SD, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
    )
    hf.vision_tower = hf.vision_tower.to(torch.float32)
    emb_w_bf16 = hf.get_input_embeddings().weight.detach().to(torch.bfloat16)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    mesh.enable_program_cache()

    _orig_to_torch = ttnn.to_torch

    def _patched_to_torch(tensor, *args, **kwargs):
        if "mesh_composer" not in kwargs and "device" not in kwargs:
            try:
                return _orig_to_torch(tensor, *args, **kwargs)
            except RuntimeError as exc:
                if "mesh composer" in str(exc) or "buffers.size() == 1" in str(exc):
                    shards = ttnn.get_device_tensors(tensor)
                    return _orig_to_torch(shards[0], *args, **kwargs)
                raise
        return _orig_to_torch(tensor, *args, **kwargs)

    ttnn.to_torch = _patched_to_torch

    dec_args = ModelArgs(
        mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096
    )
    dec_sd = dec_args.load_state_dict()
    dec_layers, dec_final = load_decoder_weights(dec_sd, mesh)

    max_pos = 8192
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    emb_rope = torch.cat((freqs, freqs), dim=-1)
    cos_full = emb_rope.cos().to(torch.bfloat16)
    sin_full = emb_rope.sin().to(torch.bfloat16)
    cos_full_tt = ttnn.from_torch(
        cos_full.unsqueeze(0).unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_full_tt = ttnn.from_torch(
        sin_full.unsqueeze(0).unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    _GLOBAL.update(
        proc=proc,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        cpu_vm=cpu_vm,
        hf=hf,
        emb_w_bf16=emb_w_bf16,
        mesh=mesh,
        dec_layers=dec_layers,
        dec_final=dec_final,
        cos_full_tt=cos_full_tt,
        sin_full_tt=sin_full_tt,
    )
    return _GLOBAL


def _build_inputs(cfg: ProbeConfig):
    g = _GLOBAL
    if cfg.native:
        img_msg = {
            "type": "image",
            "image": str(PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"),
        }
    else:
        from ocr_lab.fixed_page import fixed_page_message

        img_msg, _ = fixed_page_message(
            str(PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"),
            width=cfg.img_w,
            height=cfg.img_h,
        )
    msgs = [
        {
            "role": "user",
            "content": [
                img_msg,
                {
                    "type": "text",
                    "text": "Please output the exact text in the image.\n\nReturn plain text only.\n",
                },
            ],
        }
    ]
    text = g["proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = g["proc"](text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    pv = inputs["pixel_values"].to(torch.bfloat16)
    gt = inputs["image_grid_thw"].to(torch.int32)
    return inputs, pv, gt


def _run_vision(cfg: ProbeConfig, pv, gt):
    g = _GLOBAL
    mesh = g["mesh"]
    vis_args = VisionModelArgs(
        mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=cfg.max_seq
    )
    ref = vis_args.reference_vision_model()
    tt_vis = DropInVisionTransformer(ref, vis_args, debug=False)

    unp = (gt[:, 1] * gt[:, 2]).sum().item()
    sl = ((unp // 2048) + 1) * 2048
    pos_ids = ref.get_pos_ids_by_grid(gt.cpu())
    pos_ids = torch.cat(pos_ids, dim=0)
    mg = int(gt.cpu()[:, 1:].max().item())
    rf = ref.rotary_pos_emb(mg).cpu()
    rot = rf[pos_ids].flatten(1).float()
    ch = rot.cos().unsqueeze(1).repeat(1, 1, 2)
    sh = rot.sin().unsqueeze(1).repeat(1, 1, 2)
    cm, sm = convert_rope_style_hf_to_meta(ch, sh)
    cp = F.pad(cm, (0, 0, 0, sl - unp), value=1).unsqueeze(0).unsqueeze(0)
    sp = F.pad(sm, (0, 0, 0, sl - unp), value=0).unsqueeze(0).unsqueeze(0)
    cos_tt = ttnn.from_torch(
        cp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    sin_tt = ttnn.from_torch(
        sp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    cs, _, _, wi = qwen2_5_vision_transformer_preprocess(
        seq_len=unp,
        grid_thw=gt,
        head_dim=vis_args.vision_head_dim,
        spatial_merge_size=vis_args.hf_config.vision_config.spatial_merge_size,
        window_size=vis_args.hf_config.vision_config.window_size,
        patch_size=vis_args.hf_config.vision_config.patch_size,
    )
    cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

    if cfg.n_tt > 0:
        pe = ref.patch_embed(pv)
        x = tt_vis.tt_model.prepare_input(pe, wi, sl)
        for i in range(cfg.n_tt):
            x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
        raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
        h = raw[:, 0:1, :unp, :DIM].squeeze(0).squeeze(0).to(torch.float32)
        ttnn.deallocate(x)
    else:
        h = g["cpu_vm"].patch_embed(pv.to(torch.float32), gt).to(torch.float32)

    rotary_cpu = g["cpu_vm"].rot_pos_emb(gt)
    cu_cpu = F.pad(
        torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    with torch.no_grad():
        for i in range(cfg.n_tt, 42):
            h = g["cpu_vm"].blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
        if hasattr(g["cpu_vm"], "post_trunk_norm"):
            h = g["cpu_vm"].post_trunk_norm(h)
        vision_out = g["cpu_vm"].merger(h)
    return vision_out


def run_probe(cfg: ProbeConfig) -> ProbeResult:
    """Execute one probe run and return divergence stats."""
    g = _global_setup()
    res = ProbeResult(label=cfg.label, n_tt=cfg.n_tt, max_seq=cfg.max_seq)
    t0 = time.perf_counter()
    try:
        inputs, pv, gt = _build_inputs(cfg)
        if cfg.truncate_prompt_to is not None and cfg.truncate_prompt_to < inputs["input_ids"].shape[1]:
            cut = cfg.truncate_prompt_to
            inputs["input_ids"] = inputs["input_ids"][:, :cut]
            inputs["attention_mask"] = inputs["attention_mask"][:, :cut]
        vision_out = _run_vision(cfg, pv, gt)

        # HF prefill (fp32 reference)
        emb_w_fp32 = g["hf"].get_input_embeddings().weight.detach()
        img_mask = inputs["input_ids"] == g["hf"].config.image_token_id
        embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
        if img_mask.any():
            embeds = embeds.masked_scatter(
                img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype)
            )
        with torch.no_grad():
            outputs = g["hf"](
                input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True
            )
        first_tok = (
            outputs.logits[0, -1, : g["vocab_size"]].float().argmax(-1).item()
        )
        past_kv = outputs.past_key_values
        seq = inputs["input_ids"].shape[1]
        res.seq = seq

        # Fill TT KV cache from HF
        max_cache = seq + cfg.max_new + 32
        max_cache = ((max_cache + 31) // 32) * 32
        kv_dtype = cfg.kv_dtype if cfg.kv_dtype is not None else ttnn.bfloat16
        res.kv_dtype = str(kv_dtype)
        tt_cache = init_tt_kv_cache(g["mesh"], max_seq=max_cache, kv_dtype=kv_dtype)
        for i in range(N_LAYERS):
            torch_dtype = (
                torch.float32
                if kv_dtype == ttnn.float32
                else torch.bfloat16
            )
            k_cpu = past_kv[i][0].to(torch_dtype)
            v_cpu = past_kv[i][1].to(torch_dtype)
            k_tt_i = ttnn.from_torch(
                k_cpu,
                device=g["mesh"],
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v_tt_i = ttnn.from_torch(
                v_cpu,
                device=g["mesh"],
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill_cache(tt_cache[i][0], k_tt_i, 0)
            ttnn.fill_cache(tt_cache[i][1], v_tt_i, 0)
            ttnn.deallocate(k_tt_i)
            ttnn.deallocate(v_tt_i)

        decode_masks = precompute_attn_masks(
            start_pos=seq, num_steps=cfg.max_new, max_cache_seq=max_cache, device=g["mesh"]
        )

        # Step-by-step comparison
        cur_tok = first_tok
        cur_pos = seq
        cosine_log: dict[int, float] = {}
        argmax_log: list[tuple[int, int, int, bool]] = []
        diverged_at: Optional[int] = None
        for step in range(cfg.max_new):
            with torch.no_grad():
                hf_out = g["hf"](
                    input_ids=torch.tensor([[cur_tok]]),
                    past_key_values=past_kv,
                    use_cache=True,
                )
            hf_logits = hf_out.logits[0, -1, : g["vocab_size"]].float().cpu()
            past_kv = hf_out.past_key_values

            tok_emb = g["emb_w_bf16"][cur_tok].reshape(1, 1, 1, DIM)
            tt_tok = ttnn.from_torch(
                tok_emb,
                device=g["mesh"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_logits_t = decode_step_tt(
                tt_tok,
                g["dec_layers"],
                g["dec_final"],
                g["mesh"],
                g["cos_full_tt"],
                g["sin_full_tt"],
                tt_cache,
                cur_pos,
                attn_mask_tt=decode_masks[step],
            )
            tt_logits = ttnn.to_torch(tt_logits_t)[0, 0, 0, : g["vocab_size"]].float().cpu()

            if cfg.decode_step_hook is not None:
                tt_logits = cfg.decode_step_hook(step, hf_logits, tt_logits)

            cos = float(
                F.cosine_similarity(hf_logits.unsqueeze(0), tt_logits.unsqueeze(0)).item()
            )
            hf_a = int(hf_logits.argmax().item())
            tt_a = int(tt_logits.argmax().item())
            argmax_log.append((step, hf_a, tt_a, hf_a == tt_a))
            if step in {0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, cfg.max_new - 1}:
                cosine_log[step] = cos
            if hf_a != tt_a and diverged_at is None:
                diverged_at = step
            cur_tok = hf_a
            cur_pos += 1

        for mask in decode_masks:
            ttnn.deallocate(mask)

        res.first_divergence_step = diverged_at
        res.divergence_count = sum(1 for _, _, _, m in argmax_log if not m)
        res.total_steps = len(argmax_log)
        res.cosine_at_steps = cosine_log
        res.argmax_log = argmax_log
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        res.elapsed_s = time.perf_counter() - t0
    return res


def close_mesh() -> None:
    g = _GLOBAL
    if "mesh" in g:
        ttnn.close_mesh_device(g["mesh"])
        _GLOBAL.clear()
