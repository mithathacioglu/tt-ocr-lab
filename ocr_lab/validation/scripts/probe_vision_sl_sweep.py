#!/usr/bin/env python3
"""B1: Vision tower silent-corruption sweep.

For each cell of (image_resolution, MAX_SEQ, N_TT), this script:

  1. Builds the input image (or uses native if NATIVE=1).
  2. Runs the first N_TT vision blocks on TT (bfloat8_b weights/activations
     by default - this is the suspected error source).
  3. Runs the remaining (42 - N_TT) vision blocks on CPU fp32.
  4. Runs the HF decoder prefill in fp32 and a short greedy decode (default
     MAX_NEW=80 tokens, enough to detect Chinese characters / doubled
     suffixes that signal silent corruption).
  5. Scores the output:
       - keyword_score:   Turkish keywords from project_dots_mocr_5of5_pipeline.md
       - chinese_chars:   count of CJK code-point chars in output
       - doubled_suffix:  hits on patterns like "davalalı", "İstinaaftan"
  6. Computes per-block cosine similarity between the TT vision output
     (taken at the unpacked seq slice) and a CPU fp32 reference run for
     the FIRST CELL ONLY (per-block cosine is expensive; we only need
     it for one cell to see where bfloat8_b drift sets in).

Sweep is read from the SWEEP env var (a python literal) or from a default
small grid that includes NewMind's reported failure case (1120x1568,
MAX_SEQ=4096, N_TT=5).

Output: ocr_lab/validation/logs/vision_sl_sweep.json plus a markdown
summary table at ocr_lab/validation/logs/vision_sl_sweep.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import setup_paths, write_log

setup_paths()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal  # noqa: E402

snapshot_dir = patch_dots_mocr_config_for_tt_metal()
SD = str(snapshot_dir)

from ocr_lab.dots_model import ensure_dots_ocr_processor_compat  # noqa: E402
from ocr_lab.fixed_page import fixed_page_message  # noqa: E402

ensure_dots_ocr_processor_compat(SD)

from transformers import AutoProcessor, AutoModelForCausalLM  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from dots_tt_port.vision_tower_smoke import load_vision_model  # noqa: E402

import ttnn  # noqa: E402
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer  # noqa: E402
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs  # noqa: E402
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta  # noqa: E402
from models.demos.qwen25_vl.reference.functional import (  # noqa: E402
    qwen2_5_vision_transformer_preprocess,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

KEYWORDS = ["hükümler", "Davacı", "davalı", "Dilekçesi", "kesinleştiği"]
DOUBLED_SUFFIX_RE = re.compile(r"(?:davalalı|İstinaaftan|tarihlihüküm|tevmişolduğu|kesinleşstığı)")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ----------------------------------------------------------------------------
# Default sweep grid
# ----------------------------------------------------------------------------
# Each cell is (img_w, img_h, native, max_seq, n_tt).
# - (img_w, img_h, False) -> use fixed_page resize
# - (img_w, img_h, True)  -> use native image, w/h ignored
# Two NewMind cases come first so that even if the run is interrupted we
# capture the headline result.
DEFAULT_SWEEP: list[tuple[int, int, bool, int, int]] = [
    # NewMind reported failure
    (1120, 1568, False, 4096, 5),
    # NewMind 5/5 baseline
    (672,   952, False, 4096, 11),
    # native (seq=2791)
    (0,       0,  True, 8192, 11),
    # vision-depth axis
    (672,   952, False, 4096,  5),
    (672,   952, False, 4096, 20),
    # max_seq axis
    (672,   952, False, 2048, 11),
    (672,   952, False, 6144, 11),
    # higher resolution
    (1008, 1416, False, 4096,  5),
    (1008, 1416, False, 4096, 11),
]


def _load_sweep() -> list[tuple]:
    raw = os.environ.get("SWEEP")
    if raw:
        return list(eval(raw))  # noqa: S307 - intentional, dev-only knob
    return DEFAULT_SWEEP


# ----------------------------------------------------------------------------
# One-time setup
# ----------------------------------------------------------------------------
print("Loading processor + HF + CPU vision...", flush=True)
proc = AutoProcessor.from_pretrained(SD, trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)

cfg = json.loads((Path(SD) / "config.json").read_text())
VOCAB_SIZE = cfg.get("vocab_size", 151936)

hf = (
    AutoModelForCausalLM.from_pretrained(
        SD,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    .eval()
)
hf.vision_tower = hf.vision_tower.to(torch.float32)

cpu_vm, _ = load_vision_model(SD, "sdpa", limit_layers=0)
cpu_vm = cpu_vm.eval().to(torch.float32)

MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH_SHAPE", "1,1").split(","))
print(f"Opening mesh {MESH_SHAPE}...", flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
mesh.enable_program_cache()


_orig_to_torch = ttnn.to_torch


def _patched_to_torch(tensor, *args, **kwargs):
    if "mesh_composer" not in kwargs and "device" not in kwargs:
        try:
            return _orig_to_torch(tensor, *args, **kwargs)
        except RuntimeError as e:
            if "mesh composer" in str(e) or "buffers.size() == 1" in str(e):
                shards = ttnn.get_device_tensors(tensor)
                return _orig_to_torch(shards[0], *args, **kwargs)
            raise
    return _orig_to_torch(tensor, *args, **kwargs)


ttnn.to_torch = _patched_to_torch


# ----------------------------------------------------------------------------
# Per-cell run
# ----------------------------------------------------------------------------
def build_inputs(img_w: int, img_h: int, native: bool):
    if native:
        img_msg = {"type": "image", "image": str(TEST_IMAGE)}
    else:
        img_msg, _ = fixed_page_message(str(TEST_IMAGE), width=img_w, height=img_h)
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
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    pv = inputs["pixel_values"].to(torch.bfloat16)
    gt = inputs["image_grid_thw"].to(torch.int32)
    return inputs, pv, gt


def build_tt_vision(max_seq: int):
    vis_args = VisionModelArgs(
        mesh,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=max_seq,
    )
    ref = vis_args.reference_vision_model()
    tt_vis = DropInVisionTransformer(ref, vis_args, debug=False)
    return vis_args, ref, tt_vis


def run_vision_tt(vis_args, ref, tt_vis, pv, gt, n_tt: int):
    """Run the first n_tt blocks on TT, return CPU-side hidden state at unp.

    Also returns the per-block TT outputs for cosine measurement.
    """
    unp = (gt[:, 1] * gt[:, 2]).sum().item()
    sl = ((unp // 2048) + 1) * 2048
    dim = 1536

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
    rot_mapper = (
        ttnn.ReplicateTensorToMesh(mesh)
        if MESH_SHAPE != (1, 1)
        else ttnn.ShardTensorToMesh(mesh, dim=0)
    )
    cos_tt = ttnn.from_torch(
        cp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rot_mapper
    )
    sin_tt = ttnn.from_torch(
        sp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rot_mapper
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

    tt_hidden_per_block: list[torch.Tensor] = []
    if n_tt > 0:
        pe = ref.patch_embed(pv)
        x = tt_vis.tt_model.prepare_input(pe, wi, sl)
        for i in range(n_tt):
            x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
            if os.environ.get("PER_BLOCK_COSINE") == "1":
                snap = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
                tt_hidden_per_block.append(
                    snap[:, 0:1, :unp, :dim].squeeze(0).squeeze(0).to(torch.float32).clone()
                )
        raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
        h = raw[:, 0:1, :unp, :dim].squeeze(0).squeeze(0).to(torch.float32)
        ttnn.deallocate(x)
    else:
        h = cpu_vm.patch_embed(pv.to(torch.float32), gt).to(torch.float32)

    return h, tt_hidden_per_block, unp, sl


def run_vision_cpu_full(pv, gt, n_blocks_to_capture: int = 0):
    """Run the full CPU vision tower in fp32 and return the hidden state per
    captured block index (block i output for i in 0..n_blocks_to_capture-1)
    plus the merger output."""
    rotary_cpu = cpu_vm.rot_pos_emb(gt)
    cu_cpu = F.pad(
        torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    captures: list[torch.Tensor] = []
    with torch.no_grad():
        h = cpu_vm.patch_embed(pv.to(torch.float32), gt)
        for i in range(42):
            h = cpu_vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
            if i < n_blocks_to_capture:
                captures.append(h.clone())
        if hasattr(cpu_vm, "post_trunk_norm"):
            h = cpu_vm.post_trunk_norm(h)
        vision_out = cpu_vm.merger(h)
    return vision_out, captures


def run_cpu_tail(h_tt: torch.Tensor, gt, n_tt: int):
    rotary_cpu = cpu_vm.rot_pos_emb(gt)
    cu_cpu = F.pad(
        torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    h = h_tt
    with torch.no_grad():
        for i in range(n_tt, 42):
            h = cpu_vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
        if hasattr(cpu_vm, "post_trunk_norm"):
            h = cpu_vm.post_trunk_norm(h)
        vision_out = cpu_vm.merger(h)
    return vision_out


def run_decode(inputs, vision_out, max_new: int) -> str:
    img_mask = inputs["input_ids"] == hf.config.image_token_id
    emb_w_fp32 = hf.get_input_embeddings().weight.detach()
    embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
    embeds = embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype)
    )
    with torch.no_grad():
        gen = hf.generate(
            input_ids=inputs["input_ids"],
            inputs_embeds=embeds,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text


def score_text(text: str) -> dict:
    keyword_hits = sum(1 for k in KEYWORDS if k in text)
    chinese = len(CJK_RE.findall(text))
    doubled = len(DOUBLED_SUFFIX_RE.findall(text))
    return {
        "keyword_hits": keyword_hits,
        "keyword_total": len(KEYWORDS),
        "chinese_chars": chinese,
        "doubled_suffix_hits": doubled,
    }


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
sweep = _load_sweep()
print(f"Sweep cells: {len(sweep)}", flush=True)
results: list[dict[str, Any]] = []
MAX_NEW = int(os.environ.get("MAX_NEW", "80"))

for cell_idx, (img_w, img_h, native, max_seq, n_tt) in enumerate(sweep):
    label = f"{img_w}x{img_h}{'/native' if native else ''} max_seq={max_seq} N_TT={n_tt}"
    print(f"\n=== cell {cell_idx + 1}/{len(sweep)}: {label} ===", flush=True)
    cell: dict[str, Any] = {
        "img_w": img_w,
        "img_h": img_h,
        "native": native,
        "max_seq": max_seq,
        "n_tt": n_tt,
        "label": label,
    }
    try:
        inputs, pv, gt = build_inputs(img_w, img_h, native)
        unp = (gt[:, 1] * gt[:, 2]).sum().item()
        sl = ((unp // 2048) + 1) * 2048
        cell["unp"] = int(unp)
        cell["sl"] = int(sl)

        vis_args, ref, tt_vis = build_tt_vision(max_seq)

        if cell_idx == 0 and n_tt > 0:
            os.environ["PER_BLOCK_COSINE"] = "1"
        else:
            os.environ.pop("PER_BLOCK_COSINE", None)

        t0 = time.perf_counter()
        h_tt, tt_per_block, _, _ = run_vision_tt(vis_args, ref, tt_vis, pv, gt, n_tt)
        cell["tt_vision_ms"] = (time.perf_counter() - t0) * 1000.0

        if tt_per_block:
            t0 = time.perf_counter()
            _vo_cpu, cpu_captures = run_vision_cpu_full(pv, gt, n_blocks_to_capture=n_tt)
            cell["cpu_full_vision_ms"] = (time.perf_counter() - t0) * 1000.0
            cosines = []
            for i, (a, b) in enumerate(zip(tt_per_block, cpu_captures)):
                a_f = a.float().reshape(-1)
                b_f = b.float().reshape(-1)
                cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
                cosines.append({"block": i, "cosine_vs_cpu_fp32": cos})
            cell["per_block_cosine"] = cosines

        t0 = time.perf_counter()
        vision_out = run_cpu_tail(h_tt, gt, n_tt)
        cell["cpu_tail_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        text = run_decode(inputs, vision_out, MAX_NEW)
        cell["decode_ms"] = (time.perf_counter() - t0) * 1000.0
        cell["output_text"] = text
        cell.update(score_text(text))
        cell["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        cell["status"] = "error"
        cell["error"] = f"{type(exc).__name__}: {exc}"
    results.append(cell)
    print(json.dumps({k: v for k, v in cell.items() if k != "per_block_cosine"}, indent=2, ensure_ascii=False), flush=True)

ttnn.close_mesh_device(mesh)

write_log("vision_sl_sweep", {"sweep": results, "max_new": MAX_NEW})

# Markdown summary
md_lines = [
    "# B1 — vision-tower silent-corruption sweep",
    "",
    "| cell | unp | sl | N_TT | max_seq | kw | CJK | dbl-suf | tt_ms | text (first 100 chars) |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for c in results:
    if c.get("status") != "ok":
        md_lines.append(
            f"| {c['label']} | - | - | {c['n_tt']} | {c['max_seq']} | ERR | ERR | ERR | - | "
            f"{c.get('error','')[:100]} |"
        )
        continue
    txt = (c.get("output_text", "") or "").replace("\n", " ")[:100]
    md_lines.append(
        f"| {c['label']} | {c['unp']} | {c['sl']} | {c['n_tt']} | {c['max_seq']} | "
        f"{c['keyword_hits']}/{c['keyword_total']} | {c['chinese_chars']} | "
        f"{c['doubled_suffix_hits']} | {c['tt_vision_ms']:.0f} | {txt} |"
    )
md_path = PROJECT_ROOT / "ocr_lab" / "validation" / "logs" / "vision_sl_sweep.md"
md_path.write_text("\n".join(md_lines))
print(f"\nWrote {md_path}", flush=True)
