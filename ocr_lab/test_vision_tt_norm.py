#!/usr/bin/env python3
"""
Test: Vision tower with norms on TT (not CPU).
Reduces per-block TT↔CPU transfers from 4 to 2 (only attention).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dots_tt_port.vision_tower_smoke import (
    HybridUnaryVisionCore, TowerHybridCpuUnary,
    ensure_tt_env, load_vision_model, resolve_snapshot_dir,
    apply_norm_cpu_module, apply_rotary_pos_emb_cached_cpu,
)
from ocr_lab.tt_perf import configure_tt_runtime

SNAPSHOT_DIR = resolve_snapshot_dir("rednote-hilab/dots.mocr")


class TowerTTNorm(HybridUnaryVisionCore):
    """Like TowerHybridCpuUnary but norms stay on TT device."""

    def _tt_rms_norm(self, norm_module, x):
        eps = getattr(norm_module, "eps", 1e-5)
        # Stay in bfloat16 on TT — avoid float32 cast which can trigger sync
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        if hasattr(norm_module, "weight") and norm_module.weight is not None:
            normed = normed * norm_module.weight
        return normed

    def _hybrid_block_tt_norm(self, blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu):
        # Norm1 on TT (no transfer)
        norm1 = self._tt_rms_norm(blk.norm1, hidden_states)

        # QKV on TT
        qkv_tt = blk.attn.qkv(norm1)

        # Attention on CPU (only transfer point)
        attn_ctx_cpu = self._cpu_attention_context(blk.attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu)
        attn_ctx_tt = attn_ctx_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + blk.attn.proj(attn_ctx_tt)

        # Norm2 on TT (no transfer)
        norm2 = self._tt_rms_norm(blk.norm2, hidden_states)
        fc1_tt = blk.mlp.fc1(norm2)
        fc3_tt = blk.mlp.fc3(norm2)
        gated_tt = F.silu(fc1_tt) * fc3_tt
        hidden_states = hidden_states + blk.mlp.fc2(gated_tt)
        return hidden_states

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values, grid_thw)
        cos_cpu, sin_cpu = self._build_rotary_cache_cpu(grid_thw)
        cu_seqlens_cpu = self._build_cu_seqlens_cpu(grid_thw)

        for blk in self.model.blocks:
            hidden_states = self._hybrid_block_tt_norm(blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu)

        if self.model.config.post_norm:
            hidden_states = self._tt_rms_norm(self.model.post_trunk_norm, hidden_states)

        return self.cpu_merger(hidden_states)


def main():
    IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    ensure_tt_env("0")
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    import torch_xla
    tt_device = torch_xla.device()

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from ocr_lab.fixed_page import fixed_page_message

    proc = AutoProcessor.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    pv_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(tt_device)

    # --- Test: TT norms ---
    print("Loading TT norms vision...", flush=True)
    vm, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=0)
    tt_norm = TowerTTNorm(vm).eval().to(dtype=torch.bfloat16).to(tt_device)

    with torch.no_grad():
        print("  Cold run...", flush=True)
        t0 = time.perf_counter()
        out1 = tt_norm(pv_tt, gt_tt); torch_xla.sync(wait=True)
        cold_ms = (time.perf_counter() - t0) * 1000
        print(f"  Cold: {cold_ms:.0f}ms", flush=True)

        print("  Warm run...", flush=True)
        t0 = time.perf_counter()
        out2 = tt_norm(pv_tt, gt_tt); torch_xla.sync(wait=True)
        warm_ms = (time.perf_counter() - t0) * 1000
        out_cpu = out2.detach().cpu().float()
        print(f"  Warm: {warm_ms:.0f}ms", flush=True)

        # 3rd run
        t0 = time.perf_counter()
        out3 = tt_norm(pv_tt, gt_tt); torch_xla.sync(wait=True)
        warm2_ms = (time.perf_counter() - t0) * 1000
        print(f"  Warm2: {warm2_ms:.0f}ms", flush=True)

    print(f"  Output shape: {out_cpu.shape}")
    print(f"  Output stats: min={out_cpu.min():.4f} max={out_cpu.max():.4f} mean={out_cpu.mean():.4f}")
    print(f"\n=== TT Norm Vision: cold={cold_ms:.0f}ms warm={warm_ms:.0f}ms warm2={warm2_ms:.0f}ms ===")


if __name__ == "__main__":
    main()
