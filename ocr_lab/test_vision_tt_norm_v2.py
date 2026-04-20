#!/usr/bin/env python3
"""Test vision with norms on TT — single model, clean device."""
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
    HybridUnaryVisionCore, ensure_tt_env, load_vision_model, resolve_snapshot_dir,
    apply_rotary_pos_emb_cached_cpu,
)
from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import fixed_page_message

SNAPSHOT_DIR = resolve_snapshot_dir("rednote-hilab/dots.mocr")


class TowerTTNormV2(HybridUnaryVisionCore):
    """Norms on TT, attention on CPU. Only 1 round-trip per block (for attention)."""

    def _hybrid_block_tt_norm(self, blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu):
        residual = hidden_states

        # RMSNorm on TT (stays on device)
        eps = getattr(blk.norm1, "eps", 1e-5)
        var1 = hidden_states.pow(2).mean(-1, keepdim=True)
        norm1 = hidden_states * torch.rsqrt(var1 + eps) * blk.norm1.weight

        # QKV on TT
        qkv_tt = blk.attn.qkv(norm1)

        # Attention: TT→CPU — explicit sync before transfer
        import torch_xla
        torch_xla.sync(wait=True)
        attn_ctx_cpu = self._cpu_attention_context(blk.attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu)
        attn_ctx_tt = attn_ctx_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = residual + blk.attn.proj(attn_ctx_tt)

        # RMSNorm2 on TT
        residual2 = hidden_states
        eps2 = getattr(blk.norm2, "eps", 1e-5)
        var2 = hidden_states.pow(2).mean(-1, keepdim=True)
        norm2 = hidden_states * torch.rsqrt(var2 + eps2) * blk.norm2.weight

        # MLP fully on TT
        fc1_tt = blk.mlp.fc1(norm2)
        fc3_tt = blk.mlp.fc3(norm2)
        gated_tt = F.silu(fc1_tt) * fc3_tt
        hidden_states = residual2 + blk.mlp.fc2(gated_tt)
        return hidden_states

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values, grid_thw)
        cos_cpu, sin_cpu = self._build_rotary_cache_cpu(grid_thw)
        cu_seqlens_cpu = self._build_cu_seqlens_cpu(grid_thw)

        for blk in self.model.blocks:
            hidden_states = self._hybrid_block_tt_norm(blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu)

        if self.model.config.post_norm:
            eps = getattr(self.model.post_trunk_norm, "eps", 1e-5)
            var = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(var + eps) * self.model.post_trunk_norm.weight

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
    proc = AutoProcessor.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\nReturn plain text only.\n"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    pv_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(tt_device)

    print("Loading vision (TT norms)...", flush=True)
    vm, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=0)
    model = TowerTTNormV2(vm).eval().to(dtype=torch.bfloat16).to(tt_device)

    with torch.no_grad():
        # Cold
        print("  Cold run...", flush=True)
        t0 = time.perf_counter()
        out1 = model(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        cold = (time.perf_counter() - t0) * 1000
        print(f"  Cold: {cold:.0f}ms", flush=True)

        # Warm 1
        t0 = time.perf_counter()
        out2 = model(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        warm1 = (time.perf_counter() - t0) * 1000
        cpu_out = out2.detach().cpu().float()
        print(f"  Warm1: {warm1:.0f}ms", flush=True)

        # Warm 2
        t0 = time.perf_counter()
        out3 = model(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        warm2 = (time.perf_counter() - t0) * 1000
        print(f"  Warm2: {warm2:.0f}ms", flush=True)

    print(f"  Output shape: {cpu_out.shape}")
    print(f"  Stats: min={cpu_out.min():.4f} max={cpu_out.max():.4f} mean={cpu_out.mean():.4f}")
    print(f"\n=== TT Norm Vision: cold={cold:.0f}  warm1={warm1:.0f}  warm2={warm2:.0f}ms ===")


if __name__ == "__main__":
    main()
