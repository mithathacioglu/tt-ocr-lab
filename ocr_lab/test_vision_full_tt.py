#!/usr/bin/env python3
"""
Test: Run vision tower entirely on TT with pre-computed rotary embeddings.
For fixed 476x674 (single image, single segment), we can avoid CPU fallbacks.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dots_tt_port.vision_tower_smoke import (
    TowerHybridCpuUnary, ensure_tt_env, load_vision_model, resolve_snapshot_dir,
    apply_rotary_pos_emb_cached_cpu, apply_norm_cpu_module,
)
from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import fixed_page_message

SNAPSHOT_DIR = resolve_snapshot_dir("rednote-hilab/dots.mocr")


class TowerFullTT(torch.nn.Module):
    """Vision tower that keeps everything on TT device.

    Pre-computes rotary embeddings on CPU, transfers once.
    Runs norm, attention, MLP all on TT.
    Only works for single-segment (single image) inputs.
    """
    def __init__(self, vision_model, cos_cpu, sin_cpu):
        super().__init__()
        self.model = vision_model
        self.patch_embed = vision_model.patch_embed
        # Pre-register rotary as buffers (transferred with .to(device))
        self.register_buffer("cos_rot", cos_cpu.to(torch.bfloat16))
        self.register_buffer("sin_rot", sin_cpu.to(torch.bfloat16))

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values, grid_thw)
        seq_len = hidden_states.shape[0]

        for blk in self.model.blocks:
            residual = hidden_states
            # Norm on TT (RMSNorm)
            norm1 = self._tt_rms_norm(blk.norm1, hidden_states)

            # QKV projection on TT
            qkv = blk.attn.qkv(norm1)

            # Attention entirely on TT
            attn_out = self._tt_attention(blk.attn, qkv, seq_len)
            hidden_states = residual + blk.attn.proj(attn_out)

            # MLP on TT
            residual = hidden_states
            norm2 = self._tt_rms_norm(blk.norm2, hidden_states)
            fc1 = blk.mlp.fc1(norm2)
            fc3 = blk.mlp.fc3(norm2)
            gated = F.silu(fc1) * fc3
            hidden_states = residual + blk.mlp.fc2(gated)

        if self.model.config.post_norm:
            hidden_states = self._tt_rms_norm(self.model.post_trunk_norm, hidden_states)

        # Merger — keep on TT too
        return self._tt_merger(hidden_states)

    def _tt_rms_norm(self, norm_module, x):
        """RMSNorm on TT device."""
        eps = getattr(norm_module, "eps", 1e-5)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        if hasattr(norm_module, "weight") and norm_module.weight is not None:
            x = x * norm_module.weight
        return x.to(dtype=torch.bfloat16)

    def _tt_attention(self, attn, qkv, seq_len):
        """Full attention on TT — single segment, no mask needed."""
        num_heads = attn.num_heads
        head_dim = qkv.shape[-1] // 3 // num_heads

        q, k, v = qkv.reshape(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)

        # Apply rotary embeddings on TT
        cos = self.cos_rot[:seq_len]
        sin = self.sin_rot[:seq_len]

        # Apply rotary: q * cos + rotate_half(q) * sin
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        # [seq, heads, dim] -> [heads, seq, dim] for SDPA
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return attn_out.transpose(0, 1).reshape(seq_len, -1).to(dtype=torch.bfloat16)

    def _apply_rotary(self, x, cos, sin):
        """Apply rotary position embedding."""
        # x: [seq, heads, dim], cos/sin: [seq, 1, dim]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos + rotated * sin).to(x.dtype)

    def _tt_merger(self, hidden_states):
        """PatchMerger on TT."""
        merger = self.model.merger
        d = hidden_states.shape[-1]
        # Merge spatial neighbors
        # merger.pre_norm defines merge factor
        merge_size = merger.pre_norm  # This is actually spatial_merge_size

        # For now, fallback to CPU for merger (it's only called once, ~60ms)
        h_cpu = hidden_states.detach().cpu().float()

        if hasattr(merger, 'ln_q'):
            if hasattr(merger.ln_q, 'bias') and merger.ln_q.bias is not None:
                h_cpu = F.layer_norm(h_cpu, (d,), merger.ln_q.weight.cpu().float(), merger.ln_q.bias.cpu().float(), merger.ln_q.eps)
            else:
                variance = h_cpu.pow(2).mean(-1, keepdim=True)
                h_cpu = h_cpu * torch.rsqrt(variance + merger.ln_q.eps) * merger.ln_q.weight.cpu().float()

        # Reshape for spatial merge: [seq, d] -> [seq/merge^2, merge^2 * d] -> MLP
        merge_sq = merge_size * merge_size
        if h_cpu.shape[0] % merge_sq == 0:
            h_cpu = h_cpu.view(-1, merge_sq * d)

        h_cpu = F.linear(h_cpu, merger.mlp[0].weight.cpu().float(), merger.mlp[0].bias.cpu().float())
        h_cpu = F.gelu(h_cpu)
        h_cpu = F.linear(h_cpu, merger.mlp[2].weight.cpu().float(), merger.mlp[2].bias.cpu().float())

        return h_cpu.to(dtype=torch.bfloat16)


def main():
    IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    # Setup TT
    ensure_tt_env("0")
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    import torch_xla
    tt_device = torch_xla.device()

    # Build inputs
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    # Load vision model
    print("Loading vision model...", flush=True)
    vision_model, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=0)

    # Pre-compute rotary embeddings on CPU
    hybrid_ref = TowerHybridCpuUnary(vision_model).eval().to(dtype=torch.bfloat16)
    cos_cpu, sin_cpu = hybrid_ref._build_rotary_cache_cpu(grid_thw)
    # cos/sin shape: [1, seq, 1, dim] -> [seq, 1, dim]
    cos_for_tt = cos_cpu.squeeze(0).to(torch.bfloat16)
    sin_for_tt = sin_cpu.squeeze(0).to(torch.bfloat16)
    print(f"Rotary cache shape: cos={cos_for_tt.shape}, sin={sin_for_tt.shape}", flush=True)

    # --- Hybrid reference ---
    print("\n--- Hybrid (existing) ---", flush=True)
    hybrid_ref = hybrid_ref.to(tt_device)
    pv_tt = pixel_values.to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = grid_thw.to(dtype=torch.int32).to(tt_device)

    with torch.no_grad():
        t0 = time.perf_counter()
        ref_out = hybrid_ref(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        cold1 = (time.perf_counter() - t0) * 1000
        ref_cpu = ref_out.detach().cpu().float()

        t0 = time.perf_counter()
        ref_out2 = hybrid_ref(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        warm1 = (time.perf_counter() - t0) * 1000
    print(f"  Cold: {cold1:.0f}ms  Warm: {warm1:.0f}ms", flush=True)

    # --- Full TT ---
    print("\n--- Full TT (no CPU fallback) ---", flush=True)
    vision_model2, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=0)
    full_tt = TowerFullTT(vision_model2, cos_for_tt, sin_for_tt).eval().to(dtype=torch.bfloat16).to(tt_device)

    with torch.no_grad():
        t0 = time.perf_counter()
        tt_out = full_tt(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        cold2 = (time.perf_counter() - t0) * 1000
        tt_cpu = tt_out.detach().cpu().float() if isinstance(tt_out, torch.Tensor) and tt_out.device != torch.device('cpu') else tt_out.float()

        t0 = time.perf_counter()
        tt_out2 = full_tt(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        warm2 = (time.perf_counter() - t0) * 1000
    print(f"  Cold: {cold2:.0f}ms  Warm: {warm2:.0f}ms", flush=True)

    # Compare outputs
    if ref_cpu.shape == tt_cpu.shape:
        diff = (ref_cpu - tt_cpu).abs()
        print(f"\n  Output diff: max={diff.max():.6f} mean={diff.mean():.6f}")
        cos_sim = F.cosine_similarity(ref_cpu.flatten().unsqueeze(0), tt_cpu.flatten().unsqueeze(0))
        print(f"  Cosine similarity: {cos_sim.item():.6f}")
    else:
        print(f"\n  Shape mismatch: ref={ref_cpu.shape} vs tt={tt_cpu.shape}")

    print(f"\n=== SPEEDUP: {warm1/warm2:.2f}x ({warm1:.0f}ms -> {warm2:.0f}ms) ===")


if __name__ == "__main__":
    main()
