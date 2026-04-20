"""Optimized vision tower: compiled CPU attention + thread tuning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dots_tt_port.vision_tower_smoke import (
    HybridUnaryVisionCore,
    apply_norm_cpu_module,
)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _cpu_sdpa_with_rotary(qkv_cpu: torch.Tensor, num_heads: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused: reshape + rotary + SDPA on CPU. Compilable."""
    seq_length = qkv_cpu.shape[0]
    head_dim = qkv_cpu.shape[-1] // 3 // num_heads
    q, k, v = qkv_cpu.reshape(seq_length, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)

    # Rotary
    q_f = q.unsqueeze(0).float()
    k_f = k.unsqueeze(0).float()
    cos_f = cos.float()
    sin_f = sin.float()
    q_rot = ((q_f * cos_f) + (_rotate_half(q_f) * sin_f)).squeeze(0)
    k_rot = ((k_f * cos_f) + (_rotate_half(k_f) * sin_f)).squeeze(0)

    # SDPA
    q_t = q_rot.transpose(0, 1)
    k_t = k_rot.transpose(0, 1)
    v_t = v.float().transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
    return attn_output.transpose(0, 1).reshape(seq_length, -1).to(dtype=torch.bfloat16)


# Compiled version — will be compiled on first call
_compiled_cpu_sdpa = None


def _get_compiled_sdpa():
    global _compiled_cpu_sdpa
    if _compiled_cpu_sdpa is None:
        _compiled_cpu_sdpa = torch.compile(_cpu_sdpa_with_rotary, mode="reduce-overhead")
    return _compiled_cpu_sdpa


class TowerHybridOptimized(HybridUnaryVisionCore):
    """Vision tower with compiled CPU attention and 32 threads."""

    def __init__(self, vision_model):
        super().__init__(vision_model)
        self._orig_threads = torch.get_num_threads()

    def _optimized_cpu_attention(self, attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu):
        qkv_cpu = qkv_tt.detach().cpu().float()

        # Single segment fast path (our use case)
        if len(cu_seqlens_cpu) <= 2:
            compiled_fn = _get_compiled_sdpa()
            return compiled_fn(qkv_cpu, attn.num_heads, cos_cpu, sin_cpu)

        # Multi-segment fallback
        return self._cpu_attention_context(attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu)

    def _optimized_block(self, blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu):
        norm1 = apply_norm_cpu_module(blk.norm1, hidden_states)
        qkv_tt = blk.attn.qkv(norm1)
        attn_ctx = self._optimized_cpu_attention(blk.attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu)
        attn_ctx_tt = attn_ctx.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + blk.attn.proj(attn_ctx_tt)

        norm2 = apply_norm_cpu_module(blk.norm2, hidden_states)
        fc1_tt = blk.mlp.fc1(norm2)
        fc3_tt = blk.mlp.fc3(norm2)
        gated_tt = F.silu(fc1_tt) * fc3_tt
        hidden_states = hidden_states + blk.mlp.fc2(gated_tt)
        return hidden_states

    def forward(self, pixel_values, grid_thw):
        # Boost CPU threads for attention
        torch.set_num_threads(32)

        hidden_states = self.patch_embed(pixel_values, grid_thw)
        cos_cpu, sin_cpu = self._build_rotary_cache_cpu(grid_thw)
        cu_seqlens_cpu = self._build_cu_seqlens_cpu(grid_thw)

        for blk in self.model.blocks:
            hidden_states = self._optimized_block(blk, hidden_states, cu_seqlens_cpu, cos_cpu, sin_cpu)

        if self.model.config.post_norm:
            hidden_states = apply_norm_cpu_module(self.model.post_trunk_norm, hidden_states)

        result = self.cpu_merger(hidden_states)

        # Restore threads
        torch.set_num_threads(self._orig_threads)
        return result
