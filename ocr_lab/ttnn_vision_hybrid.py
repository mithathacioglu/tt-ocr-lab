"""
TTNN Vision Tower: TT matmuls + CPU attention (exact accuracy).

Same pattern as the decoder: heavy matmuls on TT, everything else on CPU.
This replaces the 211s pure-CPU vision with ~5-15s hybrid.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

DIM = 1536
N_HEADS = 12
HEAD_DIM = 128
INTERMEDIATE = 4224
EPS = 1e-5
N_BLOCKS = 42


def _rms_norm(x, weight, eps=EPS):
    """CPU RMS norm matching HF exactly."""
    x_f32 = x.float()
    return (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def _tt_matmul(x_bf16, weight_tt, device):
    """CPU bf16 → TT matmul → CPU. Minimal transfer."""
    # x_bf16: [seq, in_dim]
    ndim = x_bf16.dim()
    if ndim == 2:
        x_bf16 = x_bf16.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, in_dim]
    x_tt = ttnn.from_torch(x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y_tt = ttnn.linear(x_tt, weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x_tt)
    y_cpu = ttnn.to_torch(y_tt)
    ttnn.deallocate(y_tt)
    if ndim == 2:
        y_cpu = y_cpu.squeeze(0).squeeze(0)
    return y_cpu


def load_vision_weights(vision_model, device):
    """Extract weights from HF vision model and load to TT device."""
    sd = vision_model.state_dict()
    blocks = []
    for i in range(N_BLOCKS):
        p = f"blocks.{i}."
        w = {}
        # Norms stay on CPU
        w["norm1_w"] = sd[f"{p}norm1.weight"].to(torch.bfloat16)
        w["norm2_w"] = sd[f"{p}norm2.weight"].to(torch.bfloat16)
        # Matmul weights to TT (stored transposed: [in, out])
        w["qkv_w"] = ttnn.from_torch(
            sd[f"{p}attn.qkv.weight"].T.contiguous().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["qkv_b"] = sd.get(f"{p}attn.qkv.bias", None)
        if w["qkv_b"] is not None:
            w["qkv_b"] = w["qkv_b"].to(torch.bfloat16)
        w["proj_w"] = ttnn.from_torch(
            sd[f"{p}attn.proj.weight"].T.contiguous().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["proj_b"] = sd.get(f"{p}attn.proj.bias", None)
        if w["proj_b"] is not None:
            w["proj_b"] = w["proj_b"].to(torch.bfloat16)
        for name in ("fc1", "fc2", "fc3"):
            w[f"{name}_w"] = ttnn.from_torch(
                sd[f"{p}mlp.{name}.weight"].T.contiguous().unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = sd.get(f"{p}mlp.{name}.bias", None)
            w[f"{name}_b"] = b.to(torch.bfloat16) if b is not None else None
        blocks.append(w)
        if (i + 1) % 10 == 0:
            print(f"    loaded {i+1}/{N_BLOCKS} vision blocks", flush=True)

    # Post-norm + merger weights stay on CPU
    final = {}
    if "post_trunk_norm.weight" in sd:
        final["post_norm_w"] = sd["post_trunk_norm.weight"].to(torch.bfloat16)
    # Merger stays fully on CPU (small linear layers + GELU)
    return blocks, final


def _apply_rotary_pos_emb_vision(tensor, freqs):
    """Rotary for vision — matches HF exactly."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    x1 = tensor[..., :tensor.shape[-1]//2]
    x2 = tensor[..., tensor.shape[-1]//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    output = (tensor * cos) + (rotated * sin)
    return output.to(orig_dtype)


def vision_block_hybrid(x_bf16, w, device, cu_seqlens, rotary_pos_emb):
    """Single vision block: TT matmuls + CPU attention (exact).

    x_bf16: [seq, DIM] bfloat16 on CPU
    """
    seq = x_bf16.shape[0]

    # --- Attention ---
    norm1 = _rms_norm(x_bf16, w["norm1_w"])

    # QKV matmul on TT
    qkv = _tt_matmul(norm1, w["qkv_w"], device).to(torch.bfloat16)[:seq]
    if w["qkv_b"] is not None:
        qkv = qkv + w["qkv_b"]

    # Split Q, K, V + rotary (CPU)
    q, k, v = qkv.reshape(seq, 3, N_HEADS, HEAD_DIM).permute(1, 0, 2, 3).unbind(0)
    q = _apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = _apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    # Attention mask from cu_seqlens (CPU)
    attention_mask = torch.zeros([1, seq, seq], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = True

    # SDPA on CPU
    q_t = q.transpose(0, 1)  # [heads, seq, dim]
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, attention_mask, dropout_p=0.0)
    attn_out = attn_out.transpose(0, 1).reshape(seq, DIM).to(torch.bfloat16)

    # O projection on TT
    proj = _tt_matmul(attn_out, w["proj_w"], device).to(torch.bfloat16)[:seq]
    if w["proj_b"] is not None:
        proj = proj + w["proj_b"]

    # Residual
    h = x_bf16 + proj

    # --- MLP ---
    norm2 = _rms_norm(h, w["norm2_w"])

    fc1 = _tt_matmul(norm2, w["fc1_w"], device).to(torch.bfloat16)[:seq]
    if w["fc1_b"] is not None: fc1 = fc1 + w["fc1_b"]
    fc3 = _tt_matmul(norm2, w["fc3_w"], device).to(torch.bfloat16)[:seq]
    if w["fc3_b"] is not None: fc3 = fc3 + w["fc3_b"]
    gated = F.silu(fc1) * fc3
    fc2 = _tt_matmul(gated, w["fc2_w"], device).to(torch.bfloat16)[:seq]
    if w["fc2_b"] is not None: fc2 = fc2 + w["fc2_b"]

    # Residual
    return h + fc2


def vision_forward_hybrid(pixel_values, grid_thw, vision_model, blocks_tt, final_tt, device):
    """Full vision forward: patch_embed + 42 blocks (TT matmul) + merger.

    Args:
        pixel_values: [N, C*patch*patch] from processor
        grid_thw: [B, 3] grid info
        vision_model: HF vision model (for patch_embed, rotary, merger)
        blocks_tt: list of TT weight dicts from load_vision_weights
        final_tt: dict with post_norm weight
        device: ttnn device
    """
    # Patch embed on CPU (Conv2d, fast)
    with torch.no_grad():
        hidden = vision_model.patch_embed(pixel_values.to(torch.bfloat16), grid_thw)
    seq = hidden.shape[0]
    print(f"    patch_embed: seq={seq}", flush=True)

    # Rotary + cu_seqlens (CPU, reused for all blocks)
    with torch.no_grad():
        rotary_pos_emb = vision_model.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # 42 vision blocks
    x = hidden.to(torch.bfloat16)
    for i, w in enumerate(blocks_tt):
        x = vision_block_hybrid(x, w, device, cu_seqlens, rotary_pos_emb)
        if (i + 1) % 10 == 0:
            print(f"    block {i+1}/{N_BLOCKS}", flush=True)

    # Post-norm
    if "post_norm_w" in final_tt:
        x = _rms_norm(x, final_tt["post_norm_w"])

    # Merger on CPU (small, fast)
    with torch.no_grad():
        output = vision_model.merger(x.to(vision_model.merger.ln_q.weight.dtype))

    return output
