#!/usr/bin/env python3
"""
Standalone ttnn vision block test for dots.mocr.
Uses ttnn ops directly (rms_norm, linear, sdpa) without tt-metal ModelArgs.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ttnn
from dots_tt_port.vision_tower_smoke import resolve_snapshot_dir, load_vision_model


# dots.mocr vision config
DIM = 1536
N_HEADS = 12
HEAD_DIM = DIM // N_HEADS  # 128
INTERMEDIATE = 4224
EPS = 1e-5


def make_ttnn_weight(tensor, device, dtype=ttnn.bfloat16):
    """Convert PyTorch weight to ttnn tensor on device."""
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def ttnn_rms_norm(x, weight, eps=EPS):
    """RMSNorm using ttnn — stays entirely on TT device."""
    # Reshape weight for ttnn.rms_norm: [1, 1, dim//32, 32]
    return ttnn.rms_norm(x, epsilon=eps, weight=weight)


def ttnn_vision_block_forward(x, block_weights, device, seq_len):
    """One vision block forward pass using ttnn ops only."""
    w = block_weights

    # --- Attention ---
    # RMSNorm1
    norm1_out = ttnn_rms_norm(x, w["norm1"])

    # QKV projection: [seq, dim] → [seq, 3*dim]
    qkv = ttnn.linear(norm1_out, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1_out)

    # Split QKV → Q, K, V
    # qkv shape: [1, 1, seq, 3*dim]
    # We need to reshape and split
    # For now, do this on CPU (single transfer per block)
    qkv_cpu = ttnn.to_torch(qkv)
    ttnn.deallocate(qkv)

    q_cpu, k_cpu, v_cpu = qkv_cpu[..., :DIM], qkv_cpu[..., DIM:2*DIM], qkv_cpu[..., 2*DIM:]

    # Reshape for attention: [1, 1, seq, dim] → [1, n_heads, seq, head_dim]
    q_cpu = q_cpu.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)
    k_cpu = k_cpu.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)
    v_cpu = v_cpu.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)

    # TODO: Apply rotary embeddings here

    # SDPA on CPU for now (TT SDPA crashes at seq=1632)
    attn_out_cpu = torch.nn.functional.scaled_dot_product_attention(
        q_cpu.float(), k_cpu.float(), v_cpu.float(), dropout_p=0.0
    ).to(torch.bfloat16)

    # Reshape back: [1, n_heads, seq, head_dim] → [1, 1, seq, dim]
    attn_out_cpu = attn_out_cpu.permute(0, 2, 1, 3).reshape(1, 1, seq_len, DIM)

    # Transfer back to TT
    attn_out_tt = ttnn.from_torch(
        attn_out_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output projection
    proj_out = ttnn.linear(attn_out_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out_tt)

    # Residual
    h = ttnn.add(x, proj_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj_out)
    ttnn.deallocate(x)

    # --- MLP ---
    # RMSNorm2
    norm2_out = ttnn_rms_norm(h, w["norm2"])

    # fc1 (gate) and fc3 (up) projections
    fc1_out = ttnn.linear(norm2_out, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3_out = ttnn.linear(norm2_out, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2_out)

    # SiLU(fc1) * fc3 — fused op!
    gated = ttnn.mul(
        fc1_out, fc3_out,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(fc1_out)
    ttnn.deallocate(fc3_out)

    # fc2 (down) projection
    fc2_out = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)

    # Residual
    out = ttnn.add(h, fc2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h)
    ttnn.deallocate(fc2_out)

    return out


def load_block_weights(vision_state_dict, layer_num, device):
    """Load one block's weights to TT device."""
    prefix = f"blocks.{layer_num}."
    w = {}

    # RMSNorm weights: need special shape [1, 1, dim//32, 32]
    norm1_w = vision_state_dict[f"{prefix}norm1.weight"]
    norm2_w = vision_state_dict[f"{prefix}norm2.weight"]
    w["norm1"] = ttnn.from_torch(
        norm1_w.unsqueeze(0).view(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w["norm2"] = ttnn.from_torch(
        norm2_w.unsqueeze(0).view(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Linear weights: transpose for ttnn.linear (expects [out, in] → [in, out] for matmul)
    w["qkv_w"] = make_ttnn_weight(vision_state_dict[f"{prefix}attn.qkv.weight"].T, device)
    w["proj_w"] = make_ttnn_weight(vision_state_dict[f"{prefix}attn.proj.weight"].T, device)
    w["fc1_w"] = make_ttnn_weight(vision_state_dict[f"{prefix}mlp.fc1.weight"].T, device)
    w["fc2_w"] = make_ttnn_weight(vision_state_dict[f"{prefix}mlp.fc2.weight"].T, device)
    w["fc3_w"] = make_ttnn_weight(vision_state_dict[f"{prefix}mlp.fc3.weight"].T, device)

    return w


def main():
    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")

    # Load full vision state dict
    print("Loading dots.mocr vision state dict...", flush=True)
    from dots_tt_port.vision_tower_smoke import load_vision_state
    state_dict = load_vision_state(snapshot_dir, limit_layers=0)

    # Open device
    print("Opening TT device...", flush=True)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    # Load block 0 weights
    print("Loading block 0 weights to TT...", flush=True)
    t0 = time.perf_counter()
    block_w = load_block_weights(state_dict, 0, device)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Weight load: {load_ms:.0f}ms", flush=True)

    # Create test input
    seq_len = 1632  # 476x674 → 48x34 patches
    # Pad to tile boundary (32)
    padded_seq = ((seq_len + 31) // 32) * 32  # 1664

    pt_input = torch.randn(1, 1, padded_seq, DIM, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Cold run
    print("Running block 0 forward (cold)...", flush=True)
    t0 = time.perf_counter()
    tt_out = ttnn_vision_block_forward(tt_input, block_w, device, padded_seq)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"  Cold: {cold_ms:.0f}ms", flush=True)

    # Warm runs
    for i in range(3):
        tt_input_w = ttnn.from_torch(
            pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t0 = time.perf_counter()
        tt_out_w = ttnn_vision_block_forward(tt_input_w, block_w, device, padded_seq)
        warm_ms = (time.perf_counter() - t0) * 1000
        print(f"  Warm {i+1}: {warm_ms:.0f}ms", flush=True)

    out_cpu = ttnn.to_torch(tt_out_w)
    print(f"  Output shape: {out_cpu.shape}")
    print(f"  Output stats: min={out_cpu.min():.4f} max={out_cpu.max():.4f}")

    # Extrapolate
    print(f"\n  Estimated 42 blocks warm: {warm_ms * 42:.0f}ms")
    print(f"  vs current hybrid: ~2900ms")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
