#!/usr/bin/env python3
"""
Full 42-block ttnn vision tower on 1.pdf — measure warm time + compare with hybrid output.
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

import ttnn
from dots_tt_port.vision_tower_smoke import (
    TowerHybridCpuUnary, resolve_snapshot_dir, load_vision_model, load_vision_state,
)
from ocr_lab.fixed_page import fixed_page_message

DIM = 1536
N_HEADS = 12
HEAD_DIM = 128
INTERMEDIATE = 4224
EPS = 1e-5
N_BLOCKS = 42


def make_ttnn_weight(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def load_all_block_weights(state_dict, device):
    blocks = []
    for i in range(N_BLOCKS):
        prefix = f"blocks.{i}."
        w = {}
        norm1_w = state_dict[f"{prefix}norm1.weight"]
        norm2_w = state_dict[f"{prefix}norm2.weight"]
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
        w["qkv_w"] = make_ttnn_weight(state_dict[f"{prefix}attn.qkv.weight"].T, device)
        w["proj_w"] = make_ttnn_weight(state_dict[f"{prefix}attn.proj.weight"].T, device)
        w["fc1_w"] = make_ttnn_weight(state_dict[f"{prefix}mlp.fc1.weight"].T, device)
        w["fc2_w"] = make_ttnn_weight(state_dict[f"{prefix}mlp.fc2.weight"].T, device)
        w["fc3_w"] = make_ttnn_weight(state_dict[f"{prefix}mlp.fc3.weight"].T, device)
        blocks.append(w)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/{N_BLOCKS} blocks", flush=True)
    return blocks


def ttnn_vision_block(x, w, device, seq_len):
    """Single block: norm+attn+norm+mlp. Attention on CPU (SDPA), rest on TT."""
    # RMSNorm1
    norm1_out = ttnn.rms_norm(x, epsilon=EPS, weight=w["norm1"])

    # QKV on TT
    qkv = ttnn.linear(norm1_out, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1_out)

    # Attention: transfer to CPU for SDPA (seq=1632 too large for TT SDPA)
    qkv_cpu = ttnn.to_torch(qkv)
    ttnn.deallocate(qkv)

    q, k, v = qkv_cpu[..., :DIM], qkv_cpu[..., DIM:2*DIM], qkv_cpu[..., 2*DIM:]
    q = q.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)
    k = k.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)
    v = v.view(1, 1, seq_len, N_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4).reshape(1, N_HEADS, seq_len, HEAD_DIM)

    # TODO: rotary embeddings would go here

    attn_cpu = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), dropout_p=0.0
    ).to(torch.bfloat16)
    attn_cpu = attn_cpu.permute(0, 2, 1, 3).reshape(1, 1, seq_len, DIM)

    attn_tt = ttnn.from_torch(
        attn_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output projection
    proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_tt)

    # Residual
    h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj)
    ttnn.deallocate(x)

    # RMSNorm2
    norm2_out = ttnn.rms_norm(h, epsilon=EPS, weight=w["norm2"])

    # MLP: SiLU(fc1) * fc3 → fc2
    fc1 = ttnn.linear(norm2_out, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3 = ttnn.linear(norm2_out, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2_out)

    gated = ttnn.mul(fc1, fc3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                     dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(fc1)
    ttnn.deallocate(fc3)

    fc2 = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)

    out = ttnn.add(h, fc2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h)
    ttnn.deallocate(fc2)
    return out


def main():
    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")
    IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

    # Build real image inputs
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "test"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    pixel_values = inputs["pixel_values"]  # [1632, 588]
    grid_thw = inputs["image_grid_thw"]    # [[1, 48, 34]]

    # Open TT device
    print("Opening TT device...", flush=True)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    # === CPU REFERENCE (no TT, just PyTorch) ===
    print("\n=== CPU REFERENCE ===", flush=True)
    vm_ref, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    vm_ref = vm_ref.eval().to(dtype=torch.bfloat16)
    with torch.no_grad():
        t0 = time.perf_counter()
        ref_out = vm_ref(pixel_values.to(dtype=torch.bfloat16), grid_thw.to(dtype=torch.int32))
        ref_ms = (time.perf_counter() - t0) * 1000
        ref_cpu = ref_out.detach().cpu().float()
    print(f"  CPU reference: {ref_ms:.0f}ms  shape={ref_cpu.shape}", flush=True)
    hybrid_ms = 2900.0  # Known from previous benchmarks
    del vm_ref, ref_out

    # === TTNN NATIVE ===
    print("\n=== TTNN NATIVE (42 blocks) ===", flush=True)
    state_dict = load_vision_state(snapshot_dir, limit_layers=0)

    print("Loading weights to TT...", flush=True)
    t0 = time.perf_counter()
    all_blocks = load_all_block_weights(state_dict, device)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Weight load: {load_ms:.0f}ms", flush=True)

    # Patch embed on CPU (conv2d, small)
    print("Running patch embed on CPU...", flush=True)
    vm_for_patch, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    vm_for_patch = vm_for_patch.eval().to(dtype=torch.bfloat16)
    with torch.no_grad():
        hidden_cpu = vm_for_patch.patch_embed(pixel_values.to(dtype=torch.bfloat16), grid_thw.to(dtype=torch.int32))
    seq_len = hidden_cpu.shape[0]  # 1632
    padded_seq = ((seq_len + 31) // 32) * 32  # 1664
    hidden_padded = torch.nn.functional.pad(hidden_cpu, (0, 0, 0, padded_seq - seq_len)).unsqueeze(0).unsqueeze(0)
    print(f"  Patch embed: seq={seq_len}, padded={padded_seq}, shape={hidden_padded.shape}", flush=True)

    # Transfer to TT
    tt_hidden = ttnn.from_torch(
        hidden_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Cold run
    print("Running 42 blocks (cold)...", flush=True)
    t0 = time.perf_counter()
    x = tt_hidden
    for i, bw in enumerate(all_blocks):
        x = ttnn_vision_block(x, bw, device, padded_seq)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"  Cold: {cold_ms:.0f}ms", flush=True)

    # Warm run
    print("Running 42 blocks (warm)...", flush=True)
    tt_hidden2 = ttnn.from_torch(
        hidden_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t0 = time.perf_counter()
    x2 = tt_hidden2
    for i, bw in enumerate(all_blocks):
        x2 = ttnn_vision_block(x2, bw, device, padded_seq)
    warm_ms = (time.perf_counter() - t0) * 1000
    native_cpu = ttnn.to_torch(x2)
    print(f"  Warm: {warm_ms:.0f}ms", flush=True)

    # Trim padding and compare
    native_trimmed = native_cpu[0, 0, :seq_len, :].float()

    # Note: ref_cpu is post-merger output, native is pre-merger
    # For fair comparison we need merger too, but let's compare embeddings shape first
    print(f"\n  Hybrid output: {ref_cpu.shape}")
    print(f"  Native output: {native_trimmed.shape}")

    if ref_cpu.shape == native_trimmed.shape:
        diff = (ref_cpu - native_trimmed).abs()
        cos_sim = F.cosine_similarity(ref_cpu.flatten().unsqueeze(0), native_trimmed.flatten().unsqueeze(0))
        print(f"  Max diff: {diff.max():.4f}  Mean diff: {diff.mean():.4f}")
        print(f"  Cosine sim: {cos_sim.item():.6f}")
    else:
        print(f"  Shape mismatch — hybrid includes merger, native doesn't")
        print(f"  Native pre-merger stats: min={native_trimmed.min():.4f} max={native_trimmed.max():.4f}")

    print(f"\n{'='*60}")
    print(f"  HYBRID warm:  {hybrid_ms:.0f}ms")
    print(f"  NATIVE warm:  {warm_ms:.0f}ms")
    print(f"  Speedup: {hybrid_ms/warm_ms:.2f}x")
    print(f"{'='*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
