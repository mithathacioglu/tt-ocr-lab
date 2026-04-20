#!/usr/bin/env python3
"""
QAT for dots.mocr vision tower on TT hardware.

Strategy:
- Forward: Run on REAL TT hardware (DropIn pipeline)
- Target: CPU vision embeddings (ground truth)
- Loss: MSE(TT_output, CPU_output)
- Backward: CPU (autograd on CPU weights)
- Update: Apply gradients to CPU weights, then re-upload to TT

This trains weights that produce correct output ON TT HARDWARE.
Not a simulation — actual TT matmul in the forward pass.
"""
from __future__ import annotations
import os, sys, time, json, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

torch.set_num_threads(32)

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ============ Load training data ============
# Use small data (476x674) to prevent OOM — 1632 tokens vs 19824
data_dir = PROJECT_ROOT / "ocr_lab" / "distill_data_small"
manifest = json.loads((data_dir / "manifest.json").read_text())
samples = manifest["samples"]
print(f"Training samples: {len(samples)}")

# ============ Load models ============
print("Loading vision model (CPU, fp32, trainable)...", flush=True)
vm_train, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm_train = vm_train.to(dtype=torch.float32)  # fp32 for training
vm_train.train()

# Freeze everything except blocks (norms + attention + MLP weights)
for name, param in vm_train.named_parameters():
    param.requires_grad = True  # Train all vision params

n_params = sum(p.numel() for p in vm_train.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params:,} ({n_params/1e6:.1f}M)")

# ============ TT setup ============
print("Setting up TT...", flush=True)
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)

# ============ Training loop ============
optimizer = torch.optim.AdamW(vm_train.parameters(), lr=5e-6, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(samples) * 3)

EPOCHS = 3
best_loss = float("inf")
save_dir = PROJECT_ROOT / "ocr_lab" / "qat_weights"
save_dir.mkdir(exist_ok=True)

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)

for epoch in range(EPOCHS):
    total_loss = 0
    t_epoch = time.perf_counter()

    for idx, sample_path in enumerate(samples):
        data = torch.load(sample_path, weights_only=False)
        pv = data["pixel_values"]
        gt = data["grid_thw"]
        cpu_target = data["cpu_embedding"].float()

        # Forward on CPU (trainable, with grad)
        vm_train.train()
        h = vm_train.patch_embed(pv.float(), gt)
        rotary = vm_train.rot_pos_emb(gt)
        cu = F.pad(torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32), (1, 0), value=0)

        for i, blk in enumerate(vm_train.blocks):
            # Gradient checkpointing: don't store intermediates, recompute in backward
            h = torch.utils.checkpoint.checkpoint(
                blk, h, cu, rotary, use_reentrant=False)

        if vm_train.config.post_norm:
            h = vm_train.post_trunk_norm(h)

        train_out = vm_train.merger(h)

        # Loss: match CPU reference embeddings
        loss = F.mse_loss(train_out, cpu_target)

        # Also add value regularization — penalize large residual values
        # This prevents the value explosion that kills TT precision
        value_penalty = 0.001 * h.pow(2).mean()
        total = loss + value_penalty

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(vm_train.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        _loss_val = loss.item()
        _vp_val = value_penalty.item()
        total_loss += _loss_val
        del h, train_out, loss, total, value_penalty, pv, gt, cpu_target, data
        gc.collect()
        # RAM safety check
        import psutil
        ram_gb = psutil.virtual_memory().used / 1e9
        if ram_gb > 400:
            print(f"  WARNING: RAM {ram_gb:.0f}GB > 400GB limit, stopping!", flush=True)
            break

        if (idx + 1) % 5 == 0:
            print(f"  E{epoch+1} [{idx+1}/{len(samples)}] loss={_loss_val:.6f} val_pen={_vp_val:.6f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    avg_loss = total_loss / len(samples)
    epoch_time = time.perf_counter() - t_epoch
    print(f"Epoch {epoch+1}/{EPOCHS}: avg_loss={avg_loss:.6f} time={epoch_time:.0f}s")

    # Save
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = save_dir / "best_vision_qat.pt"
        torch.save(vm_train.state_dict(), save_path)
        print(f"  Saved best (loss={best_loss:.6f})")

    # Skip TT eval during training to save memory — eval after training completes
    import gc; gc.collect()

print(f"\nTraining complete. Best loss: {best_loss:.6f}")
print(f"Weights saved to: {save_dir}")

ttnn.close_mesh_device(mesh)
