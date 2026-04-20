#!/usr/bin/env python3
"""
Knowledge Distillation: fine-tune vision tower so TT output matches CPU output.

Strategy:
1. Generate (image, CPU_vision_embedding) pairs = training data
2. Forward through TT vision to get TT_vision_embedding
3. Loss = MSE(TT_output, CPU_output)
4. Backward through CPU (emulate TT precision with noise injection)
5. Update weights

Since TT matmul has slightly different rounding, we inject simulated
quantization noise during training so weights become robust to it.

This is essentially Quantization-Aware Training (QAT) for TT hardware.
"""
from __future__ import annotations
import os, sys, time, json, glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

torch.set_num_threads(32)

# ============ Step 1: Generate training data ============
def generate_training_data(pdf_dir, output_dir, max_pages=50, dpi=200):
    """Extract pages from PDFs and compute CPU vision embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import fitz
    from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    snapshot = resolve_snapshot_dir("rednote-hilab/dots.mocr")
    proc = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True)

    # Load vision model fp32
    vm, _ = load_vision_model(snapshot, "sdpa", limit_layers=0)
    vm = vm.eval().to(dtype=torch.float32)

    pdfs = sorted(glob.glob(str(Path(pdf_dir) / "*.pdf")))
    print(f"Found {len(pdfs)} PDFs")

    samples = []
    for pdf_path in pdfs:
        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                if len(samples) >= max_pages:
                    break
                pix = doc[page_idx].get_pixmap(dpi=dpi)
                img_path = output_dir / f"page_{len(samples):04d}.png"
                pix.save(str(img_path))

                # Process image
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": "test"},
                ]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(msgs)
                inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

                # CPU vision embedding
                pv = inputs["pixel_values"].to(torch.float32)
                gt = inputs["image_grid_thw"].to(torch.int32)

                with torch.no_grad():
                    cpu_emb = vm(pv, gt, bf16=False)

                # Save
                emb_path = output_dir / f"emb_{len(samples):04d}.pt"
                meta = {
                    "image": str(img_path),
                    "pixel_values_shape": list(pv.shape),
                    "grid_thw": gt.tolist(),
                    "embedding_shape": list(cpu_emb.shape),
                }
                torch.save({
                    "pixel_values": pv.cpu(),
                    "grid_thw": gt.cpu(),
                    "cpu_embedding": cpu_emb.cpu(),
                    "meta": meta,
                }, emb_path)

                samples.append(str(emb_path))
                print(f"  [{len(samples)}/{max_pages}] {pdf_path}:p{page_idx+1} pv={pv.shape} emb={cpu_emb.shape}", flush=True)
            doc.close()
        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")

    # Save manifest
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples, "count": len(samples)}, indent=2))
    print(f"\nGenerated {len(samples)} training samples in {output_dir}")
    return samples


class VisionDistillDataset(Dataset):
    def __init__(self, sample_paths):
        self.paths = sample_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        return data["pixel_values"], data["grid_thw"], data["cpu_embedding"]


# ============ Step 2: Simulated TT quantization noise ============
class SimulatedTTNoise(nn.Module):
    """Inject noise that simulates TT matmul precision difference.

    TT Wormhole bf16 matmul differs from CPU bf16 by ~0.001 per element per layer.
    After 42 layers this compounds. We simulate this by adding scaled noise
    during training so the model learns to be robust.
    """
    def __init__(self, noise_scale=0.001):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_scale * x.abs().mean()
            return x + noise
        return x


# ============ Step 3: Training loop ============
def train_distill(
    vision_model,
    train_loader,
    epochs=5,
    lr=1e-5,
    noise_scale=0.001,
    save_dir="ocr_lab/distilled_weights",
):
    """Fine-tune vision model with TT noise injection."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Add noise injection after each block
    noise_layers = nn.ModuleList([SimulatedTTNoise(noise_scale) for _ in range(42)])

    optimizer = torch.optim.AdamW(vision_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    vision_model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        t_epoch = time.perf_counter()

        for batch_idx, (pv, gt, cpu_emb) in enumerate(train_loader):
            # pv: [N, C], gt: [1, 3], cpu_emb: [merged_seq, dim]
            pv = pv.squeeze(0).float()
            gt = gt.squeeze(0).to(torch.int32)
            cpu_emb = cpu_emb.squeeze(0).float()

            # Forward with noise injection
            # Manually run blocks with noise
            h = vision_model.patch_embed(pv, gt)
            rotary = vision_model.rot_pos_emb(gt)
            cu = F.pad(
                torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32),
                (1, 0), value=0
            )

            for i, blk in enumerate(vision_model.blocks):
                h = blk(h, cu_seqlens=cu, rotary_pos_emb=rotary)
                # Inject simulated TT noise after each block
                h = noise_layers[i](h)

            if vision_model.config.post_norm:
                h = vision_model.post_trunk_norm(h)

            tt_emb = vision_model.merger(h)

            # Loss: match CPU embeddings
            loss = F.mse_loss(tt_emb, cpu_emb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vision_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.6f} LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.perf_counter() - t_epoch
        print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.6f} time={epoch_time:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / "best_vision_model.pt"
            torch.save(vision_model.state_dict(), save_path)
            print(f"  Saved best model (loss={best_loss:.6f}) to {save_path}")

    return vision_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "train", "both"], default="both")
    parser.add_argument("--pdf-dir", default="/home/mlops/Desktop/dosyalar")
    parser.add_argument("--data-dir", default="ocr_lab/distill_data")
    parser.add_argument("--save-dir", default="ocr_lab/distilled_weights")
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise-scale", type=float, default=0.002)
    args = parser.parse_args()

    if args.phase in ("generate", "both"):
        print("=" * 60)
        print("Phase 1: Generate training data")
        print("=" * 60)
        samples = generate_training_data(args.pdf_dir, args.data_dir, max_pages=args.max_pages)

    if args.phase in ("train", "both"):
        print("\n" + "=" * 60)
        print("Phase 2: Train with distillation")
        print("=" * 60)

        # Load manifest
        manifest = json.loads(Path(args.data_dir, "manifest.json").read_text())
        sample_paths = manifest["samples"]
        print(f"Training samples: {len(sample_paths)}")

        dataset = VisionDistillDataset(sample_paths)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Load vision model
        from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir
        snapshot = resolve_snapshot_dir("rednote-hilab/dots.mocr")
        vm, _ = load_vision_model(snapshot, "sdpa", limit_layers=0)
        vm = vm.to(dtype=torch.float32)

        trained = train_distill(vm, loader, epochs=args.epochs, lr=args.lr,
                                 noise_scale=args.noise_scale, save_dir=args.save_dir)
        print("\nTraining complete!")
