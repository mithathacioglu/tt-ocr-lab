#!/usr/bin/env python3
"""
Test tt-metal native VisionBlock with dots.mocr weights.
This uses ttnn directly — no CPU transfers, everything on TT.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL_ROOT = PROJECT_ROOT / "tt-xla" / "third_party" / "tt-mlir" / "src" / "tt-mlir" / "third_party" / "tt-metal" / "src" / "tt-metal"

# Add tt-metal models to path
sys.path.insert(0, str(TT_METAL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dots_tt_port.vision_tower_smoke import resolve_snapshot_dir as _resolve
_SNAPSHOT = str(_resolve("rednote-hilab/dots.mocr"))
os.environ.setdefault("HF_MODEL", _SNAPSHOT)

import ttnn
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys_multimodal
from models.common.utility_functions import comp_pcc

from dots_tt_port.vision_tower_smoke import resolve_snapshot_dir, load_vision_model


class DotsVisionConfigCompat:
    """Wraps dots.mocr vision config to match Qwen2.5-VL expected attributes."""
    def __init__(self, dots_vision_config):
        self._cfg = dots_vision_config
        # Map dots.mocr names to Qwen2.5-VL names
        self.hidden_size = dots_vision_config.hidden_size
        self.num_heads = dots_vision_config.num_attention_heads
        self.num_attention_heads = dots_vision_config.num_attention_heads
        self.intermediate_size = dots_vision_config.intermediate_size
        self.patch_size = dots_vision_config.patch_size
        self.spatial_merge_size = dots_vision_config.spatial_merge_size
        self.temporal_patch_size = getattr(dots_vision_config, 'temporal_patch_size', 1)
        self.num_hidden_layers = dots_vision_config.num_hidden_layers
        self.depth = dots_vision_config.num_hidden_layers
        self.embed_dim = dots_vision_config.hidden_size
        # Qwen2.5-VL specific - dots.mocr doesn't have these
        self.window_size = None  # dots.mocr uses full attention everywhere
        self.fullatt_block_indexes = list(range(dots_vision_config.num_hidden_layers))  # all blocks use full attn
        self.rms_norm_eps = getattr(dots_vision_config, 'rms_norm_eps', 1e-5)

    def __getattr__(self, name):
        return getattr(self._cfg, name)


class DotsHFConfigCompat:
    """Wraps full dots.mocr config to look like Qwen2.5-VL HF config."""
    def __init__(self, hf_config):
        self._cfg = hf_config
        self.vision_config = DotsVisionConfigCompat(hf_config.vision_config)

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def main():
    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")

    # --- Load dots.mocr vision model (reference) ---
    print("Loading dots.mocr vision model (CPU reference)...", flush=True)
    vision_model, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    vision_model = vision_model.eval().to(dtype=torch.float32)

    # --- Get state dict from block 0 ---
    block0_ref = vision_model.blocks[0]
    block0_state = block0_ref.state_dict()
    print(f"Block 0 state dict keys: {list(block0_state.keys())}")

    # --- Convert to tt-metal format ---
    print("Converting state dict to tt-metal format...", flush=True)
    # First standardize HF keys
    std_state = standardize_hf_keys_multimodal(block0_state)
    print(f"After standardize: {list(std_state.keys())}")

    # Convert HF to meta (split qkv, rename keys)
    head_dim = 128  # 1536 / 12 = 128
    meta_state = convert_hf_to_meta(std_state, head_dim)
    print(f"After convert_hf_to_meta: {list(meta_state.keys())}")

    # Add state dict prefix for layer 0
    prefixed_state = {}
    for k, v in meta_state.items():
        prefixed_state[k] = v
    print(f"Final state dict keys: {list(prefixed_state.keys())}")

    # --- Setup TT device ---
    print("Opening TT device...", flush=True)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    mesh_device.enable_program_cache()

    # --- Create model args ---
    # We need to create VisionModelArgs from dots.mocr config
    # But VisionModelArgs expects HF config path. Let's use it with the snapshot dir.
    print("Creating model args...", flush=True)

    # For now, test with dummy weights first to verify the pipeline
    # Then we'll load real weights
    image_grid_thw = torch.tensor([[1, 48, 34]])  # 476x674 → 48x34 patches
    ref_seq_len = int(image_grid_thw[0, 1] * image_grid_thw[0, 2])  # 1632

    # Patch HF config before creating model args
    from transformers import AutoConfig
    real_config = AutoConfig.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    compat_config = DotsHFConfigCompat(real_config)

    # Temporarily monkey-patch AutoConfig to return our compat config
    _orig_from_pretrained = AutoConfig.from_pretrained
    AutoConfig.from_pretrained = lambda *a, **kw: compat_config

    try:
        model_args = VisionModelArgs(
            mesh_device,
            instruct=False,
            dummy_weights=True,  # We'll load weights manually
            max_batch_size=1,
            max_seq_len=1664,
        )
    finally:
        AutoConfig.from_pretrained = _orig_from_pretrained

    print(f"  vision_dim: {model_args.vision_dim}")
    print(f"  vision_n_heads: {model_args.vision_n_heads}")
    print(f"  vision_head_dim: {model_args.vision_head_dim}")
    print(f"  vision_hidden_dim: {model_args.vision_hidden_dim}")
    print(f"  CKPT_DIR: {model_args.CKPT_DIR}")

    # --- Preprocess (rotary embeddings, cu_seqlens) ---
    print("Computing rotary embeddings and cu_seqlens...", flush=True)
    seq_len = ((ref_seq_len // 128) + 1) * 128  # pad to 128

    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=model_args.vision_head_dim,
        spatial_merge_size=model_args.hf_config.vision_config.spatial_merge_size,
        window_size=getattr(model_args.hf_config.vision_config, 'window_size', None),
        patch_size=model_args.hf_config.vision_config.patch_size,
    )

    cos, sin = position_embeddings
    cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
    cos_tt = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    sin_tt = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    rot_mats = [cos_tt, sin_tt]

    transformation_mat_torch = get_rot_transformation_mat(model_args.vision_head_dim)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    # --- Get real weights for block 0 ---
    print("Loading real weights for block 0...", flush=True)
    ref_model = model_args.reference_vision_model(depth=1)  # only 1 block
    ref_block = ref_model.blocks[0]
    state_dict = standardize_hf_keys_multimodal(ref_block.state_dict())
    state_dict = convert_hf_to_meta(state_dict, model_args.vision_head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("VisionBlock", 0)
    state_dict = {f"{state_dict_prefix}{k}": v for k, v in state_dict.items()}
    print(f"  Prefixed keys: {list(state_dict.keys())[:5]}...")

    # --- Create TT VisionBlock ---
    print("Creating TT VisionBlock...", flush=True)
    tt_block = VisionBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        transformation_mats=transformation_mats,
        args=model_args,
    )

    # --- Create test input ---
    print("Running forward pass...", flush=True)
    pt_input = torch.randn(1, 1, ref_seq_len, model_args.vision_dim)
    tt_input = pt_input.clone()
    tt_input = torch.nn.functional.pad(tt_input, (0, 0, 0, seq_len - ref_seq_len))
    tt_input = model_args.prepare_residual_tensor_prefill(tt_input.squeeze(0), force_replicated=True)

    cu_seqlens_tt = ttnn.from_torch(cu_seqlens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # --- Cold run ---
    t0 = time.perf_counter()
    tt_out = tt_block(tt_input, cu_seqlens=cu_seqlens_tt, rot_mats=rot_mats)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"  Cold: {cold_ms:.0f}ms", flush=True)

    # --- Warm run ---
    tt_input2 = model_args.prepare_residual_tensor_prefill(
        torch.nn.functional.pad(pt_input.clone(), (0, 0, 0, seq_len - ref_seq_len)).squeeze(0),
        force_replicated=True,
    )
    t0 = time.perf_counter()
    tt_out2 = tt_block(tt_input2, cu_seqlens=cu_seqlens_tt, rot_mats=rot_mats)
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  Warm: {warm_ms:.0f}ms", flush=True)

    # --- Convert output and compare with reference ---
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    tt_output_torch = tt_output_torch[:, 0:1, :, :model_args.vision_dim].view(1, seq_len, -1)
    tt_output_torch = tt_output_torch[0, :ref_seq_len, :]

    # Reference
    ref_output = ref_block(
        pt_input.squeeze(0).squeeze(0),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=position_embeddings,
    )

    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(f"\n  PCC: {pcc_msg}")
    print(f"  Pass: {passing}")
    print(f"\n=== RESULT: cold={cold_ms:.0f}ms warm={warm_ms:.0f}ms PCC={pcc_msg} ===")

    # Extrapolate for 42 blocks
    print(f"\n  Extrapolated 42 blocks warm: {warm_ms * 42:.0f}ms")
    print(f"  vs hybrid CPU: ~2900ms")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
