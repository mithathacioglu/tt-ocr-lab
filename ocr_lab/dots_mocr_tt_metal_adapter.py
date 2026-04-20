"""
Adapter: Make dots.mocr config compatible with tt-metal's Qwen2.5-VL ModelArgs/VisionModelArgs.
Patches HF config + ModelArgs + VisionModelArgs to handle dots.mocr's custom config class.
"""
from __future__ import annotations
import os
from pathlib import Path
from dots_tt_port.vision_tower_smoke import resolve_snapshot_dir


def patch_dots_mocr_config_for_tt_metal():
    """
    Patches everything needed so tt-metal's ModelArgs/VisionModelArgs
    can load dots.mocr. Call BEFORE creating any ModelArgs.

    Returns the snapshot dir path.
    """
    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")
    snapshot_path = str(snapshot_dir)

    # HF_MODEL must point to actual checkpoint dir
    os.environ["HF_MODEL"] = snapshot_path

    # --- Patch 1: trust_remote_code globally ---
    import transformers.dynamic_module_utils as _dmu
    _dmu.resolve_trust_remote_code = lambda *a, **kw: True

    # --- Patch 2: AutoConfig — add missing Qwen2.5-VL fields ---
    from transformers import AutoConfig
    _orig_from_pretrained = AutoConfig.from_pretrained

    def _patched_from_pretrained(*args, **kwargs):
        kwargs["trust_remote_code"] = True
        config = _orig_from_pretrained(*args, **kwargs)

        if hasattr(config, 'vision_config'):
            vc = config.vision_config
            if not hasattr(vc, 'num_heads'):
                vc.num_heads = getattr(vc, 'num_attention_heads', 12)
            if not hasattr(vc, 'depth'):
                vc.depth = getattr(vc, 'num_hidden_layers', 42)
            if not hasattr(vc, 'window_size') or vc.window_size is None:
                # dots.mocr: no windowed attention. Set to a large value so
                # qwen2_5_vl_get_window_index doesn't divide by None.
                # All blocks are in fullatt_block_indexes anyway, so window path is never used.
                vc.window_size = 9999
            if not hasattr(vc, 'fullatt_block_indexes'):
                n = getattr(vc, 'num_hidden_layers', 42)
                vc.fullatt_block_indexes = list(range(n))
            if not hasattr(vc, 'out_hidden_size'):
                vc.out_hidden_size = getattr(vc, 'hidden_size', 1536)

        try:
            if not hasattr(config, 'dim'):
                config.dim = getattr(config, 'hidden_size', 1536)
            if not hasattr(config, 'n_layers'):
                config.n_layers = getattr(config, 'num_hidden_layers', 28)
            if not hasattr(config, 'n_heads'):
                config.n_heads = getattr(config, 'num_attention_heads', 12)
            if not hasattr(config, 'n_kv_heads'):
                config.n_kv_heads = getattr(config, 'num_key_value_heads', 2)
        except (AttributeError, TypeError, Exception):
            pass
        return config

    AutoConfig.from_pretrained = _patched_from_pretrained

    # --- Patch 3: ModelArgs.__init__ — fix model_name + extend decoder opts ---
    import ttnn as _ttnn
    from models.tt_transformers.tt import model_config as mc
    _orig_init = mc.ModelArgs.__init__

    def _patched_init(self, *args, **kwargs):
        self.trust_remote_code_hf = True
        _orig_init(self, *args, **kwargs)
        # Force bfloat16 weights for vision tower (prevents bfloat8_b overflow at deep layers)
        import ttnn as _t
        self._vision_weight_dtype = _t.bfloat16
        if self.model_name and ('dots' in self.model_name.lower() or len(self.model_name) > 30):
            self.model_name = "dots.mocr"
            if hasattr(self, '_base_model_name'):
                self._base_model_name = "dots.mocr"
        # HF-style RoPE (no_qkv_permute weights + ttnn.experimental.rotary_embedding)
        self.use_hf_rope = True
        # Extend DECODERS_OPTIMIZATIONS for 42 vision layers
        # Force activation_dtype=bfloat16 to prevent bfloat8_b clamp at deep layers
        import ttnn as _ttnn
        model_cfg = self.get_model_config()
        if "DECODERS_OPTIMIZATIONS" in model_cfg:
            dec_opt = model_cfg["DECODERS_OPTIMIZATIONS"]
            if hasattr(dec_opt, 'decoder_optimizations'):
                existing = dec_opt.decoder_optimizations
                if len(existing) < 42:
                    last_entry = existing[len(existing) - 1]
                    for i in range(len(existing), 42):
                        existing[i] = last_entry
                # Force bfloat16 activations for all layers (prevents bfloat8_b overflow at deep layers)
                for i in existing:
                    opt = existing[i]
                    if hasattr(opt, 'activation_dtype'):
                        opt.activation_dtype = _ttnn.bfloat16
                    if hasattr(opt, 'qkv_dtype'):
                        opt.qkv_dtype = _ttnn.bfloat16
                    if hasattr(opt, 'wo_dtype'):
                        opt.wo_dtype = _ttnn.bfloat16

    mc.ModelArgs.__init__ = _patched_init

    # --- Patch 3b: get_hf_model_cls — dots.mocr not in AutoModel mapping ---
    _orig_get_hf_model_cls = mc.ModelArgs.get_hf_model_cls

    def _patched_get_hf_model_cls(self):
        try:
            return _orig_get_hf_model_cls(self)
        except ValueError:
            # dots.mocr custom config not in standard mapping — use AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM

    mc.ModelArgs.get_hf_model_cls = _patched_get_hf_model_cls

    # --- Patch 3c: load_state_dict — handle dots.mocr safetensors ---
    _orig_load_state_dict = mc.ModelArgs.load_state_dict

    def _patched_load_state_dict(self):
        from models.tt_transformers.tt.load_checkpoints import (
            load_hf_state_dict, standardize_hf_keys, convert_hf_to_meta_no_qkv_permute as convert_hf_to_meta,
        )
        from loguru import logger

        logger.info(f"Loading dots.mocr state dict from {self.CKPT_DIR}")
        raw_state = load_hf_state_dict(self.CKPT_DIR)

        # dots.mocr keys: model.layers.X.self_attn.q_proj.weight etc.
        # Strip "model." prefix if present
        stripped = {}
        for k, v in raw_state.items():
            if k.startswith("model."):
                stripped[k[len("model."):]] = v
            else:
                stripped[k] = v

        # Standardize + convert to meta format
        std = standardize_hf_keys(stripped)
        meta = convert_hf_to_meta(std, self.head_dim)
        logger.info(f"  Loaded {len(meta)} keys, sample: {list(meta.keys())[:3]}")
        return meta

    mc.ModelArgs.load_state_dict = _patched_load_state_dict

    # --- Patch 4: VisionModelArgs.reference_vision_model — use dots.mocr's own vision model ---
    from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

    def _dots_reference_vision_model(self, depth=None):
        """Load dots.mocr vision model directly instead of Qwen2.5-VL's."""
        from dots_tt_port.vision_tower_smoke import load_vision_model
        import torch

        actual_depth = depth if depth is not None else self.hf_config.vision_config.depth
        vision_model, _ = load_vision_model(
            Path(self.CKPT_DIR), "sdpa", limit_layers=actual_depth
        )
        vision_model = vision_model.eval().to(dtype=torch.bfloat16)
        return vision_model

    VisionModelArgs.reference_vision_model = _dots_reference_vision_model

    # --- Patch 5: DropInVisionTransformer — fix state_dict conversion for dots.mocr ---
    from models.demos.qwen25_vl.tt import model as qwen_model
    from models.tt_transformers.tt.load_checkpoints import (
        convert_hf_to_meta_no_qkv_permute, standardize_hf_keys_multimodal,
    )

    _orig_dropin_init = qwen_model.DropInVisionTransformer.__init__

    def _patched_dropin_init(self, reference_model, model_args, dtype=None, debug=False):
        import torch
        super(qwen_model.DropInVisionTransformer, self).__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        raw_state = reference_model.state_dict()

        # Add "visual." prefix
        prefixed_state = {}
        for k, v in raw_state.items():
            prefixed_state[f"visual.{k}"] = v

        # Standardize + convert WITH QKV permutation (Meta format to match rotary_embedding_llama)
        from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta
        std_state = standardize_hf_keys_multimodal(prefixed_state)
        meta_state = convert_hf_to_meta(std_state, model_args.vision_head_dim)

        # Fix MLP key names: fc1/fc2/fc3 → w1/w2/w3
        import torch as _torch
        final_state = {}
        for k, v in meta_state.items():
            k = k.replace(".feed_forward.fc1.", ".feed_forward.w1.")
            k = k.replace(".feed_forward.fc2.", ".feed_forward.w2.")
            k = k.replace(".feed_forward.fc3.", ".feed_forward.w3.")
            final_state[k] = v

        # Add zero biases (dots.mocr has none, tt-metal MLP expects them)
        for i in range(42):
            prefix = f"visual.blocks.{i}.feed_forward"
            if f"{prefix}.w1.bias" not in final_state:
                final_state[f"{prefix}.w1.bias"] = _torch.zeros(4224)
            if f"{prefix}.w2.bias" not in final_state:
                final_state[f"{prefix}.w2.bias"] = _torch.zeros(1536)
            if f"{prefix}.w3.bias" not in final_state:
                final_state[f"{prefix}.w3.bias"] = _torch.zeros(4224)
            attn_prefix = f"visual.blocks.{i}.attention"
            for proj in ["wq", "wk", "wv"]:
                if f"{attn_prefix}.{proj}.bias" not in final_state:
                    final_state[f"{attn_prefix}.{proj}.bias"] = _torch.zeros(1536)
            if f"{attn_prefix}.wo.bias" not in final_state:
                final_state[f"{attn_prefix}.wo.bias"] = _torch.zeros(1536)

        if dtype is None:
            import ttnn
            dtype = ttnn.bfloat16  # All vision weights in bf16 for precision

        self.tt_model = qwen_model.VisionTransformer(
            args=model_args,
            state_dict=final_state,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

    qwen_model.DropInVisionTransformer.__init__ = _patched_dropin_init

    # --- Patch 6: DropIn.forward — use dots.mocr rotary instead of Qwen2.5-VL ---
    _orig_dropin_forward = qwen_model.DropInVisionTransformer.forward

    def _patched_dropin_forward(self, pixel_values, grid_thw):
        import torch
        import ttnn as _ttnn
        from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []

        for grid_thw_single in all_grid_thw:
            grid_thw_single = grid_thw_single.unsqueeze(0)
            n_pixels = grid_thw_single.prod().item()
            pv = all_pixel_values[:n_pixels, :]
            all_pixel_values = all_pixel_values[n_pixels:]

            unpadded_seq_len = (grid_thw_single[:, 1] * grid_thw_single[:, 2]).sum().item()
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048

            # dots.mocr rotary: own pos_ids + own rotary_pos_emb, then convert HF→Meta
            ref_model = self.reference_model
            pos_ids = ref_model.get_pos_ids_by_grid(grid_thw_single.cpu())
            pos_ids = torch.cat(pos_ids, dim=0)
            max_grid = int(grid_thw_single.cpu()[:, 1:].max().item())
            rotary_full = ref_model.rotary_pos_emb(max_grid).cpu()
            rotary = rotary_full[pos_ids].flatten(1).float()
            # HF format: [c0,c1,...,c63,c0,c1,...,c63]
            cos_hf = rotary.cos().unsqueeze(1).repeat(1, 1, 2)
            sin_hf = rotary.sin().unsqueeze(1).repeat(1, 1, 2)
            # Convert to Meta format for rotary_embedding_llama: [c0,c0,c1,c1,...,c63,c63]
            cos_meta, sin_meta = convert_rope_style_hf_to_meta(cos_hf, sin_hf)

            cos_padded = torch.nn.functional.pad(cos_meta, (0, 0, 0, seq_len - unpadded_seq_len), value=1).unsqueeze(0).unsqueeze(0)
            sin_padded = torch.nn.functional.pad(sin_meta, (0, 0, 0, seq_len - unpadded_seq_len), value=0).unsqueeze(0).unsqueeze(0)
            cos_tt = _ttnn.from_torch(cos_padded, dtype=_ttnn.bfloat16, layout=_ttnn.TILE_LAYOUT,
                                       device=self.model_args.mesh_device,
                                       mesh_mapper=_ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0))
            sin_tt = _ttnn.from_torch(sin_padded, dtype=_ttnn.bfloat16, layout=_ttnn.TILE_LAYOUT,
                                       device=self.model_args.mesh_device,
                                       mesh_mapper=_ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0))
            rot_mats = [cos_tt, sin_tt]

            # cu_seqlens + window_index from Qwen2.5-VL preprocessing
            from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
            cu_seqlens, cu_window_seqlens, _, window_index = qwen2_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw_single,
                head_dim=self.model_args.vision_head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
                window_size=self.model_args.hf_config.vision_config.window_size,
                patch_size=self.model_args.hf_config.vision_config.patch_size,
            )

            # Patch embed using dots.mocr reference model
            patch_input = ref_model.patch_embed(pv)

            # Prepare input
            tt_input = self.tt_model.prepare_input(patch_input, window_index, seq_len)

            # Full attention window info: single window = entire sequence
            # This makes attention use regular SDPA instead of windowed SDPA fallback
            full_win = {"uniform": True, "window_size": seq_len, "num_windows": 1}

            # Forward
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,
                cu_seqlens=_ttnn.from_torch(cu_seqlens, dtype=_ttnn.uint32, layout=_ttnn.ROW_MAJOR_LAYOUT, device=self.model_args.mesh_device),
                cu_window_seqlens=_ttnn.from_torch(cu_window_seqlens, dtype=_ttnn.uint32, layout=_ttnn.ROW_MAJOR_LAYOUT, device=self.model_args.mesh_device),
                windowed_window_info=full_win,
                full_window_info=full_win,
            )

            # Cleanup
            _ttnn.deallocate(tt_input)
            _ttnn.deallocate(cos_tt)
            _ttnn.deallocate(sin_tt)

            # To torch
            tt_output_torch = _ttnn.to_torch(tt_out, mesh_composer=_ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1))
            _ttnn.deallocate(tt_out)

            out_hidden = self.model_args.hf_config.vision_config.out_hidden_size
            tt_output_torch = tt_output_torch[:, 0:1, :, :out_hidden].squeeze(0).squeeze(0)

            # Reverse window index
            reverse_indices = torch.argsort(window_index)
            final_output = tt_output_torch[reverse_indices, :]
            final_outputs.append(final_output)

        return torch.cat(final_outputs, dim=0)

    qwen_model.DropInVisionTransformer.forward = _patched_dropin_forward

    # --- Patch 7: (reserved for precision fixes) ---
    pass

    return snapshot_dir
