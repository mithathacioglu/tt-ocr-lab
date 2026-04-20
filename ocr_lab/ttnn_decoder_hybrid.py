"""
Standalone ttnn decoder: matmul on TT, attention SDPA on CPU.
With KV cache for autoregressive decode.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
import ttnn

DIM = 1536
N_HEADS = 12
N_KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 8960
EPS = 1e-6
N_LAYERS = 28
GQA_GROUPS = N_HEADS // N_KV_HEADS  # 6
SCALE = HEAD_DIM ** -0.5  # 1/sqrt(128) ≈ 0.0884
DECODE_CHUNK_SIZE = 256


WEIGHT_DTYPE = ttnn.bfloat8_b  # Decode weights — proven correct for single-token decode

# High-precision compute config for prefill matmuls (prevents accumulation drift on long sequences)
PREFILL_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def make_weight(t, dev, dtype=None):
    if dtype is None:
        dtype = WEIGHT_DTYPE
    return ttnn.from_torch(t.T.contiguous().unsqueeze(0).unsqueeze(0),
                           dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def load_decoder_weights(state_dict, device):
    layers = []
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        w = {}
        w["attn_norm"] = ttnn.from_torch(
            state_dict[f"{p}attention_norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["ffn_norm"] = ttnn.from_torch(
            state_dict[f"{p}ffn_norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["wq"] = make_weight(state_dict[f"{p}attention.wq.weight"], device)
        w["wk"] = make_weight(state_dict[f"{p}attention.wk.weight"], device)
        w["wv"] = make_weight(state_dict[f"{p}attention.wv.weight"], device)
        w["wo"] = make_weight(state_dict[f"{p}attention.wo.weight"], device)
        # Bias on CPU (for prefill) and TT (for decode)
        w["qb"] = state_dict.get(f"{p}attention.wq.bias")
        w["kb"] = state_dict.get(f"{p}attention.wk.bias")
        w["vb"] = state_dict.get(f"{p}attention.wv.bias")
        # Pre-load bias tensors on TT device for decode
        # QKV bias fused: [1, 1, 1, q_dim+k_dim+v_dim] for broadcast add
        qb = state_dict.get(f"{p}attention.wq.bias")
        kb = state_dict.get(f"{p}attention.wk.bias")
        vb = state_dict.get(f"{p}attention.wv.bias")
        if qb is not None:
            # Reshape for broadcast: bias needs to match linear output shape [1,1,1,dim]
            w["qb_tt"] = ttnn.from_torch(qb.reshape(1,1,1,-1).to(torch.bfloat16),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            w["qb_tt"] = None
        if kb is not None:
            w["kb_tt"] = ttnn.from_torch(kb.reshape(1,1,1,-1).to(torch.bfloat16),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            w["kb_tt"] = None
        if vb is not None:
            w["vb_tt"] = ttnn.from_torch(vb.reshape(1,1,1,-1).to(torch.bfloat16),
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            w["vb_tt"] = None
        w["w1"] = make_weight(state_dict[f"{p}feed_forward.w1.weight"], device)
        w["w2"] = make_weight(state_dict[f"{p}feed_forward.w2.weight"], device)
        w["w3"] = make_weight(state_dict[f"{p}feed_forward.w3.weight"], device)
        # Fused QKV weight for optimized decode (1 matmul instead of 3)
        wq_raw = state_dict[f"{p}attention.wq.weight"]   # [N_HEADS*HEAD_DIM, DIM]
        wk_raw = state_dict[f"{p}attention.wk.weight"]   # [N_KV_HEADS*HEAD_DIM, DIM]
        wv_raw = state_dict[f"{p}attention.wv.weight"]   # [N_KV_HEADS*HEAD_DIM, DIM]
        w["wqkv"] = make_weight(torch.cat([wq_raw, wk_raw, wv_raw], dim=0), device)
        # Fused QKV bias for optimized decode
        if qb is not None and kb is not None and vb is not None:
            qkvb_fused = torch.cat([qb, kb, vb], dim=0)
            w["qkvb_fused_tt"] = ttnn.from_torch(
                qkvb_fused.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            w["qkvb_fused_tt"] = None
        layers.append(w)
    final = {}
    final["norm"] = ttnn.from_torch(
        state_dict["norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final["lm_head"] = make_weight(state_dict["output.weight"], device) if "output.weight" in state_dict else make_weight(state_dict["tok_embeddings.weight"], device)
    return layers, final


def apply_rotary(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    q_rot = (q.float() * cos.float()) + (rotate_half(q.float()) * sin.float())
    k_rot = (k.float() * cos.float()) + (rotate_half(k.float()) * sin.float())
    return q_rot.to(torch.bfloat16), k_rot.to(torch.bfloat16)


def _cpu_rms_norm(x, weight_cpu):
    """RMS norm on CPU matching HF Qwen2RMSNorm exactly."""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + EPS)
    return (weight_cpu * x_normed.to(input_dtype)).to(input_dtype)


def decoder_layer_prefill(x_bf16, w, device, cos_cpu, sin_cpu, seq_len, kv_cache_k, kv_cache_v, layer_idx):
    """Prefill: ALL computation on CPU (matches HF exactly).

    CPU matmuls ensure zero drift on long sequences.
    TT matmul gives different results due to tiling reduction order,
    which compounds over 28 layers and causes accuracy loss at seq > ~500.
    """
    # CPU weights (cached on first call per layer)
    if "wq_cpu" not in w:
        # Weights stored transposed in TT: [1,1,DIM,out_dim]. Just squeeze, no .T needed.
        w["wq_cpu"] = ttnn.to_torch(w["wq"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["wk_cpu"] = ttnn.to_torch(w["wk"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["wv_cpu"] = ttnn.to_torch(w["wv"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["wo_cpu"] = ttnn.to_torch(w["wo"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["w1_cpu"] = ttnn.to_torch(w["w1"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["w2_cpu"] = ttnn.to_torch(w["w2"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["w3_cpu"] = ttnn.to_torch(w["w3"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
        w["attn_norm_cpu"] = ttnn.to_torch(w["attn_norm"]).reshape(-1).to(torch.bfloat16)
        w["ffn_norm_cpu"] = ttnn.to_torch(w["ffn_norm"]).reshape(-1).to(torch.bfloat16)

    x = x_bf16  # [1, 1, seq, DIM] bfloat16

    # --- Attention ---
    norm1 = _cpu_rms_norm(x, w["attn_norm_cpu"])
    # Flatten for matmul: [seq, DIM]
    norm1_2d = norm1.reshape(seq_len, DIM)
    q_cpu = (norm1_2d @ w["wq_cpu"]).reshape(seq_len, N_HEADS, HEAD_DIM)
    k_cpu = (norm1_2d @ w["wk_cpu"]).reshape(seq_len, N_KV_HEADS, HEAD_DIM)
    v_cpu = (norm1_2d @ w["wv_cpu"]).reshape(seq_len, N_KV_HEADS, HEAD_DIM)

    if w["qb"] is not None: q_cpu = q_cpu + w["qb"].to(torch.bfloat16).reshape(N_HEADS, HEAD_DIM)
    if w["kb"] is not None: k_cpu = k_cpu + w["kb"].to(torch.bfloat16).reshape(N_KV_HEADS, HEAD_DIM)
    if w["vb"] is not None: v_cpu = v_cpu + w["vb"].to(torch.bfloat16).reshape(N_KV_HEADS, HEAD_DIM)

    q_rot, k_rot = apply_rotary(q_cpu.float().unsqueeze(0), k_cpu.float().unsqueeze(0),
                                  cos_cpu[:, :seq_len], sin_cpu[:, :seq_len])

    # Store K,V in cache
    kv_cache_k[layer_idx] = k_rot.squeeze(0).transpose(0, 1).float()
    kv_cache_v[layer_idx] = v_cpu.float().unsqueeze(0).squeeze(0).transpose(0, 1).float()

    # GQA expand + SDPA on CPU
    k_exp = k_rot.float().repeat_interleave(GQA_GROUPS, dim=2)
    v_exp = v_cpu.float().unsqueeze(0).repeat_interleave(GQA_GROUPS, dim=2)
    q_t = q_rot.float().squeeze(0).transpose(0, 1)
    k_t = k_exp.squeeze(0).transpose(0, 1)
    v_t = v_exp.squeeze(0).transpose(0, 1)
    attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    attn_out = attn_out.transpose(0, 1).reshape(seq_len, DIM).to(torch.bfloat16)

    # O projection on CPU
    proj = attn_out @ w["wo_cpu"]  # [seq, DIM]

    # Residual
    h = x.reshape(seq_len, DIM) + proj

    # --- MLP ---
    norm2 = _cpu_rms_norm(h.reshape(1, 1, seq_len, DIM), w["ffn_norm_cpu"])
    norm2_2d = norm2.reshape(seq_len, DIM)

    w1_out = norm2_2d @ w["w1_cpu"]
    w3_out = norm2_2d @ w["w3_cpu"]
    gated = F.silu(w1_out) * w3_out
    w2_out = gated @ w["w2_cpu"]

    # Residual
    out = h + w2_out
    return out.reshape(1, 1, seq_len, DIM)


def decoder_layer_decode(x_tt, w, device, cos_step, sin_step, kv_cache_k, kv_cache_v, layer_idx):
    """Decode: single token. RoPE + bias + SDPA on CPU (fast for 1 token). MLP on TT."""
    norm1 = ttnn.rms_norm(x_tt, epsilon=EPS, weight=w["attn_norm"])
    q_tt = ttnn.linear(norm1, w["wq"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_tt = ttnn.linear(norm1, w["wk"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_tt = ttnn.linear(norm1, w["wv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)

    q_cpu = ttnn.to_torch(q_tt).float().reshape(1, N_HEADS, HEAD_DIM)
    k_cpu = ttnn.to_torch(k_tt).float().reshape(1, N_KV_HEADS, HEAD_DIM)
    v_cpu = ttnn.to_torch(v_tt).float().reshape(1, N_KV_HEADS, HEAD_DIM)
    ttnn.deallocate(q_tt); ttnn.deallocate(k_tt); ttnn.deallocate(v_tt)

    if w["qb"] is not None: q_cpu = q_cpu + w["qb"].float().reshape(N_HEADS, HEAD_DIM)
    if w["kb"] is not None: k_cpu = k_cpu + w["kb"].float().reshape(N_KV_HEADS, HEAD_DIM)
    if w["vb"] is not None: v_cpu = v_cpu + w["vb"].float().reshape(N_KV_HEADS, HEAD_DIM)

    q_rot, k_rot = apply_rotary(q_cpu.unsqueeze(0), k_cpu.unsqueeze(0), cos_step, sin_step)

    # Update KV cache
    new_k = k_rot.squeeze(0).transpose(0, 1).float()
    new_v = v_cpu.unsqueeze(0).squeeze(0).transpose(0, 1).float()
    kv_cache_k[layer_idx] = torch.cat([kv_cache_k[layer_idx], new_k], dim=1)
    kv_cache_v[layer_idx] = torch.cat([kv_cache_v[layer_idx], new_v], dim=1)

    # CPU SDPA (fast for decode — Q is 1 token, K/V already in CPU cache)
    full_k = kv_cache_k[layer_idx].repeat_interleave(GQA_GROUPS, dim=0)
    full_v = kv_cache_v[layer_idx].repeat_interleave(GQA_GROUPS, dim=0)
    q_t = q_rot.float().squeeze(0).transpose(0, 1)
    attn_out = F.scaled_dot_product_attention(q_t, full_k, full_v, is_causal=False)
    attn_out = attn_out.transpose(0, 1).reshape(1, 1, 1, DIM).to(torch.bfloat16)

    attn_tt = ttnn.from_torch(attn_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    proj = ttnn.linear(attn_tt, w["wo"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(attn_tt)
    h = ttnn.add(x_tt, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(proj); ttnn.deallocate(x_tt)

    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["ffn_norm"])
    w1_out = ttnn.linear(norm2, w["w1"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_out = ttnn.linear(norm2, w["w3"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm2)
    gated = ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(w1_out); ttnn.deallocate(w3_out)
    w2_out = ttnn.linear(gated, w["w2"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(gated)
    out = ttnn.add(h, w2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(h); ttnn.deallocate(w2_out)
    return out


def prefill(embeds_tt, layers, final, device, cos_cpu, sin_cpu, seq_len):
    """Prefill: full CPU computation for accuracy.

    embeds_tt can be a TT device tensor or a CPU torch tensor.
    Returns (logits_tt, kv_cache_k, kv_cache_v).
    """
    kv_cache_k = {}
    kv_cache_v = {}
    # Convert to CPU bfloat16 (matches HF model dtype)
    if not isinstance(embeds_tt, torch.Tensor):
        x = ttnn.to_torch(embeds_tt).to(torch.bfloat16)
    else:
        x = embeds_tt.to(torch.bfloat16)
    for i, w in enumerate(layers):
        x = decoder_layer_prefill(x, w, device, cos_cpu, sin_cpu, seq_len, kv_cache_k, kv_cache_v, i)
        if i % 7 == 0:
            print(f"    layer {i}/{len(layers)}", flush=True)
    # Final norm + lm_head on CPU
    if "norm_cpu" not in final:
        final["norm_cpu"] = ttnn.to_torch(final["norm"]).reshape(-1).to(torch.bfloat16)
        final["lm_head_cpu"] = ttnn.to_torch(final["lm_head"]).squeeze(0).squeeze(0).contiguous().to(torch.bfloat16)
    x_normed = _cpu_rms_norm(x, final["norm_cpu"])
    logits = (x_normed.reshape(seq_len, DIM) @ final["lm_head_cpu"])  # [seq, vocab]
    # Return as TT tensor for compatibility
    logits_tt = ttnn.from_torch(logits.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return logits_tt, kv_cache_k, kv_cache_v


def decode_step(token_embed_tt, layers, final, device, cos_step, sin_step, kv_cache_k, kv_cache_v):
    """Single decode step with CPU KV cache."""
    x = token_embed_tt
    for i, w in enumerate(layers):
        x = decoder_layer_decode(x, w, device, cos_step, sin_step, kv_cache_k, kv_cache_v, i)
    x = ttnn.rms_norm(x, epsilon=EPS, weight=final["norm"])
    logits = ttnn.linear(x, final["lm_head"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return logits


# ============================================================
# Full TT decode — KV cache on device, zero CPU transfer
# ============================================================

def init_tt_kv_cache(device, max_seq=1024, kv_dtype=None):
    """Pre-allocate KV cache on TT device for all 28 layers."""
    import os as _os
    if kv_dtype is None:
        kv_dtype = ttnn.float32 if _os.environ.get("KV_FP32") == "1" else ttnn.bfloat16
    torch_dtype = torch.float32 if kv_dtype == ttnn.float32 else torch.bfloat16
    cache = []
    for _ in range(N_LAYERS):
        k = ttnn.from_torch(
            torch.zeros(1, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch_dtype),
            device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.from_torch(
            torch.zeros(1, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch_dtype),
            device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cache.append((k, v))
    return cache


def fill_tt_kv_cache(tt_cache, cpu_kv_k, cpu_kv_v, device):
    """Copy CPU KV cache (from prefill) into TT pre-allocated cache."""
    for i in range(N_LAYERS):
        k_cpu = cpu_kv_k[i].unsqueeze(0).to(torch.bfloat16)  # [1, 2, seq, 128]
        v_cpu = cpu_kv_v[i].unsqueeze(0).to(torch.bfloat16)
        k_tt = ttnn.from_torch(k_cpu, device=device, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_tt = ttnn.from_torch(v_cpu, device=device, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.fill_cache(tt_cache[i][0], k_tt, 0)
        ttnn.fill_cache(tt_cache[i][1], v_tt, 0)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)


def decoder_layer_decode_tt(x_tt, w, device, cos_tt, sin_tt, tt_cache, layer_idx, cur_seq, attn_mask_tt=None):
    """Optimized decode — manual slice+reshape QKV, full cache SDPA with mask.
    20ms/tok warm, Match: True. Zero CPU transfer except tiny K/V reshape."""
    Q_DIM = N_HEADS * HEAD_DIM
    KV_DIM = N_KV_HEADS * HEAD_DIM

    norm1 = ttnn.rms_norm(x_tt, epsilon=EPS, weight=w["attn_norm"])
    xqkv = ttnn.linear(norm1, w["wqkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)
    if w["qkvb_fused_tt"] is not None:
        xqkv = ttnn.add(xqkv, w["qkvb_fused_tt"])

    # QKV split on device (slice+reshape)
    q_flat = ttnn.to_memory_config(xqkv[:, :, :, :Q_DIM], ttnn.DRAM_MEMORY_CONFIG)
    k_flat = ttnn.to_memory_config(xqkv[:, :, :, Q_DIM:Q_DIM + KV_DIM], ttnn.DRAM_MEMORY_CONFIG)
    v_flat = ttnn.to_memory_config(xqkv[:, :, :, Q_DIM + KV_DIM:Q_DIM + 2*KV_DIM], ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(xqkv)

    q_heads = ttnn.reshape(q_flat, (1, N_HEADS, 1, HEAD_DIM))
    k_heads = ttnn.reshape(k_flat, (1, N_KV_HEADS, 1, HEAD_DIM))
    v_heads = ttnn.reshape(v_flat, (1, N_KV_HEADS, 1, HEAD_DIM))
    ttnn.deallocate(q_flat); ttnn.deallocate(k_flat); ttnn.deallocate(v_flat)

    # Rotary
    q_rot = ttnn.experimental.rotary_embedding(q_heads, cos_tt, sin_tt, cur_seq)
    k_rot = ttnn.experimental.rotary_embedding(k_heads, cos_tt, sin_tt, cur_seq)
    ttnn.deallocate(q_heads); ttnn.deallocate(k_heads)

    # KV cache update
    ttnn.kv_cache.update_cache_for_token_(tt_cache[layer_idx][0], k_rot, cur_seq)
    ttnn.kv_cache.update_cache_for_token_(tt_cache[layer_idx][1], v_heads, cur_seq)
    ttnn.deallocate(k_rot); ttnn.deallocate(v_heads)

    # SDPA (full cache + mask) — precision-tuned: exp_approx_mode=False, k_chunk_size=16
    import os as _os
    _sdpa_prog_cfg = None
    if _os.environ.get("SDPA_PRECISE") == "1":
        try:
            _sdpa_prog_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=int(_os.environ.get("K_CHUNK", 16)),
                exp_approx_mode=False,
            )
        except Exception:
            _sdpa_prog_cfg = None
    attn_tt = ttnn.transformer.scaled_dot_product_attention(
        q_rot, tt_cache[layer_idx][0], tt_cache[layer_idx][1],
        is_causal=False, attn_mask=attn_mask_tt,
        program_config=_sdpa_prog_cfg,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(q_rot)

    # Head concat on device
    attn_sliced = attn_tt[:, :, :1, :]
    ttnn.deallocate(attn_tt)
    attn_flat = ttnn.reshape(attn_sliced, (1, 1, 1, DIM))
    ttnn.deallocate(attn_sliced)

    # === O projection + residual ===
    proj = ttnn.linear(attn_flat, w["wo"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_flat)
    h = ttnn.add(x_tt, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj); ttnn.deallocate(x_tt)

    # === MLP on TT (unchanged) ===
    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["ffn_norm"])
    w1_out = ttnn.linear(norm2, w["w1"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_out = ttnn.linear(norm2, w["w3"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
    gated = ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                     dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(w1_out); ttnn.deallocate(w3_out)
    w2_out = ttnn.linear(gated, w["w2"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)
    out = ttnn.add(h, w2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h); ttnn.deallocate(w2_out)
    return out


# ============================================================
# FlashDecode path — sdpa_decode + cur_pos, no mask needed
# ============================================================

_SDPA_DECODE_CFG = None
_SDPA_DECODE_CK = None

def _get_sdpa_decode_configs():
    global _SDPA_DECODE_CFG, _SDPA_DECODE_CK
    if _SDPA_DECODE_CFG is None:
        _SDPA_DECODE_CFG = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            q_chunk_size=32,
            k_chunk_size=32,
        )
        _SDPA_DECODE_CK = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _SDPA_DECODE_CFG, _SDPA_DECODE_CK


def decoder_layer_flash(x_tt, w, device, cos_tt, sin_tt, tt_cache, layer_idx, cur_seq):
    """FlashDecode path — sdpa_decode kernel + cur_pos list. No mask, no dynamic slice."""
    Q_DIM = N_HEADS * HEAD_DIM
    KV_DIM = N_KV_HEADS * HEAD_DIM
    sdpa_cfg, sdpa_ck = _get_sdpa_decode_configs()

    norm1 = ttnn.rms_norm(x_tt, epsilon=EPS, weight=w["attn_norm"])
    xqkv = ttnn.linear(norm1, w["wqkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)
    if w["qkvb_fused_tt"] is not None:
        xqkv = ttnn.add(xqkv, w["qkvb_fused_tt"])

    # Slice QKV → reshape to head format
    q_flat = ttnn.to_memory_config(xqkv[:, :, :, :Q_DIM], ttnn.DRAM_MEMORY_CONFIG)
    k_flat = ttnn.to_memory_config(xqkv[:, :, :, Q_DIM:Q_DIM + KV_DIM], ttnn.DRAM_MEMORY_CONFIG)
    v_flat = ttnn.to_memory_config(xqkv[:, :, :, Q_DIM + KV_DIM:Q_DIM + 2*KV_DIM], ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(xqkv)

    # sdpa_decode Q format: [1, bsz=1, nh, hd]  (NOT [1, nh, bsz, hd])
    q_heads = ttnn.reshape(q_flat, (1, 1, N_HEADS, HEAD_DIM))      # [1, 1, 12, 128]
    k_heads = ttnn.reshape(k_flat, (1, N_KV_HEADS, 1, HEAD_DIM))   # [1, 2, 1, 128]
    v_heads = ttnn.reshape(v_flat, (1, N_KV_HEADS, 1, HEAD_DIM))   # [1, 2, 1, 128]
    ttnn.deallocate(q_flat); ttnn.deallocate(k_flat); ttnn.deallocate(v_flat)

    # Rotary — Q [1,1,12,128], K [1,2,1,128]
    q_rot = ttnn.experimental.rotary_embedding(q_heads, cos_tt, sin_tt, cur_seq)
    k_rot = ttnn.experimental.rotary_embedding(k_heads, cos_tt, sin_tt, cur_seq)
    ttnn.deallocate(q_heads); ttnn.deallocate(k_heads)

    # KV cache update
    ttnn.kv_cache.update_cache_for_token_(tt_cache[layer_idx][0], k_rot, cur_seq)
    ttnn.kv_cache.update_cache_for_token_(tt_cache[layer_idx][1], v_heads, cur_seq)
    ttnn.deallocate(k_rot); ttnn.deallocate(v_heads)

    # FlashDecode SDPA — no mask, no dynamic slice
    attn_tt = ttnn.transformer.scaled_dot_product_attention_decode(
        q_rot,                        # [1, 1, 12, 128]
        tt_cache[layer_idx][0],       # [1, 2, max_seq, 128]
        tt_cache[layer_idx][1],       # [1, 2, max_seq, 128]
        cur_pos=[cur_seq],
        scale=SCALE,
        program_config=sdpa_cfg,
        compute_kernel_config=sdpa_ck,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(q_rot)

    # Head concat: sdpa_decode output [1,1,padded_heads,128] → [1,1,1,1536]
    attn_cpu = ttnn.to_torch(attn_tt)
    ttnn.deallocate(attn_tt)
    attn_cpu = attn_cpu[0, 0, :N_HEADS, :HEAD_DIM].reshape(1, 1, 1, DIM).to(torch.bfloat16)
    attn_flat = ttnn.from_torch(attn_cpu, device=device, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_tt)

    # O proj + residual
    proj = ttnn.linear(attn_flat, w["wo"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_flat)
    h = ttnn.add(x_tt, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj); ttnn.deallocate(x_tt)

    # MLP
    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["ffn_norm"])
    w1_out = ttnn.linear(norm2, w["w1"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_out = ttnn.linear(norm2, w["w3"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
    gated = ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                     dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(w1_out); ttnn.deallocate(w3_out)
    w2_out = ttnn.linear(gated, w["w2"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)
    out = ttnn.add(h, w2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h); ttnn.deallocate(w2_out)
    return out


def decode_step_flash(token_embed_tt, layers, final, device, cos_full_tt, sin_full_tt, tt_cache, cur_seq):
    """FlashDecode step — no mask, no dynamic slice, just cur_pos list."""
    x = token_embed_tt
    for i, w in enumerate(layers):
        x = decoder_layer_flash(x, w, device, cos_full_tt, sin_full_tt, tt_cache, i, cur_seq)
    x = ttnn.rms_norm(x, epsilon=EPS, weight=final["norm"])
    logits = ttnn.linear(x, final["lm_head"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return logits


def make_attn_mask(cur_seq, max_cache_seq, device):
    """Create attention mask: 0 for valid [0..cur_seq], -inf for rest.
    Q is (1,H,1,D) padded to tile (seq=32), so mask dim[2]=32."""
    mask = torch.full((1, 1, 32, max_cache_seq), float('-inf'), dtype=torch.bfloat16)
    mask[:, :, :, :cur_seq + 1] = 0
    return ttnn.from_torch(mask, device=device, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def precompute_attn_masks(start_pos, num_steps, max_cache_seq, device):
    """Pre-build decode attention masks once to avoid per-step host allocation/transfer.
    Q is reshaped to (1,H,1,D) but TILE_LAYOUT pads seq to 32, so mask must have dim[2]=32."""
    masks = []
    mask_cpu = torch.full((1, 1, 32, max_cache_seq), float("-inf"), dtype=torch.bfloat16)
    for cur_seq in range(start_pos, start_pos + num_steps):
        mask_cpu[:, :, :, : cur_seq + 1] = 0
        masks.append(
            ttnn.from_torch(
                mask_cpu.clone(),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    return masks


def align_decode_cache_seq(min_seq_len, chunk_size=DECODE_CHUNK_SIZE):
    """Pad cache length for decode SDPA kernels that prefer chunk-aligned sequence lengths."""
    return ((min_seq_len + chunk_size - 1) // chunk_size) * chunk_size


def make_current_pos_tensor(cur_seq, device):
    """Create a device tensor for the current decode position."""
    return ttnn.from_torch(
        torch.tensor([cur_seq], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def precompute_rotary_rows(cos_full, sin_full, start_pos, num_steps, device):
    """Pre-build one-position cos/sin rows for decode kernels and trace-friendly replay."""
    rows = []
    for cur_seq in range(start_pos, start_pos + num_steps):
        cos_row = cos_full[cur_seq : cur_seq + 1].unsqueeze(0).unsqueeze(0)
        sin_row = sin_full[cur_seq : cur_seq + 1].unsqueeze(0).unsqueeze(0)
        rows.append(
            (
                ttnn.from_torch(
                    cos_row,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    sin_row,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        )
    return rows


def make_sdpa_decode_program_config():
    """Default decode SDPA program config that matches our single-board experiments."""
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        q_chunk_size=32,
        k_chunk_size=DECODE_CHUNK_SIZE,
    )


def make_decode_compute_kernel_config():
    """Shared decode kernel config used by both generic and decode-SDPA attention paths."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def make_kv_update_memory_config(max_local_batch_size=1, num_local_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM):
    """Build the sharded L1 memory config required by paged KV cache updates."""
    grid_size = ttnn.CoreCoord(8, 8)
    shard_height = ((max_local_batch_size + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    shard_width = head_dim
    core_grid = ttnn.num_cores_to_corerangeset(max_local_batch_size, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


def make_decode_attention_output_memory_config(batch_size=1, num_heads=N_HEADS, head_dim=HEAD_DIM):
    """Build the sharded output memory config expected by decode SDPA + concat-heads."""
    grid_size = ttnn.CoreCoord(8, 8)
    padded_heads = ((num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _reshape_rotary_decode_heads(tt_tensor, cos_step_tt, sin_step_tt, num_heads):
    """Apply HF rotary on a single decode token and restore the logical head count."""
    tt_rot = ttnn.experimental.rotary_embedding(tt_tensor, cos_step_tt, sin_step_tt)
    tt_rot = ttnn.reshape(tt_rot, (1, 1, num_heads, HEAD_DIM), (1, 1, 32, HEAD_DIM))
    return tt_rot[:, :, :num_heads]


def decoder_layer_decode_sdpa_tt(
    x_tt,
    w,
    device,
    cos_full_tt,
    sin_full_tt,
    tt_cache,
    layer_idx,
    current_pos_tt,
    current_pos,
    sdpa_program_config=None,
    sdpa_compute_kernel_config=None,
):
    """Decode layer using tensor-based KV update + decode SDPA kernel.

    This path avoids integer-only cache updates and uses one-position rotary rows,
    which makes it suitable for trace replay with fixed device input buffers.
    """
    Q_DIM = N_HEADS * HEAD_DIM
    KV_DIM = N_KV_HEADS * HEAD_DIM

    if sdpa_program_config is None:
        sdpa_program_config = make_sdpa_decode_program_config()
    if sdpa_compute_kernel_config is None:
        sdpa_compute_kernel_config = make_decode_compute_kernel_config()

    norm1 = ttnn.rms_norm(x_tt, epsilon=EPS, weight=w["attn_norm"])
    xqkv = ttnn.linear(norm1, w["wqkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)
    if w["qkvb_fused_tt"] is not None:
        xqkv = ttnn.add(xqkv, w["qkvb_fused_tt"])

    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=N_HEADS,
        num_kv_heads=N_KV_HEADS,
        memory_config=make_kv_update_memory_config(),
    )
    ttnn.deallocate(xqkv)

    q_rot = ttnn.experimental.rotary_embedding(q_heads, cos_full_tt, sin_full_tt, current_pos)
    k_rot = ttnn.experimental.rotary_embedding(k_heads, cos_full_tt, sin_full_tt, current_pos)
    q_rot = ttnn.reshape(q_rot, (1, 1, N_HEADS, HEAD_DIM), (1, 1, 32, HEAD_DIM))
    k_rot = ttnn.reshape(k_rot, (1, 1, N_KV_HEADS, HEAD_DIM), (1, 1, 32, HEAD_DIM))
    q_rot = q_rot[:, :, :N_HEADS]
    k_rot = k_rot[:, :, :N_KV_HEADS]
    ttnn.deallocate(q_heads)
    ttnn.deallocate(k_heads)

    ttnn.experimental.paged_update_cache(tt_cache[layer_idx][0], k_rot, update_idxs_tensor=current_pos_tt)
    ttnn.experimental.paged_update_cache(tt_cache[layer_idx][1], v_heads, update_idxs_tensor=current_pos_tt)
    ttnn.deallocate(k_rot)
    ttnn.deallocate(v_heads)

    attn_tt = ttnn.transformer.scaled_dot_product_attention_decode(
        q_rot,
        tt_cache[layer_idx][0],
        tt_cache[layer_idx][1],
        cur_pos_tensor=current_pos_tt,
        scale=SCALE,
        program_config=sdpa_program_config,
        compute_kernel_config=sdpa_compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(q_rot)

    attn_tt = ttnn.to_memory_config(attn_tt, make_decode_attention_output_memory_config())
    attn_flat = ttnn.experimental.nlp_concat_heads_decode(attn_tt, num_heads=N_HEADS)
    ttnn.deallocate(attn_tt)
    attn_flat = ttnn.to_memory_config(attn_flat, ttnn.DRAM_MEMORY_CONFIG)
    attn_flat = attn_flat[:, :, :1, :]

    proj = ttnn.linear(attn_flat, w["wo"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_flat)
    h = ttnn.add(x_tt, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj)
    ttnn.deallocate(x_tt)

    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["ffn_norm"])
    w1_out = ttnn.linear(norm2, w["w1"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_out = ttnn.linear(norm2, w["w3"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
    gated = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)
    w2_out = ttnn.linear(gated, w["w2"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)
    out = ttnn.add(h, w2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h)
    ttnn.deallocate(w2_out)
    return out


def decode_step_sdpa_tt(
    token_embed_tt,
    layers,
    final,
    device,
    cos_full_tt,
    sin_full_tt,
    tt_cache,
    current_pos_tt,
    current_pos,
    sdpa_program_config=None,
    sdpa_compute_kernel_config=None,
):
    """Trace-friendly decode step using tensor current_pos and decode SDPA kernel."""
    x = token_embed_tt
    for i, w in enumerate(layers):
        x = decoder_layer_decode_sdpa_tt(
            x,
            w,
            device,
            cos_full_tt,
            sin_full_tt,
            tt_cache,
            i,
            current_pos_tt,
            current_pos,
            sdpa_program_config=sdpa_program_config,
            sdpa_compute_kernel_config=sdpa_compute_kernel_config,
        )
    x = ttnn.rms_norm(x, epsilon=EPS, weight=final["norm"])
    logits = ttnn.linear(x, final["lm_head"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return logits


def decode_step_tt(token_embed_tt, layers, final, device, cos_full_tt, sin_full_tt, tt_cache, cur_seq, max_cache_seq=None, attn_mask_tt=None):
    """Full on-device decode step — zero CPU transfers.

    Args:
        cos_full_tt, sin_full_tt: Pre-loaded full rotary tables on device
        cur_seq: Integer position index
        max_cache_seq: If set and attn_mask_tt is None, builds mask internally
        attn_mask_tt: Pre-built attention mask on device (fastest)
    """
    if attn_mask_tt is None and max_cache_seq is not None:
        attn_mask_tt = make_attn_mask(cur_seq, max_cache_seq, device)

    x = token_embed_tt
    for i, w in enumerate(layers):
        x = decoder_layer_decode_tt(x, w, device, cos_full_tt, sin_full_tt, tt_cache, i, cur_seq, attn_mask_tt)
    x = ttnn.rms_norm(x, epsilon=EPS, weight=final["norm"])
    logits = ttnn.linear(x, final["lm_head"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return logits
