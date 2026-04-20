from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F


def build_inputs_embeds_cpu(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    *,
    embedding_weight_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    img_mask = input_ids == model.config.image_token_id
    if embedding_weight_cpu is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)
    else:
        inputs_embeds = F.embedding(input_ids, embedding_weight_cpu)
    return inputs_embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        vision_embeddings.to(inputs_embeds.dtype),
    )


def inject_vision_embeddings(model, input_ids: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:
    return build_inputs_embeds_cpu(model, input_ids, vision_embeddings)


def build_rotary_cache_cpu(rotary_emb: torch.nn.Module, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    inv_freq = rotary_emb.inv_freq.detach().cpu().float()
    pos_cpu = position_ids.float()
    freqs = (inv_freq[None, :, None].expand(pos_cpu.shape[0], -1, 1) @ pos_cpu[:, None, :]).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    scaling = float(rotary_emb.attention_scaling)
    cos_cpu = (emb.cos() * scaling).to(torch.float32)
    sin_cpu = (emb.sin() * scaling).to(torch.float32)
    return cos_cpu, sin_cpu


def _get_decode_target(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def _decode_token_ids(processor_or_tokenizer, token_ids: list[int], skip_special_tokens: bool) -> str:
    tokenizer = _get_decode_target(processor_or_tokenizer)
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)


def _normalize_eos_ids(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(x) for x in eos_token_id}


def _tt_poly2_silu_approx(x_tt: torch.Tensor) -> torch.Tensor:
    x_sq_tt = x_tt * x_tt
    return (x_tt * 0.5) + (x_sq_tt * 0.25)


def _tt_poly7_datafit_silu_approx(x_tt: torch.Tensor) -> torch.Tensor:
    # Degree-7 polynomial fitted on real decoder gate activations from the DOTS OCR dump.
    # p(x) = a1*x + a2*x^2 + ... + a7*x^7
    coeffs = (
        0.4006877839565277,
        0.1404726207256317,
        0.011149781756103039,
        -0.0005008771549910307,
        -7.543880201410502e-05,
        -2.4412229322479106e-06,
        -2.4767205175635354e-08,
    )

    acc_tt = x_tt * coeffs[-1]
    for coef in reversed(coeffs[1:-1]):
        acc_tt = (acc_tt + coef) * x_tt
    return (acc_tt + coeffs[0]) * x_tt


def _apply_decoder_mlp(
    layer,
    norm2_tt: torch.Tensor,
    device,
    *,
    mlp_mode: str,
) -> torch.Tensor:
    gate_tt = layer.mlp.gate_proj(norm2_tt)
    up_tt = layer.mlp.up_proj(norm2_tt)

    if mlp_mode == "cpu_silu":
        gated_cpu = F.silu(gate_tt.detach().cpu().float()) * up_tt.detach().cpu().float()
        return layer.mlp.down_proj(gated_cpu.to(device=device, dtype=torch.bfloat16))

    if mlp_mode == "cpu_silu_tt_mul":
        # Keep the exact SiLU on CPU, but move the elementwise multiply back onto TT so
        # we stop shipping both gate and up activations to the host on every layer.
        act_cpu = F.silu(gate_tt.detach().cpu().float())
        act_tt = act_cpu.to(device=device, dtype=torch.bfloat16)
        gated_tt = act_tt * up_tt
        return layer.mlp.down_proj(gated_tt)

    if mlp_mode == "tt_exact":
        act_tt = F.silu(gate_tt)
        gated_tt = act_tt * up_tt
        return layer.mlp.down_proj(gated_tt)

    if mlp_mode == "tt_poly2":
        act_tt = _tt_poly2_silu_approx(gate_tt)
        gated_tt = act_tt * up_tt
        return layer.mlp.down_proj(gated_tt)

    if mlp_mode == "tt_poly7_datafit":
        act_tt = _tt_poly7_datafit_silu_approx(gate_tt)
        gated_tt = act_tt * up_tt
        return layer.mlp.down_proj(gated_tt)

    raise ValueError(f"unsupported mlp_mode: {mlp_mode}")


def _hybrid_decoder_prefill_core(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    device,
    *,
    sync_fn: Callable[[], None] | None = None,
    rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    embedding_weight_cpu: torch.Tensor | None = None,
    return_cache: bool = False,
    mlp_mode: str = "cpu_silu",
):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_kv_groups = num_heads // num_kv_heads
    seq_len = input_ids.shape[1]

    if rotary_cache is not None and seq_len in rotary_cache:
        cos_cpu, sin_cpu = rotary_cache[seq_len]
    else:
        cos_cpu, sin_cpu = build_rotary_cache_cpu(model.model.rotary_emb, seq_len)
        if rotary_cache is not None:
            rotary_cache[seq_len] = (cos_cpu, sin_cpu)

    inputs_embeds = build_inputs_embeds_cpu(
        model,
        input_ids,
        vision_embeddings,
        embedding_weight_cpu=embedding_weight_cpu,
    )
    hidden_tt = inputs_embeds.to(device=device, dtype=torch.bfloat16)
    layer_caches: list[dict[str, torch.Tensor]] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for layer in model.model.layers:
            residual_tt = hidden_tt
            norm1_tt = layer.input_layernorm(hidden_tt)
            q_tt = layer.self_attn.q_proj(norm1_tt)
            k_tt = layer.self_attn.k_proj(norm1_tt)
            v_tt = layer.self_attn.v_proj(norm1_tt)

            q_cpu = q_tt.detach().cpu().float().view(1, seq_len, num_heads, head_dim).transpose(1, 2)
            k_cpu = k_tt.detach().cpu().float().view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_cpu = v_tt.detach().cpu().float().view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

            q_cpu, k_cpu = apply_rotary_pos_emb(q_cpu, k_cpu, cos_cpu, sin_cpu)
            if return_cache:
                layer_caches.append(
                    {
                        "k": k_cpu.detach().cpu().to(dtype=torch.float32),
                        "v": v_cpu.detach().cpu().to(dtype=torch.float32),
                    }
                )
            k_attn_cpu = repeat_kv(k_cpu, num_kv_groups)
            v_attn_cpu = repeat_kv(v_cpu, num_kv_groups)
            attn_out_cpu = F.scaled_dot_product_attention(q_cpu, k_attn_cpu, v_attn_cpu, dropout_p=0.0, is_causal=True)
            attn_out_cpu = attn_out_cpu.transpose(1, 2).reshape(1, seq_len, -1).to(dtype=torch.bfloat16)
            attn_out_tt = layer.self_attn.o_proj(attn_out_cpu.to(device))
            hidden_tt = residual_tt + attn_out_tt

            residual_tt = hidden_tt
            norm2_tt = layer.post_attention_layernorm(hidden_tt)
            down_tt = _apply_decoder_mlp(
                layer,
                norm2_tt,
                device,
                mlp_mode=mlp_mode,
            )
            hidden_tt = residual_tt + down_tt

        hidden_tt = model.model.norm(hidden_tt)
        logits_tt = model.lm_head(hidden_tt)
        if sync_fn is not None:
            sync_fn()
        t1 = time.perf_counter()

    logits_cpu = logits_tt.detach().cpu().float()
    elapsed_ms = (t1 - t0) * 1000.0
    if return_cache:
        return logits_cpu, elapsed_ms, layer_caches
    return logits_cpu, elapsed_ms


def hybrid_decoder_prefill_logits(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    device,
    *,
    sync_fn: Callable[[], None] | None = None,
    rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    embedding_weight_cpu: torch.Tensor | None = None,
    mlp_mode: str = "cpu_silu",
) -> tuple[torch.Tensor, float]:
    return _hybrid_decoder_prefill_core(
        model,
        input_ids,
        vision_embeddings,
        device,
        sync_fn=sync_fn,
        rotary_cache=rotary_cache,
        embedding_weight_cpu=embedding_weight_cpu,
        mlp_mode=mlp_mode,
        return_cache=False,
    )


def hybrid_decoder_prefill_with_cache(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    device,
    *,
    sync_fn: Callable[[], None] | None = None,
    rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    embedding_weight_cpu: torch.Tensor | None = None,
    mlp_mode: str = "cpu_silu",
):
    return _hybrid_decoder_prefill_core(
        model,
        input_ids,
        vision_embeddings,
        device,
        sync_fn=sync_fn,
        rotary_cache=rotary_cache,
        embedding_weight_cpu=embedding_weight_cpu,
        mlp_mode=mlp_mode,
        return_cache=True,
    )


def hybrid_decoder_decode_step(
    model,
    token_id: int,
    device,
    layer_caches: list[dict[str, torch.Tensor]],
    *,
    sync_fn: Callable[[], None] | None = None,
    rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    embedding_weight_cpu: torch.Tensor,
    mlp_mode: str = "cpu_silu",
):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model.config.hidden_size // num_heads
    total_seq_len = layer_caches[0]["k"].shape[2] + 1

    if rotary_cache is not None and total_seq_len in rotary_cache:
        cos_cpu, sin_cpu = rotary_cache[total_seq_len]
    else:
        cos_cpu, sin_cpu = build_rotary_cache_cpu(model.model.rotary_emb, total_seq_len)
        if rotary_cache is not None:
            rotary_cache[total_seq_len] = (cos_cpu, sin_cpu)
    cos_step = cos_cpu[:, -1:, :]
    sin_step = sin_cpu[:, -1:, :]

    token_tensor = torch.tensor([[token_id]], dtype=torch.long)
    hidden_tt = F.embedding(token_tensor, embedding_weight_cpu).to(device=device, dtype=torch.bfloat16)
    new_layer_caches: list[dict[str, torch.Tensor]] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            residual_tt = hidden_tt
            norm1_tt = layer.input_layernorm(hidden_tt)
            q_tt = layer.self_attn.q_proj(norm1_tt)
            k_tt = layer.self_attn.k_proj(norm1_tt)
            v_tt = layer.self_attn.v_proj(norm1_tt)

            q_cpu = q_tt.detach().cpu().float().view(1, 1, num_heads, head_dim).transpose(1, 2)
            prev_k_cpu = layer_caches[layer_idx]["k"]
            prev_v_cpu = layer_caches[layer_idx]["v"]
            k_cpu = k_tt.detach().cpu().float().view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            v_cpu = v_tt.detach().cpu().float().view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            q_cpu, k_cpu = apply_rotary_pos_emb(q_cpu, k_cpu, cos_step, sin_step)

            full_k_cpu = torch.cat([prev_k_cpu, k_cpu], dim=2)
            full_v_cpu = torch.cat([prev_v_cpu, v_cpu], dim=2)
            new_layer_caches.append({"k": full_k_cpu, "v": full_v_cpu})

            full_k_attn_cpu = repeat_kv(full_k_cpu, num_kv_groups)
            full_v_attn_cpu = repeat_kv(full_v_cpu, num_kv_groups)
            attn_out_cpu = F.scaled_dot_product_attention(
                q_cpu,
                full_k_attn_cpu,
                full_v_attn_cpu,
                dropout_p=0.0,
                is_causal=False,
            )
            attn_out_cpu = attn_out_cpu.transpose(1, 2).reshape(1, 1, -1).to(dtype=torch.bfloat16)
            attn_out_tt = layer.self_attn.o_proj(attn_out_cpu.to(device))
            hidden_tt = residual_tt + attn_out_tt

            residual_tt = hidden_tt
            norm2_tt = layer.post_attention_layernorm(hidden_tt)
            down_tt = _apply_decoder_mlp(
                layer,
                norm2_tt,
                device,
                mlp_mode=mlp_mode,
            )
            hidden_tt = residual_tt + down_tt

        hidden_tt = model.model.norm(hidden_tt)
        logits_tt = model.lm_head(hidden_tt)
        if sync_fn is not None:
            sync_fn()
        t1 = time.perf_counter()

    return logits_tt.detach().cpu().float(), (t1 - t0) * 1000.0, new_layer_caches


def hybrid_greedy_generate(
    model,
    processor_or_tokenizer,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    device,
    *,
    max_new_tokens: int,
    sync_fn: Callable[[], None] | None = None,
    eos_token_id=None,
    embedding_weight_cpu: torch.Tensor | None = None,
    mlp_mode: str = "cpu_silu",
) -> dict:
    generated_token_ids: list[int] = []
    token_timings_ms: list[float] = []
    rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    eos_ids = _normalize_eos_ids(eos_token_id)
    logits_cpu, elapsed_ms, layer_caches = hybrid_decoder_prefill_with_cache(
        model,
        input_ids,
        vision_embeddings,
        device,
        sync_fn=sync_fn,
        rotary_cache=rotary_cache,
        embedding_weight_cpu=embedding_weight_cpu,
        mlp_mode=mlp_mode,
    )
    next_token_id = int(logits_cpu[:, -1, :].argmax(-1).item())
    generated_token_ids.append(next_token_id)
    token_timings_ms.append(round(float(elapsed_ms), 3))

    while len(generated_token_ids) < max_new_tokens and next_token_id not in eos_ids:
        logits_cpu, elapsed_ms, layer_caches = hybrid_decoder_decode_step(
            model,
            next_token_id,
            device,
            layer_caches,
            sync_fn=sync_fn,
            rotary_cache=rotary_cache,
            embedding_weight_cpu=embedding_weight_cpu,
            mlp_mode=mlp_mode,
        )
        next_token_id = int(logits_cpu[:, -1, :].argmax(-1).item())
        generated_token_ids.append(next_token_id)
        token_timings_ms.append(round(float(elapsed_ms), 3))

    text = _decode_token_ids(processor_or_tokenizer, generated_token_ids, skip_special_tokens=True)
    raw_text = _decode_token_ids(processor_or_tokenizer, generated_token_ids, skip_special_tokens=False)

    return {
        "generated_token_ids": generated_token_ids,
        "token_timings_ms": token_timings_ms,
        "generated_text": text,
        "generated_text_raw": raw_text,
        "total_ms": round(sum(token_timings_ms), 3),
        "num_generated_tokens": len(generated_token_ids),
    }
