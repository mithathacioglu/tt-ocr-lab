#!/usr/bin/env python3
"""
End-to-end: ttnn native vision (42 blocks) + merger + CPU decoder generate on 1.pdf.
Compares text output with CPU-only reference.
"""
from __future__ import annotations
import os, sys, time, types
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
    resolve_snapshot_dir, load_vision_model, load_vision_state,
    apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
)
from ocr_lab.fixed_page import fixed_page_message

DIM = 1536
N_HEADS = 12
HEAD_DIM = 128
EPS = 1e-5
N_BLOCKS = 42


def make_ttnn_weight(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def load_all_block_weights(state_dict, device):
    blocks = []
    for i in range(N_BLOCKS):
        p = f"blocks.{i}."
        w = {}
        w["norm1"] = ttnn.from_torch(
            state_dict[f"{p}norm1.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["norm2"] = ttnn.from_torch(
            state_dict[f"{p}norm2.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["qkv_w"] = make_ttnn_weight(state_dict[f"{p}attn.qkv.weight"].T, device)
        w["proj_w"] = make_ttnn_weight(state_dict[f"{p}attn.proj.weight"].T, device)
        w["fc1_w"] = make_ttnn_weight(state_dict[f"{p}mlp.fc1.weight"].T, device)
        w["fc2_w"] = make_ttnn_weight(state_dict[f"{p}mlp.fc2.weight"].T, device)
        w["fc3_w"] = make_ttnn_weight(state_dict[f"{p}mlp.fc3.weight"].T, device)
        blocks.append(w)
    return blocks


def ttnn_vision_block(x, w, device, seq_len, cos_cpu, sin_cpu):
    """One block: TT norm+linear, CPU attention with rotary, TT MLP."""
    # RMSNorm1 on TT
    norm1 = ttnn.rms_norm(x, epsilon=EPS, weight=w["norm1"])

    # QKV on TT
    qkv = ttnn.linear(norm1, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)

    # Attention: CPU (with rotary embedding)
    qkv_cpu = ttnn.to_torch(qkv).float()
    ttnn.deallocate(qkv)

    actual_seq = qkv_cpu.shape[-2]
    q_cpu = qkv_cpu[..., :DIM].reshape(actual_seq, N_HEADS, HEAD_DIM)
    k_cpu = qkv_cpu[..., DIM:2*DIM].reshape(actual_seq, N_HEADS, HEAD_DIM)
    v_cpu = qkv_cpu[..., 2*DIM:].reshape(actual_seq, N_HEADS, HEAD_DIM)

    # Apply rotary
    q_cpu = apply_rotary_pos_emb_cached_cpu(q_cpu.unsqueeze(0), cos_cpu[:, :actual_seq], sin_cpu[:, :actual_seq]).squeeze(0)
    k_cpu = apply_rotary_pos_emb_cached_cpu(k_cpu.unsqueeze(0), cos_cpu[:, :actual_seq], sin_cpu[:, :actual_seq]).squeeze(0)

    # SDPA
    q_t = q_cpu.transpose(0, 1)  # [heads, seq, dim]
    k_t = k_cpu.transpose(0, 1)
    v_t = v_cpu.transpose(0, 1).float()
    attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
    attn_out = attn_out.transpose(0, 1).reshape(1, 1, actual_seq, DIM).to(torch.bfloat16)

    attn_tt = ttnn.from_torch(attn_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Proj on TT
    proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_tt)
    h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj)
    ttnn.deallocate(x)

    # RMSNorm2 on TT
    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["norm2"])

    # MLP on TT (fused SiLU)
    fc1 = ttnn.linear(norm2, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3 = ttnn.linear(norm2, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
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
    PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
    MAX_NEW_TOKENS = 64

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))
    from transformers import AutoProcessor, AutoModelForCausalLM
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # --- Load decoder (CPU) ---
    print("Loading decoder model...", flush=True)
    decoder = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()

    original_forward = decoder.forward
    def patched_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is None and inputs_embeds is not None:
            input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
    decoder.forward = types.MethodType(patched_forward, decoder)

    emb_weight = decoder.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    # --- Load vision model for patch_embed + rotary ---
    print("Loading vision model (for patch_embed + rotary)...", flush=True)
    vision_model, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    vision_model = vision_model.eval().to(dtype=torch.bfloat16)

    # Patch embed
    with torch.no_grad():
        hidden = vision_model.patch_embed(pixel_values.to(dtype=torch.bfloat16), grid_thw.to(dtype=torch.int32))
    seq_len = hidden.shape[0]  # 1632

    # Rotary embeddings
    pos_ids = vision_model.get_pos_ids_by_grid(grid_thw.cpu())
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid = int(grid_thw.cpu()[:, 1:].max().item())
    rotary_full = vision_model.rotary_pos_emb(max_grid).cpu()
    rotary = rotary_full[pos_ids].flatten(1).float()
    cos_cpu = rotary.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    sin_cpu = rotary.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)

    # Post-trunk norm weight (for after 42 blocks)
    post_norm_weight = vision_model.post_trunk_norm.weight.detach().cpu()

    # Merger
    cpu_merger = CpuPatchMerger(vision_model.merger)
    del vision_model

    # --- Open TT device ---
    print("Opening TT device...", flush=True)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()

    # Load block weights
    print("Loading 42 block weights to TT...", flush=True)
    state_dict = load_vision_state(snapshot_dir, limit_layers=0)
    all_blocks = load_all_block_weights(state_dict, device)

    # Post-norm weight
    post_norm_tt = ttnn.from_torch(
        post_norm_weight.unsqueeze(0).view(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # --- ttnn Vision Forward ---
    hidden_tt = ttnn.from_torch(
        hidden.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Cold run
    print("Running 42 blocks (cold)...", flush=True)
    t0 = time.perf_counter()
    x = hidden_tt
    for bw in all_blocks:
        x = ttnn_vision_block(x, bw, device, seq_len, cos_cpu, sin_cpu)
    # Post-trunk norm
    x = ttnn.rms_norm(x, epsilon=EPS, weight=post_norm_tt)
    cold_ms = (time.perf_counter() - t0) * 1000
    vision_out_cpu = ttnn.to_torch(x).squeeze(0).squeeze(0)[:seq_len].float()
    print(f"  Cold: {cold_ms:.0f}ms", flush=True)

    # Warm run
    hidden_tt2 = ttnn.from_torch(
        hidden.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("Running 42 blocks (warm)...", flush=True)
    t0 = time.perf_counter()
    x2 = hidden_tt2
    for bw in all_blocks:
        x2 = ttnn_vision_block(x2, bw, device, seq_len, cos_cpu, sin_cpu)
    x2 = ttnn.rms_norm(x2, epsilon=EPS, weight=post_norm_tt)
    warm_ms = (time.perf_counter() - t0) * 1000
    vision_out_warm = ttnn.to_torch(x2).squeeze(0).squeeze(0)[:seq_len].float()
    print(f"  Warm: {warm_ms:.0f}ms", flush=True)

    # Close TT — we need it free for potential torch_xla later
    ttnn.close_device(device)

    # --- Merger (CPU) ---
    print("Running merger...", flush=True)
    t0 = time.perf_counter()
    vision_embeddings = cpu_merger(vision_out_warm.to(torch.bfloat16))
    merger_ms = (time.perf_counter() - t0) * 1000
    print(f"  Merger: {merger_ms:.0f}ms  shape={vision_embeddings.shape}", flush=True)

    # --- Inject into decoder and generate ---
    print("Generating text...", flush=True)
    img_mask = input_ids == decoder.config.image_token_id
    inputs_embeds = F.embedding(input_ids, emb_weight)
    inputs_embeds = inputs_embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        vision_embeddings.to(inputs_embeds.dtype),
    )

    with torch.no_grad():
        t0 = time.perf_counter()
        gen_ids = decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        gen_ms = (time.perf_counter() - t0) * 1000

    gen_trimmed = gen_ids[0, input_ids.shape[1]:]
    tokenizer = getattr(proc, "tokenizer", proc)
    gen_text = tokenizer.decode(gen_trimmed, skip_special_tokens=True)
    print(f"  Generate: {gen_ms:.0f}ms", flush=True)
    print(f"  Text: {repr(gen_text[:200])}", flush=True)

    # --- Reference ---
    REF_TEXT = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"
    match = gen_text.strip() == REF_TEXT.strip()

    total_ms = warm_ms + merger_ms + gen_ms
    print(f"\n{'='*60}")
    print(f"  Vision (ttnn warm): {warm_ms:.0f}ms")
    print(f"  Merger:             {merger_ms:.0f}ms")
    print(f"  Decoder (CPU):      {gen_ms:.0f}ms")
    print(f"  TOTAL:              {total_ms:.0f}ms")
    print(f"  Text match:         {match}")
    print(f"  Reference:          {repr(REF_TEXT[:80])}")
    print(f"  Output:             {repr(gen_text[:80])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
