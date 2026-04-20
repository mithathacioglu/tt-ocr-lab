#!/usr/bin/env python3
"""2-chip parallel: each chip processes 1 page independently."""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from multiprocessing import Process, Queue
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

PAGES = [PROJECT_ROOT / "ocr_lab" / "tmp_batch4" / f"page_{i}.png" for i in range(1, 5)]
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 256


def process_page_on_device(device_id, page_path, result_queue):
    """Complete pipeline for 1 page on 1 device."""
    import torch, torch.nn.functional as F
    TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
    sys.path.insert(0, str(TT_METAL))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))

    from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
    snapshot_dir = patch_dots_mocr_config_for_tt_metal()

    from transformers import AutoProcessor, AutoModelForCausalLM
    from qwen_vl_utils import process_vision_info
    from dots_tt_port.vision_tower_smoke import (
        load_vision_model, load_vision_state,
        apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
    )
    from ocr_lab.fixed_page import fixed_page_message
    import ttnn

    t_total = time.perf_counter()

    # --- Load processor + embeddings ---
    proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    tokenizer = getattr(proc, "tokenizer", proc)
    hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
    emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    image_token_id = hf.config.image_token_id
    vocab_size = hf.config.vocab_size
    del hf

    # --- Prepare inputs ---
    img_msg, _ = fixed_page_message(page_path, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    # === VISION (ttnn on assigned device) ===
    DIM_V, N_HEADS_V, HEAD_DIM_V, EPS_V, N_BLOCKS_V = 1536, 12, 128, 1e-5, 42
    device = ttnn.open_device(device_id=device_id)
    device.enable_program_cache()

    vision_model, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    vision_model = vision_model.eval().to(dtype=torch.bfloat16)
    vs = load_vision_state(snapshot_dir, limit_layers=0)
    post_norm_w = vision_model.post_trunk_norm.weight.detach().cpu()
    cpu_merger = CpuPatchMerger(vision_model.merger)

    def make_vw(t): return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    vblocks = []
    for i in range(N_BLOCKS_V):
        p = f"blocks.{i}."
        w = {}
        w["norm1"] = ttnn.from_torch(vs[f"{p}norm1.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["norm2"] = ttnn.from_torch(vs[f"{p}norm2.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["qkv_w"] = make_vw(vs[f"{p}attn.qkv.weight"].T)
        w["proj_w"] = make_vw(vs[f"{p}attn.proj.weight"].T)
        w["fc1_w"] = make_vw(vs[f"{p}mlp.fc1.weight"].T)
        w["fc2_w"] = make_vw(vs[f"{p}mlp.fc2.weight"].T)
        w["fc3_w"] = make_vw(vs[f"{p}mlp.fc3.weight"].T)
        vblocks.append(w)
    post_norm_tt = ttnn.from_torch(post_norm_w.unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def vision_block(x, w, dev, seq, cos, sin):
        norm1 = ttnn.rms_norm(x, epsilon=EPS_V, weight=w["norm1"])
        qkv = ttnn.linear(norm1, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm1)
        qkv_cpu = ttnn.to_torch(qkv).float(); ttnn.deallocate(qkv)
        q = qkv_cpu[..., :DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
        k = qkv_cpu[..., DIM_V:2*DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
        v = qkv_cpu[..., 2*DIM_V:].reshape(seq, N_HEADS_V, HEAD_DIM_V)
        q = apply_rotary_pos_emb_cached_cpu(q.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
        k = apply_rotary_pos_emb_cached_cpu(k.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
        attn = F.scaled_dot_product_attention(q.transpose(0,1), k.transpose(0,1), v.float().transpose(0,1), dropout_p=0.0)
        attn = attn.transpose(0,1).reshape(1,1,seq,DIM_V).to(torch.bfloat16)
        attn_tt = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(attn_tt)
        h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(proj); ttnn.deallocate(x)
        norm2 = ttnn.rms_norm(h, epsilon=EPS_V, weight=w["norm2"])
        fc1 = ttnn.linear(norm2, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fc3 = ttnn.linear(norm2, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm2)
        gated = ttnn.mul(fc1, fc3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(fc1); ttnn.deallocate(fc3)
        fc2 = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(gated)
        out = ttnn.add(h, fc2, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(h); ttnn.deallocate(fc2)
        return out

    # Vision forward
    t0 = time.perf_counter()
    pv = inputs["pixel_values"].to(torch.bfloat16)
    gt = inputs["image_grid_thw"]
    with torch.no_grad():
        hidden_v = vision_model.patch_embed(pv, gt.to(torch.int32))
    seq_v = hidden_v.shape[0]
    pos_ids = torch.cat(vision_model.get_pos_ids_by_grid(gt.cpu()), dim=0)
    rotary_full = vision_model.rotary_pos_emb(int(gt.cpu()[:, 1:].max().item())).cpu()
    rotary = rotary_full[pos_ids].flatten(1).float()
    cos_v = rotary.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    sin_v = rotary.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)

    tt_h = ttnn.from_torch(hidden_v.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for bw in vblocks:
        tt_h = vision_block(tt_h, bw, device, seq_v, cos_v, sin_v)
    tt_h = ttnn.rms_norm(tt_h, epsilon=EPS_V, weight=post_norm_tt)
    vo = ttnn.to_torch(tt_h).squeeze(0).squeeze(0)[:seq_v].float()
    vision_emb = cpu_merger(vo.to(torch.bfloat16))
    vision_ms = (time.perf_counter() - t0) * 1000

    ttnn.close_device(device)
    del vs, vblocks, post_norm_tt, vision_model

    # === DECODER (ttnn on mesh) ===
    from models.tt_transformers.tt.model_config import ModelArgs
    from ocr_lab.ttnn_decoder_hybrid import (
        load_decoder_weights, prefill, decode_step_tt,
        init_tt_kv_cache, fill_tt_kv_cache, DIM,
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    mesh.enable_program_cache()
    args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
    sd = args.load_state_dict()
    dec_layers, dec_final = load_decoder_weights(sd, mesh)

    cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
    rope_theta = cfg.get("rope_theta", 1e6)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    freqs = torch.outer(torch.arange(2048, dtype=torch.float32), inv_freq)
    emb_rope = torch.cat((freqs, freqs), dim=-1)
    cos_full = emb_rope.cos().to(torch.bfloat16)
    sin_full = emb_rope.sin().to(torch.bfloat16)
    cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
    sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)
    cos_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Build embeds
    img_mask = inputs["input_ids"] == image_token_id
    embeds = F.embedding(inputs["input_ids"], emb_w)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_emb.to(embeds.dtype))
    seq = embeds.shape[1]

    # Prefill
    t0 = time.perf_counter()
    embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
    first_tok = logits_cpu.argmax(-1).item()

    tt_cache = init_tt_kv_cache(mesh, max_seq=seq + MAX_NEW_TOKENS + 32)
    fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
    del kv_k, kv_v

    # Decode
    generated = [first_tok]
    cur_pos = seq
    next_token = first_tok
    t0 = time.perf_counter()
    for step in range(1, MAX_NEW_TOKENS):
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh, cos_tt, sin_tt, tt_cache, cur_pos)
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
        next_token = logits_d.argmax(-1).item()
        generated.append(next_token)
        cur_pos += 1
        if next_token in (151643, 151645):
            break
    decode_ms = (time.perf_counter() - t0) * 1000

    for k, v in tt_cache:
        ttnn.deallocate(k); ttnn.deallocate(v)
    ttnn.close_mesh_device(mesh)

    total_ms = (time.perf_counter() - t_total) * 1000
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)

    result_queue.put({
        "device": device_id,
        "page": str(page_path.name),
        "vision_ms": vision_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tokens": len(generated),
        "ms_per_tok": decode_ms / max(len(generated) - 1, 1),
        "total_ms": total_ms,
        "text_preview": gen_text[:150],
    })


if __name__ == "__main__":
    # Use 2 chips on the same N300 card (PCIe slot 0)
    DEVICE_IDS = [0, 4]  # Same card: chip 0 and chip 1
    pages_to_run = PAGES[:2]  # 2 pages for 2 chips

    print(f"=== 2-Chip Parallel Test ===")
    print(f"Devices: {DEVICE_IDS}")
    print(f"Pages: {[p.name for p in pages_to_run]}")
    print()

    result_queue = Queue()
    processes = []

    t_start = time.perf_counter()
    for dev_id, page in zip(DEVICE_IDS, pages_to_run):
        p = Process(target=process_page_on_device, args=(dev_id, page, result_queue))
        processes.append(p)
        p.start()
        print(f"  Started device {dev_id} → {page.name}", flush=True)

    for p in processes:
        p.join()

    wall_time = (time.perf_counter() - t_start) * 1000

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda x: x["device"])

    print(f"\n{'='*60}")
    print(f"  2-CHIP PARALLEL RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  Device {r['device']} ({r['page']}):")
        print(f"    Vision={r['vision_ms']:.0f}ms Prefill={r['prefill_ms']:.0f}ms "
              f"Decode={r['decode_ms']:.0f}ms ({r['tokens']}tok, {r['ms_per_tok']:.0f}ms/tok)")
        print(f"    Total={r['total_ms']:.0f}ms")
        print(f"    Text: {repr(r['text_preview'][:100])}")
    seq_time = sum(r["total_ms"] for r in results)
    print(f"  ─────────────────────────────")
    print(f"  Wall clock:     {wall_time:.0f}ms ({wall_time/1000:.1f}s)")
    print(f"  Sequential:     {seq_time:.0f}ms ({seq_time/1000:.1f}s)")
    print(f"  Speedup:        {seq_time/wall_time:.2f}x")
    print(f"  Pages/second:   {len(pages_to_run)/(wall_time/1000):.3f}")
    print(f"{'='*60}")
