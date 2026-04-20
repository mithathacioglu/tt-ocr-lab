#!/usr/bin/env python3
"""Generate ops-level perf CSV using ttnn built-in profiler."""
from __future__ import annotations
import os, sys

# MUST set before any ttnn import
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["ENABLE_PROFILER"] = "1"

import time, json, csv
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.fixed_page import fixed_page_message
import ttnn

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    ve = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
vocab_size = hf.config.vocab_size
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), ve.to(embeds.dtype))
seq = embeds.shape[1]
del hf

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
max_pos = seq + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)
cos_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Prefill
embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
first_tok = logits_cpu.argmax(-1).item()
tt_cache = init_tt_kv_cache(mesh, max_seq=seq + 32)
fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
del kv_k, kv_v

# Warm-up
next_token = first_tok
cur_pos = seq
for _ in range(2):
    te = emb_w[next_token].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(te, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_l = decode_step_tt(tt_tok, dec_layers, dec_final, mesh, cos_tt, sin_tt, tt_cache, cur_pos)
    ld = ttnn.to_torch(tt_l)[0, 0, 0, :vocab_size].float()
    next_token = ld.argmax(-1).item()
    cur_pos += 1

print("Warm-up done. Running 2 profiled decode steps...", flush=True)

# Measured decode (just 2 steps)
for step in range(2):
    te = emb_w[next_token].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(te, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t0 = time.perf_counter()
    tt_l = decode_step_tt(tt_tok, dec_layers, dec_final, mesh, cos_tt, sin_tt, tt_cache, cur_pos)
    ms = (time.perf_counter() - t0) * 1000
    ld = ttnn.to_torch(tt_l)[0, 0, 0, :vocab_size].float()
    next_token = ld.argmax(-1).item()
    cur_pos += 1
    print(f"  Step {step}: {ms:.0f}ms, token={next_token}", flush=True)

# Export perf data
print("\nExporting perf data...", flush=True)
perf = ttnn.get_all_programs_perf_data()
print(f"  Type: {type(perf)}, length: {len(perf) if perf else 0}")
if perf:
    outdir = PROJECT_ROOT / "ocr_lab" / "perf_reports"
    outdir.mkdir(exist_ok=True)
    # Print first few entries to understand format
    if isinstance(perf, str):
        with open(outdir / "ops_perf_raw.txt", "w") as f:
            f.write(perf)
        print(f"  Saved raw string to ops_perf_raw.txt ({len(perf)} chars)")
        print(f"  First 500 chars: {perf[:500]}")
    elif isinstance(perf, list):
        print(f"  First entry type: {type(perf[0]) if perf else 'empty'}")
        if perf and isinstance(perf[0], str):
            with open(outdir / "ops_perf.csv", "w") as f:
                for line in perf:
                    f.write(line + "\n")
            print(f"  Saved {len(perf)} lines to ops_perf.csv")
            for line in perf[:5]:
                print(f"    {line[:200]}")
        elif perf and isinstance(perf[0], dict):
            keys = list(perf[0].keys())
            print(f"  Keys: {keys}")
            with open(outdir / "ops_perf.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(perf)
            print(f"  Saved {len(perf)} rows to ops_perf.csv")
else:
    print("  No perf data returned")

# Also check generated directory
import glob
csvs = glob.glob("generated/profiler/**/*.csv", recursive=True) + \
       glob.glob("/home/mlops/dll_project/.tracy_artifacts/**/*.csv", recursive=True)
print(f"\nFound CSV files: {csvs}")

ttnn.close_mesh_device(mesh)
