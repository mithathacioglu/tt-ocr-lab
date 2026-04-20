#!/usr/bin/env python3
"""Full generation: ttnn native decoder prefill + decode loop on 1.pdf."""
from __future__ import annotations
import os, sys, time, types, json
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

# Build vision embeddings + inputs_embeds
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.fixed_page import fixed_page_message

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    ve = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), ve.to(embeds.dtype))
seq = embeds.shape[1]  # 427
del hf

MAX_NEW_TOKENS = 64
padded = ((seq + 127) // 128) * 128
embeds_padded = F.pad(embeds, (0, 0, 0, padded - seq))

# TT setup
import ttnn
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.rope import HfRotarySetup
from models.tt_transformers.tt.common import Mode

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()
model = TTTransformer(args=args, mesh_device=mesh, dtype=ttnn.bfloat16,
                       state_dict=sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat16),
                       attention_class=Attention, rope_setup_class=HfRotarySetup)
print(f"Model: {len(model.layers)} layers", flush=True)

# cos/sin
cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = padded + MAX_NEW_TOKENS + 8
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos_full = emb.cos().to(torch.bfloat16)
sin_full = emb.sin().to(torch.bfloat16)

# Prefill
print(f"Prefill ({padded} tokens)...", flush=True)
cos_pf = ttnn.from_torch(cos_full[:padded].unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
sin_pf = ttnn.from_torch(sin_full[:padded].unsqueeze(0).unsqueeze(0), device=mesh, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
tt_in = ttnn.from_torch(embeds_padded.unsqueeze(1), device=mesh, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape))

last_real = seq - 1
last_idx = (last_real // 32) * 32

t0 = time.perf_counter()
tt_logits = model.forward(tt_in, current_pos=None, rot_mats_global=[cos_pf, sin_pf],
                           mode=Mode.PREFILL, get_last_token=last_idx)
prefill_ms = (time.perf_counter() - t0) * 1000
logits_cpu = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
offset = last_real - last_idx
next_token = logits_cpu[0, 0, offset, :args.vocab_size].float().argmax(-1).item()
print(f"  Prefill: {prefill_ms:.0f}ms, first token: {next_token} = {repr(tokenizer.decode([next_token]))}", flush=True)

# Decode loop
generated = [next_token]
cur_pos = seq
decode_times = []

print(f"Decoding {MAX_NEW_TOKENS-1} tokens...", flush=True)
for step in range(1, MAX_NEW_TOKENS):
    # Embed the token
    token_embed = emb_w[next_token].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1536]

    # Get rot_mats for this position
    pos_idx = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=mesh, dtype=ttnn.int32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    rot_mats_decode = model.rope_setup.get_rot_mats(pos_idx)

    # Input
    tt_token = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape))

    t0 = time.perf_counter()
    tt_logits_d = model.forward(tt_token, current_pos=pos_idx, rot_mats_global=rot_mats_decode,
                                 mode=Mode.DECODE)
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_times.append(decode_ms)

    logits_d = ttnn.to_torch(tt_logits_d, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    next_token = logits_d[0, 0, 0, :args.vocab_size].float().argmax(-1).item()
    generated.append(next_token)
    cur_pos += 1

    if step <= 3 or step % 10 == 0:
        print(f"  Step {step}: {decode_ms:.0f}ms, token={next_token}={repr(tokenizer.decode([next_token]))}", flush=True)

# Results
gen_text = tokenizer.decode(generated, skip_special_tokens=True)
REF = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"
avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0

print(f"\n{'='*60}")
print(f"  Prefill:     {prefill_ms:.0f}ms")
print(f"  Avg decode:  {avg_decode:.0f}ms/token")
print(f"  Total decode: {sum(decode_times):.0f}ms ({len(generated)} tokens)")
print(f"  Generated:   {repr(gen_text[:200])}")
print(f"  Reference:   {repr(REF[:200])}")
print(f"  Match:       {gen_text.strip() == REF.strip()}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh)
