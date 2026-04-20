#!/usr/bin/env python3
"""Hybrid vision E2E: first N blocks on TT, rest on CPU. Full OCR test."""
import os, sys, time, json, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.dots_model import ensure_dots_ocr_processor_compat
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

N_TT_BLOCKS = int(os.environ.get("N_TT_BLOCKS", "5"))
IMAGE = P + "/ocr_lab/tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"

# Native resolution inputs
ensure_dots_ocr_processor_compat(str(snapshot_dir))
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
msgs = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE},
    {"type": "text", "text": PROMPT},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"].to(torch.int32)
print(f"pixel_values: {pv.shape}, grid_thw: {gt}")

# Load vision model for CPU blocks
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)

# ============ HYBRID VISION: TT first N blocks + CPU rest ============
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
ref = args.reference_vision_model()
tt_vision = DropInVisionTransformer(ref, args, debug=False)

unpadded_seq = (gt[:,1] * gt[:,2]).sum().item()
seq_len = ((unpadded_seq // 2048) + 1) * 2048
out_dim = args.hf_config.vision_config.out_hidden_size

pos_ids = ref.get_pos_ids_by_grid(gt.cpu())
pos_ids = torch.cat(pos_ids, dim=0)
max_grid = int(gt.cpu()[:,1:].max().item())
rotary_full = ref.rotary_pos_emb(max_grid).cpu()
rotary_tt = rotary_full[pos_ids].flatten(1).float()
cos_orig = rotary_tt.cos().unsqueeze(1).repeat(1, 1, 2)
sin_orig = rotary_tt.sin().unsqueeze(1).repeat(1, 1, 2)
cos_meta, sin_meta = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
cos_padded = F.pad(cos_meta, (0, 0, 0, seq_len - unpadded_seq), value=1).unsqueeze(0).unsqueeze(0)
sin_padded = F.pad(sin_meta, (0, 0, 0, seq_len - unpadded_seq), value=0).unsqueeze(0).unsqueeze(0)
cos_tt = ttnn.from_torch(cos_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
sin_tt = ttnn.from_torch(sin_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
cu_seqlens_cpu, cu_window_seqlens, _, window_index = qwen2_5_vision_transformer_preprocess(
    seq_len=unpadded_seq, grid_thw=gt, head_dim=args.vision_head_dim,
    spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size,
    patch_size=args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cu_seqlens_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# CPU rotary + cu_seqlens for remaining CPU blocks
cpu_rotary = vm.rot_pos_emb(gt)
cpu_cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)

# Cold run
print(f"\n=== Hybrid Vision: {N_TT_BLOCKS} TT + {42-N_TT_BLOCKS} CPU ===")
patch_embed = ref.patch_embed(pv)
tt_input = tt_vision.tt_model.prepare_input(patch_embed, window_index, seq_len)
x_tt = tt_input
for i in range(N_TT_BLOCKS):
    x_tt = tt_vision.tt_model.blocks[i](x_tt, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
tt_cpu_raw = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
ttnn.deallocate(x_tt)
print("  Cold done")

# Warm run
t_vis_start = time.perf_counter()
patch_embed = ref.patch_embed(pv)
tt_input = tt_vision.tt_model.prepare_input(patch_embed, window_index, seq_len)
x_tt = tt_input
for i in range(N_TT_BLOCKS):
    x_tt = tt_vision.tt_model.blocks[i](x_tt, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
tt_cpu_raw = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
tt_ms = (time.perf_counter() - t_vis_start) * 1000
ttnn.deallocate(x_tt)

# Continue on CPU
tt_hidden = tt_cpu_raw[:, 0:1, :unpadded_seq, :out_dim].squeeze(0).squeeze(0).to(torch.bfloat16)

t_cpu_start = time.perf_counter()
h = tt_hidden
with torch.no_grad():
    for i in range(N_TT_BLOCKS, 42):
        h = vm.blocks[i](h, cu_seqlens=cpu_cu, rotary_pos_emb=cpu_rotary)
    if hasattr(vm, 'post_trunk_norm'):
        h = vm.post_trunk_norm(h)
    vision_embeddings = vm.merger(h.to(vm.merger.ln_q.weight.dtype))
cpu_ms = (time.perf_counter() - t_cpu_start) * 1000
total_vis_ms = tt_ms + cpu_ms

print(f"  TT ({N_TT_BLOCKS} blocks): {tt_ms:.0f}ms")
print(f"  CPU ({42-N_TT_BLOCKS} blocks): {cpu_ms:.0f}ms")
print(f"  Total vision: {total_vis_ms:.0f}ms ({total_vis_ms/1000:.1f}s)")
print(f"  Vision embeddings: {vision_embeddings.shape}")

# Close TT (not needed for decoder)
ttnn.close_mesh_device(mesh)

# ============ CPU Decoder (HF model) ============
print("\n=== HF Decoder ===")
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
# We don't need vision tower in HF model, just decoder
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_embeddings.to(embeds.dtype))

# HF prefill (decoder only, vision already done)
t_pf = time.perf_counter()
with torch.no_grad():
    outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True)
prefill_ms = (time.perf_counter() - t_pf) * 1000

from pathlib import Path
cfg = json.loads(Path(str(snapshot_dir)).joinpath("config.json").read_text())
vocab_size = cfg.get('vocab_size', 151936)
first_tok = outputs.logits[0, -1, :vocab_size].float().argmax(-1).item()
past_kv = outputs.past_key_values
print(f"  Prefill: {prefill_ms:.0f}ms, first={repr(tokenizer.decode([first_tok]))}")

# Decode
generated = [first_tok]
next_token = first_tok
with torch.no_grad():
    for step in range(1, 300):
        out = hf(input_ids=torch.tensor([[next_token]]), past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[0, -1, :vocab_size].float().argmax(-1).item()
        generated.append(next_token)
        if next_token in (151643, 151645):
            break
del hf

gen_text = tokenizer.decode(generated, skip_special_tokens=True)
REF_KEYWORDS = ["hükümler", "davalı", "olunmuş", "Dilekçesi", "kesinleştiği"]
found = [kw for kw in REF_KEYWORDS if kw in gen_text]

print(f"\n{'='*60}")
print(f"  Vision ({N_TT_BLOCKS} TT + {42-N_TT_BLOCKS} CPU): {total_vis_ms:.0f}ms ({total_vis_ms/1000:.1f}s)")
print(f"  Decoder prefill: {prefill_ms:.0f}ms")
print(f"  Tokens: {len(generated)}")
print(f"  Keywords: {found} / {REF_KEYWORDS}")
print(f"  Text: {repr(gen_text[:300])}")
print(f"{'='*60}")
