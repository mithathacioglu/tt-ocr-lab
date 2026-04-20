#!/usr/bin/env python3
"""
Persistent OCR service: torch.compile CPU vision + TT vision hybrid.
First page: JIT compile overhead. Subsequent pages: compiled graph reused.
Usage: python ocr_lab/run_persistent_service.py
"""
import os, sys, time, json, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")
torch.set_num_threads(32)

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
sd = patch_dots_mocr_config_for_tt_metal()
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.dots_model import ensure_dots_ocr_processor_compat
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from ocr_lab.inference_config import *

ensure_dots_ocr_processor_compat(str(sd))
proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
tokenizer = getattr(proc, 'tokenizer', proc)

# ========== TT Setup ==========
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, precompute_attn_masks,
    init_tt_kv_cache, DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1)); mesh.enable_program_cache()

N_TT = int(os.environ.get("N_TT", 11))
MAX_SEQ = int(os.environ.get("MAX_SEQ", 8192))
vis_args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=MAX_SEQ)
ref = vis_args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref, vis_args, debug=False)

# CPU vision (fp32) + torch.compile
vm, _ = load_vision_model(sd, 'sdpa', limit_layers=0); vm = vm.eval().to(torch.float32)

COMPILE = os.environ.get("COMPILE_VIS", "1") == "1"
if COMPILE:
    print("Compiling CPU vision blocks...", flush=True)
    for i in range(N_TT, 42):
        vm.blocks[i] = torch.compile(vm.blocks[i], mode="reduce-overhead", dynamic=True)
    print(f"Compiled blocks {N_TT}-41", flush=True)

# HF decoder
cfg = json.loads(__import__('pathlib').Path(str(sd), 'config.json').read_text())
vocab_size = cfg.get('vocab_size', 151936)
rope_theta = cfg.get("rope_theta", 1e6)
hf = AutoModelForCausalLM.from_pretrained(str(sd), trust_remote_code=True,
    torch_dtype=torch.float32, attn_implementation='sdpa').eval()
hf.vision_tower = hf.vision_tower.to(torch.float32)

# Decoder weights on TT (use ModelArgs for correct key format)
from models.tt_transformers.tt.model_config import ModelArgs
dec_args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
dec_sd = dec_args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(dec_sd, mesh)
emb_w = hf.get_input_embeddings().weight.detach().to(torch.bfloat16)

EOS_TOKEN_IDS = {151643, 151645}
KEYWORDS = ['hükümler', 'davalı', 'olunmuş', 'Dilekçesi', 'kesinleştiği']


def process_page(image_path, page_num=1):
    """Process one page end-to-end. Returns (text, timing_dict)."""
    t_total = time.perf_counter()

    # Input: native resolution
    img_msg = {"type": "image", "image": image_path}
    msgs = [{"role": "user", "content": [img_msg,
        {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    pv = inputs["pixel_values"].to(torch.bfloat16); gt = inputs["image_grid_thw"].to(torch.int32)
    seq = inputs["input_ids"].shape[1]

    # TT vision prep
    unp = (gt[:,1]*gt[:,2]).sum().item(); dim = 1536
    sl = ((unp // 2048) + 1) * 2048
    pos_ids = ref.get_pos_ids_by_grid(gt.cpu()); pos_ids = torch.cat(pos_ids, dim=0)
    mg = int(gt.cpu()[:,1:].max().item()); rf = ref.rotary_pos_emb(mg).cpu()
    rot = rf[pos_ids].flatten(1).float()
    ch = rot.cos().unsqueeze(1).repeat(1,1,2); sh = rot.sin().unsqueeze(1).repeat(1,1,2)
    cm, sm = convert_rope_style_hf_to_meta(ch, sh)
    cp = F.pad(cm, (0,0,0,sl-unp), value=1).unsqueeze(0).unsqueeze(0)
    sp = F.pad(sm, (0,0,0,sl-unp), value=0).unsqueeze(0).unsqueeze(0)
    cos_tt = ttnn.from_torch(cp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    sin_tt = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    cs, cws, _, wi = qwen2_5_vision_transformer_preprocess(
        seq_len=unp, grid_thw=gt, head_dim=vis_args.vision_head_dim,
        spatial_merge_size=vis_args.hf_config.vision_config.spatial_merge_size,
        window_size=vis_args.hf_config.vision_config.window_size,
        patch_size=vis_args.hf_config.vision_config.patch_size)
    cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    rotary_cpu = vm.rot_pos_emb(gt)
    cu_cpu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32),
                   (1,0), value=0)

    # 1. TT vision
    t0 = time.perf_counter()
    pe = ref.patch_embed(pv); x = tt_vis.tt_model.prepare_input(pe, wi, sl)
    for i in range(N_TT):
        x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
    raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    h = raw[:,0:1,:unp,:dim].squeeze(0).squeeze(0).to(torch.float32)
    ttnn.deallocate(x)
    tt_vis_ms = (time.perf_counter()-t0)*1000

    # 2. CPU vision (compiled)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N_TT, 42):
            h = vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
        if hasattr(vm, 'post_trunk_norm'): h = vm.post_trunk_norm(h)
        vision_out = vm.merger(h)
    cpu_vis_ms = (time.perf_counter()-t0)*1000

    # 3. HF prefill
    emb_w_fp32 = hf.get_input_embeddings().weight.detach()
    img_mask = inputs["input_ids"] == hf.config.image_token_id
    embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype))
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True)
    prefill_ms = (time.perf_counter()-t0)*1000
    first_tok = outputs.logits[0, -1, :vocab_size].float().argmax(-1).item()
    past_kv = outputs.past_key_values

    # 4. HF greedy decode
    generated = [first_tok]; next_token = first_tok
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(1, 300):
            out = hf(input_ids=torch.tensor([[next_token]]), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = int(out.logits[0, -1, :vocab_size].float().argmax(-1).item())
            generated.append(next_token)
            if next_token in EOS_TOKEN_IDS: break
    decode_ms = (time.perf_counter()-t0)*1000
    total_ms = (time.perf_counter()-t_total)*1000

    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    found = [k for k in KEYWORDS if k in gen_text]

    timings = {
        "tt_vis_ms": tt_vis_ms, "cpu_vis_ms": cpu_vis_ms,
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "total_ms": total_ms, "tokens": len(generated),
        "keywords": f"{len(found)}/{len(KEYWORDS)}",
    }

    # Cleanup TT tensors
    for t in [cos_tt, sin_tt, cu_tt]:
        try: ttnn.deallocate(t)
        except: pass

    return gen_text, timings


# ========== Run multiple pages ==========
IMAGE = P + '/ocr_lab/tmp_1pdf_page-1.png'
N_RUNS = int(os.environ.get("N_RUNS", 3))

print(f"\n{'='*60}")
print(f"Persistent service: N_TT={N_TT}, compile={COMPILE}, runs={N_RUNS}")
print(f"{'='*60}\n")

for run in range(N_RUNS):
    text, t = process_page(IMAGE, page_num=run+1)
    found = [k for k in KEYWORDS if k in text]
    label = "COLD" if run == 0 else "WARM"
    print(f"[{label} run {run+1}] {t['total_ms']:.0f}ms total | "
          f"TT vis {t['tt_vis_ms']:.0f}ms | CPU vis {t['cpu_vis_ms']:.0f}ms | "
          f"prefill {t['prefill_ms']:.0f}ms | decode {t['decode_ms']:.0f}ms ({t['tokens']} tok) | "
          f"keywords {t['keywords']} {found}", flush=True)

print(f"\n{'='*60}")
ttnn.close_mesh_device(mesh)
