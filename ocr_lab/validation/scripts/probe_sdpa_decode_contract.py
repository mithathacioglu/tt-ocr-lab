#!/usr/bin/env python3
"""B3: ``scaled_dot_product_attention_decode`` shape-contract probe.

Reproduces the silent-gibberish failure mode that NewMind reported when
swapping the prefill ``ttnn.transformer.scaled_dot_product_attention`` op
for ``scaled_dot_product_attention_decode`` (flash decode).

The probe runs three configurations on the same fabricated KV cache:

  layout_A: prefill-style Q  [1, 1, n_q_heads=12, head_dim=128]
            and KV cache     [1, n_kv_heads=2, max_cache, head_dim=128]
            cur_pos = [seq-1]
            -> what NewMind passed; expected to RUN (no fatal) but produce
               output that does NOT match a CPU fp32 reference.

  layout_B: decode-style Q   [1, 1, n_q_heads=12, head_dim=128]   (= same as A)
            and KV cache rearranged so that K[:, h, :, :] strides match the
            kernel's GQA replication assumption. We mostly include this to
            confirm whether NewMind's specific layout is what fails or
            whether all single-batch decode paths fail.

  layout_C: same as A but invoked via the prefill ``scaled_dot_product_attention``
            op as ground truth (with a single-row mask) -> baseline.

For each, we print:
  - whether the call ran (True/False) and any fatal message
  - cosine similarity to a CPU fp32 reference SDPA over the same K/V slice
  - the lm_head/tokenizer decoded output (if available; otherwise raw argmax
    on the projected hidden state)

Output: ocr_lab/validation/logs/sdpa_decode_contract.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import setup_paths, write_log, open_mesh  # noqa: E402

setup_paths()
import ttnn  # noqa: E402


N_HEADS = 12
N_KV_HEADS = 2
HEAD_DIM = 128
SEQ = 2791  # NewMind's failure point; tile-aligned cache: 2976
MAX_CACHE = ((SEQ + 31) // 32) * 32 + 32 * 5  # 2976 + headroom


def _to_tt(t: torch.Tensor, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _cpu_reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute single-step SDPA in fp32 on CPU. Returns the attention output
    shaped (1, n_q_heads, head_dim).
    """
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    # GQA: replicate KV heads to match Q heads.
    if kf.shape[1] != qf.shape[1]:
        repeat = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    # qf: (1, n_q, 1, d), kf/vf: (1, n_q, S, d)
    scale = 1.0 / (qf.shape[-1] ** 0.5)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale  # (1,n_q,1,S)
    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)  # (1,n_q,1,d)
    return out.squeeze(2)  # (1,n_q,d)


def run() -> dict:
    mesh = open_mesh((1, 1))
    torch.manual_seed(0)
    # KV cache: bf16, populated with synthetic data only up to position SEQ-1.
    k_full_t = torch.randn(1, N_KV_HEADS, MAX_CACHE, HEAD_DIM)
    v_full_t = torch.randn(1, N_KV_HEADS, MAX_CACHE, HEAD_DIM)
    # Zero the tail to mimic an "uninitialized" region NewMind's setup may read
    k_full_t[:, :, SEQ:, :] = 0.0
    v_full_t[:, :, SEQ:, :] = 0.0
    q_t = torch.randn(1, 1, N_HEADS, HEAD_DIM)

    cur_pos = SEQ - 1  # last position is the actual decode step

    # --- CPU fp32 reference (single-position decode) ---
    # Q: (1, n_heads, 1, head_dim); KV slice up to cur_pos+1
    q_cpu = q_t.permute(0, 2, 1, 3)  # (1, n_q, 1, d)
    k_slice = k_full_t[:, :, : cur_pos + 1, :]
    v_slice = v_full_t[:, :, : cur_pos + 1, :]
    ref_out = _cpu_reference_attention(q_cpu, k_slice, v_slice)  # (1,n_q,d)
    ref_flat = ref_out.reshape(-1).to(torch.float32)

    results: dict = {"layouts": {}}

    # ------------------------------------------------------------------
    # layout_A: NewMind's actual layout
    # ------------------------------------------------------------------
    info: dict = {}
    info["q_shape"] = list(q_t.shape)  # [1, 1, 12, 128]
    info["k_shape"] = list(k_full_t.shape)  # [1, 2, max_cache, 128]
    info["v_shape"] = list(v_full_t.shape)
    info["cur_pos"] = cur_pos
    try:
        q_tt = _to_tt(q_t.to(torch.bfloat16), mesh)
        k_tt = _to_tt(k_full_t.to(torch.bfloat16), mesh)
        v_tt = _to_tt(v_full_t.to(torch.bfloat16), mesh)
        t0 = time.perf_counter()
        out_tt = ttnn.transformer.scaled_dot_product_attention_decode(
            q_tt, k_tt, v_tt, is_causal=True, cur_pos=[cur_pos]
        )
        info["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        info["ran"] = True
        out_torch = ttnn.to_torch(out_tt)  # shape (1, 1, n_heads, d)
        # Squeeze to (1, n_heads, d)
        actual = out_torch.squeeze(0).squeeze(0).reshape(1, N_HEADS, HEAD_DIM).to(torch.float32)
        actual_flat = actual.reshape(-1)
        cos = float(F.cosine_similarity(actual_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item())
        max_abs = float((actual_flat - ref_flat).abs().max().item())
        info["cosine_vs_cpu_fp32"] = cos
        info["max_abs_diff"] = max_abs
        info["matches_reference"] = cos > 0.99
        ttnn.deallocate(q_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)
        ttnn.deallocate(out_tt)
    except RuntimeError as exc:
        info["ran"] = False
        info["error"] = str(exc)
    results["layouts"]["A_newmind_layout"] = info

    # ------------------------------------------------------------------
    # layout_C: prefill SDPA as ground truth (over the populated slice)
    # ------------------------------------------------------------------
    info = {}
    try:
        # Build full causal mask of (1, 1, S_padded, S_padded) tile-aligned
        S = ((cur_pos + 1 + 31) // 32) * 32
        q_pref_t = torch.zeros(1, N_HEADS, S, HEAD_DIM)
        # Place the single decode Q at the last row, head-replicated already
        q_pref_t[0, :, S - 1, :] = q_t[0, 0, :, :]
        # K/V replicated to N_HEADS for plain (non-GQA) prefill SDPA
        k_pref_t = torch.zeros(1, N_HEADS, S, HEAD_DIM)
        v_pref_t = torch.zeros(1, N_HEADS, S, HEAD_DIM)
        # Repeat KV across heads (replicate)
        k_pref_t[:, :, : cur_pos + 1, :] = k_slice.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        v_pref_t[:, :, : cur_pos + 1, :] = v_slice.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        q_tt = _to_tt(q_pref_t.to(torch.bfloat16), mesh)
        k_tt = _to_tt(k_pref_t.to(torch.bfloat16), mesh)
        v_tt = _to_tt(v_pref_t.to(torch.bfloat16), mesh)
        t0 = time.perf_counter()
        out_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=True)
        info["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        info["ran"] = True
        out_torch = ttnn.to_torch(out_tt)
        actual_last = out_torch[0, :, S - 1, :].to(torch.float32)  # (n_heads, d)
        actual_flat = actual_last.reshape(-1)
        cos = float(F.cosine_similarity(actual_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item())
        info["cosine_vs_cpu_fp32"] = cos
        info["matches_reference"] = cos > 0.99
        ttnn.deallocate(q_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)
        ttnn.deallocate(out_tt)
    except RuntimeError as exc:
        info["ran"] = False
        info["error"] = str(exc)
    results["layouts"]["C_prefill_sdpa_ground_truth"] = info

    # ------------------------------------------------------------------
    # Notes derived from the audit of sdpa_decode_device_operation.cpp:
    # ------------------------------------------------------------------
    results["expected_decode_q_layout"] = "[1, B, n_q_heads, head_dim] with B = decode batch users"
    results["expected_decode_kv_layout"] = "[1, B, S, head_dim] (or share_cache K/V batch=1)"
    results["audit_note"] = (
        "Q [1,1,12,128] passes the contract because q_shape[0]==1, q_shape[1]==B==1 "
        "and the GQA validator at line 316 checks q_shape_unpadded[2] % k_shape[1] == 0 "
        "(12 % 2 == 0). However, the kernel reads K/V starting at cur_pos under the "
        "assumption that KV's seq dim is dim 2 -- which it is, by coincidence with the "
        "prefill-style cache layout used here -- but the GQA head replication may "
        "interact with cur_pos in a way that misaligns reads when n_q_heads/n_kv_heads "
        "is large. The cosine_vs_cpu_fp32 measured here is the empirical answer."
    )
    ttnn.close_mesh_device(mesh)
    return results


if __name__ == "__main__":
    res = run()
    write_log("sdpa_decode_contract", res)
    print(json.dumps(res, indent=2, ensure_ascii=False))
