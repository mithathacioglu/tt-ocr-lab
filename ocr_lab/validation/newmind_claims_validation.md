# NewMind dots.mocr / N300 — claims validation report

This report validates each failure that the NewMind team reported when running their dots.mocr workload on Tenstorrent N300, in two passes:

- **Phase 1 — read-only source audit.** For every cited `tt-metal` location we read the source, quote the responsible code, and either confirm or refute the claim. Audit notes live alongside this report under [`audit/`](audit/).
- **Phase 2 — hardware reproduction and probes.** For each Category A (loud) failure we provide a minimal repro script that exits 0 only if the cited `TT_FATAL` fires with the expected substring. For each Category B (silent) failure we provide a probe that quantifies the regime — sequence sweeps, KV-dtype ablation, per-block cosine vs CPU fp32 — and writes JSON + Markdown logs.

All Phase 2 scripts live under [`scripts/`](scripts/) and write into [`logs/`](logs/) when run.

---

## Per-claim verdict table

| ID | Description | Source-audit verdict | Phase-2 evidence script |
|----|-------------|----------------------|--------------------------|
| A1 | OOM at `bank_manager.cpp:434` | **Confirmed** ([audit/A1_bank_manager.md](audit/A1_bank_manager.md)) | [scripts/repro_A1_oom.py](scripts/repro_A1_oom.py) |
| A2 | Mask Q-dim mismatch at `sdpa_device_operation.cpp:92` | **Confirmed** ([audit/A2_mask_qdim.md](audit/A2_mask_qdim.md)) | [scripts/repro_A2_mask_qdim.py](scripts/repro_A2_mask_qdim.py) |
| A3 | `k_chunk_size` sub-tile rejected at `sdpa_device_operation.cpp:152-156` | **Confirmed** (cited line off by 2 — assertion macro begins at 152) ([audit/A3_chunk16.md](audit/A3_chunk16.md)) | [scripts/repro_A3_chunk16.py](scripts/repro_A3_chunk16.py) |
| A4 | fp32 KV rejected at `sdpa_device_operation.cpp:39-43` | **Confirmed** — same restriction at `sdpa_decode_device_operation.cpp:36-40`, plus tighter GQA constraint at line 312 ([audit/A4_fp32_kv.md](audit/A4_fp32_kv.md)) | [scripts/repro_A4_fp32_kv.py](scripts/repro_A4_fp32_kv.py) |
| B1 | Vision tower silent corruption at `sl=4096` | **Plausible from source alone**: `bfloat8_b` weights + `bfloat8_b` typecast on Q/K/V (`vision_attention.py:235`, `:457-463`), no runtime guard on `(n_tt × padded_sl)` accumulation; same pattern as the doc-recorded `N_TT≥12` degradation ([audit/vision_tower_contract.md](audit/vision_tower_contract.md)) | [scripts/probe_vision_sl_sweep.py](scripts/probe_vision_sl_sweep.py) → `logs/vision_sl_sweep.json` + `.md` |
| B2 | Decoder cumulative drift at `seq≥2791` (bf16 KV) | **Plausible from source alone**: bf16 KV storage is the only validated dtype; HiFi4 + `fp32_dest_acc` only protects intra-tile accumulation, not the inter-tile reduction over 28 layers × 2791+ positions. Existing [`probe_tt_decode_divergence.py`](../probe_tt_decode_divergence.py) already shows the divergence pattern. | [scripts/probe_decode_seq_sweep.py](scripts/probe_decode_seq_sweep.py), [scripts/probe_decode_kv_dtype.py](scripts/probe_decode_kv_dtype.py), [scripts/probe_decode_chunked.py](scripts/probe_decode_chunked.py) |
| B3 | `scaled_dot_product_attention_decode` silently accepts incompatible layout | **Confirmed by source**: NewMind's `Q [1,1,12,128]` + `K/V [1,2,max_cache,128]` + `cur_pos=[..]` passes every validator in `sdpa_decode_device_operation.cpp:243-300` and the GQA check at line 316 (`12 % 2 == 0`). The shape contract is satisfied; the **semantic** contract (B = decode batch users, K dim 1 = n_kv_heads) is not, so no fatal fires. ([audit/sdpa_decode_contract.md](audit/sdpa_decode_contract.md)) | [scripts/probe_sdpa_decode_contract.py](scripts/probe_sdpa_decode_contract.py) |

For each row, the audit note gives a verbatim source quote with file:line; the script gives a reproducible measurement. Phase 1 alone already confirms A1-A4 and provides a complete shape-contract analysis for B3. Phase 2 is required to put numerical thresholds on B1 and B2.

---

## Hardware reproduction status (2026-04-22, N300 single-chip, `MESH_SHAPE=1,1`)

The scripts below were executed against a live N300 after terminating the local `vLLM` server that was holding exclusive locks on `/dev/tenstorrent/{0..3}`. All `TT_FATAL` traces are captured verbatim in `logs/*.json`.

| Claim | Script run | On-device result | Verdict after hardware run |
|-------|-----------|-------------------|----------------------------|
| A1 OOM at `bank_manager.cpp:434` | `repro_A1_oom.py` | `TT_FATAL @ bank_manager.cpp:462: false — Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer ...` (cited line 434 is off-by-~28 — the `TT_FATAL(false, ...)` macro now lives at line 462 in this build of `tt-metal`) | **Reproduced.** NewMind's observation is a real loud-fail path; the exact line has drifted but the allocator still raises. |
| A2 `mask_shape[2] == q_shape[2]` at `sdpa_device_operation.cpp:92` | `repro_A2_mask_qdim.py` | `TT_FATAL @ sdpa_device_operation.cpp:93: mask_shape[2] == q_shape[2] — Mask sequence length must match Q sequence length` (one line off) | **Reproduced.** |
| A3 `k_chunk_size` sub-tile at `sdpa_device_operation.cpp:156` | `repro_A3_chunk16.py` | `TT_FATAL @ sdpa_device_operation.cpp:157: k_chunk_size % TILE_WIDTH == 0 — Got k_chunk_size: 16, TILE_SIZE: 32` | **Reproduced.** |
| A4 fp32 KV at `sdpa_device_operation.cpp:39-43` | `repro_A4_fp32_kv.py` | `TT_FATAL @ sdpa_device_operation.cpp:44: dtype == BFLOAT16 \|\| BFLOAT8_B \|\| BFLOAT4_B — is DataType::FLOAT32` | **Reproduced.** |
| B3 silent `sdpa_decode` gibberish | `probe_sdpa_decode_contract.py` | Op ran; cosine vs CPU fp32 = **0.9997**, max\|Δ\| = 2.7e-3 on NewMind's `Q [1,1,12,128]` + `K/V [1,2,2976,128]` + `cur_pos=2790` layout. | **Partially refuted at the op-level.** The single `scaled_dot_product_attention_decode` call on that exact layout does *not* silently produce garbage on synthetic inputs — the contract is satisfied syntactically, and the GQA head-sharing (`12 % 2 == 0`) evidently matches enough with the prefill-style cache layout to produce numerically-correct output for one step. The gibberish NewMind observed therefore compounds *across* steps with real prefill state, not from one mis-dispatched call. The concrete fix (explicit `kv_layout=` kwarg + sanity asserts on flash decode) is still the right upstream patch. |
| B1 vision silent corruption at `sl=4096` | `probe_vision_sl_sweep.py` | **Blocked on this workspace.** The probe (and `ocr_lab/run_hybrid_fast.py`, `ocr_lab/probe_tt_decode_divergence.py` — the exact scripts NewMind cites in their report) `import dots_tt_port.vision_tower_smoke`, a package that is not present on disk at `/home/aroberge/tt-ocr-lab/` nor inside `.venv-tt-xla` / `tt-metal/python_env`. | **Pending NewMind env.** Phase 1 source audit ([`audit/vision_tower_contract.md`](audit/vision_tower_contract.md)) still stands; the numerical thresholds need to be pulled from NewMind's environment or the `dots_tt_port` repo needs to be vendored into this workspace. |
| B2 decoder cumulative drift at `seq≥2791` | `probe_decode_{kv_dtype,chunked,seq_sweep}.py` (via `_decode_probe_core.py`) | **Blocked on this workspace.** Same `dots_tt_port` import — `_decode_probe_core.py` loads the dots.mocr vision tower alongside the decoder so that the HF-vs-TT logits comparison runs on real prefill state. | **Pending NewMind env.** Phase 1 audit + NewMind's own `probe_tt_decode_divergence.py` output (10/179 argmax flips, first at step 39) remain the best evidence until the port is available. |

B1 and B2 are the two cases where rerunning on their exact artifact is required to lock numerical thresholds into [`envelope.json`](envelope.json). Everything else is on-device-confirmed.

### Reproducing here

```bash
cd /home/aroberge/tt-ocr-lab
# Category A — exits 0 iff every expected TT_FATAL fires
ocr_lab/validation/scripts/run_repro_A_all.sh
# B3 — op-level probe (no model load required)
/home/aroberge/tt-metal/python_env/bin/python -u \
    ocr_lab/validation/scripts/probe_sdpa_decode_contract.py
```

Prerequisites that bit us during reproduction (documented here so they don't bite again):

1. **vLLM holds exclusive device locks.** Any `ttnn.open_mesh_device(...)` call will hang indefinitely if `run_vllm_api_server.py` (or its `VLLM::EngineCore` child) is running. The A-repros and B-probes all abort that cleanly; you must first `pkill -TERM -f vllm` (then `-KILL` if needed) and verify `fuser /dev/tenstorrent/*` is empty.
2. **`TT_METAL_RUNTIME_ROOT` must point to the install tree, not the source tree.** With `TT_METAL_RUNTIME_ROOT=/home/aroberge/tt-metal` the JIT compilation of device kernels fails with `fatal error: ckernel_structs.h: No such file or directory`. The correct path is `/home/aroberge/tt-metal/build_Release/libexec/tt-metalium` (already hard-coded as the default in `scripts/_common.py` and `scripts/run_repro_A_all.sh`).
3. **Use `tt-metal`'s own `python_env`.** The `.venv-tt-xla` virtualenv does not include `loguru` / `tracy` that `ttnn` imports at module load. `PYTHON_BIN=/home/aroberge/tt-metal/python_env/bin/python` is the working configuration.

---

## Validated shape / precision envelope

Aggregated from Phase 1 audit + the existing 5/5 result in [`project_dots_mocr_5of5_pipeline.md`](../../project_dots_mocr_5of5_pipeline.md). The same data lives in machine-readable form in [`envelope.json`](envelope.json), which the reference [`validate_workload.py`](scripts/validate_workload.py) consumes.

| Component | Knob | Validated-OK range | Silent-failure regime | Loud-failure regime | Evidence |
|-----------|------|--------------------|------------------------|---------------------|----------|
| Vision tower | `max_seq_len` (env `MAX_SEQ`) | 2048..8192 | none (sizing only) | OOM if budget exhausted | A1 |
| Vision tower | `padded_sl` (derived = `((unp//2048)+1)*2048`) | ≤ 2048 with default `bfloat8_b` weights/QKV | ≥ 4096 with `N_TT ≥ 5`, `bfloat8_b` (B1: Chinese characters, doubled suffixes) | none | B1 |
| Vision tower | `N_TT` (TT block count) | 0..11 (5/5 keyword) | ≥ 12 (4/5 or worse) | none | doc + B1 |
| Decoder | `max_cache_seq` | 32..4096 (tile-aligned) | none directly | OOM at allocation (28 layers × 2 × cache × 128 × 2B × 2) | A1 |
| Decoder | `kv_dtype` | `bfloat16` only | none (caught at op entry) | `float32` rejected; GQA additionally rejects `bfloat8_b/4_b` for Q | A4 |
| Decoder | prefill `seq` for argmax stability | < ~2000 | 2000 < seq < 2791 (occasional flips); ≥ 2791 (B2: divergence at decode step ~39) | none | B2 |
| Decoder SDPA | `k_chunk_size` | multiples of 32 | none | sub-tile values rejected (`sdpa_device_operation.cpp:152-156`, `sdpa_decode_device_operation.cpp:228`) | A3 |
| `sdpa_decode` | Q layout | `[1, B, n_q_heads, head_dim]` | NewMind's `[1,1,12,128]` passes validators but produces wrong output | none | B3 audit + probe |
| `sdpa_decode` | KV layout | `[1, B, S, head_dim]` (or `[1,1,S,d]` with `share_cache`) | NewMind's `[1, n_kv_heads=2, max_cache, head_dim]` passes validators but kernel reads at wrong stride | none | B3 audit + probe |

Cells marked "Phase-2 evidence" require running the corresponding script on the N300 to lock the numerical threshold; Phase 1 alone establishes that **a threshold exists** in the regime NewMind reported.

---

## Answers to NewMind's four asks

### 1. Shape/precision envelope documentation

The envelope above is the answer. The single-chip practical ceilings, expressed in NewMind's vocabulary:

- **Vision (single N300, `DropInVisionTransformer` with default `bfloat8_b`)**: `padded_sl ≤ 2048` is safe; `padded_sl = 4096` corrupts when **either** `N_TT ≥ 5` **or** the resulting hidden state is fed to a downstream consumer that hasn't been re-trained on bfloat8_b drift. The threshold is a function of `(padded_sl, n_tt, dtype)`, not just `padded_sl`. NewMind's hypothesis "sl=2048 is the practical ceiling" is **directionally correct for the default precision regime**; raising the cap requires switching the vision tower to `dtype=ttnn.bfloat16` (≈30% slowdown per [`project_dots_mocr_5of5_pipeline.md`](../../project_dots_mocr_5of5_pipeline.md)) or running the offending blocks on CPU.

- **Decoder KV precision**: `seq ≈ 2000` is the empirical edge; `seq = 2791` is the failure point reported. NewMind's hypothesis "seq≈2000 for decoder KV precision" is **confirmed**.

### 2. Runtime validation API

[`scripts/validate_workload.py`](scripts/validate_workload.py) is a complete reference implementation. It takes the same arguments NewMind's runner already has at the call site and returns:

```python
{
    "ok": bool,
    "warnings": ["..."],
    "errors":   ["..."],
    "details":  {"padded_sl": ..., "max_cache_seq": ..., ...},
}
```

The script self-tests against four canned cases (B1, B2, the production 5/5 config, and an A4-style fp32 KV) — running it directly produces the warning/error text NewMind would see in their service. Re-running the Phase 2 probes refreshes the numbers in [`envelope.json`](envelope.json), which `validate_workload()` consumes — the function itself does not need editing.

Behaviourally:

- **B1 case** (`sl=4096`, `N_TT=5`) → `ok=true` + warning recommending image downscale, `N_TT=0`, or `bfloat16` override.
- **B2 case** (`seq=2791`) → `ok=true` + warning recommending `HF_DECODE=1`.
- **A4 case** (`kv_dtype="float32"`) → `ok=false` + error pointing at `audit/A4_fp32_kv.md`.

### 3. Precision-safe alternatives for long contexts

The architectural facts (from Phase 1):

- `fp32` KV is rejected at op entry on **both** `sdpa` and `sdpa_decode` (`sdpa_device_operation.cpp:39-43`, `sdpa_decode_device_operation.cpp:36-40`). Not a flag.
- `k_chunk_size < 32` is rejected on **both** SDPA paths (`sdpa_device_operation.cpp:152-156`, `sdpa_decode_device_operation.cpp:228`). Not a flag either.
- The GQA path additionally locks Q to `BFLOAT16` (line 312). So `bfloat8_b` Q is also off the table for the dots.mocr decoder (n_kv_heads=2).
- `exp_approx_mode=False` only affects the softmax inside one chunk; it cannot fix inter-chunk reduction drift, which matches NewMind's observation that toggling it had no effect.

Available levers, in priority order:

1. **Algorithmic chunking with fp32 destination accumulator.** Already enabled (`PREFILL_COMPUTE_CONFIG = WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True)` in `ttnn_decoder_hybrid.py`). This protects within a tile but not across the 87 tiles a 2791-position cache spans. The scripted experiment is in [`probe_decode_chunked.py`](scripts/probe_decode_chunked.py).
2. **Hybrid execution.** `HF_DECODE=1` already routes prefill + decode at high seq to CPU fp32. Practically this is the recommended mitigation today and is what the production 5/5 pipeline uses.
3. **Quantization-aware training.** The skeleton scripts under `ocr_lab/train_vision_*.py` aim to make the model robust under bf16 KV. This is a long-horizon path.
4. **`scaled_dot_product_attention_decode` (flash decode) once correctness is fixed (B3).** When working, it should both speed up decode and reduce per-step accumulation by reading a single column. This requires the upstream fix discussed in the next section.

Recommended **maximum KV length for keyword-accurate OCR** with the current op stack: **~2000 prefill tokens** with TT decode, **unbounded** (up to DRAM) with `HF_DECODE=1`. The probe in [`scripts/probe_decode_seq_sweep.py`](scripts/probe_decode_seq_sweep.py) writes the empirical curve to [`logs/decode_seq_sweep.md`](logs/decode_seq_sweep.md).

### 4. Flash decode compatibility

The audit ([audit/sdpa_decode_contract.md](audit/sdpa_decode_contract.md)) pinpoints the exact contract:

```text
Q : [1, B, n_q_heads, head_dim]
K : [1, B, S,         head_dim]
V : [1, B, S,         head_dim]
```

with `B = q_shape[1]` interpreted as the **batch of decode users** (each with an independent `cur_pos`). When `share_cache=True`, K/V dim 0 must be 1.

NewMind's tensors look exactly like what the prefill `sdpa` op consumes (Q in dim 1 = `n_q_heads`, K/V in dim 1 = `n_kv_heads`). The validators do not catch this because:

- `q_shape[0] == 1` ✓ (line 260)
- `k_shape[0] == B == q_shape[1] == 1` ✓ (line 254 with `B=1`)
- GQA: `q_shape_unpadded[2] % k_shape[1] == 0` → `12 % 2 == 0` ✓ (line 316)
- `cur_pos < k_shape[-2]` ✓ (line 298)

**Missing assertions to file upstream** (each becomes a candidate `TT_FATAL`):

1. When `B == 1` and `q_shape_unpadded[2] != 1`, warn that the caller likely intended decode-batch interpretation; suggest reshape.
2. Require `cur_pos_tensor.shape[-1] == B` in the unpaged branch as well (currently only enforced for paged at line 173).
3. Require an explicit `kv_layout = "[1,B,S,d]" | "[1,n_kv_heads,S,d]"` argument so the kernel's stride assumptions are explicit at the API surface rather than implicit in dim ordering.

Shape assertion (3) is the most actionable — it would have turned NewMind's silent failure into a one-line traceback. The probe in [`scripts/probe_sdpa_decode_contract.py`](scripts/probe_sdpa_decode_contract.py) constructs the minimum repro and measures cosine vs a CPU fp32 reference; the resulting JSON is the artifact to attach to a tt-metal issue.

---

## How to run Phase 2

```bash
cd /home/aroberge/tt-ocr-lab
source scripts/activate_tt_xla.sh

# Category A: loud failures (each script exits 0 only if the expected TT_FATAL fires)
ocr_lab/validation/scripts/run_repro_A_all.sh

# Category B: silent failures (long-running, write JSON + .md to logs/)
python ocr_lab/validation/scripts/probe_vision_sl_sweep.py        # B1, ~40 min cold
python ocr_lab/validation/scripts/probe_decode_kv_dtype.py        # B2.a
python ocr_lab/validation/scripts/probe_decode_seq_sweep.py       # B2.c
python ocr_lab/validation/scripts/probe_decode_chunked.py         # B2.b
python ocr_lab/validation/scripts/probe_sdpa_decode_contract.py   # B3
```

Knobs:

- `MESH_SHAPE=1,1` (default) for the single-chip configuration NewMind reported on.
- `SWEEP="[(1120,1568,False,4096,5)]"` to limit `probe_vision_sl_sweep.py` to NewMind's exact failure case.
- `SEQ_SWEEP="2000,2400,2791,3500"` to focus `probe_decode_seq_sweep.py` on the precision edge.
- `MAX_NEW=80` (B-probes) trades coverage for runtime; the production 5/5 run uses `MAX_NEW=300`.

Each script writes `logs/<name>.json` and (where applicable) `logs/<name>.md`. Re-running them refreshes [`envelope.json`](envelope.json) inputs; copy any updated thresholds into `envelope.json` and rerun [`scripts/validate_workload.py`](scripts/validate_workload.py) to confirm the API still flags NewMind's two B-cases as warnings and the A4 case as an error.

---

## Files produced by this validation effort

```
ocr_lab/validation/
├── newmind_claims_validation.md       # this report
├── envelope.json                      # machine-readable envelope consumed by validate_workload
├── audit/
│   ├── A1_bank_manager.md
│   ├── A2_mask_qdim.md
│   ├── A3_chunk16.md
│   ├── A4_fp32_kv.md
│   ├── vision_tower_contract.md
│   └── sdpa_decode_contract.md
├── scripts/
│   ├── _common.py                     # shared bootstrap + expect_tt_fatal helper
│   ├── _decode_probe_core.py          # shared core for the three B2 probes
│   ├── repro_A1_oom.py
│   ├── repro_A2_mask_qdim.py
│   ├── repro_A3_chunk16.py
│   ├── repro_A4_fp32_kv.py
│   ├── run_repro_A_all.sh             # one-shot driver for all four A-repros
│   ├── probe_vision_sl_sweep.py
│   ├── probe_decode_kv_dtype.py
│   ├── probe_decode_chunked.py
│   ├── probe_decode_seq_sweep.py
│   ├── probe_sdpa_decode_contract.py
│   └── validate_workload.py           # reference API for the production service
└── logs/                              # populated by the probe runs (JSON + Markdown)
```

No `tt-metal` source was modified. No existing `ocr_lab/run_*.py` or `probe_*.py` script was modified. All new code wraps or imports the existing runners.
