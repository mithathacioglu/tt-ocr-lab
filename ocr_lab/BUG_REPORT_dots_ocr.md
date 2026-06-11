# dots.ocr on N300 — Three Production-Blocking Issues

**Branch:** `alnah005/dots_ocr_N300` @ `ed252877`
**Dependencies:** `transformers==4.56.1`, `tt-metal` @ `c09f09c35a` (ign/dots_ocr_opt baseline build)
**Mesh:** `MESH_DEVICE=N300` (1, 2 — TP=2 to match `n_kv_heads=2`)
**Mode:** `TT_SYMBIOTE_RUN_MODE=TRACED`
**Reproducer:** https://github.com/mithathacioglu/tt-ocr-lab

---

## What works (baseline)

- `demo_image1.jpg` (your lymphoma table): **22.4s for 512 tokens**, output identical to your posted reference.
- Single Turkish A4 court ruling (`tmp_1pdf_page-1.png`, 2807 ISL / ~180 OSL): **10.1s for 213 tokens, ~40 ms/tok**, semantically correct (8/10 keyword fidelity at `max_new_tokens=2000`).

The pipeline behaves correctly for a **single** `pipeline.generate()` call. The three issues below all surface when running multiple pages or multiple workers.

---

## Issue 1: State degradation across successive `pipeline.generate()` calls

**Symptom:** After 2–3 successful `generate()` calls on the same warm pipeline (same trace shape, no shape changes between calls), throughput collapses ~1000×:

| Call | Time | Tokens | ms/tok |
|---|---|---|---|
| Page 1 | 25.97 s | 746 | 34.8 |
| Page 2 | 39.13 s | 1161 | 33.7 |
| Page 3 | 42.62 s | 1287 | 33.1 |
| Page 4 | **never completes (18+ min observed, ~30,000 ms/tok)** | — | — |

No exception. Trace replay continues in the device log — kernels just stall. Reproduced three independent ways:

1. Sequential warm server (single N300, hung at page 4 after pages 1–3 succeeded)
2. 4-board DP (worker assigned pages 6,7 — page 6 OK at 41 s, page 7 hung)
3. Worker 2 isolated retry (pages 6, 7 — page 6 OK, page 7 hung)

**Suspected location:** `models/experimental/tt_symbiote/models/dots_ocr.py:445` calls `self.paged_cache.reset()` and `self._decode_cache_position = None`. In `modules/attention.py:312` the comment says:

> "Device buffer addresses are preserved so traces that reference them remain valid. The stale values in the cache are harmless: prefill overwrites positions 0..seq_len-1, and paged_sdpa_decode uses cur_pos_tensor to limit attention to valid positions."

The stale values do **not** appear harmless after ~3 calls. Either DRAM/L1 leaks between calls, or paged_sdpa_decode sometimes reads invalid positions.

**What we tried:** Adding `ttnn.deallocate(self._decode_cache_position)` between calls in the server made it worse — hung at page 1, confirming the trace-buffer-preservation reasoning. So the fix needs to live inside the cache/trace lifecycle, not at the Python entry point.

---

## Issue 2: `TT_METAL_PCI_BUS_IDS` doesn't isolate workers across boards

**Symptom:** When we launch 4 worker processes, each pinned to its own N300 board's PCI bus:

```bash
WORKER_ID=0 TT_METAL_PCI_BUS_IDS=0000:31:00.0 python -m pytest ... &
WORKER_ID=1 TT_METAL_PCI_BUS_IDS=0000:4b:00.0 python -m pytest ... &
WORKER_ID=2 TT_METAL_PCI_BUS_IDS=0000:b1:00.0 python -m pytest ... &
WORKER_ID=3 TT_METAL_PCI_BUS_IDS=0000:ca:00.0 python -m pytest ... &
```

All four workers stall on `Waiting for lock 'CHIP_IN_USE_0_PCIe' which is currently held by thread TID: ...`. They all race for chip 0 regardless of the bus ID we set. We saw two failure modes:

- Lock-wait deadlock (forever)
- `Timeout waiting for Ethernet core service remote IO request` on the second worker once W0 grabs chips

We added 15 s and then 60 s stagger between launches — didn't help. Looks like the env var is not the right knob.

**Question for the call:** What's the correct way to bind a worker process to a specific N300 board? Is there a `physical_device_ids` equivalent at the env-var or `open_mesh_device` level we should be using?

---

## Issue 3: Repeated test cycles degrade hardware, `tt-smi -r` doesn't always recover

**Symptom:** After a series of test cycles (run → SIGKILL on hang → reset → retry), the N300 cards fall into a state where:

- `tt-smi -r` reports success but subsequent device opens fail
- `risc_firmware_initializer.cpp:1133: tt::exception` ("Timeout waiting for physical cores to finish")
- `RuntimeError: ARC core (0, 10) failed to start.`

Only fully resolved by `sudo modprobe -r tenstorrent && sudo modprobe tenstorrent`. After ~5–10 reset cycles, even modprobe reload sometimes isn't enough — needs a host reboot.

This is mostly self-inflicted (we SIGKILL'd hung workers from Issue 1), so the upstream fix for #1 reduces this. But it would be useful to know if `tt-smi` has a deeper-reset mode we should be using.

---

## What we'd like to validate on the call

1. Walk through the page-3 / page-4 transition with a live reproducer (we have a 9-page Turkish PDF + `test_dots_ocr_t1_full.py` ready).
2. Confirm worker-isolation pattern: how to actually pin a process to one specific N300 board.
3. Hand off the bf16/HiFi4 weight precision check — Suhail's latest message says the demo "should've passed with the synthetic images" with the bumped precision; we have the synthetic-doc DP test runner ready to verify once #2 is sorted.

Throughput target on our side: **4 pages/second sustained** across 4×N300 (1 page/sec/card). Single-card warm throughput today is `~1.6 pages/min` (limited by per-page decode + ~3 successful calls before degradation). The gap is mostly Issue 1.

---

## Reproducible test files (in the repo)

- `ocr_lab/test_dots_ocr_turkish.py` — single Turkish page, validates 10 keywords
- `ocr_lab/test_dots_ocr_t1_full.py` — 9-page Turkish PDF sequential, shows degradation
- `ocr_lab/test_dots_ocr_t1_dp.py` — per-worker DP slice
- `ocr_lab/launch_t1_dp.sh` — 4-board DP orchestration with SIGTERM-graceful shutdown
- `ocr_lab/ocr_server_warm.py` — warm Flask server (single N300), demonstrates Issue 1 cleanly
- `ocr_lab/ocr_pdf_dp.py` — CLI/server wrapper with PDF→pages splitting
