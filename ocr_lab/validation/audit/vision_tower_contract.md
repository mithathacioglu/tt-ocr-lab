# Vision tower contract — `models/demos/qwen25_vl/tt`

Audit of the precision and shape contract that governs `DropInVisionTransformer` for dots.mocr workloads. Relevant to claim **B1** (silent garbage at `sl=4096`).

## How `seq_len` (`sl`) is chosen

[/home/aroberge/tt-metal/models/demos/qwen25_vl/tt/model.py](/home/aroberge/tt-metal/models/demos/qwen25_vl/tt/model.py), `DropInVisionTransformer.forward`:

```229:232:tt-metal/models/demos/qwen25_vl/tt/model.py
            unpadded_seq_len = (grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
            # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048
```

The same arithmetic appears in `VisionTransformer.prepare_input` (line 107). NewMind's reported case `unp=2259` therefore lands at `sl=2*2048=4096`. The padding is unconditional (`+1` always), so `unp=2048` lands at `sl=4096`, `unp=4096` at `sl=6144`, etc.

`max_seq_len` (the `MAX_SEQ` env var in NewMind's runners) is **not** consulted here. It only sizes pre-allocated buffers in `VisionAttention` (line 62, 82) and `VisionBlock` (line 31). If `seq_len > max_seq_len`, the run will likely OOM or under-allocate, but the `seq_len` the kernels actually see is the formula above.

## Precision settings inside each block

Both the QKV matmul weights and the post-RoPE Q/K/V activations are stored / quantized to `bfloat8_b` on the device:

[/home/aroberge/tt-metal/models/demos/qwen25_vl/tt/vision_attention.py](/home/aroberge/tt-metal/models/demos/qwen25_vl/tt/vision_attention.py):

```235:242:tt-metal/models/demos/qwen25_vl/tt/vision_attention.py
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=ttnn.bfloat8_b,
            ...
            cache_file_name=cache_name("wqkv"),
        )
```

```457:463:tt-metal/models/demos/qwen25_vl/tt/vision_attention.py
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ...
        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
```

The default `dtype` passed to `DropInVisionTransformer.__init__` is also `ttnn.bfloat8_b` (model.py:166).

## Why this matters for B1

The cumulative error in a sequence of `bfloat8_b` matmuls grows roughly linearly with sequence length (more dot-product terms per output element) and with depth (more chained quantizations). The pinned project doc already identified this for the deeper blocks:

```22:23:tt-ocr-lab/project_dots_mocr_5of5_pipeline.md
- **N_TT=11**: 10→5/5 (112.9s), 11→5/5 (108.5s), 12→4/5, 15→4/5. bfloat8_b accumulation in TT vision blocks 12+ degrades "hükümler" word detection. 11 is sweet spot.
```

NewMind's B1 case is the same effect at a different point on the same surface: `N_TT=5` (small depth, was safe at sl=2048) but `sl=4096` (large per-block error). Both axes feed the same accumulator.

## Silent-failure hypothesis

The forward path has **no runtime guard** on either axis (depth × seq_len) of bfloat8_b error. The block code does not consult any `max_seq_len`-derived precision threshold; it just feeds whatever `seq_len` the caller chose into the matmul / SDPA sequence with `bfloat8_b` activations. As soon as the accumulated drift in early blocks pushes the hidden state out of the manifold the CPU fp32 tail expects, the merger and downstream decoder happily produce plausible-looking garbage.

This explains why NewMind sees Chinese characters and doubled Turkish suffixes: the corrupted hidden states still decode to valid token IDs through the LM head, just not to the right ones.

## Suggested probes (driven into Phase 2)

- Sweep `(N_TT, sl)` and measure cosine similarity between the TT vision output (at the unpacked `unp` slice) and a CPU fp32 reference at each block boundary. If cosine drops below ~0.95 at the merger input, the OCR text is unreliable.
- Try forcing `dtype=ttnn.bfloat16` (instead of the default `bfloat8_b`) on `DropInVisionTransformer` to confirm bfloat8_b is the dominant error source. This is a one-line change in the runner; it costs perf but should restore accuracy.

## Verdict

Vision-tower silent corruption at `sl=4096` is **plausible from source inspection alone**: the precision regime (bfloat8_b weights and bfloat8_b activations into SDPA) plus the linear-in-`sl` accumulation behaviour, combined with the absence of any runtime guard, exactly matches NewMind's symptom. Phase 2 sweeps will confirm the threshold.
