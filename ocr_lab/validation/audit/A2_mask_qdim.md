# A2 — `mask_shape[2] == q_shape[2]` at `sdpa_device_operation.cpp:92`

NewMind's reported error: `Mask sequence length must match Q sequence length`.

## Source location

File: [/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp), lines 81-93:

```81:93:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp
            const auto& mask_shape = mask.logical_shape();
            const auto q_shape = q.logical_shape();
            const auto k_shape = k.logical_shape();

            TT_FATAL(
                mask_shape[0] == 1 || mask_shape[0] == q_shape[0],
                "Mask batch dim must either be 1 (to be broadcasted across all batches) or must match Q batch "
                "dimension");
            TT_FATAL(
                mask_shape[1] == 1 || mask_shape[1] == q_shape[1],
                "Mask num_heads must either be 1 (to be broadcasted across all heads) or must match Q heads dimension");
            TT_FATAL(mask_shape[2] == q_shape[2], "Mask sequence length must match Q sequence length");
            TT_FATAL(mask_shape[3] == k_shape[2], "Mask sequence length must match K sequence length");
```

The check uses `logical_shape()`, not `padded_shape()`. This is what makes the prefill SDPA reject a `(1,1,1,N)` mask when Q's logical seq is already 32 (because the caller reshape-padded Q to a tile but kept the mask at logical seq 1).

## Cross-reference

The pinned project doc records exactly this trap:

```31:31:tt-ocr-lab/project_dots_mocr_5of5_pipeline.md
- Mask shape (1,1,1,N) "fix": SDPA asserts because Q is (1,12,1,128) but reshape to tile makes Q logical seq=32; original (1,1,32,N) mask is correct
```

Note the asymmetry the project doc captures and the source confirms: Q logical may go up to 32 implicitly via the reshape, but the mask's logical seq must be set to 32 by the caller to satisfy line 92. The op does not auto-align mask to padded Q.

## Verdict

**CONFIRMED.** Cited line is exact. Behaviour as described.

## Categorization

Category-A (loud) failure. No silent path here — the mask must be supplied at the right logical seq.
