# A3 — `k_chunk_size must be divisible by TILE_SIZE` at `sdpa_device_operation.cpp:154`

NewMind's reported error: `k_chunk_size must be divisible by TILE_SIZE` (they cited line 156; the actual TT_FATAL is at line 152-156).

## Source location

File: [/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp), lines 143-156:

```143:156:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp
        if (attrs.program_config.has_value()) {
            auto q_chunk_size = attrs.program_config->q_chunk_size;
            auto k_chunk_size = attrs.program_config->k_chunk_size;

            TT_FATAL(
                q_chunk_size % tt::constants::TILE_WIDTH == 0,
                "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
                q_chunk_size,
                tt::constants::TILE_WIDTH);
            TT_FATAL(
                k_chunk_size % tt::constants::TILE_WIDTH == 0,
                "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
                k_chunk_size,
                tt::constants::TILE_WIDTH);
        }
```

## Note on line number

NewMind reported "line 156"; the actual `TT_FATAL` macro begins at line 152, with the message string at line 154. The 2-line drift is likely a reporting artifact (some compilers report the closing-paren line). The check itself is exactly as described.

There is also an equivalent check in `sdpa_decode` at line 228 of [sdpa_decode_device_operation.cpp](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp):

```227:230:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp
        TT_FATAL(
            operation_attributes.k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, got: {}",
            operation_attributes.k_chunk_size);
```

So sub-tile chunking is rejected on **both** the prefill and decode flash-attention paths. NewMind's mitigation idea of `k_chunk_size<32` cannot land at the kernel API surface; any chunked-attention precision recipe must keep `k_chunk_size` a multiple of 32 and instead reduce per-chunk work via (a) more chunks, (b) higher math fidelity, or (c) fp32 dest accumulator.

## Verdict

**CONFIRMED** (with a 2-line citation offset noted above).

## Categorization

Category-A (loud).
