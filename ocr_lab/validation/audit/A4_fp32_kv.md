# A4 — `Data type must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B` at `sdpa_device_operation.cpp:39-43`

NewMind's reported error: trying to feed an fp32 KV cache to SDPA is rejected.

## Source location — prefill SDPA

File: [/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp), lines 35-45:

```35:45:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp
    for (const auto* input_tensor : {&q, &k, &v}) {
        TT_FATAL(input_tensor->storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device");
        TT_FATAL(input_tensor->buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device");
        TT_FATAL((input_tensor->layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor->dtype() == DataType::BFLOAT16 || input_tensor->dtype() == DataType::BFLOAT8_B ||
                input_tensor->dtype() == DataType::BFLOAT4_B,
            "Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B and is {}",
            input_tensor->dtype());
        TT_FATAL(!input_tensor->is_sharded(), "Operands to SDPA need to be DRAM/L1 interleaved");
    }
```

The mask path enforces the same set at lines 72-75.

## Source location — decode SDPA

The same restriction holds for `sdpa_decode`: [/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:36-40](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp):

```36:40:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                input_tensor.dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.dtype());
```

GQA (any `k_shape[1] > 1`, which is the dots.mocr decoder case with `n_kv_heads=2`) tightens this further at line 312:

```311:314:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp
        TT_FATAL(
            input_tensors.at(0).dtype() == DataType::BFLOAT16,
            "GQA expects BFLOAT16 input tensor, but got {}",
            input_tensors.at(0).dtype());
```

So under GQA, **Q must be exactly bf16** (bfp8/bfp4 are rejected, fp32 is rejected, only bf16 is accepted), while K/V may be any of the three bf-family types.

## Implication for NewMind's "fp32 KV" ask

The dtype contract is enforced at op entry on the device, at every program-cache miss. There is no runtime flag or kernel option that admits fp32 storage for Q/K/V/mask in either SDPA flavor. Mitigation must be:

1. **Algorithmic** — chunk the K dimension into pieces, run each chunk on TT with fp32 destination accumulator (`WormholeComputeKernelConfig(fp32_dest_acc_en=True)`, already enabled in `ttnn_decoder_hybrid.py`'s `PREFILL_COMPUTE_CONFIG`), and reduce across chunks on host or with an extra TT pass at higher fidelity. Note: this still does **not** raise the storage precision of K-cache to fp32; it only avoids accumulator truncation within a chunk.
2. **Hybrid** — keep the high-seq portion of decode on HF (CPU fp32) and the steady-state on TT, which is what NewMind's `HF_DECODE=1` flag already does.
3. **Quantization-aware training** — distill the model to be robust under bf16 KV (the existing [ocr_lab/train_vision_qat.py](tt-ocr-lab/ocr_lab/train_vision_qat.py) and [ocr_lab/train_vision_distill.py](tt-ocr-lab/ocr_lab/train_vision_distill.py) sketches go in this direction).

## Verdict

**CONFIRMED.** Cited lines are exact and the contract is enforced at both op entry points. The "fp32 KV cache" approach is a hard architectural constraint, not a flag.

## Categorization

Category-A (loud).
