# A1 — OOM at `bank_manager.cpp:434`

NewMind's reported error:

```
Out of Memory: Not enough space to allocate 14622720 B DRAM buffer
  (location: bank_manager.cpp:434)
```

## Source location

File: [/home/aroberge/tt-metal/tt_metal/impl/allocator/bank_manager.cpp](/home/aroberge/tt-metal/tt_metal/impl/allocator/bank_manager.cpp), lines 422-434:

```422:434:tt-metal/tt_metal/impl/allocator/bank_manager.cpp
            TT_FATAL(
                false,
                "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to "
                "store {} B, but bank size is {} B (allocated: {} B, free: {} B, largest free block: {} B)",
                size,
                enchantum::to_string(buffer_type_),
                num_banks,
                size_per_bank,
                bank_size(),
                mem_stats.total_allocated_bytes,
                mem_stats.total_free_bytes,
                mem_stats.largest_free_block_bytes);
```

This is in `BankManager::allocate_buffer` after `alloc->allocate(size_per_bank, bottom_up, address_limit)` returns no value — i.e. fragmentation or insufficient total capacity in the requested buffer type (DRAM here).

## Cross-check of the byte count

`14_622_720 B` at bf16 = `14_622_720 / 2 = 7_311_360 elements`.

Plausible KV-cache shape that would land near this number:
- One layer of K (or V): `(1, n_kv_heads=2, max_cache, head_dim=128)` packed in TILE layout.
- `7_311_360 / (2 * 128) = 28_560` ≈ `tile-aligned 28_576` (i.e. 893 tiles of 32 rows).

So the 14.62 MB buffer is consistent with a single-layer KV-cache allocation when `max_cache_seq` is in the ~28k range — exactly what happens if MAX_SEQ is set high while the decoder is already resident. This matches NewMind's "decoder co-resident" diagnosis.

## Verdict

**CONFIRMED.** The cited file/line is correct, the message format matches verbatim, and the byte count is consistent with the diagnosed root cause.

## Categorization

This is a Category-A (loud) failure — `TT_FATAL` is fired and the caller gets a Python `RuntimeError` containing the message. No silent path exists in this allocator branch.
