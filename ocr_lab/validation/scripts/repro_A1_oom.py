#!/usr/bin/env python3
"""Reproduce Category-A1: OOM allocating a multi-megabyte DRAM buffer.

The original NewMind report cited a 14_622_720 B buffer, which is consistent
with one decoder-layer KV slab at ``max_cache_seq ~= 28_560`` (bf16, n_kv=2,
head_dim=128). To trigger the same allocator path deterministically, we
allocate progressively larger DRAM buffers until ``BankManager::allocate_buffer``
returns the expected ``Out of Memory: Not enough space to allocate ...`` fatal.

Exit 0  -> assertion fired with the expected substring in the message.
Exit 1  -> no assertion or wrong message (UNEXPECTED).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import setup_paths, open_mesh, expect_tt_fatal, write_log

setup_paths()
import ttnn  # noqa: E402

EXPECTED = ["Out of Memory", "Not enough space to allocate", "DRAM"]


def run() -> dict:
    mesh = open_mesh((1, 1))

    def _attempt() -> None:
        # Fill DRAM with ~2 GiB slabs (no deallocation between them) until the
        # allocator refuses. Single-chip wormhole_b0 DRAM is ~12 GiB, so the
        # allocator should fail after 6-7 slabs.
        import torch

        held = []
        # Each slab is 65536 * 16384 bf16 = 2 GiB.
        slab = torch.zeros((1, 1, 65536, 16384), dtype=torch.bfloat16)
        for i in range(16):
            tt = ttnn.from_torch(
                slab,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            held.append(tt)
        # If nothing raised, free everything to avoid leaking.
        for tt in held:
            ttnn.deallocate(tt)

    res = expect_tt_fatal(_attempt, EXPECTED)
    ttnn.close_mesh_device(mesh)
    res["expected_substrings"] = EXPECTED
    res["script"] = "repro_A1_oom.py"
    res["citation"] = "tt-metal/tt_metal/impl/allocator/bank_manager.cpp:423-434"
    return res


if __name__ == "__main__":
    res = run()
    write_log("A1_oom", res)
    print(f"ok={res['ok']} raised={res['raised']} missing={res['missing']}")
    sys.exit(0 if res["ok"] else 1)
