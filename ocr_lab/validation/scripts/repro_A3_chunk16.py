#!/usr/bin/env python3
"""Reproduce Category-A3: prefill SDPA rejects k_chunk_size that is not a
multiple of TILE_SIZE (32).

Triggers:
    k_chunk_size must be divisible by TILE_SIZE
    (sdpa_device_operation.cpp:152-156)

Exit 0 -> assertion fired with the expected substring.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import setup_paths, open_mesh, expect_tt_fatal, write_log

setup_paths()
import torch  # noqa: E402
import ttnn  # noqa: E402
from ttnn.operations.transformer import SDPAProgramConfig  # noqa: E402

EXPECTED = ["k_chunk_size must be divisible by TILE_SIZE"]


def _to_tt(tensor: torch.Tensor, mesh, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run() -> dict:
    mesh = open_mesh((1, 1))

    def _attempt() -> None:
        # Q/K/V must share seq len for causal SDPA.
        S = 128
        q = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh)
        k = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh)
        v = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh)
        # k_chunk_size = 16 (sub-tile) -> rejected
        prog = SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=16,
            exp_approx_mode=True,
        )
        ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, program_config=prog
        )

    res = expect_tt_fatal(_attempt, EXPECTED)
    ttnn.close_mesh_device(mesh)
    res["expected_substrings"] = EXPECTED
    res["script"] = "repro_A3_chunk16.py"
    res["citation"] = (
        "tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/"
        "sdpa_device_operation.cpp:152-156"
    )
    return res


if __name__ == "__main__":
    res = run()
    write_log("A3_chunk16", res)
    print(f"ok={res['ok']} raised={res['raised']} missing={res['missing']}")
    sys.exit(0 if res["ok"] else 1)
