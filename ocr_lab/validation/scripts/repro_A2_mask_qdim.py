#!/usr/bin/env python3
"""Reproduce Category-A2: prefill SDPA rejects a mask whose logical seq dim
does not match Q's logical seq dim.

We construct Q with logical shape (1, 12, 1, 128) and a mask with logical
shape (1, 1, 1, S). The op fires:
    Mask sequence length must match Q sequence length
    (sdpa_device_operation.cpp:92)

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

EXPECTED = ["Mask sequence length must match Q sequence length"]


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
        S = 64
        # Q logical (1, 12, 1, 128) - prefill SDPA wants seq in dim 2.
        q = _to_tt(torch.randn(1, 12, 1, 128).to(torch.bfloat16), mesh)
        k = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh)
        v = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh)
        # Mismatched mask: logical seq dim is 1 instead of 1 (would actually match)
        # so we use 32 to provoke the assertion - mask logical (1,1,32,S), Q logical seq=1
        mask = _to_tt(torch.zeros(1, 1, 32, S).to(torch.bfloat16), mesh)
        ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False
        )

    res = expect_tt_fatal(_attempt, EXPECTED)
    ttnn.close_mesh_device(mesh)
    res["expected_substrings"] = EXPECTED
    res["script"] = "repro_A2_mask_qdim.py"
    res["citation"] = (
        "tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/"
        "sdpa_device_operation.cpp:92"
    )
    return res


if __name__ == "__main__":
    res = run()
    write_log("A2_mask_qdim", res)
    print(f"ok={res['ok']} raised={res['raised']} missing={res['missing']}")
    sys.exit(0 if res["ok"] else 1)
