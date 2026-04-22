#!/usr/bin/env python3
"""Reproduce Category-A4: SDPA rejects fp32 inputs.

Triggers:
    Data type of input tensor must be BFLOAT16, BFLOAT8_B, or BFLOAT4_B
    (sdpa_device_operation.cpp:39-43)

We pass a fp32 K to the prefill SDPA op and confirm the dtype check fires
before any compute runs. This proves the "fp32 KV cache" mitigation NewMind
proposed cannot be expressed at the device-op surface.

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

EXPECTED = ["BFLOAT16", "BFLOAT8_B", "BFLOAT4_B"]


def _to_tt(tensor: torch.Tensor, mesh, dtype):
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
        q = _to_tt(torch.randn(1, 12, 32, 128).to(torch.bfloat16), mesh, ttnn.bfloat16)
        # K as fp32 - rejected
        k = _to_tt(torch.randn(1, 12, S, 128), mesh, ttnn.float32)
        v = _to_tt(torch.randn(1, 12, S, 128).to(torch.bfloat16), mesh, ttnn.bfloat16)
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)

    res = expect_tt_fatal(_attempt, EXPECTED)
    ttnn.close_mesh_device(mesh)
    res["expected_substrings"] = EXPECTED
    res["script"] = "repro_A4_fp32_kv.py"
    res["citation"] = (
        "tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/"
        "sdpa_device_operation.cpp:39-43"
    )
    return res


if __name__ == "__main__":
    res = run()
    write_log("A4_fp32_kv", res)
    print(f"ok={res['ok']} raised={res['raised']} missing={res['missing']}")
    sys.exit(0 if res["ok"] else 1)
