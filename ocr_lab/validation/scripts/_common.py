"""Shared bootstrap for validation scripts.

Each script imports ``setup_paths`` and ``open_mesh`` from here. The path
contract mirrors the existing runners under ``ocr_lab/`` so that ``ttnn`` and
the ``tt-metal`` Python packages resolve correctly when running under the
``.venv-tt-xla`` virtualenv activated by ``scripts/activate_tt_xla.sh``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CANDIDATE_TT_METAL_SRC_PATHS = [
    os.environ.get("TT_METAL_SRC"),
    "/home/aroberge/tt-metal",
    str(PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"),
]
TT_METAL_SRC = next(
    (Path(p) for p in _CANDIDATE_TT_METAL_SRC_PATHS if p and Path(p, "ttnn/ttnn/__init__.py").exists()),
    Path("/home/aroberge/tt-metal"),
)

_CANDIDATE_TT_METAL_RUNTIME_ROOTS = [
    os.environ.get("TT_METAL_RUNTIME_ROOT"),
    str(TT_METAL_SRC / "build_Release/libexec/tt-metalium"),
    str(TT_METAL_SRC),
]
TT_METAL_RUNTIME_ROOT = next(
    (
        Path(p)
        for p in _CANDIDATE_TT_METAL_RUNTIME_ROOTS
        if p
        and Path(p, "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_structs.h").exists()
    ),
    Path(str(TT_METAL_SRC / "build_Release/libexec/tt-metalium")),
)

TT_METAL = TT_METAL_SRC


def setup_paths() -> None:
    """Mirror sys.path / env setup used by ocr_lab/run_*.py scripts."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
    os.environ.setdefault("ARCH_NAME", "wormhole_b0")
    os.environ.setdefault("TT_METAL_HOME", str(TT_METAL_RUNTIME_ROOT))
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", str(TT_METAL_RUNTIME_ROOT))

    if str(TT_METAL_SRC) not in sys.path:
        sys.path.insert(0, str(TT_METAL_SRC))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    shims = PROJECT_ROOT / "ocr_lab" / "shims"
    if str(shims) not in sys.path:
        sys.path.insert(0, str(shims))


def open_mesh(shape: tuple[int, int] = (1, 1)):
    """Open a mesh device using the given shape, returning the mesh handle."""
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    mesh.enable_program_cache()
    return mesh


def expect_tt_fatal(callable_, expected_substrings: list[str]) -> dict[str, Any]:
    """Run ``callable_``; expect it to raise with ALL of ``expected_substrings`` in the message.

    Returns a result dict suitable for JSON dump:
        ok           - True if the expected fatal fired with all substrings
        raised       - True if any RuntimeError fired
        message      - the captured error message (or '' on success)
        missing      - substrings that were expected but not found
        elapsed_ms   - wallclock ms
    """
    t0 = time.perf_counter()
    raised = False
    message = ""
    try:
        callable_()
    except RuntimeError as exc:
        raised = True
        message = str(exc)
    except Exception as exc:  # noqa: BLE001 - capture for diagnosis
        raised = True
        message = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    missing = [s for s in expected_substrings if s not in message]
    return {
        "ok": raised and not missing,
        "raised": raised,
        "message": message,
        "missing": missing,
        "elapsed_ms": elapsed_ms,
    }


def write_log(name: str, payload: dict[str, Any]) -> Path:
    import json

    log_dir = PROJECT_ROOT / "ocr_lab" / "validation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out
