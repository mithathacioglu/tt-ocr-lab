from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path(os.environ.get("DOTS_TT_CACHE_ROOT", PROJECT_ROOT / "ocr_lab" / ".tt_xla_cache"))


def default_cache_dir(cache_name: str, device_index: str | int) -> Path:
    return DEFAULT_CACHE_ROOT / cache_name / f"device_{device_index}"


def configure_tt_runtime(
    *,
    cache_dir: str | Path | None,
    optimization_level: int = 2,
    enable_trace: bool = True,
    enable_program_cache: bool = True,
    trace_region_size: str = "50000000",
) -> dict:
    if enable_program_cache:
        os.environ.setdefault("TT_RUNTIME_ENABLE_PROGRAM_CACHE", "1")
    if enable_trace:
        current = os.environ.get("TT_RUNTIME_TRACE_REGION_SIZE")
        if current is None:
            os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = trace_region_size
        else:
            try:
                if int(current) < int(trace_region_size):
                    os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = trace_region_size
            except ValueError:
                os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = trace_region_size

    import torch
    import torch_xla
    import torch_xla.runtime as xr

    try:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
    except Exception:
        pass

    compile_options = {"optimization_level": str(optimization_level)}
    if enable_trace:
        compile_options["enable_trace"] = "true"
    torch_xla.set_custom_compile_options(compile_options)

    cache_dir_str = None
    if cache_dir:
        cache_dir_path = Path(cache_dir).resolve()
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_dir_str = str(cache_dir_path)
        try:
            xr.initialize_cache(cache_dir_str, readonly=False)
        except TypeError:
            xr.initialize_cache(cache_dir_str)

    return {
        "cache_dir": cache_dir_str,
        "compile_options": compile_options,
        "program_cache_enabled": enable_program_cache,
        "trace_enabled": enable_trace,
        "trace_region_size": trace_region_size if enable_trace else None,
    }
