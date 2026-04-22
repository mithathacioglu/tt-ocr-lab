#!/usr/bin/env bash
# Run all four Category-A reproductions in sequence.
# Each script is self-contained and exits 0 only if the cited TT_FATAL
# fires with the expected substring.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/ocr_lab/validation/scripts"
LOG_DIR="$ROOT/ocr_lab/validation/logs"
mkdir -p "$LOG_DIR"

# Pick python + TT env. Prefer an explicit PYTHON_BIN override; otherwise
# fall back to the tt-metal bundled python_env which has ttnn's runtime deps.
PY="${PYTHON_BIN:-/home/aroberge/tt-metal/python_env/bin/python}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export TT_METAL_SRC="${TT_METAL_SRC:-/home/aroberge/tt-metal}"
# Runtime root must be the install tree so the JIT finds kernel headers
# under `tt_metal/tt-llk/tt_llk_wormhole_b0/...`.
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-/home/aroberge/tt-metal/build_Release/libexec/tt-metalium}"
export TT_METAL_HOME="${TT_METAL_HOME:-$TT_METAL_RUNTIME_ROOT}"
export TT_METAL_LOGGER_LEVEL="${TT_METAL_LOGGER_LEVEL:-ERROR}"

declare -A RESULTS
overall=0
for s in repro_A1_oom.py repro_A2_mask_qdim.py repro_A3_chunk16.py repro_A4_fp32_kv.py; do
  echo "=== $s ==="
  if "$PY" -u "$SCRIPT_DIR/$s" 2>&1 | tee "$LOG_DIR/${s%.py}.log"; then
    RESULTS[$s]="PASS"
  else
    RESULTS[$s]="FAIL"
    overall=1
  fi
done

echo
echo "=== Summary ==="
for s in "${!RESULTS[@]}"; do
  printf "  %-25s %s\n" "$s" "${RESULTS[$s]}"
done
exit "$overall"
