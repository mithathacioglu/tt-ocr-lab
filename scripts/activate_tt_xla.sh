#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-tt-xla"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing venv at ${VENV_DIR}. Run scripts/setup_tt_xla_env.sh first." >&2
  exit 1
fi

if [ -x "${HOME}/.local/bin/uv" ]; then
  PYTHON312_BIN="$("${HOME}/.local/bin/uv" python find 3.12)"
else
  PYTHON312_BIN="$(command -v python3.12)"
fi

if [ -z "${PYTHON312_BIN}" ]; then
  echo "Python 3.12 not found. Run scripts/setup_tt_xla_env.sh first." >&2
  exit 1
fi

UV_PYTHON_LIB="$(dirname "$(dirname "$(readlink -f "${PYTHON312_BIN}")")")/lib"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

TORCH_LIB="$(python - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

export LD_LIBRARY_PATH="${UV_PYTHON_LIB}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export TT_XLA_LOCAL_ENV=1

echo "Activated ${VENV_DIR}"
echo "LD_LIBRARY_PATH now includes:"
echo "  ${UV_PYTHON_LIB}"
echo "  ${TORCH_LIB}"
