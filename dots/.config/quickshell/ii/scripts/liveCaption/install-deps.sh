#!/usr/bin/env bash
set -euo pipefail

VENV_RAW="${ILLOGICAL_IMPULSE_VIRTUAL_ENV:-${XDG_STATE_HOME:-$HOME/.local/state}/quickshell/.venv}"
VENV_DIR="$(eval echo "$VENV_RAW")"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[liveCaption] ERROR: Quickshell venv not found at ${VENV_DIR}" >&2
  echo "[liveCaption] Set ILLOGICAL_IMPULSE_VIRTUAL_ENV or re-run the install to create the venv." >&2
  exit 1
fi

echo "[liveCaption] Using venv: ${VENV_DIR}" >&2

source "${VENV_DIR}/bin/activate"

if command -v uv >/dev/null 2>&1; then
  echo "[liveCaption] Installing python deps via uv..." >&2
  uv pip install --upgrade sherpa-onnx numpy
else
  echo "[liveCaption] Installing python deps via pip..." >&2
  python -m pip install --upgrade pip
  python -m pip install --upgrade sherpa-onnx numpy
fi

python - <<'PY'
import sherpa_onnx
print(getattr(sherpa_onnx, "__version__", "unknown"))
PY

