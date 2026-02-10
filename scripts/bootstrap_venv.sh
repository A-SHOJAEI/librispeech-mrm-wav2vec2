#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
CACHE_DIR=".cache"
GET_PIP_PY="${CACHE_DIR}/get-pip.py"

mkdir -p "${CACHE_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[setup] creating venv at ${VENV_DIR} (without pip)"
  python3 -m venv --without-pip "${VENV_DIR}"
fi

if [[ ! -f "${GET_PIP_PY}" ]]; then
  echo "[setup] downloading get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${GET_PIP_PY}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${GET_PIP_PY}" https://bootstrap.pypa.io/get-pip.py
  else
    echo "ERROR: need curl or wget to download get-pip.py" >&2
    exit 1
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/pip" ]]; then
  echo "[setup] bootstrapping pip into venv"
  "${VENV_DIR}/bin/python" "${GET_PIP_PY}" --no-warn-script-location
fi

# Keep tooling modern enough to handle wheels; do not touch the system interpreter.
"${VENV_DIR}/bin/pip" install --upgrade "pip>=23.3" "setuptools>=68" "wheel>=0.41"

# Install repo dependencies and the package itself (editable).
"${VENV_DIR}/bin/pip" install -r requirements.txt
"${VENV_DIR}/bin/pip" install -e .

echo "[setup] done"
