#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-sherpa-onnx-streaming-zipformer-en-2023-06-26}"
DEST_BASE="${2:-${XDG_DATA_HOME:-$HOME/.local/share}/sherpa-onnx/models}"

URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"
DEST_DIR="${DEST_BASE}/${MODEL_NAME}"

mkdir -p "${DEST_BASE}"

if [[ -d "${DEST_DIR}" ]]; then
  echo "${DEST_DIR}"
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "[liveCaption] Downloading ${URL}" >&2
curl -L --fail --retry 3 --retry-delay 1 --silent --show-error -o "${tmpdir}/model.tar.bz2" "${URL}"

echo "[liveCaption] Extracting to ${DEST_BASE}" >&2
tar -xjf "${tmpdir}/model.tar.bz2" -C "${DEST_BASE}"

if [[ ! -d "${DEST_DIR}" ]]; then
  echo "[liveCaption] WARNING: expected ${DEST_DIR} to exist after extraction." >&2
fi

echo "${DEST_DIR}"
