#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/build/reproduction}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1 &&
      "${candidate}" -c \
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Python 3.10 or newer is required" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/predecessor"

"${PYTHON_BIN}" "${ROOT}/code/rebuild_submission_hpac.py" \
  --base-archive "${ROOT}/artifacts/base/int5_delta_archive.zip" \
  --hpac "${ROOT}/artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.bin.xz" \
  --tokens "${ROOT}/artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.tokens.bin" \
  --submission-dir "${OUTPUT_DIR}/predecessor" \
  --report "${OUTPUT_DIR}/predecessor.json"

PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT}/code/compress.sh" \
  --source-archive "${OUTPUT_DIR}/predecessor/archive.zip" \
  --output "${OUTPUT_DIR}/archive.zip" \
  --report "${OUTPUT_DIR}/cpr1.json"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/archive.zip" \
  "${ROOT}/artifacts/final/archive.zip" <<'PY'
import hashlib
import sys
from pathlib import Path

actual = Path(sys.argv[1]).read_bytes()
expected = Path(sys.argv[2]).read_bytes()
if actual != expected:
    raise SystemExit(
        "CPR1 rebuild differs from the committed canonical archive: "
        f"actual={hashlib.sha256(actual).hexdigest()} "
        f"expected={hashlib.sha256(expected).hexdigest()}"
    )
print(
    "CPR1 byte-identical:",
    len(actual),
    hashlib.sha256(actual).hexdigest(),
)
PY
