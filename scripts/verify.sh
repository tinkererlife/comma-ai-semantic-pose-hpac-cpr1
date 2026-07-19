#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/cpr1-verify.XXXXXX")"

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

cleanup() {
  rm -rf "${SCRATCH}"
}
trap cleanup EXIT

cd "${ROOT}"
"${PYTHON_BIN}" scripts/audit_repo.py
PYTHONPYCACHEPREFIX="${SCRATCH}/pycache" \
  "${PYTHON_BIN}" -m compileall -q code scripts tests
bash scripts/reproduce.sh "${SCRATCH}/reproduction"

SEMANTIC_POSE_SOURCE_ARCHIVE="${SCRATCH}/reproduction/predecessor/archive.zip" \
  "${PYTHON_BIN}" -m pytest -q code/test_carrier_codec.py tests/test_provenance.py

echo "CPR1 repository verification passed"
