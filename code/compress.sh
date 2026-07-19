#!/usr/bin/env bash
# Reproduce the exact CPR1 archive from the frozen predecessor archive.
#
# This is the final lossless compression stage used by the submitted artifact.
# It is not the multi-day model-training pipeline. The predecessor archive is a
# compression-side input and is never used by inflate.sh.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_URL="https://github.com/fesalfayed/comma_video_compression_challenge/releases/download/semantic-pose-landslide-selfcompress/archive.zip"
SOURCE_SHA256="f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451"
SOURCE_BYTES="194380"
OUTPUT_SHA256="0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
OUTPUT_BYTES="191052"

SOURCE_ARCHIVE=""
OUTPUT_ARCHIVE="${HERE}/archive.zip"
REPORT_PATH=""
TEMP_DIR=""
OUTPUT_TMP=""
REPORT_TMP=""

usage() {
  cat <<'USAGE'
Usage: compress.sh [--source-archive PATH] [--output PATH] [--report PATH]

Reproduces the exact submitted CPR1 archive. If --source-archive is omitted,
the frozen 194,380-byte predecessor archive is downloaded and SHA-256 checked.

Options:
  --source-archive PATH  Use a local frozen predecessor archive.
  --output PATH          Output archive (default: <submission-dir>/archive.zip).
  --report PATH          Optional JSON report from the CPR1 repacker.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-archive)
      [[ $# -ge 2 ]] || { echo "ERROR: --source-archive requires a path" >&2; exit 2; }
      SOURCE_ARCHIVE="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "ERROR: --output requires a path" >&2; exit 2; }
      OUTPUT_ARCHIVE="$2"
      shift 2
      ;;
    --report)
      [[ $# -ge 2 ]] || { echo "ERROR: --report requires a path" >&2; exit 2; }
      REPORT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

python_is_supported() {
  command -v "$1" >/dev/null 2>&1 &&
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
}

PYTHON=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if python_is_supported "$PYTHON_BIN"; then
    PYTHON="$PYTHON_BIN"
  fi
else
  for candidate in python python3; do
    if python_is_supported "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3.10 or newer is required" >&2
  exit 2
fi

cleanup() {
  if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
  fi
  if [[ -n "$OUTPUT_TMP" ]]; then
    rm -f "$OUTPUT_TMP"
  fi
  if [[ -n "$REPORT_TMP" ]]; then
    rm -f "$REPORT_TMP"
  fi
}
trap cleanup EXIT

if [[ -z "$SOURCE_ARCHIVE" ]]; then
  command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required when --source-archive is omitted" >&2
    exit 127
  }
  TEMP_DIR="$(mktemp -d)"
  SOURCE_ARCHIVE="${TEMP_DIR}/source-archive-194380.zip"
  curl --fail --location --retry 3 --output "$SOURCE_ARCHIVE" "$SOURCE_URL"
fi

PATH_ARGS=("$SOURCE_ARCHIVE" "$OUTPUT_ARCHIVE")
if [[ -n "$REPORT_PATH" ]]; then
  PATH_ARGS+=("$REPORT_PATH")
fi
"$PYTHON" - "${PATH_ARGS[@]}" <<'PY'
import sys
import unicodedata
from pathlib import Path

resolved = [Path(value).resolve() for value in sys.argv[1:]]
path_keys = [
    unicodedata.normalize("NFC", str(path)).casefold()
    for path in resolved
]
if len(path_keys) != len(set(path_keys)):
    raise SystemExit(
        "ERROR: source archive, output archive, and report must use distinct paths"
    )
if len(resolved) == 3:
    output_parts = tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in resolved[1].parts
    )
    report_parts = tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in resolved[2].parts
    )
    shared = min(len(output_parts), len(report_parts))
    if output_parts[:shared] == report_parts[:shared]:
        raise SystemExit(
            "ERROR: output archive and report paths must not contain one another"
        )
for label, path in zip(("source", "output", "report"), resolved):
    if label != "source" and path.is_dir():
        raise SystemExit(f"ERROR: {label} path is a directory: {path}")
PY

verify_file() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
expected_bytes = int(sys.argv[3])
blob = path.read_bytes()
actual_sha256 = hashlib.sha256(blob).hexdigest()
if len(blob) != expected_bytes:
    raise SystemExit(
        f"size mismatch for {path}: {len(blob)} != {expected_bytes}"
    )
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
    )
print(f"verified {path}: {len(blob)} bytes, sha256={actual_sha256}")
PY
}

atomic_replace() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
}

verify_file "$SOURCE_ARCHIVE" "$SOURCE_SHA256" "$SOURCE_BYTES"

OUTPUT_DIR="$(dirname -- "$OUTPUT_ARCHIVE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_TMP="$(mktemp "${OUTPUT_DIR}/.cpr1-archive.XXXXXX")"
REPACK_ARGS=("$SOURCE_ARCHIVE" "$OUTPUT_TMP")
if [[ -n "$REPORT_PATH" ]]; then
  REPORT_DIR="$(dirname -- "$REPORT_PATH")"
  mkdir -p "$REPORT_DIR"
  REPORT_TMP="$(mktemp "${REPORT_DIR}/.cpr1-report.XXXXXX")"
  REPACK_ARGS+=(--report "$REPORT_TMP")
fi
"$PYTHON" "$HERE/repack_carrier.py" "${REPACK_ARGS[@]}"

verify_file "$OUTPUT_TMP" "$OUTPUT_SHA256" "$OUTPUT_BYTES"
if [[ -n "$REPORT_PATH" ]]; then
  atomic_replace "$REPORT_TMP" "$REPORT_PATH"
  REPORT_TMP=""
fi
atomic_replace "$OUTPUT_TMP" "$OUTPUT_ARCHIVE"
OUTPUT_TMP=""
echo "CPR1 archive reproduced exactly: $OUTPUT_ARCHIVE"
