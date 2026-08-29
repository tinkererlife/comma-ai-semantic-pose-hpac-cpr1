#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 CHALLENGE_ROOT WORK_DIR" >&2
  exit 2
fi

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CHALLENGE_ROOT="$(cd "$1" && pwd)"
readonly WORK_DIR="$2"
readonly ARCHIVE="${ROOT}/artifacts/final/archive.zip"
readonly EXPECTED_COMMIT="d3f688f84f555c5aaebee7d2c4203efc8a9051e2"
readonly EXPECTED_ARCHIVE_SHA256="0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"

if [[ "$(git -C "${CHALLENGE_ROOT}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "golden preflight requires challenge commit ${EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${CHALLENGE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "golden preflight requires a clean pinned challenge checkout" >&2
  exit 2
fi

readonly PYTHON="${CHALLENGE_ROOT}/.venv/bin/python"
"${PYTHON}" - <<'PY'
import sys
import torch
import nvidia.dali
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.cuda.is_available(), "CUDA is required"
print(sys.version)
print(torch.__version__, torch.cuda.get_device_name(0))
PY

mkdir -p "${WORK_DIR}"
"${PYTHON}" "${ROOT}/code/stage_submission.py" \
  --archive "${ARCHIVE}" \
  --code-root "${ROOT}/code" \
  --submission-dir "${WORK_DIR}/submission" \
  --report "${WORK_DIR}/stage-report.json"

PATH="${CHALLENGE_ROOT}/.venv/bin:${PATH}" \
  bash "${CHALLENGE_ROOT}/evaluate.sh" \
    --submission-dir "${WORK_DIR}/submission" \
    --device cuda

"${PYTHON}" "${ROOT}/code/verify_official_report.py" \
  --report "${WORK_DIR}/submission/report.txt" \
  --archive "${WORK_DIR}/submission/archive.zip" \
  --out "${WORK_DIR}/official-metrics.json" \
  --expected-archive-sha256 "${EXPECTED_ARCHIVE_SHA256}" \
  --expected-pose 0.00002331 \
  --expected-seg 0.00029660 \
  --expected-score 0.17214129749189644 \
  --pose-atol 0.00001 \
  --seg-atol 0.00001 \
  --score-atol 0.01

echo "GOLDEN_ARCHIVE_PASS"
