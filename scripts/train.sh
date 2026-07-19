#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="${ROOT}/code"
WORK_DIR="${WORK_DIR:-${ROOT}/work}"
DEVICE="${DEVICE:-cuda}"
ACTION="${1:-}"

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

usage() {
  cat <<'USAGE'
Usage: CHALLENGE_ROOT=/path/to/challenge bash scripts/train.sh ACTION

Actions:
  prepare        Restore and verify the two exact training caches
  semantic       Replay the selected 6,000-step semantic QAT tail
  carrier        Replay the selected 4,000-step int6 carrier tail
  hpac-init      Extract the exact pre-self-compression HPAC initialization
  hpac           Replay the selected 60-epoch HPAC self-compression run
  pack-hpac      Pack the canonical or HPAC_CHECKPOINT checkpoint
  encode-tokens  Encode all 600 maps from the canonical or HPAC_CHECKPOINT checkpoint
  all            Run semantic, carrier, hpac, pack-hpac, and encode-tokens
USAGE
}

if [[ -z "${ACTION}" ]]; then
  usage
  exit 2
fi

case "${ACTION}" in
  semantic|carrier|all)
    if [[ -z "${CHALLENGE_ROOT:-}" ]]; then
      echo "CHALLENGE_ROOT is required for ${ACTION}" >&2
      exit 2
    fi
    ;;
esac

mkdir -p "${WORK_DIR}/caches" "${WORK_DIR}/checkpoints" "${WORK_DIR}/reports" \
  "${WORK_DIR}/artifacts"

verify_file() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_hash = sys.argv[2]
expected_bytes = int(sys.argv[3])
blob = path.read_bytes()
actual_hash = hashlib.sha256(blob).hexdigest()
if len(blob) != expected_bytes or actual_hash != expected_hash:
    raise SystemExit(
        f"verification failed for {path}: "
        f"{len(blob)} bytes {actual_hash}"
    )
print(f"verified {path}: {len(blob)} bytes {actual_hash}")
PY
}

prepare() {
  xz -dc "${ROOT}/artifacts/caches/gt_cache_600.pt.xz" \
    > "${WORK_DIR}/caches/gt_cache_600.pt"
  xz -dc "${ROOT}/artifacts/caches/gt_cache_600_official_ada.pt.xz" \
    > "${WORK_DIR}/caches/gt_cache_600_official_ada.pt"
  verify_file \
    "${WORK_DIR}/caches/gt_cache_600.pt" \
    "8248a60da56119eb4b3ad76bfa32f5498dee849eaf4b83b304275064141fd828" \
    117981133
  verify_file \
    "${WORK_DIR}/caches/gt_cache_600_official_ada.pt" \
    "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195" \
    117981301
}

ensure_caches() {
  if [[ ! -f "${WORK_DIR}/caches/gt_cache_600.pt" ||
        ! -f "${WORK_DIR}/caches/gt_cache_600_official_ada.pt" ]]; then
    prepare
  fi
}

semantic() {
  ensure_caches
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/train_semantic_quantized.py" \
    --challenge-root "${CHALLENGE_ROOT}" \
    --cache "${WORK_DIR}/caches/gt_cache_600.pt" \
    --init "${ROOT}/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt" \
    --bits 4 \
    --steps 6000 \
    --batch-size 2 \
    --eval-batch-size 8 \
    --eval-every 250 \
    --lr 2e-7 \
    --ce-fraction 0.0 \
    --softplus-fraction -999.0 \
    --seed 20260716 \
    --device "${DEVICE}" \
    --out "${WORK_DIR}/reports/semantic_tail6k.json" \
    --save "${WORK_DIR}/checkpoints/semantic_tail6k.pt"
}

carrier() {
  ensure_caches
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/train_pose_carrier_full.py" \
    --challenge-root "${CHALLENGE_ROOT}" \
    --target-cache "${WORK_DIR}/caches/gt_cache_600_official_ada.pt" \
    --master-checkpoint \
      "${ROOT}/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt" \
    --init-carrier \
      "${ROOT}/artifacts/checkpoints/archive_carrier_int6_stable_s8k.pt" \
    --master-cache "${WORK_DIR}/caches/archive_master_exact.pt" \
    --reuse-master-cache \
    --cache-masters-on-device \
    --steps 4000 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --render-batch-size 4 \
    --eval-every 250 \
    --lr-basis 1e-6 \
    --lr-coeff 3e-4 \
    --basis-freeze-fraction 1.0 \
    --basis-train-until-fraction 0.0 \
    --qat-fraction 0.0 \
    --coeff-qat-fraction 0.0 \
    --always-metric-loss \
    --metric-normalized-weight 0.0 \
    --hard-mining-power 0.75 \
    --hard-mining-max 6.0 \
    --basis-bits 6 \
    --coeff-bits 12 \
    --amplitude 64.0 \
    --master-carrier-amplitude 0.0 \
    --carrier-base gray \
    --seed 20260722 \
    --device "${DEVICE}" \
    --out "${WORK_DIR}/reports/carrier_int6_coefftail_s4k.json" \
    --save "${WORK_DIR}/checkpoints/carrier_int6_coefftail_s4k.pt"
}

hpac_init() {
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/extract_integer_hpac_archive.py" \
    --archive "${ROOT}/artifacts/base/int5_delta_archive.zip" \
    --channels 64 \
    --patch 64 \
    --delta 2 \
    --frame-dim 8 \
    --out "${WORK_DIR}/checkpoints/hpac_p64_exact_from_archive.pt"
}

hpac() {
  ensure_caches
  hpac_init
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/train_hpac_self_compress.py" \
    --cache "${WORK_DIR}/caches/gt_cache_600_official_ada.pt" \
    --init "${WORK_DIR}/checkpoints/hpac_p64_exact_from_archive.pt" \
    --epochs 60 \
    --batch-size 8 \
    --eval-batch-size 4 \
    --eval-every 2 \
    --lr 0.003 \
    --lr-exponent 0.0002 \
    --lr-bits 0.01 \
    --bit-eps 1e-6 \
    --rate-lambda 1.0 \
    --qat-fraction 0.5 \
    --init-bits 8.0 \
    --channels 64 \
    --patch 64 \
    --delta 2 \
    --frame-dim 8 \
    --norm-mode none \
    --activation relu \
    --frame-scale \
    --weight-bound 127 \
    --activation-bound 127 \
    --weight-scales \
    --weight-exponent-min -6 \
    --spm \
    --target-mode raw \
    --seed 20260716 \
    --device "${DEVICE}" \
    --save "${WORK_DIR}/checkpoints/hpac_selfcompress_e60.pt" \
    --out "${WORK_DIR}/reports/hpac_selfcompress_e60.json"
}

canonical_hpac_checkpoint() {
  if [[ -n "${HPAC_CHECKPOINT:-}" ]]; then
    printf '%s\n' "${HPAC_CHECKPOINT}"
  else
    printf '%s\n' \
      "${ROOT}/artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt"
  fi
}

pack_hpac() {
  local checkpoint
  checkpoint="$(canonical_hpac_checkpoint)"
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/pack_hpac_self_compress.py" \
    --checkpoint "${checkpoint}" \
    --channels 64 \
    --patch 64 \
    --delta 2 \
    --frame-dim 8 \
    --weight-bound 127 \
    --activation-bound 127 \
    --weight-exponent-min -6 \
    --device "${PACK_DEVICE:-cpu}" \
    --blob "${WORK_DIR}/artifacts/hpac.bin.xz" \
    --report "${WORK_DIR}/reports/hpac.pack.json"
  if [[ -z "${HPAC_CHECKPOINT:-}" ]]; then
    verify_file \
      "${WORK_DIR}/artifacts/hpac.bin.xz" \
      "ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee" \
      15164
  fi
}

encode_tokens() {
  ensure_caches
  local checkpoint
  checkpoint="$(canonical_hpac_checkpoint)"
  PYTHONPATH="${CODE}" "${PYTHON_BIN}" "${CODE}/codec_hpac_integer.py" \
    --checkpoint "${checkpoint}" \
    --cache "${WORK_DIR}/caches/gt_cache_600_official_ada.pt" \
    --channels 64 \
    --patch 64 \
    --delta 2 \
    --frame-dim 8 \
    --norm-mode none \
    --activation relu \
    --frame-scale \
    --weight-bound 127 \
    --activation-bound 127 \
    --weight-scales \
    --weight-exponent-min -6 \
    --spm \
    --sparse \
    --self-compress \
    --target-mode raw \
    --frames 600 \
    --device "${DEVICE}" \
    --tokens-out "${WORK_DIR}/artifacts/tokens.bin" \
    --report "${WORK_DIR}/reports/tokens.codec.json"
  if [[ -z "${HPAC_CHECKPOINT:-}" ]]; then
    verify_file \
      "${WORK_DIR}/artifacts/tokens.bin" \
      "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb" \
      116980
  fi
}

case "${ACTION}" in
  prepare) prepare ;;
  semantic) semantic ;;
  carrier) carrier ;;
  hpac-init) hpac_init ;;
  hpac) hpac ;;
  pack-hpac) pack_hpac ;;
  encode-tokens) encode_tokens ;;
  all)
    semantic
    carrier
    hpac
    pack_hpac
    encode_tokens
    ;;
  *)
    usage
    exit 2
    ;;
esac
