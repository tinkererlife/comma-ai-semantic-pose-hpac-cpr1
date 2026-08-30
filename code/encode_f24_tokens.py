#!/usr/bin/env python3
"""Teacher-force arbitrary token grids through #135's exact F24S RC64 rail."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_renderer(code_dir: Path):
    path = code_dir / "inflate.py"
    spec = importlib.util.spec_from_file_location("_f24_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(code_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def compile_backend(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(output)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-archive-stream", action="store_true")
    args = parser.parse_args()

    root = args.submission_root.resolve()
    sys.path.insert(0, str(root))
    from rc64 import NativeEncoder
    from runtime.hpac_inference import (
        configure_cuda_reproducibility,
        optimize_sparse_evaluator,
    )
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import (
        NUM_CLASSES,
        _boundary_buckets,
        _probability_table,
        _sparse_class,
        read_residual_archive,
    )

    renderer = load_renderer(root / "cpr1")
    expected = renderer.N * renderer.EVAL_H * renderer.EVAL_W
    raw = args.tokens.read_bytes()
    if len(raw) != expected:
        raise ValueError(f"token grid has {len(raw)} bytes; expected {expected}")
    tokens = np.frombuffer(raw, dtype=np.uint8).reshape(
        renderer.N, renderer.EVAL_H, renderer.EVAL_W
    )
    if not args.library.is_file():
        compile_backend(Path(__file__).with_name("rc64_backend.c"), args.library)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("F24S token encoding requires CUDA")
    configure_cuda_reproducibility()
    parts = read_residual_archive(args.archive)
    base_hpac = materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = _sparse_class(root / "cpr1")(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    optimize_sparse_evaluator(sparse)
    encoder = NativeEncoder(args.library)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append((
            torch.from_numpy(flat_positions).to(device), flat_positions
        ))

    corrected_digest = hashlib.sha256()
    probability_digest = hashlib.sha256()
    started = time.time()
    with torch.inference_mode():
        previous = torch.zeros(
            (1, renderer.EVAL_H, renderer.EVAL_W),
            dtype=torch.long,
            device=device,
        )
        for frame in range(renderer.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(
                    device="cpu", dtype=torch.uint8
                ).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(
                    renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8
                )
            target = torch.from_numpy(tokens[frame].astype(np.int64)).to(device)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = _probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                probability_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                symbols = np.ascontiguousarray(
                    tokens[frame].reshape(-1)[flat_positions], dtype=np.int32
                )
                encoder.encode(symbols, probability)
                current.reshape(-1)[device_positions] = target.reshape(-1)[
                    device_positions
                ]
            previous = current
            if frame == 0 or (frame + 1) % 25 == 0:
                print(json.dumps({
                    "encoded_frames": frame + 1,
                    "elapsed_seconds": round(time.time() - started, 1),
                }), flush=True)

    stream = encoder.finish()
    matches_archive = stream == parts.token_stream
    if args.require_archive_stream and not matches_archive:
        raise RuntimeError("re-encoded stream differs from the archive stream")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(stream)
    result = {
        "decoded_token_sha256": hashlib.sha256(raw).hexdigest(),
        "stream_bytes": len(stream),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "matches_archive_stream": matches_archive,
        "corrected_logit_sha256": corrected_digest.hexdigest(),
        "probability_sha256": probability_digest.hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
