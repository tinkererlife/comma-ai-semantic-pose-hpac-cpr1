# CPR1 winning recipe

This private preservation repository isolates the semantic-renderer, pose-carrier,
integer-HPAC, and CPR1 code path that produced the
`semantic_pose_landslide_selfcompress` submission.

The canonical charged artifact is:

```text
bytes:   191052
sha256:  0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
member:  p
```

## What is exact

`scripts/reproduce.sh` starts from the included 197,228-byte int5 semantic/pose
archive, installs the final packed HPAC model and exact token stream, and then
applies the CPR1 Huffman/Rice carrier repack. It must reproduce both frozen
stage hashes:

```text
194380 bytes  f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451
191052 bytes  0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
```

Run the complete local integrity suite:

```bash
bash scripts/verify.sh
```

That checks every retained artifact, audits repository size, duplicates, and
common secret formats, rebuilds CPR1 byte-for-byte, and runs the CPR1 golden,
randomized, and malformed-stream tests.

## Repository map

| Path | Purpose |
| --- | --- |
| `code/` | Exact production trainers, packers, codec, repacker, and submission runtime |
| `artifacts/checkpoints/` | Canonical semantic, carrier, and HPAC stage boundaries |
| `artifacts/caches/` | Two exact 113 MB target caches, losslessly XZ-compressed to about 515 KB each |
| `artifacts/base/` | Minimal non-circular int5 archive needed to rebuild the predecessor |
| `artifacts/hpac/` | Exact packed HPAC and 600-frame arithmetic-coded token stream |
| `artifacts/final/` | Canonical CPR1 archive |
| `evidence/` | Training reports and official source-archive validation records |
| `recipe/` | Provenance, artifact lock, and exact historical commands |
| `scripts/` | Reproduction, training replay, and audit entry points |

## Training replay

The historical commands are encoded in `scripts/train.sh` and explained in
[`recipe/TRAINING.md`](recipe/TRAINING.md). They run inside the official
challenge environment at commit
`d3f688f84f555c5aaebee7d2c4203efc8a9051e2`.

Examples:

```bash
CHALLENGE_ROOT=/path/to/comma_video_compression_challenge \
  bash scripts/train.sh prepare

CHALLENGE_ROOT=/path/to/comma_video_compression_challenge \
  bash scripts/train.sh semantic

CHALLENGE_ROOT=/path/to/comma_video_compression_challenge \
  bash scripts/train.sh carrier

CHALLENGE_ROOT=/path/to/comma_video_compression_challenge \
  bash scripts/train.sh hpac
```

The included checkpoints are the authoritative stage boundaries. CUDA training
is hardware-sensitive, so a replay is not represented as a promise of a
byte-identical newly trained checkpoint. The final archive assembly is
byte-identical and is enforced as such.

## Deliberate exclusions

This repository does not duplicate public challenge videos, evaluator weights,
the multi-gigabyte inflated output, generated camera-frame master caches,
individual `p` payload copies, or the reproducible 194,380-byte predecessor.
It also omits unrelated probes and failed experiments. The retained repository
is under 5 MB and has no Git LFS dependency.

See [`recipe/PROVENANCE.md`](recipe/PROVENANCE.md) for the audited lineage and
the one upstream record inconsistency that was intentionally not propagated.
