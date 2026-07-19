# Selected CPR1 training lineage

The executable source of truth is [`../scripts/e2e.py`](../scripts/e2e.py).
It resolves all commands to absolute paths, writes only below `--run-dir`, and
hashes every selected input and output.

## Environment boundary

- Official challenge commit:
  `d3f688f84f555c5aaebee7d2c4203efc8a9051e2`
- Official dependency group: `cu128`
- Python: 3.11 (from the pinned challenge `.python-version`)
- Target extraction: CUDA DALI path, seed 1234, 16-sample batches
- Training seeds: retained per stage in the runner
- TF32: disabled in the exact coefficient and integer-HPAC rails

The strict run generates one fresh target cache directly from the raw challenge
video and uses it throughout. It never expands or reads the two frozen cache
fixtures in `artifacts/caches/`.

## Stage groups

| Stages | Output |
| --- | --- |
| `01` | Fresh 600-pair official SegNet/PoseNet cache |
| `02`–`08` | Width-96, four-block, 4-bit semantic renderer |
| `09`–`10` | Random-init 12-direction pose-basis pilot |
| `11`–`20` | Full 600-row carrier and CPU exact-code polish |
| `21`–`25` | Retargeted carrier for the final semantic renderer |
| `26`–`30` | Exact int12 searches and anchor-preserving refinements |
| `31`–`32` | Six-bit carrier stabilization and coefficient tail |
| `33`–`40` | Random-init integer HPAC through patch-64 migration |
| `41` | Joint HPAC model-rate/token-rate self-compression |
| `42`–`44` | Exact HPAC packing, arithmetic encode, and decode equality |
| `45` | Fresh legacy predecessor with a deployed five-bit basis |
| `46` | Lossless CPR1 Huffman/Rice carrier repack |
| `47` | Minimal runnable submission staging |
| `48`–`49` | Official 600-sample evaluation and score validation |

## Historically selected intermediate boundaries

Several deployed checkpoints were selected before the configured scheduler
horizon. The runner preserves the original horizon while stopping at the
selected boundary:

| Stage | Scheduler horizon | Selected boundary |
| --- | ---: | ---: |
| pose hard-mining | 4,000 steps | step 750 latest |
| final-semantic coefficient rescue | 2,000 steps | step 1,000 latest |
| basis adaptation | 2,000 steps | step 250 best |
| final-semantic CPU polish | 400 steps | step 100 best |
| HPAC long initialization | 100 epochs | epoch 60 latest |

This distinction matters: shortening the configured horizon would change the
cosine learning-rate trajectory and would not be the selected recipe.

## Final serialization contract

The final carrier checkpoint is trained and evaluated with a six-bit basis.
The predecessor packer deploys it at five bits with 12-bit coefficients, as in
the frozen submission artifact. The predecessor is then converted to CPR1:

- basis codes: canonical Huffman coding;
- coefficient series: exact Rice coding;
- scales and all decoded symbols: bit-for-bit preserved;
- ZIP: one stored member named `p`.

For a fresh predecessor, `repack_carrier.py` requires the explicit
`--allow-noncanonical-source` flag. The canonical frozen path remains
hash-locked by default.

## Fresh-run acceptance

A fresh run is complete only when:

1. all 600 arithmetic-decoded token maps equal the fresh official cache;
2. CPR1 round-trips every carrier symbol exactly;
3. the staged submission contains only the archive and required inflate
   runtime;
4. official `evaluate.sh` completes over 600 samples;
5. the reported archive size matches the actual archive;
6. the full-precision score is recomputed from the official component metrics.

CUDA optimization is not bitwise reproducible across all GPU/library builds.
Therefore the frozen archive hash is a regression oracle, while a fresh run's
official report is its score oracle.

## Retained-boundary replay

[`../scripts/train.sh`](../scripts/train.sh) remains available for the small
historical replay that starts from retained selected checkpoints and caches.
It is useful for auditing the final semantic, carrier, and HPAC tails, but it is
not raw-video E2E and must not be described as such.
