# semantic-pose-HPAC_CPR1

This is the public reproducibility repository for
[`semantic-pose-HPAC_CPR1`](https://github.com/commaai/comma_video_compression_challenge/pull/130).
This fork learns a hard five-symbol token grid, searches exact rate-distortion moves, then losslessly ports [#135](https://github.com/commaai/comma_video_compression_challenge/pull/135)'s RC64 token coder.
The pinned L40S rail scored `0.165896` (`-2.70%` versus same-machine #130), projected rank 3 just behind #133's exact `0.165780`; every new rail must pass the frozen #130 control and final claims require T4 validation.
It preserves two separate guarantees:

1. a byte-exact rebuild of the frozen 191,052-byte CPR1 submission artifact;
   and
2. a strict, non-circular reconstruction of the selected training lineage from
   the raw challenge video through official scoring.

The canonical archive, submission runtime, and official review request are in
[PR #130](https://github.com/commaai/comma_video_compression_challenge/pull/130).
The complete predecessor attribution and originality boundaries are in
[`LINEAGE_AND_CITATIONS.md`](LINEAGE_AND_CITATIONS.md). A narrative technical
write-up is available at
<https://fesalfayed.com/blog/semantic-pose-compression/>.

The completed official 600-sample `linux-nvidia-t4` evaluation reports a
full-precision score of `0.1677066352` (displayed as `0.17`).

The canonical charged artifact is:

```text
bytes:   191052
sha256:  0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
member:  p
```

## Full raw-video E2E

[`scripts/e2e.py`](scripts/e2e.py) is the source of truth. Its 49 resumable
stages perform:

```text
raw 0.mkv
  -> official 600-pair SegNet/PoseNet targets
  -> semantic renderer from random initialization
  -> pose basis and 600 coefficient rows from the selected pilot lineage
  -> integer HPAC from random initialization
  -> exact arithmetic token stream
  -> fresh legacy predecessor archive
  -> lossless CPR1 Huffman/Rice repack
  -> minimal submission directory
  -> official 600-sample evaluation and full-precision score check
```

The runner rejects:

- any training input under this repository's frozen `artifacts/` tree;
- a challenge checkout other than commit
  `d3f688f84f555c5aaebee7d2c4203efc8a9051e2`;
- tracked modifications in the pinned challenge checkout;
- missing Git LFS video/model files;
- a run outside Python 3.11, missing CUDA/DALI dependencies, or no CUDA GPU;
- fewer than 10 GiB of free working space;
- a token decode that differs from the freshly extracted semantic maps;
- a partial, malformed, or archive-size-mismatched official report.

Prepare the pinned official environment:

```bash
git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge
git checkout --detach d3f688f84f555c5aaebee7d2c4203efc8a9051e2
git lfs install
git lfs pull
uv sync --group cu128
```

Preview every resolved command without training:

```bash
CHALLENGE=/path/to/comma_video_compression_challenge
RECIPE=/path/to/comma-ai-semantic-pose-hpac-cpr1

cd "$CHALLENGE"
uv run --group cu128 python "$RECIPE/scripts/e2e.py" plan \
  --challenge-root "$CHALLENGE" \
  --run-dir "$RECIPE/work/e2e"
```

Run the complete pipeline:

```bash
uv run --group cu128 python "$RECIPE/scripts/e2e.py" run \
  --challenge-root "$CHALLENGE" \
  --run-dir "$RECIPE/work/e2e"
```

Every completed stage records its command, input/output SHA-256 hashes, elapsed
time, and log. Re-running the same command verifies and skips valid stages.
Inspect progress or resume at a named boundary:

```bash
uv run --group cu128 python "$RECIPE/scripts/e2e.py" status \
  --challenge-root "$CHALLENGE" \
  --run-dir "$RECIPE/work/e2e"

uv run --group cu128 python "$RECIPE/scripts/e2e.py" run \
  --challenge-root "$CHALLENGE" \
  --run-dir "$RECIPE/work/e2e" \
  --from-stage 33_hpac_smoke
```

Successful completion leaves:

```text
work/e2e/submission/archive.zip
work/e2e/submission/report.txt
work/e2e/reports/49_official_metrics.json
work/e2e/e2e-manifest.json
```

This reproduces the selected method and all data dependencies from raw video.
Fresh CUDA optimization is hardware- and software-sensitive, so it does not
promise the same checkpoint bytes, archive bytes, or score as the frozen
reference artifact. The final `report.txt`, produced by the official
evaluator, is the truth for a fresh run.

See [`recipe/TRAINING.md`](recipe/TRAINING.md) for the stage groups and exact
selection boundaries.

## Frozen byte-exact rebuild

[`scripts/reproduce.sh`](scripts/reproduce.sh) starts from the retained
197,228-byte int5 semantic/pose archive, installs the frozen final HPAC model
and exact token stream, then applies CPR1. It must reproduce:

```text
194380 bytes  f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451
191052 bytes  0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
```

Run the complete local integrity suite:

```bash
bash scripts/verify.sh
```

Before experiments on a fresh GPU machine, run `scripts/evaluate_golden.sh`; it
refuses to proceed unless the pinned #130 archive reproduces its official score band.

It audits size, duplicates, common secret formats, all retained artifact
hashes, the full frozen rebuild, CPR1 randomized/malformed-stream tests, the
strict E2E dependency graph, and official-report parsing.

## Repository map

| Path | Purpose |
| --- | --- |
| `scripts/e2e.py` | Strict raw-video-to-official-score runner |
| `code/` | Trainers, packers, exact codecs, submission runtime, and validators |
| `recipe/` | Exact stage map, provenance, and frozen artifact lock |
| `artifacts/` | Frozen regression fixtures only; forbidden as strict E2E inputs |
| `evidence/` | Preserved reports for the frozen historical artifact |
| `scripts/train.sh` | Short retained-boundary replay, not the full E2E |
| `scripts/reproduce.sh` | Byte-exact frozen CPR1 reconstruction |
| `LINEAGE_AND_CITATIONS.md` | Predecessor attribution and claim boundaries |
| `CITATION.cff` | Machine-readable citation metadata |

## Deliberate exclusions

The repository does not duplicate public challenge videos or evaluator
weights, generated camera-frame caches, inflated raw output, or fresh training
runs. Generated state stays under ignored `work/`. The repository remains
under 5 MB and does not require Git LFS.

## License

The code is released under the [MIT License](LICENSE). The lineage document
separately records the upstream projects, papers, and challenge submissions
that informed this work.
