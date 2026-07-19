# Historical training recipe

Run these commands from the repository root in the official challenge
environment at commit `d3f688f84f555c5aaebee7d2c4203efc8a9051e2`.

The exact command lines are implemented by `scripts/train.sh`. The two cache
archives expand under ignored `work/caches/`; no generated training output is
committed.

## Prepare

```bash
CHALLENGE_ROOT=/path/to/challenge bash scripts/train.sh prepare
```

This restores and verifies:

| Cache | Bytes | SHA-256 |
| --- | ---: | --- |
| Original semantic cache | 117,981,133 | `8248a60da56119eb4b3ad76bfa32f5498dee849eaf4b83b304275064141fd828` |
| Official Ada cache | 117,981,301 | `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195` |

## Semantic renderer

The final 6,000-step, 4-bit QAT tail starts from
`semantic_renderer_w96_b4_qat4_12k.pt` and uses the original target cache.
Its authoritative output is the included
`semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`.

## Pose carrier

The final 4,000-step coefficient tail uses:

- the authoritative final semantic checkpoint;
- the official Ada target cache;
- `archive_carrier_int6_stable_s8k.pt` as initialization;
- 6-bit basis, 12-bit coefficients, amplitude 64;
- frozen basis, exact raw PoseNet metric loss, and seed `20260722`.

The authoritative output is
`archive_carrier_int6_coefftail_s4k.pt`.

## Integer HPAC

The 60-epoch self-compression run uses:

- an initialization extracted from the base archive;
- the official Ada semantic maps;
- 64 channels, patch 64, delta 2, frame embedding 8;
- raw-token mode, per-channel learned bit depths, and seed `20260716`.

Epoch 40 was selected. The authoritative output is
`hpac_selfcompress_l1_fastbits_e60.pt`.

The `pack-hpac` and `encode-tokens` actions reproduce the deployment
serialization and the arithmetic-coded stream from the authoritative
checkpoint. Encoding requires the exact integer inference path and is expected
to produce:

```text
packed HPAC:
ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee

token stream:
948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb
```

## Reproducibility claim

The commands, input caches, initialization checkpoints, selected outputs,
packing artifacts, and reports are retained. CUDA optimization can vary by
hardware and library build, so freshly trained tensors are not claimed to be
bit-identical across machines. The included selected checkpoints and final
archive chain are the frozen truth.
