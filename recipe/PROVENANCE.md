# CPR1 provenance

## Canonical chain

1. The 4-bit width-96 semantic renderer produces frame 2.
2. The 12-direction 24x32 pose carrier produces frame 1.
3. The final carrier basis is stored at 5 bits and its 600x12 coefficients at
   12 bits.
4. The self-compressed integer-lattice HPAC encodes all 600 semantic maps.
5. `rebuild_submission_hpac.py` combines the exact semantic/pose bytes with the
   final HPAC model and token stream.
6. `repack_carrier.py` losslessly converts the carrier to CPR1 using canonical
   Huffman coding for basis symbols and exact Rice coding for coefficient
   series.

The final two stages are deterministic and byte-exact on the retained files.

## Verified identities

| Item | Verification |
| --- | --- |
| Final semantic checkpoint | Packs to the exact 40,252-byte semantic blob embedded in the base archive |
| Int6 carrier checkpoint | Matches the deployed int6 basis, scales, and 7,200 coefficient codes after reversing the lossless delta/zigzag transform |
| HPAC checkpoint | Re-packs to the exact 15,164-byte XZ artifact with zero logit difference |
| Token stream | 116,980 bytes; full 600-frame CPU replay previously matched the frozen raw-token hash |
| Predecessor | Rebuilt locally at 194,380 bytes with SHA-256 `f4457de0...a1451` |
| CPR1 | Rebuilt locally at 191,052 bytes with SHA-256 `0491d5df...c7cd` |

Detailed hashes and sizes are locked in `artifacts.json` and
`evidence/cpr1_verification.json`.

## Training inputs

Two target caches are retained losslessly:

- `gt_cache_600.pt.xz` is the original semantic-training cache.
- `gt_cache_600_official_ada.pt.xz` is the independent official Ada cache used
  for the final carrier and HPAC stages.

Their uncompressed hashes are verified during `scripts/train.sh prepare`.
Public videos and evaluator weights remain sourced from the pinned official
challenge checkout.

The pose trainer's generated `archive_master_exact.pt` is intentionally absent.
It is a large camera-frame cache and is deterministically regenerated from the
included semantic checkpoint when needed.

The HPAC initialization checkpoint is also intentionally absent. It is
losslessly extracted from the included int5 archive by
`extract_integer_hpac_archive.py`.

## Audited upstream inconsistency

The copied development directory for the int5 candidate contained an old
`rebuild_int5_report.json` describing a 197,404-byte intermediate, while the
actual retained int5 archive is the later lossless coefficient-delta form:

```text
197228 bytes
e16abacf3a83062f96139ef980fe95d9fd2061a5ce89d1d31c80dcfe52d65051
```

That stale report and its copied README were excluded. The actual archive,
decoded payload, final HPAC artifacts, and both downstream stage hashes are the
authoritative lineage.

## Evaluation boundary

The 194,380-byte predecessor completed official-path validation on RTX 2000 Ada
and RTX A4500. CPR1 changes only the lossless carrier representation, and its
legacy-versus-CPR1 decoded tensors were proven equal. The GitHub
`linux-nvidia-t4` run was still recorded as pending in the preserved evidence;
this repository does not rewrite that historical status.
