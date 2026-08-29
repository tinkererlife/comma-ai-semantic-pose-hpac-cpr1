# External review brief: lossy semantic token grid

## Goal and hypothesis

PR #130 losslessly transmits 600 semantic maps of shape 384x512 with five token
IDs through HPAC, then renders the decoded IDs back to video.  Our hypothesis
was that those IDs do not have to equal the original SegNet labels: they may be
an arbitrary discrete latent grid as long as the official total score improves.

The official objective is

```text
100 * SegNet distortion + sqrt(10 * PoseNet distortion) + 25 * archive rate
```

"Lossy" therefore means lossy with respect to the intermediate semantic maps.
The chosen modified grid is still transmitted losslessly by HPAC.  The specific
trade we originally wanted to test was: sacrifice some perceptual accuracy when
the resulting reduction in encoded bytes is worth more.

## What we actually implemented

1. Reproduced the frozen #130 archive and official 600-frame evaluator.
2. Replaced integer token lookup during optimization with hard one-hot values in
   the forward pass and a straight-through softmax gradient in the backward
   pass.
3. Used backpropagation through the frozen renderer, SegNet and PoseNet to rank
   categorical token changes.  Proposals are not random or manually selected.
4. Added the deployed integer HPAC model as a per-symbol rate oracle, including
   spatial context and the changed frame's effect on the next frame.
5. Accepted proposed hard changes only after exact SegNet/PoseNet evaluation
   plus projected HPAC ideal-bit cost, with batch backtracking when a group
   failed.
6. Encoded the accepted grid with the real #130 coder and ran the official
   evaluator.
7. As a separate coordinate-descent phase, froze the 159 accepted tokens and
   quantization-aware fine-tuned the shared renderer on all 600 frames.  The
   carrier and per-frame embedding stayed frozen.  We retained the best exact
   int4 checkpoint, not the last training checkpoint.
8. Starting from that checkpoint, ran a strict rate-first pass over all 600
   frames.  HPAC alone proposed one categorical move per frame.  A move was
   accepted only if the complete affected HPAC ideal rate fell and the exact
   combined rate-distortion score improved.  This explicitly permits a
   perception loss when the rate saving is larger.

The core token search is in `learned_token_mvp.py` and
`hpac_token_search.py`.  The separated renderer phase is in
`finetune_renderer_on_tokens.py`.  `replace_archive_tokens.py` replaces only the
semantic renderer and token stream while preserving the carrier and HPAC bytes.

## Results

| Variant | Official/recomputed score | Archive | Delta vs #130 |
|---|---:|---:|---:|
| Frozen #130 | 0.1721412975 | 191,052 B | baseline |
| Token grid only, 159 flips | 0.1706872406 | 191,516 B | -0.8447% |
| Tokens + best full-600 int4 renderer | 0.1702539624 | 191,504 B | -1.0964% |
| Strict full-600 rate-first pass | 0.1700018955 | 191,256 B | -1.2428% |
| Four category-aware checkerboard sweeps | 0.1692241979 | 191,084 B | -1.6946% |
| Twenty vectorized rate sweeps + two joint sweeps | 0.1677066352 | 190,664 B | -2.5762% |

The first strict pass's score decomposition was important:

```text
                             #130          final          change
rate contribution          0.127214      0.127350       +0.000136
distortion contribution    0.044928      0.042652       -0.002275
total score                0.172141      0.170002       -0.002139
```

The rate-first pass tested 600 candidates and accepted 111.  All 111 reduced the
recomputed affected HPAC ideal rate; 40 individually worsened perception but
still improved total score, exactly the motivating trade.  The real range-coded
stream shrank by 248 bytes versus the renderer-hardened artifact.  Although the
40 deliberately lossy moves exist, the other 71 moves improved perception
enough that the aggregate final artifact is a Pareto improvement over the prior
artifact: both its rate and distortion contribution are lower.

The category-aware follow-up started from that artifact, tested 2,400 further
candidates and accepted 175, including 39 perception-for-rate trades.  Its four
cumulative projected score deltas were `-0.0000403`, `-0.0002515`,
`-0.0005434`, and `-0.0007829`; there was no clear saturation after four
sweeps.  The real token stream shrank by 172 bytes and the official full-score
improvement was `-0.0007777`, only `0.0000052` away from the projection.

The scalability follow-up continued the exact search for twenty sweeps.  It
tested 12,000 moves and accepted 747, including 149 explicit perception-for-rate
trades.  The marginal projected gain fell from roughly `1.5e-4` per sweep around
sweeps 5--10 to `4.8e-5` over the last five, so more identical single-token
sweeps do not extrapolate to first place.  Two calibrated joint-backprop sweeps
accepted 35 further moves but were slower and added only `6.3e-5` projected
score improvement.  The final real stream is 116,604 bytes, decodes exactly,
and the official 600-sample T4 score is `0.1677066352`.

## What happened in the smaller experiments

- Unchecked persistent soft-token optimization failed.  The differentiable
  optimizer sees mixtures of token embeddings, while deployment takes an
  `argmax`.  Once many logits crossed at the same time, tens of thousands of
  hard IDs changed discontinuously and the exact score collapsed.
- Switching to plain gradient-ranked hard proposals plus an exact acceptance
  gate stabilized the search.  A 4-frame run accepted 1 of 32 proposals; a
  32-frame run accepted 2 of 63.  This says the gradient is a useful ranker, but
  not a trustworthy direct optimizer of the discrete objective.
- Renderer fine-tuning on four frames looked extremely good locally but scored
  about 0.1764 over all 600 frames: clear overfitting.
- Full-600 renderer training at `1e-6` did not beat the fixed-grid checkpoint.
  At `2e-7`, epoch 1 improved; epochs 2-4 progressively regressed.  The final
  renderer uses epoch 1.
- A strict 32-frame rate-first control accepted 11 of 32 proposals, including
  four explicit perception-for-rate trades.  The corresponding 600-frame pass
  accepted 111 of 600, including 40 such trades, and survived real encoding and
  official evaluation.

## Exact-search scalability follow-up

The next iteration fixes the full-600 pose gate (frame contributions are now
replaced in the global mean before the nonlinear square root), remembers
attempted token categories rather than blacklisting a pixel, and keeps HPAC
probability math on the GPU.  Its localized HPAC recomputation was checked
against full-frame recomputation on ten random and boundary edits: the maximum
ideal-bit difference was `5.4e-13`, with a measured mean rate-recheck speedup of
`6.63x`.

The same probe disproved the proposed renderer crop optimization.  Four spatial
GroupNorm layers make a one-token edit affect the renderer globally; after
camera rounding, 1,932--12,439 pixels changed in the sampled frames.  We therefore
batch temporally independent even/odd frames but retain full renderer,
SegNet and PoseNet evaluation.  A 32-frame decision-equivalent control fell
from 168.9 s in the original path to 73.7 s on T4.  Increasing the batch from
4 to 16 reduced that only to 68.8 s because exact localized HPAC rechecks remain
serial.  An A100-SXM4 run with TF32 disabled reproduced the same accepted move
and ideal-bit delta in 44.0 s (`1.56x` faster than T4 batch 16); leaving TF32 on
changed the accept/reject decisions and is not used.

The remaining localized rechecks were then vectorized across same-parity frames.
On 16 identical edits the new path was exactly bit-identical and took 2.08 s
instead of 17.03 s (`8.20x`).  Four full 600-frame sweeps fell from 2,825.9 s to
387.5 s (`7.29x`) while reproducing the previous official score within
`2.2e-6`.  Batch 75 is the largest useful search batch on a 40 GiB A100; batch
100 exceeds memory once before/after temporal contexts are materialized.

The next test evaluates eight independent pixel/category alternatives per frame
instead of committing the rate model's first choice.  On the same 32 frames,
K=1 accepted 0/32 in 13.9 s while K=8 accepted 4/256 in 28.9 s and improved the
projected score by `0.00014199`.  The decisive 600-frame L40S pass evaluated
4,800 alternatives in 472.2 s, accepted 38 (13 deliberate perception-for-rate
trades), and improved the projected official score by only `0.00004072` with a
24.54 GiB peak.  This is a real candidate-quality win but far below the
predeclared `0.0002--0.0003` go threshold, so more identical single-token sweeps
are not the route to first place.  Checkpoints now include the compressed exact
category-attempt history; the 600-frame file records all 4,800 evaluations.

The structural follow-up replaces one pixel proposal with a constant-category
rectangle while leaving the exact HPAC/SegNet/PoseNet acceptance gate unchanged.
Five full 600-frame sweeps tested 24,000 rectangles from `1x2` through `9x9` and
accepted 142.  Their projected deltas were `-0.00012632`, `-0.00002859`,
`-0.00005236`, `-0.00003079`, and `-0.00001362`.  A `13x13`--`17x17` smoke test
made no final accepted change, while the largest allowed shape dominated at
both `5x5` and `9x9`; useful scale therefore ends somewhere around this range
for the current rectangular family.

The final real range-coded stream is 116,428 bytes and the archive is 190,488
bytes, 176 bytes below the previous best.  A same-L40S official A/B evaluation
is required because DALI target decoding shifts absolute SegNet/PoseNet values
between T4 and L40S.  On L40S the unchanged control scored `0.1682625922` and
the structural grid scored `0.1681259044`, an actual improvement of
`0.0001366878` (`0.0812%`).  Applying only that measured A/B delta to the prior
T4 result projects `0.1675699474`, or `2.6556%` below #130 and still rank 3.
The internal ideal-bit/perception projection predicted `0.0002516870`, so it
overstated the gain by about 46%; the byte prediction was close (188 predicted
versus 176 actual), and nearly all error came from the cached perception gate.

The RunPod reproduction found the perception-gate mismatch.  Official inflate
disables TF32 while the separate evaluator retains cuDNN TF32 and constructs its
input via a contiguous BHWC buffer followed by a CHW view.  Our combined search
process had disabled TF32 for both renderer and metric models and bypassed that
layout.  The gate now preserves exact inflate rendering but locally enables the
official metric path.  Its no-op control matches the official full-precision
metric, and a fresh 600-frame multiscale sweep accepted 215/4,800 proposals (34
explicit perception-for-rate trades).  The real stream fell from 116,604 to
116,356 bytes; official L40S evaluation gives an approximately `0.167426` score,
`-0.0008375` (`-0.50%`) versus the same-machine control and rank 3 on the current
leaderboard.  The archived proxy delta was `-0.0008534`, within about `1.9%` of
the realized improvement.

A second category-aware sweep accepted 61/4,800 more proposals (10 explicit
perception-for-rate trades).  The deployed stream fell another 40 bytes to
116,316 bytes and the archive to 190,376 bytes.  Official L40S evaluation gives
approximately `0.167289`: `-0.0009741` (`-0.58%`) versus the same-machine
control, `-2.82%` versus #130, and still rank 3.  Its marginal gain was only
about 16% of pass 1, evidence that this fixed proposal family is beginning to
saturate.

## Why this is still only a conservative MVP

The grid contains 117,964,800 token positions.  The previous deployed artifact
differs from #130 in 875 positions and the final structural grid in 1,202.  We
did not perform a global joint optimization over all tokens and the decoder.
The production-scale search phases tried only 20,400 single-token and 24,000
rectangular proposals.  In particular, we did not implement:

- a persistent freely learned grid over all 600 frames with a scalable discrete
  optimizer;
- invalidation and selective re-ranking of attempt history after nearby
  accepted changes alter the HPAC context;
- freely growing connected regions, contours or temporal-tube moves instead of
  the fixed rectangular family;
- exact final arithmetic-coded byte length inside every acceptance decision
  (the oracle matches deployed HPAC probabilities and ideal bits);
- retraining HPAC for the changed token distribution;
- a new codebook, decoder/world model or a port to the current #1 architecture.

The previous phrase "full potential" should only be read as the measured result
of this narrow implementation.  It is not an estimate of the idea's ceiling.

## Highest-value review questions

Please focus on missed algorithmic potential rather than style:

1. Is the corrected evaluator-matched gate portable across the official T4 and
   L40S paths, or should acceptance be calibrated separately per GPU family?
2. The latest HPAC ideal-bit delta predicted about 188 saved bytes and the real
   range coder saved 176.  Can acceptance cheaply target those final bytes?
3. Which cheap proposal model can rank structural candidates better than direct
   one-symbol HPAC surprise without putting an unreliable gradient in the gate?
4. What is the simplest scalable rate-first search over structural moves:
   contours, connected regions, blocks, or temporal tubes?
5. Should optimization alternate between (A) metric-neutral byte removal and
   (B) renderer/distortion repair, instead of combining both gradients at once?
6. Can sparse discrete state, coordinate descent, policy gradients or a learned
   encoder optimize all 118M choices without the soft/hard collapse and without
   exceeding a 16 GB T4?
7. Is continuing on #130 scientifically useful, or should the method first be
   ported to the current leading wire format and decoder?

## Reproduction and evidence

Review the complete fork delta with:

```bash
git diff 2f94596..HEAD -- README.md experiments/learned-token-grid-mvp
```

`official-result.json` records the final metrics and archive hash.  Detailed
reports, failed-run logs and final archives were downloaded separately so that
failed experiments do not bloat the public source repository.
