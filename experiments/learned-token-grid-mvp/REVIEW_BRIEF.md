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

The final score decomposition is important:

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

## Why this is still only a conservative MVP

The grid contains 117,964,800 token positions, but the final artifact differs
from #130 in only 142.  We did not perform a global joint optimization over all
tokens and the decoder.  The rate-first pass tried only one candidate per frame,
or 600 out of roughly 472 million possible single-token category changes.  In
particular, we did not implement:

- a persistent freely learned grid over all 600 frames with a scalable discrete
  optimizer;
- second/third/fourth category candidates or multiple rate-first passes;
- region, contour, block or temporal-tube moves instead of isolated pixels;
- exact final arithmetic-coded byte length inside every acceptance decision
  (the oracle matches deployed HPAC probabilities and ideal bits);
- retraining HPAC for the changed token distribution;
- a new codebook, decoder/world model or a port to the current #1 architecture.

The previous phrase "full potential" should only be read as the measured result
of this narrow implementation.  It is not an estimate of the idea's ceiling.

## Highest-value review questions

Please focus on missed algorithmic potential rather than style:

1. Does the gradient and exact-evaluation path optimize the official objective
   without a scale, indexing, context or projection error?
2. HPAC ideal bits predicted about 449 saved bytes while the real range coder
   saved 248.  Can acceptance cheaply target actual coder bytes more closely?
3. Does permanently blacklisting an attempted pixel wrongly prevent trying its
   other three token categories later under a changed context?
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
