# HLTD Signed Matched-Betti Causal Gate

## Question

The positive-alpha matched-Betti gate found a coexact split: the branch tended
to support the observed next token while reducing the coarse semantic
target-control margin. This signed follow-up asks whether that split is an
oriented response or only sign-symmetric perturbation magnitude.

For each prompt-level branch-minus-random gap:

```text
odd(a)  = (gap(+a) - gap(-a)) / 2
even(a) = (gap(+a) + gap(-a)) / 2
```

`odd` is the direction-dependent part. `even` is compatible with curvature,
branch-specific magnitude, or generic nonlinear disruption.

## Fixed Contract

- model: local GPT-2 small
- prompts: all 20 bundled prompts, five per family
- chart: normalized hidden states, PCA 32
- vector field: centered differences
- layer: L5
- graph: k=16
- complex: orthogonal matched-Betti decomposition
- target Betti-1 fraction: 0.5
- selected position: middle interior node
- branches: exact, coexact, harmonic
- null: eight random-tangent seeds, 0 through 7
- random reference: `max_full_branch_node_speed`
- strengths: alpha -1.0, -0.5, -0.25, 0.25, 0.5, and 1.0
- inference unit: prompt after null-seed and selected-position averaging
- uncertainty: 5,000 prompt bootstrap draws, seed 1729

The run contains 3,840 steering rows. The realized Betti-1 fraction is
0.5000-0.5022 (mean 0.5013). Mean energy is exact 0.1056, coexact 0.5203,
and harmonic 0.3741. Exact and harmonic are active for all 20 prompts.
Coexact is active for 18; `metaphor_03` and `ontology_05` have no coexact
direction at the selected middle node and are omitted rather than zero-imputed.

The signed pair is formed only after averaging the eight random seeds within a
prompt. Random directions are repeated null measurements, not independent
samples.

## Run

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_matched_betti_signed_causal_full20_s8 \
  --layers 5 \
  --k 16 \
  --complex-mode matched_betti \
  --target-betti-1-fraction 0.5 \
  --steering-components exact coexact harmonic random_tangent \
  --token-selectors middle \
  --alphas -1.0 -0.5 -0.25 0.25 0.5 1.0 \
  --seeds 0 1 2 3 4 5 6 7 \
  --target-set-file data/hltd_semantic_targets.json \
  --device mps

python3 scripts/plot_hltd_matched_betti_causal.py \
  --summary spiral_out_hltd_matched_betti_signed_causal_full20_s8/summary.csv \
  --output-root spiral_out_hltd_matched_betti_signed_causal_full20_s8/causal \
  --require-signed \
  --expected-prompts 20 \
  --expected-null-seeds 8 \
  --expected-alpha-magnitudes 3
```

`--require-signed` fails if either direction is missing, if positive and
negative rows use different seed/position repeat sets, or if any prompt-level
alpha magnitude is absent. It prevents a partial run from silently becoming a
signed result. The explicit expected-count flags also reject a truncated
prompt suite or a globally shortened seed/strength schedule.

## Odd And Even Read

![Signed odd/even causal contrasts](../spiral_out_hltd_matched_betti_signed_causal_full20_s8/causal/plots/matched_betti_signed_causal_odd_even.png)

The central oriented results are:

| readout | branch | \|alpha\| | prompts | odd mean | 95% prompt CI |
| --- | --- | ---: | ---: | ---: | ---: |
| observed next-token support | coexact | 0.25 | 18 | +0.08683 | [-0.00060, +0.18903] |
| observed next-token support | coexact | 0.50 | 18 | +0.16084 | [+0.01600, +0.33848] |
| observed next-token support | coexact | 1.00 | 18 | +0.32811 | [+0.03662, +0.65391] |
| semantic target-control margin | coexact | 0.25 | 18 | -0.09431 | [-0.19244, -0.00835] |
| semantic target-control margin | coexact | 0.50 | 18 | -0.17587 | [-0.36566, -0.01268] |
| semantic target-control margin | coexact | 1.00 | 18 | -0.30746 | [-0.68018, +0.01150] |
| semantic target-control margin | exact | 0.25 | 20 | -0.10618 | [-0.17628, -0.03966] |
| semantic target-control margin | exact | 0.50 | 20 | -0.19600 | [-0.33824, -0.06913] |
| semantic target-control margin | exact | 1.00 | 20 | -0.38717 | [-0.68926, -0.09313] |

At alpha +1, the earlier coexact next-token gap of +0.35322 decomposes into
+0.32811 odd and +0.02511 even. Its semantic-margin gap of -0.35189 decomposes
into -0.30746 odd and -0.04443 even. The positive-alpha result is therefore
mostly directional, not mostly sign-symmetric disruption.

Eleven of 27 odd intervals exclude zero, versus two of 27 even intervals. The
cells are dependent across strengths and this count is not a
multiple-comparison-corrected test. The two isolated even intervals are a small
negative harmonic KL contrast and a small negative coexact next-token contrast
at |alpha|=0.25; neither persists across strengths.

Harmonic has no stable odd effect on the primary next-token or semantic
readouts. It remains `open-cycle residual`, not a demonstrated concept ring.

## Response Scaling

The signed trajectories also permit a low-order response check. For each
prompt, odd values are fit through zero against |alpha| and even values against
|alpha| squared. The resulting first-order slope and second-order curvature are
then bootstrapped across prompts.

![Signed response coefficients](../spiral_out_hltd_matched_betti_signed_causal_full20_s8/causal/plots/matched_betti_signed_response_coefficients.png)

| response coefficient | branch | prompts | mean | 95% prompt CI | mean relative RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| odd next-token slope | coexact | 18 | +0.32780 | [+0.03047, +0.67021] | 0.351 |
| odd semantic-margin slope | coexact | 18 | -0.31921 | [-0.69064, +0.00610] | 0.300 |
| odd semantic-margin slope | exact | 20 | -0.38988 | [-0.68302, -0.10960] | 0.230 |
| odd semantic-margin slope | harmonic | 20 | +0.04124 | [-0.24304, +0.35184] | 0.276 |

The coexact next-token coefficient survives the trajectory-level bootstrap.
The coexact semantic coefficient keeps the negative mean but narrowly crosses
zero after fitting all three strengths. Exact semantic movement remains
negative. No harmonic coefficient separates from random tangent.

KL should be locally second-order around the unsteered logits, so its odd terms
are not the main semantic readout. None of the trajectory-level KL odd or even
coefficient intervals excludes zero.

## Family And Prompt Heterogeneity

The descriptive odd coexact next-token slopes are -0.036 for literal-stable,
+0.225 for identity-stress, +0.458 for metaphor-shift, and +0.781 for
ontology-collapse. The corresponding coexact semantic slopes are -0.523,
+0.112, -0.423, and -0.500. Each family cell has only four or five active
prompts and is not separately powered.

The main signs are not produced by one prompt. Across leave-one-prompt-out
means:

- coexact next-token slope remains positive, +0.204 to +0.384
- coexact semantic slope remains negative, -0.378 to -0.209
- exact semantic slope remains negative, -0.452 to -0.321

Prompt heterogeneity is still substantial. Coexact next-token slopes range
from strongly negative literal prompts to a large positive `ontology_01`.
That variation is part of the result and motivates position and family
localization rather than a universal branch label.

## Decision

This gate passes the narrow orientation question:

> At the L5/k16 matched-Betti middle node, the positive coexact direction has a
> direction-dependent advantage over norm-matched random tangents for support
> of the actually observed next token.

It does not pass a stronger target-directed semantic-circulation claim. The
same coexact orientation generally moves away from the current coarse semantic
target-control sets. This could mean that local predictive transport rotates
across, rather than toward, those family axes, or that the lexical target sets
do not resolve the transported feature.

The `exact/presence` name also remains a hypothesis label. Positive exact flow
is fixed by the observed decomposition, but it is not calibrated to mean
"increase identity"; its negative semantic slope cautions against that reading.

The all-interior signed follow-up is complete in
[`hltd_signed_position_gate.md`](hltd_signed_position_gate.md). It finds a
positive odd next-token coefficient in the early, middle, and late thirds,
with a paired early-minus-late advantage, while the coarse semantic-margin
effect is not broad across positions. The next robustness gate should freeze
those phases and repeat them across layers before signed learned-probe and
closed-loop tests are treated as confirmatory.

## Artifacts

- `causal/summary_branch_minus_random_pairs.csv`: seed-matched raw gaps
- `causal/summary_prompt_branch_gaps.csv`: prompt rows before sign pairing
- `causal/summary_signed_prompt_contrasts.csv`: prompt odd/even contrasts
- `causal/summary_signed_prompt_bootstrap.csv`: per-strength prompt inference
- `causal/summary_signed_prompt_response_coefficients.csv`: per-prompt slopes and curvatures
- `causal/summary_signed_response_coefficient_bootstrap.csv`: coefficient inference
- `causal/summary_signed_causal_report.md`: generated compact report
- `causal/plots/matched_betti_signed_causal_odd_even.png`: per-strength figure
- `causal/plots/matched_betti_signed_response_coefficients.png`: response-coefficient figure
