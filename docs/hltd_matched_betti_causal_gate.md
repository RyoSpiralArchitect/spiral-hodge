# HLTD Matched-Betti One-Step Causal Gate

## Question

The topology-filtration gate found that the real field retains more non-exact
energy than matched vector-shuffle and random-tangent nulls at an interior
complex. This follow-up asks a narrower causal question:

> At the pre-registered interior topology, do exact, coexact, or harmonic node
> directions move next-token logits more coherently than a norm-matched random
> tangent?

This is a one-step logit gate. It does not test closed-loop generation,
readability preservation, ontology drift, or semantic return around a loop.

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
- strengths: alpha 0.25, 0.5, and 1.0
- inference unit: prompt after null-seed averaging
- uncertainty: 5,000 prompt bootstrap draws, seed 1729

The random-tangent direction uses `max_full_branch_node_speed` as its activity
reference. All active interventions are normalized to the same natural hidden
step norm. This prevents a zero coexact vector from silently turning the null
into a no-op. Inactive Hodge branches are omitted only from their own contrast;
they are never zero-imputed.

Across the 20 prompt graphs, the realized Betti-1 fraction is 0.5000-0.5022
(mean 0.5013). The mean energy split is exact 0.1056, coexact 0.5203, and
harmonic 0.3741. Exact and harmonic are active for all 20 prompts. Coexact is
active for 18; `metaphor_03` and `ontology_05` have an exactly zero coexact
vector at the selected middle node.

## Run

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_matched_betti_causal_full20_s8_v2 \
  --layers 5 \
  --k 16 \
  --complex-mode matched_betti \
  --target-betti-1-fraction 0.5 \
  --steering-components exact coexact harmonic random_tangent \
  --token-selectors middle \
  --alphas 0.25 0.5 1.0 \
  --seeds 0 1 2 3 4 5 6 7 \
  --target-set-file data/hltd_semantic_targets.json \
  --device mps

python3 scripts/plot_hltd_matched_betti_causal.py \
  --summary spiral_out_hltd_matched_betti_causal_full20_s8_v2/summary.csv \
  --output-root spiral_out_hltd_matched_betti_causal_full20_s8_v2/causal
```

The plotting pass pairs each branch row with the same prompt, position, alpha,
and random seed. It then averages repeated null seeds within each prompt before
bootstrap inference. Random seeds are therefore not counted as independent
prompts.

## Result

![Matched-Betti one-step causal branch gaps](../spiral_out_hltd_matched_betti_causal_full20_s8_v2/causal/plots/matched_betti_causal_branch_gaps.png)

Positive values are branch minus its paired random tangent. The alpha=1
prompt-level estimates are:

| readout | branch | prompts | mean gap | 95% prompt CI | positive prompts |
| --- | --- | ---: | ---: | ---: | ---: |
| KL movement | exact | 20 | +0.08278 | [-0.01093, +0.20130] | 10/20 |
| KL movement | coexact | 18 | -0.03158 | [-0.08740, +0.03130] | 4/18 |
| KL movement | harmonic | 20 | +0.00188 | [-0.05495, +0.07340] | 9/20 |
| observed next-token support | exact | 20 | +0.10948 | [-0.16391, +0.38460] | 13/20 |
| observed next-token support | coexact | 18 | +0.35322 | [-0.02969, +0.76392] | 14/18 |
| observed next-token support | harmonic | 20 | +0.27566 | [-0.07770, +0.71346] | 11/20 |
| semantic target-control margin | exact | 20 | -0.30059 | [-0.63831, +0.01151] | 7/20 |
| semantic target-control margin | coexact | 18 | -0.35189 | [-0.72884, -0.02193] | 6/18 |
| semantic target-control margin | harmonic | 20 | +0.07327 | [-0.24540, +0.43953] | 10/20 |

The coexact next-token trend grows with alpha, but every next-token interval
still crosses zero. Its semantic target-control margin is negative at every
strength and excludes zero at alpha 0.25, 0.5, and 1.0. Exact has the same
negative semantic-margin result at alpha 0.25 and 0.5. Harmonic is largely
indistinguishable from random tangent; its only excluding interval is a small
negative KL gap at alpha 0.25.

## Family Heterogeneity

At alpha=1, the descriptive coexact next-token gaps are +0.108 for
identity-stress, -0.132 for literal-stable, +0.506 for metaphor-shift, and
+1.113 for ontology-collapse. The corresponding semantic-margin gaps are
-0.018, -0.464, -0.446, and -0.535. These family cells contain only four or
five active prompts and are not separately powered inferential tests.

Several aggregate means also contain large prompt effects. For example,
`literal_01` contributes opposite large coexact and harmonic semantic-margin
effects, while `metaphor_02` contributes a large negative exact/coexact effect.
The prompt dots in the figure are therefore part of the result, not decoration.

## Decision

The observational result survives: the matched-topology decomposition yields
nontrivial exact, coexact, and harmonic branches, and the non-exact excess over
structural nulls remains a valid causal candidate-selection result.

The present causal gate does **not** support a clean semantic-circulation or
harmonic concept-ring claim. Coexact may be helping the locally observed next
token without moving toward the coarse family target set, but that is a
hypothesis generated by the split readouts, not a demonstrated mechanism.
Harmonic should remain named `open-cycle residual` at this stage.

## Next Signed Gate

Keep the same model, prompt suite, L5, k=16, matched Betti-1 0.5, middle node,
and eight null seeds. Add alpha `-1.0 -0.5 -0.25 0.25 0.5 1.0` and separate:

```text
odd(a)  = (gap(+a) - gap(-a)) / 2
even(a) = (gap(+a) + gap(-a)) / 2
```

The odd contrast tests oriented branch transport. The even contrast captures
sign-symmetric perturbation magnitude. Prompt bootstrap remains the inference
unit. Only after an oriented effect survives this gate should the experiment
expand to all-interior position bins, learned disjoint probes, and closed-loop
generation.

## Artifacts

- `causal/summary_branch_minus_random_pairs.csv`: seed-matched row pairs
- `causal/summary_prompt_branch_gaps.csv`: one row per prompt contrast
- `causal/summary_prompt_bootstrap.csv`: prompt-bootstrap estimates
- `causal/summary_causal_report.md`: generated compact report
- `causal/plots/matched_betti_causal_branch_gaps.png`: primary figure
