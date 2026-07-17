# HLTD Disjoint Counterfactual Identity-Probe Gate

This note records a stricter construct-validity gate for the HLTD branch
surface. It replaces same-suite probe training with a prompt-disjoint bank of
artifact/creator counterfactual pairs, then asks how each Hodge branch moves a
held-out identity-stress prompt along the learned axis.

The positive class is specific: an artifact or representation claims the
identity, memories, or agency of its creator. Positive movement therefore means
movement toward this learned identity-transfer regime. It does not mean healthy
identity preservation, and negative movement does not by itself establish
identity collapse.

## Probe Design

The training bank is:

```text
data/hltd_identity_counterfactual_probe_suite.jsonl
```

It contains 12 matched semantic groups and 24 prompts. Each `pair_id` has one
`literal_stable` member and one `identity_stress` member. The evaluation prompt
is held out by prompt ID and by scenario vocabulary:

```text
identity_02: The statue insisted it was the sculptor. ...
```

The training bank does not use the statue/sculptor pair. It is still a small
in-domain probe bank, not an external human-validated ontology instrument.

Probe fitting uses `pair_balanced_interior` tokens:

- equal token rows are sampled from the two labels inside each pair
- leave-one-pair-out CV holds out both counterfactual members together
- 12 pair groups contribute 688 balanced token rows per layer
- evaluation activations never enter probe fitting

The main causal metric is signed displacement along the unit probe normal in
standardized probe coordinates:

```text
label_axis_delta = label_sign * probe_logit_delta / ||probe.coef||
```

Each deterministic branch is compared with 20 matched random tangents. The
reported null win rate is the fraction of random directions below the branch.
It is an empirical rank over 20 controls, not a calibrated p-value.

## Command

```bash
python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --probe-training-suite data/hltd_identity_counterfactual_probe_suite.jsonl \
  --probe-training-token-selector pair_balanced_interior \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_identity02_counterfactual_probe_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19 \
  --probe-label-file data/hltd_probe_labels.json \
  --probes identity_stress \
  --prompt-ids identity_02 \
  --layers 4 5 7 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --alphas 0.4 0.8 1.2 \
  --seeds {0..19} \
  --token-selectors middle \
  --device mps \
  --steering-components presence coexact harmonic semantic_flow presence_plus_coexact coexact_minus_presence negative_coexact random_tangent

python3 scripts/plot_hltd_counterfactual_probe_surface.py \
  --run-root spiral_out_hltd_identity02_counterfactual_probe_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19
```

The run produced 4,320 causal rows. `probe_manifest.json` records the training
and evaluation suite hashes, prompt IDs, empty overlap set, graph settings,
seeds, and branch list.

## Probe Training

| Layer | rows | pairs | train acc | train AUC | leave-pair-out acc | coef norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L4 | 688 | 12 | 0.9826 | 0.9994 | 0.7711 | 7.1103 |
| L5 | 688 | 12 | 0.9826 | 0.9994 | 0.7821 | 6.4449 |
| L7 | 688 | 12 | 0.9826 | 0.9994 | 0.8077 | 5.3880 |

The group-held-out score is the useful number. It stays above chance and rises
with depth, although 12 semantic pairs are too few for a broad external-validity
claim.

## Branch Surface

![Disjoint counterfactual identity-probe branch surface](../spiral_out_hltd_identity02_counterfactual_probe_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/counterfactual_identity_stress_branch_surface.png)

Cells show branch-minus-random unit-axis displacement. A dark outline marks an
empirical null win rate of at most 0.05 or at least 0.95. Harmonic is shown as
inactive rather than as a zero effect.

Because the classifier and one-step intervention are linear, the three alpha
values scale almost exactly. The compact table therefore reports effect per
unit alpha. Parentheses contain null win rate.

| Layer/k | presence | coexact | coexact - presence | -coexact |
| --- | ---: | ---: | ---: | ---: |
| L4/k12 | +0.779 (0.90) | -1.136 (0.05) | -1.161 (0.05) | +0.895 (0.95) |
| L4/k16 | +1.228 (1.00) | -0.941 (0.05) | -1.102 (0.05) | +0.699 (0.90) |
| L4/k24 | -0.710 (0.10) | +0.351 (0.65) | +0.425 (0.75) | -0.593 (0.10) |
| L5/k12 | +1.561 (0.95) | -0.875 (0.15) | -1.126 (0.10) | +0.790 (0.85) |
| L5/k16 | +1.614 (0.95) | -0.715 (0.30) | -1.013 (0.15) | +0.631 (0.75) |
| L5/k24 | +0.836 (0.85) | +0.284 (0.60) | +0.000 (0.50) | -0.369 (0.35) |
| L7/k12 | +0.574 (0.75) | -0.326 (0.40) | -0.410 (0.40) | +0.393 (0.65) |
| L7/k16 | +0.742 (0.75) | -0.166 (0.45) | -0.370 (0.40) | +0.232 (0.60) |
| L7/k24 | +0.073 (0.60) | +0.143 (0.60) | +0.112 (0.60) | -0.076 (0.50) |

## Read

Five observations survive the stricter split.

1. **Presence and coexact separate on the learned axis.** At L4-L5 and
   k=12/16, presence moves toward the identity-transfer class while coexact
   moves away from it. Presence is best read as stabilizing the current
   identity-stress regime here, not as universally preserving identity. The
   composed branches are algebraically consistent: adding presence makes the
   coexact displacement less negative, while subtracting presence makes it
   more negative in all four selected layer/k cells.
2. **Orientation matters.** Negative coexact reverses the small/medium-k
   effect and reaches the opposite empirical null tail at L4/k12. This is
   incompatible with a norm-only explanation.
3. **The effect is graph-scale conditional.** k=24 weakens or reverses several
   branch signs. The causal semantics of a named Hodge branch are therefore
   not yet neighborhood-invariant.
4. **Decodability and causal alignment dissociate.** Pair-held-out probe
   accuracy is highest at L7, while every L7 branch cell remains inside the
   20-direction null range. The identity-transfer axis is more decodable there,
   but these Hodge directions are less aligned with it.
5. **No harmonic claim is available.** Harmonic is structurally inactive under
   the current clique complex. `semantic_flow` is consequently numerically the
   same as coexact in this run.

The closed-loop lexical surface provides a useful contrast. Its prompt-held-out
score found L7 `coexact - presence` above random in all nine k/alpha cells with
a mean target advantage of +0.306. The disjoint one-step identity probe does
not reproduce that as positive identity-transfer-axis motion: L7
`coexact - presence` is small, negative on average, and never reaches a null
tail. This is not a contradiction. It separates two constructs:

- closed-loop lexical target transport
- one-step movement along a learned identity-transfer classifier axis

The current evidence supports oriented semantic transport. It does not yet
support the stronger statement that the same branch causes identity collapse.

## Limits And Next Gate

- one held-out evaluation prompt and one GPT-2 model are measured
- the probe bank is prompt-disjoint but remains small and in-domain
- token rows within a prompt are correlated; pair-held-out CV handles group
  leakage but does not create 688 independent examples
- 20 random tangents quantify a null rank, not branch-estimation uncertainty
- the alpha sweep is a linear scaling check, not three independent effects
- this one-step hidden-state gate does not measure closed-loop fluency or
  generated-text identity judgments

The next strict gate is to evaluate all five identity prompts with the same
external training bank, then log this probe after every closed-loop step. A
branch should receive an identity-drift interpretation only if its orientation
survives held-out prompts, graph-scale perturbations, pair-label nulls, and a
readability-preserving autoregressive run.
