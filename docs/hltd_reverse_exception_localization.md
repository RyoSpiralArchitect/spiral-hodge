# HLTD Reverse-Branch Exception Localization

This note localizes the two prompt-local `negative_coexact` exceptions found in
the closed-loop sign-control panel: `identity_04` and `ontology_05`.

The goal is not to update the family-level branch ledger. This is a targeted
exception probe, so its rows should stay separate from the main branch-role
matrix unless a later run balances the full prompt families across the same
layer/k grid.

## Command

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01 \
  --layers 5 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 \
  --prompt-ids identity_04 ontology_05 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01 \
  --output-dir spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/plots \
  --components negative_coexact random_tangent
```

Runtime: 18 prompt/layer/k cells in 1093.4 seconds on CPU.

## Outputs

- `spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/closed_loop_metrics.csv`
- `spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/closed_loop_contrasts.csv`
- `spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/closed_loop_prompt_layer_k_summary.csv`
- `spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/plots/closed_loop_prompt_layer_k_map.png`
- `spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/plots/plot_manifest.json`

![Reverse exception layer-k map](../spiral_out_hltd_closed_loop_reverse_exception_lk_probe_l5_l8_k12_k24_a08_s01/plots/closed_loop_prompt_layer_k_map.png)

## Layer-k Read

`branch_specific_gate_rate` uses the same strict closed-loop definition as the
main branch ledger: drift at least 0.5, positive target margin, and matched
random-tangent not better on drift or target margin.

| prompt | L5 k12 | L5 k16 | L5 k24 | L7 k12 | L7 k16 | L7 k24 | L8 k12 | L8 k16 | L8 k24 | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity_04 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | k>=16 at L5/L7, disappears by L8 |
| ontology_05 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | late and k-broad, strongest at L8 |

Target-margin advantage has a different shape:

| prompt | L5 k12 | L5 k16 | L5 k24 | L7 k12 | L7 k16 | L7 k24 | L8 k12 | L8 k16 | L8 k24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity_04 | 0.22 | 0.23 | 0.22 | 0.25 | 0.48 | 0.34 | -0.02 | 0.00 | -0.05 |
| ontology_05 | -0.09 | -0.12 | -0.09 | 0.10 | 0.15 | 0.30 | 0.38 | 0.48 | 0.48 |

## Interpretation

- `identity_04` is not just the original L7/k16 cell. The reverse branch is a
  middle-layer, medium/high-k effect: L5 and L7 pass at k=16/24, while k=12
  lacks decoded drift and L8 collapses back to the baseline.
- `ontology_05` is a late-layer reverse branch. L5 can produce surface moon
  drift at k=12, but target-random stays negative, so it fails the strict gate.
  From L7 onward the target advantage turns positive, and by L8 all tested k
  values pass.
- This separates the two exceptions mechanistically. `identity_04` looks like a
  mid-layer identity-mask branch that requires enough graph neighborhood width.
  `ontology_05` looks like a late ontology/moon attractor that strengthens with
  layer depth and survives the k sweep.

## Five-Seed Passing Bands

The passing bands were then rerun with seeds `0..4`, while avoiding the cells
that already failed in the 2-seed localization.

Identity band:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_reverse_identity04_passing_band_l5_l7_k16_k24_a08_s04 \
  --layers 5 7 \
  --k 16 24 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids identity_04 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Ontology band:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_reverse_ontology05_passing_band_l7_l8_k12_k24_a08_s04 \
  --layers 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids ontology_05 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Runtimes:

- `identity_04` passing band: 4 prompt/layer/k cells in 625.6 seconds on CPU.
- `ontology_05` passing band: 6 prompt/layer/k cells in 940.6 seconds on CPU.

Summary outputs:

- `spiral_out_hltd_closed_loop_reverse_exception_passing_band_s04/passing_band_negative_coexact_summary.csv`
- `spiral_out_hltd_closed_loop_reverse_identity04_passing_band_l5_l7_k16_k24_a08_s04/plots/closed_loop_prompt_layer_k_map.png`
- `spiral_out_hltd_closed_loop_reverse_ontology05_passing_band_l7_l8_k12_k24_a08_s04/plots/closed_loop_prompt_layer_k_map.png`

![Identity reverse passing band](../spiral_out_hltd_closed_loop_reverse_identity04_passing_band_l5_l7_k16_k24_a08_s04/plots/closed_loop_prompt_layer_k_map.png)

![Ontology reverse passing band](../spiral_out_hltd_closed_loop_reverse_ontology05_passing_band_l7_l8_k12_k24_a08_s04/plots/closed_loop_prompt_layer_k_map.png)

Five-seed strict gate rates:

| prompt | layer | k | branch gate | branch-specific gate | random gate | target-random | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity_04 | L5 | 16 | 1.00 | 0.80 | 0.00 | 0.1166 | robust, but one seed loses target-random specificity |
| identity_04 | L5 | 24 | 1.00 | 0.80 | 0.00 | 0.1126 | same as L5/k16 |
| identity_04 | L7 | 16 | 1.00 | 1.00 | 0.00 | 0.3630 | strongest identity reverse cell |
| identity_04 | L7 | 24 | 1.00 | 1.00 | 0.00 | 0.2265 | robust identity reverse cell |
| ontology_05 | L7 | 12 | 1.00 | 0.40 | 0.20 | 0.0458 | weak/partial, random-sensitive |
| ontology_05 | L7 | 16 | 1.00 | 0.40 | 0.20 | 0.0977 | weak/partial, random-sensitive |
| ontology_05 | L7 | 24 | 1.00 | 0.80 | 0.20 | 0.2400 | robustens as k widens |
| ontology_05 | L8 | 12 | 1.00 | 0.80 | 0.20 | 0.3411 | late-layer robust cell |
| ontology_05 | L8 | 16 | 1.00 | 0.80 | 0.20 | 0.4419 | strongest ontology reverse band |
| ontology_05 | L8 | 24 | 1.00 | 0.80 | 0.20 | 0.4427 | strongest ontology reverse band |

The five-seed run preserves the mechanism split:

- `identity_04` is strongest at L7. L5 still passes, but one seed fails the
  stricter matched-random target advantage, lowering specificity to 0.80.
- `ontology_05` is strongest at L8. L7 remains real but weaker and
  random-sensitive; k=24 is the clearest L7 cell.
- Random tangent never passes for `identity_04` in the passing band. For
  `ontology_05`, random tangent passes one seed in each tested cell, so
  branch-specific scoring remains necessary.

## Next Probe

Vary the semantic target set to test whether the exceptions are target-set
sensitive or truly branch-geometric:

- For `identity_04`, compare the current identity target set with object/door,
  mirror/mask, and generic ontology-control vocabularies.
- For `ontology_05`, compare the current ontology target set with moon/celestial,
  map/road, and generic surreal-control vocabularies.

## Identity Target-Set Sensitivity

The first target-set sensitivity probe reran the strongest identity reverse cell
(`identity_04`, L7/k16, `alpha=0.8`, seeds `0..4`) with three alternate scoring
vocabularies:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_identity04_l7_k16_mirror_mask_a08_s04 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids identity_04 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json \
  --target-set-key identity_mirror_mask
```

The same command was used for `identity_door_object` and
`identity_generic_control` by changing `--target-set-key` and `--output-root`.

The comparison plot is regenerated with:

```bash
python3 scripts/plot_hltd_target_sensitivity.py \
  --source identity_stress=spiral_out_hltd_closed_loop_reverse_identity04_passing_band_l5_l7_k16_k24_a08_s04 \
  --source identity_door_object=spiral_out_hltd_closed_loop_target_sensitivity_identity04_l7_k16_door_object_a08_s04 \
  --source identity_mirror_mask=spiral_out_hltd_closed_loop_target_sensitivity_identity04_l7_k16_mirror_mask_a08_s04 \
  --source identity_generic_control=spiral_out_hltd_closed_loop_target_sensitivity_identity04_l7_k16_generic_control_a08_s04 \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_summary \
  --prompt-id identity_04 \
  --layer 7 \
  --k 16 \
  --component negative_coexact \
  --output-prefix target_sensitivity_identity04_l7_k16_full
```

![Identity target-set sensitivity](../spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_identity04_l7_k16_full.png)

Five-seed target-set sensitivity:

| target set | branch gate | branch-specific gate | random gate | target margin | target-random | drift | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity_stress | 1.00 | 1.00 | 0.00 | 0.2482 | 0.3630 | 0.50 | original identity target passes cleanly |
| identity_door_object | 1.00 | 1.00 | 0.20 | 0.3595 | 0.3099 | 0.50 | object/door vocabulary also passes |
| identity_mirror_mask | 1.00 | 1.00 | 0.00 | 0.1809 | 0.3599 | 0.50 | mirror/mask vocabulary also passes |
| identity_generic_control | 0.00 | 0.00 | 0.20 | -0.3247 | -0.4199 | 0.50 | drift remains, but target direction fails |

This is a useful specificity pattern. The negative-coexact branch changes the
tokens at the same rate for all target sets (`0.50`), but the target margin only
supports identity/object/mirror vocabularies. The generic-control vocabulary
turns the target margin negative, while the matched random tangent has the same
small drift baseline (`0.10`). That makes the current read stronger than "the
branch merely causes any decoded movement": the decoded movement is target-set
selective in this identity reverse cell.

## Ontology Target-Set Sensitivity

The same target-set sensitivity probe was then run on the strongest ontology
reverse cell (`ontology_05`, L8/k16, `alpha=0.8`, seeds `0..4`):

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_ontology05_l8_k16_moon_celestial_a08_s04 \
  --layers 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids ontology_05 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json \
  --target-set-key ontology_moon_celestial
```

The same command was used for `ontology_map_road` and
`ontology_surreal_control` by changing `--target-set-key` and `--output-root`.

The comparison plot is regenerated with:

```bash
python3 scripts/plot_hltd_target_sensitivity.py \
  --source ontology_collapse=spiral_out_hltd_closed_loop_reverse_ontology05_passing_band_l7_l8_k12_k24_a08_s04 \
  --source ontology_moon_celestial=spiral_out_hltd_closed_loop_target_sensitivity_ontology05_l8_k16_moon_celestial_a08_s04 \
  --source ontology_map_road=spiral_out_hltd_closed_loop_target_sensitivity_ontology05_l8_k16_map_road_a08_s04 \
  --source ontology_surreal_control=spiral_out_hltd_closed_loop_target_sensitivity_ontology05_l8_k16_surreal_control_a08_s04 \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_summary \
  --prompt-id ontology_05 \
  --layer 8 \
  --k 16 \
  --component negative_coexact \
  --output-prefix target_sensitivity_ontology05_l8_k16_full
```

![Ontology target-set sensitivity](../spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_ontology05_l8_k16_full.png)

Five-seed target-set sensitivity:

| target set | branch gate | branch-specific gate | random gate | target margin | target-random | drift | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ontology_collapse | 1.00 | 0.80 | 0.20 | 0.4380 | 0.4419 | 0.50 | original ontology target passes |
| ontology_moon_celestial | 1.00 | 0.80 | 0.20 | 1.0242 | 1.0779 | 0.50 | strongest semantic support; moon-local |
| ontology_map_road | 1.00 | 0.60 | 0.20 | 0.2399 | 0.1844 | 0.50 | passes but weaker and less specific |
| ontology_surreal_control | 0.00 | 0.00 | 0.00 | -0.3987 | -0.4271 | 0.50 | drift remains, but target direction fails |

This mirrors the identity result. The negative-coexact branch keeps the same
decoded drift rate (`0.50`) across all ontology target sets, but the target
margin is strongly positive for moon/celestial, positive but weaker for
map/road, and negative for the generic surreal-control vocabulary. The matched
random tangent stays at `0.20` drift with near-zero or negative target support.
So the L8 ontology reverse branch is not only a generic closed-loop breaker; it
preferentially steers into the moon/ontology semantic neighborhood measured by
the target vocabulary.

## Reverse Exception Specificity Figure

The two target-set probes can be compressed into one figure:

```bash
python3 scripts/plot_hltd_reverse_specificity.py \
  --panel 'identity_04 L7/k16=spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_identity04_l7_k16_full.csv' \
  --panel 'ontology_05 L8/k16=spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_ontology05_l8_k16_full.csv' \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_summary \
  --component negative_coexact \
  --output-prefix reverse_exception_specificity_identity_ontology
```

![Reverse exception specificity](../spiral_out_hltd_closed_loop_target_sensitivity_summary/reverse_exception_specificity_identity_ontology.png)

This figure shows the shared specificity pattern directly:

- In both exception cells, decoded token drift remains fixed at `0.50` across
  semantic and control target sets.
- Branch-specific gate support is high for semantically aligned vocabularies
  and collapses to `0.00` for the generic control vocabulary.
- Target-random semantic margin is positive for aligned vocabularies and
  negative for the controls.

The result is a stronger reverse-branch read than a surface-drift claim. The
negative-coexact reverse exceptions are branch-geometric enough to move tokens,
but their measured semantic direction is target-set selective rather than
generic.
