# HLTD Branch Hodge Ledger

This note reconnects the structural graph-Hodge runs with the later causal
steering and learned-probe branches.

## Command

```bash
python3 scripts/summarize_hltd_branch_hodge.py \
  --hodge-root spiral_out_hltd_invariance \
  --steering-root spiral_out_hltd_dissociation_steering_ksweep_mps \
  --probe-root spiral_out_hltd_dissociation_probe_ksweep_mps \
  --output-root spiral_out_hltd_branch_hodge \
  --closed-loop-roots \
    spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08 \
    spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08 \
  --topology triangles \
  --k 16 \
  --structural-ks 12 16 24 \
  --causal-ks 12 16 24 \
  --layers 4 5 6 7 8 \
  --selector middle \
  --compare-selectors middle max_component \
  --reverse-specificity-csv spiral_out_hltd_closed_loop_target_sensitivity_summary/reverse_exception_specificity_identity_ontology.csv
```

Outputs:

- `spiral_out_hltd_branch_hodge/hodge_family_k.csv`
- `spiral_out_hltd_branch_hodge/hodge_family_k_sweep.csv`
- `spiral_out_hltd_branch_hodge/hodge_k_sweep.csv`
- `spiral_out_hltd_branch_hodge/hodge_topology_family_k.csv`
- `spiral_out_hltd_branch_hodge/hodge_family_layer.csv`
- `spiral_out_hltd_branch_hodge/hodge_layer.csv`
- `spiral_out_hltd_branch_hodge/causal_hodge_join.csv`
- `spiral_out_hltd_branch_hodge/causal_k_scoreboard.csv`
- `spiral_out_hltd_branch_hodge/selector_delta_scoreboard.csv`
- `spiral_out_hltd_branch_hodge/family_branch_join.csv`
- `spiral_out_hltd_branch_hodge/branch_scoreboard.csv`
- `spiral_out_hltd_branch_hodge/closed_loop_prompt_join.csv`
- `spiral_out_hltd_branch_hodge/closed_loop_branch_scoreboard.csv`
- `spiral_out_hltd_branch_hodge/branch_role_summary.csv`
- `spiral_out_hltd_branch_hodge/branch_role_diagnostics.csv`
- `spiral_out_hltd_branch_hodge/branch_layer_condition_summary.csv`
- `spiral_out_hltd_branch_hodge/branch_layer_transition_summary.csv`
- `spiral_out_hltd_branch_hodge/branch_condition_summary.csv`
- `spiral_out_hltd_branch_hodge/branch_band_candidate_scoreboard.csv`
- `spiral_out_hltd_branch_hodge/reverse_exception_specificity.csv`
- `spiral_out_hltd_branch_hodge/summary_report.md`

## Branch Plots

The ledger can be rendered into structural and causal branch plots with:

```bash
python3 scripts/plot_hltd_branch_hodge.py \
  --summary-root spiral_out_hltd_branch_hodge \
  --output-dir spiral_out_hltd_branch_hodge/plots \
  --probe ontology_collapse \
  --selector middle \
  --components coexact coexact_minus_presence presence presence_plus_coexact negative_coexact
```

Outputs:

- `spiral_out_hltd_branch_hodge/plots/hodge_layer_spine.png`
- `spiral_out_hltd_branch_hodge/plots/hodge_k_sweep.png`
- `spiral_out_hltd_branch_hodge/plots/hodge_topology_contrast.png`
- `spiral_out_hltd_branch_hodge/plots/ontology_collapse_middle_causal_split.png`
- `spiral_out_hltd_branch_hodge/plots/ontology_collapse_middle_branch_phase.png`
- `spiral_out_hltd_branch_hodge/plots/closed_loop_branch_specific_scoreboard.png`
- `spiral_out_hltd_branch_hodge/plots/branch_role_summary.png`
- `spiral_out_hltd_branch_hodge/plots/branch_role_matrix.png`
- `spiral_out_hltd_branch_hodge/plots/family_branch_atlas.png`
- `spiral_out_hltd_branch_hodge/plots/branch_role_diagnostics.png`
- `spiral_out_hltd_branch_hodge/plots/branch_layer_condition_summary.png`
- `spiral_out_hltd_branch_hodge/plots/branch_layer_transition_summary.png`
- `spiral_out_hltd_branch_hodge/plots/branch_condition_summary.png`
- `spiral_out_hltd_branch_hodge/plots/branch_band_candidate_scoreboard.png`
- `spiral_out_hltd_branch_hodge/plots/closed_loop_prompt_branch_heatmap.png`
- `spiral_out_hltd_branch_hodge/plots/reverse_exception_specificity.png`
- `spiral_out_hltd_branch_hodge/plots/plot_manifest.json`

![Hodge layer spine](../spiral_out_hltd_branch_hodge/plots/hodge_layer_spine.png)

![Hodge k sweep](../spiral_out_hltd_branch_hodge/plots/hodge_k_sweep.png)

![Hodge topology contrast](../spiral_out_hltd_branch_hodge/plots/hodge_topology_contrast.png)

![Ontology causal branch split](../spiral_out_hltd_branch_hodge/plots/ontology_collapse_middle_causal_split.png)

![Ontology branch phase](../spiral_out_hltd_branch_hodge/plots/ontology_collapse_middle_branch_phase.png)

![Closed-loop branch-specific scoreboard](../spiral_out_hltd_branch_hodge/plots/closed_loop_branch_specific_scoreboard.png)

![Branch role summary](../spiral_out_hltd_branch_hodge/plots/branch_role_summary.png)

![Branch role matrix](../spiral_out_hltd_branch_hodge/plots/branch_role_matrix.png)

![Family/probe branch atlas](../spiral_out_hltd_branch_hodge/plots/family_branch_atlas.png)

![Family-local branch role diagnostics](../spiral_out_hltd_branch_hodge/plots/branch_role_diagnostics.png)

![Branch condition layer spine](../spiral_out_hltd_branch_hodge/plots/branch_layer_condition_summary.png)

![Branch layer transitions](../spiral_out_hltd_branch_hodge/plots/branch_layer_transition_summary.png)

![Family-component branch conditions](../spiral_out_hltd_branch_hodge/plots/branch_condition_summary.png)

![Branch-band candidate scoreboard](../spiral_out_hltd_branch_hodge/plots/branch_band_candidate_scoreboard.png)

![Closed-loop prompt branch heatmap](../spiral_out_hltd_branch_hodge/plots/closed_loop_prompt_branch_heatmap.png)

![Reverse exception specificity](../spiral_out_hltd_branch_hodge/plots/reverse_exception_specificity.png)

The phase plot gives the cleanest branch map in one figure:

- `coexact` stays on the traversal-positive side across k.
- `presence` stays stabilization-positive but traversal-negative/weak.
- `presence_plus_coexact` lands in the combined traversal/stabilization quadrant.
- `coexact_minus_presence` keeps traversal while dropping stabilization.
- `negative_coexact` remains the sign-control branch, with prompt-local reverse
  exceptions rather than family-wide traversal support.

The closed-loop scoreboard adds the stricter autoregressive gate from
`docs/hltd_closed_loop_gate.md`. It keeps raw branch gates separate from
`branch_specific_gate_rate`, which requires matched random-tangent
drift/target advantage at the same prompt/layer/k/seed/alpha.

The role summary compresses the same ledger into one branch map: x is mean
next-token traversal over the causal k-sweep, y is mean ontology-probe margin,
point size is closed-loop branch-specific gate support, and color is
closed-loop target margin advantage over random tangent.

The role matrix extends that read across probes. The top panel shows
coexact-like branches carrying next-token traversal across all probes; the
middle panel shows presence carrying probe stabilization, with
`presence_plus_coexact` combining traversal and stabilization most clearly for
ontology. The bottom panel now has strict closed-loop gates for
`affordance_stress`, `identity_stress`, and `ontology_collapse` across all five
branches, including the `-coexact` sign-control branch. The prompt heatmap makes
the exception structure explicit: `-coexact` is clean for the measured
affordance prompts, while only `identity_04` and `ontology_05` pass the strict
reverse-branch gate in the full-family sign-control runs. The
reverse-specificity panel keeps those prompt-local exceptions in the same
ledger: for both exception cells, `-coexact` preserves `0.50` token drift across
aligned and control target vocabularies, but the target-random margin turns
negative for the generic control vocabularies.

The family/probe branch atlas keeps the layer-averaged family structure visible
instead of compressing it into a single role row. It shows the strongest
next-token traversal in the `ontology_collapse` family, where `coexact`,
`coexact_minus_presence`, and `presence_plus_coexact` all move positive while
`-coexact` moves negative. The literal family is the counterpoint: presence and
`presence_plus_coexact` are the next-token positive branches there, while the
semantic-margin panel remains negative for the coexact-like branches. This is a
useful warning that a branch label is not globally semantic by itself; the
family-local geometry still matters.

The branch-role diagnostics panel turns that warning into a mechanical gate.
It scores each family/probe/component cell against a simple expected role map:
`coexact` should carry traversal, `presence` should carry stabilization,
`presence_plus_coexact` should combine the two, `coexact_minus_presence` should
carry traversal without probe stabilization, and `-coexact` should behave like
a reverse/control direction. In the current ledger, `41/60` cells pass. The
cleanest columns are `coexact` and `-coexact`; the informative breaks cluster
in the composite branches. `presence_plus_coexact` often becomes pure
traversal in `ontology_collapse`, and `coexact_minus_presence` flips negative
in `literal_stable` and `identity_stress`. That is exactly the place to look
next if we want branch roles to become conditional rules rather than global
labels.

The family-component condition panel is the first version of those conditional
rules. It compresses the diagnostics over probes and marks each family/branch
as `stable`, `mostly`, `mixed`, or `partial`. Two branches are currently
family-stable: `coexact` as traversal and `-coexact` as reverse/control.
`presence` is stable except in `literal_stable`, where it becomes a combined
branch because next-token movement turns positive. The conditional branches are
more selective: `coexact_minus_presence` holds in `ontology_collapse` and mostly
in `metaphor_shift`, but reverses in `identity_stress` and `literal_stable`;
`presence_plus_coexact` is mostly combined only in `metaphor_shift` and becomes
pure traversal in `ontology_collapse`.

The layer condition spine expands those rules back over L4-L8. This is the
more honest view when a family-level row looks stable: `coexact` and
`-coexact` are globally stable after probe averaging, but `coexact` still drops
at selected layers in `identity_stress`, `literal_stable`, and
`metaphor_shift`; `-coexact` flips at literal L4/L6. The ontology family is the
cleanest coexact spine: `coexact` passes at every measured layer, and
`coexact_minus_presence` passes from L5 onward. The largest composite-branch
failure remains `presence_plus_coexact`: it is early/mid mostly-good in
`metaphor_shift`, but in `ontology_collapse` it becomes traversal without probe
stabilization, especially at L7-L8.

The branch layer transition table compresses that spine into stable and broken
layer spans. The cleanest turn-on bands are `ontology_collapse/coexact`
(`L4-L8`), `ontology_collapse/coexact_minus_presence` (`L5-L8` after a mixed
L4), and `metaphor_shift/coexact` (`L5-L8` after an L4 break). The most
fragmented rows are the identity and literal controls: `identity_stress/coexact`
is stable at `L4-L5` and `L7` but breaks at `L6/L8`, while literal `coexact`
only holds at `L5-L6`. This suggests the next causal pass should treat layer
bands as branch conditions rather than picking a single middle-layer default.

The branch-band candidate scoreboard turns that suggestion into an explicit
queue. It joins each family/component transition row with weighted closed-loop
branch-specific support. The top rows are mostly structural rather than fully
causal: `identity_stress/-coexact`, `ontology_collapse/coexact`,
`ontology_collapse/-coexact`, and `metaphor_shift/-coexact` all have full
L4-L8 structural bands, but only modest closed-loop gates. The most interesting
causal exception is `identity_stress/coexact`: its stable layers are fragmented
(`L4-L5 L7`), yet it has positive closed-loop gate and target-random support.
That makes it a good next probe for whether narrow layer bands carry stronger
autoregressive specificity than the global structural rows.

The candidate queue can be converted into closed-loop follow-up commands with:

```bash
python3 scripts/plan_hltd_branch_band_runs.py \
  --candidate-csv spiral_out_hltd_branch_hodge/branch_band_candidate_scoreboard.csv \
  --output-root spiral_out_hltd_branch_band_plan \
  --run-output-root spiral_out_hltd_branch_band_runs \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --suite data/hltd_prompt_suite.jsonl \
  --target-set-file data/hltd_semantic_targets.json \
  --k 16 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --top-n 8 \
  --min-priority 0.25 \
  --device mps
```

This writes:

- `spiral_out_hltd_branch_band_plan/branch_band_run_plan.csv`
- `spiral_out_hltd_branch_band_plan/branch_band_run_plan.md`
- `spiral_out_hltd_branch_band_plan/run_branch_band_plan.sh`
- `spiral_out_hltd_branch_band_plan/rank_scripts/run_rank_*.sh`
- `spiral_out_hltd_branch_band_plan/branch_band_run_plan_manifest.json`

The generated shell script also runs the combined result summarizer after all
branch-band follow-ups finish:

```bash
python3 scripts/summarize_hltd_branch_band_runs.py \
  --plan-csv spiral_out_hltd_branch_band_plan/branch_band_run_plan.csv \
  --output-root spiral_out_hltd_branch_band_results
```

That result pass writes:

- `spiral_out_hltd_branch_band_results/branch_band_result_scoreboard.csv`
- `spiral_out_hltd_branch_band_results/branch_band_layer_result_summary.csv`
- `spiral_out_hltd_branch_band_results/branch_band_result_report.md`
- `spiral_out_hltd_branch_band_results/plots/branch_band_result_scoreboard.png`
- `spiral_out_hltd_branch_band_results/plots/branch_band_layer_result_summary.png`
  when completed layer rows exist
- `spiral_out_hltd_branch_band_results/plots/plot_manifest.json`

![Branch-band result scoreboard](../spiral_out_hltd_branch_band_results/plots/branch_band_result_scoreboard.png)

Before the branch-band runs are executed, the result scoreboard intentionally
contains `missing_run` rows. After any subset of roots is produced and
summarized, the same command fills in branch-specific gate, random gate,
token-drift, and target-random support for the completed rows while leaving the
rest visible as pending.

The result plot uses gray bars for pending rows, component-colored outlines for
branch identity, a vertical tick for the planned closed-loop gate from earlier
runs, and filled points for newly measured branch-specific gates once the
follow-up roots exist. If layer-level rows are present, the layer plot shows
branch-specific gate and target-random support over the exact planned layer
band.

`run_branch_band_plan.sh` is resume-safe. For each branch-band root it checks:

- `closed_loop_metrics.csv`: skips the expensive closed-loop run if metrics
  already exist
- `closed_loop_prompt_layer_k_summary.csv`: skips summarization if the closed
  loop has already been summarized
- `plots/plot_manifest.json`: skips per-root plotting if plots already exist

This means the full branch-band queue can be restarted after interruption
without re-running completed branch roots. The final combined result summary
and result plot are always refreshed.

For narrower execution, use the generated per-rank scripts. They run exactly
one branch-band row and then refresh the combined result table/plot. For
example, the current narrow causal-exception probe is:

```bash
bash spiral_out_hltd_branch_band_plan/rank_scripts/run_rank_07__identity_stress__coexact.sh
```

The current top-8 follow-up plan is:

| rank | family | component | layers | read |
| ---: | --- | --- | --- | --- |
| 1 | `identity_stress` | `negative_coexact` | L4-L8 | full-band sign-control stress test |
| 2 | `ontology_collapse` | `coexact` | L4-L8 | structural coexact band with weak current closed-loop target support |
| 3 | `ontology_collapse` | `negative_coexact` | L4-L8 | ontology reverse-control exception check |
| 4 | `metaphor_shift` | `negative_coexact` | L4-L8 | unmeasured closed-loop sign-control band |
| 5 | `ontology_collapse` | `coexact_minus_presence` | L5-L8 | strongest current ontology closed-loop gate, but target-random still near zero |
| 6 | `metaphor_shift` | `coexact` | L5-L8 | structural metaphor swirl band needing closed-loop coverage |
| 7 | `identity_stress` | `coexact` | L4-L5 L7 | narrow causal-exception band with positive target-random support |
| 8 | `literal_stable` | `negative_coexact` | L5 L7-L8 | literal sign-control narrow-band probe |

The first layer/k localization of those two reverse-branch exceptions is in:

```text
docs/hltd_reverse_exception_localization.md
```

It keeps the targeted exception rows separate from the family-level branch
ledger, because mixing only exception prompts into the main matrix would bias
family-level closed-loop rates.

The five-seed passing-band follow-up keeps the split: `identity_04` is strongest
at L7/k16-24, while `ontology_05` strengthens at L8 across k=12/16/24.

## Structural Branches

At `k=16`, triangles enabled:

| Family | exact/presence L5-L8 | coexact L5-L8 | harmonic max | coexact-shuffle |
| --- | ---: | ---: | ---: | ---: |
| literal_stable | 0.1426 | 0.8574 | 0.0000 | 0.0348 |
| metaphor_shift | 0.1252 | 0.8748 | 0.0000 | 0.0782 |
| identity_stress | 0.1199 | 0.8801 | 0.0000 | 0.0816 |
| ontology_collapse | 0.1113 | 0.8887 | 0.0000 | 0.1004 |

The structural Hodge branch remains coexact-dominant in all prompt families,
with the largest real-minus-shuffle separation in `ontology_collapse`.

The structural branch also survives the current `k=12/16/24` sweep:

| k | exact/presence mean | coexact mean | harmonic max | coexact-shuffle mean | coexact-random mean | reverse gap max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.1093 | 0.8907 | 0.0016 | 0.0876 | 0.2333 | 0.0000 |
| 16 | 0.1247 | 0.8753 | 0.0000 | 0.0737 | 0.1576 | 0.0000 |
| 24 | 0.1520 | 0.8480 | 0.0000 | 0.0535 | 0.2702 | 0.0000 |

So the coexact branch is not a single-k accident. Increasing `k` raises the
exact/presence share and lowers coexact somewhat, which fits the idea that a
larger neighborhood smooths more of the local circulation into basin-like
structure. The same-graph reverse branch remains invariant in this summary.

## Causal k-Sweep

The dissociation steering/probe gates were rerun with the same layers,
components, seeds, and middle-token selector at `k=12/16/24`:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_dissociation_steering_ksweep_mps \
  --layers 4 5 6 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_dissociation_probe_ksweep_mps \
  --layers 4 5 6 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent
```

Run sizes:

- steering: 300 prompt/layer/k runs, 7200 rows, 149.4 seconds
- probe: 21600 rows, 69.2 seconds

Middle-token ontology branch read:

| k | component | Hodge coexact | next | semantic margin | ontology probe |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | coexact | 0.8907 | 0.3523 | -0.0460 | -0.2715 |
| 12 | presence | 0.8907 | -0.1188 | -0.0708 | 1.2362 |
| 12 | presence_plus_coexact | 0.8907 | 0.3464 | -0.0231 | 0.4156 |
| 16 | coexact | 0.8753 | 0.3593 | -0.1481 | -0.0230 |
| 16 | presence | 0.8753 | -0.1076 | -0.2442 | 1.1151 |
| 16 | presence_plus_coexact | 0.8753 | 0.3735 | -0.1356 | 0.6041 |
| 24 | coexact | 0.8480 | 0.3784 | -0.0672 | 0.2227 |
| 24 | presence | 0.8480 | -0.1073 | -0.1866 | 0.7811 |
| 24 | presence_plus_coexact | 0.8480 | 0.3156 | -0.0398 | 0.4627 |

This makes the branch split more robust:

- `coexact` keeps positive next-token traversal at every tested `k`.
- `presence` keeps negative/weak next-token support but positive learned-probe
  stabilization at every tested `k`.
- `presence_plus_coexact` remains the combined branch: positive next-token
  support plus positive ontology-probe margin.
- `coexact_minus_presence` keeps some next-token support but remains negative
  on ontology-probe margin.
- `negative_coexact` remains a useful sign check because it is strongly
  negative for next-token support at all tested `k`.

## Selector Comparison

The k-sweep outputs include both `middle` and `max_component` token selectors.
The ledger now writes:

```text
spiral_out_hltd_branch_hodge/selector_delta_scoreboard.csv
```

where `max_component - middle` is reported for each branch. Ontology branch
read:

| k | component | middle next | max next | next diff | middle probe | max probe | probe diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | coexact | 0.3523 | 0.0908 | -0.2615 | -0.2715 | -0.0012 | 0.2703 |
| 12 | presence | -0.1188 | 0.1414 | 0.2602 | 1.2362 | 0.9836 | -0.2526 |
| 12 | presence_plus_coexact | 0.3464 | 0.1582 | -0.1882 | 0.4156 | 0.2097 | -0.2059 |
| 16 | coexact | 0.3593 | 0.0489 | -0.3104 | -0.0230 | -0.0868 | -0.0638 |
| 16 | presence | -0.1076 | 0.0723 | 0.1800 | 1.1151 | 1.4586 | 0.3435 |
| 16 | presence_plus_coexact | 0.3735 | 0.0796 | -0.2939 | 0.6041 | 0.1357 | -0.4684 |
| 24 | coexact | 0.3784 | 0.1089 | -0.2695 | 0.2227 | 0.4284 | 0.2058 |
| 24 | presence | -0.1073 | 0.1217 | 0.2290 | 0.7811 | 1.4892 | 0.7081 |
| 24 | presence_plus_coexact | 0.3156 | 0.1299 | -0.1857 | 0.4627 | 0.8070 | 0.3444 |

The selector read is now:

- `middle` is the cleaner traversal selector for coexact and
  `presence_plus_coexact`; it gives larger next-token support across k.
- `max_component` can increase presence/probe stabilization, especially at
  larger k, but it weakens coexact next-token transport.
- This means selector choice is not just convenience. It changes which branch
  aspect is emphasized: trajectory transport near a stable token position
  versus locally strongest component norm.

## Closed-Loop Branch-Specific Ledger

The branch ledger now also ingests closed-loop prompt summaries:

```text
spiral_out_hltd_branch_hodge/closed_loop_branch_scoreboard.csv
```

For `ontology_collapse` at structural `k=16`, the family-level Hodge branch is
strongly coexact (`0.8887`) while the stricter closed-loop gate separates branch
effects from random-tangent prompt sensitivity:

| source | component | raw gate | branch-specific gate | target-random | read |
| --- | --- | ---: | ---: | ---: | --- |
| five_prompt | coexact_minus_presence | 0.8000 | 0.6000 | 0.0933 | strongest current closed-loop branch |
| five_prompt | coexact | 0.6000 | 0.4000 | 0.0039 | traversal-positive but less specific |
| five_prompt | presence_plus_coexact | 0.2000 | 0.2000 | -0.1730 | one prompt-specific combined branch |
| five_prompt | presence | 0.0000 | 0.0000 | -0.2134 | not a traversal branch |
| ontology01_05 | coexact_minus_presence | 0.5000 | 0.4000 | -0.1030 | positive prompt-specific cells, but averaged down by `ontology_01` |
| ontology01_05 | coexact | 0.0000 | 0.0000 | -0.3175 | no closed-loop branch-specific support |
| identity5 | coexact | 0.4000 | 0.4000 | 0.1549 | strongest identity closed-loop branch |
| identity5 | coexact_minus_presence | 0.2000 | 0.2000 | 0.2709 | larger target advantage, fewer gate cells |
| identity5 | presence_plus_coexact | 0.2000 | 0.2000 | 0.0073 | prompt-specific branch |
| identity5 | presence | 0.0000 | 0.0000 | -0.0593 | stabilization branch |
| affordance5 | coexact | 0.4000 | 0.2000 | 0.0193 | weak positive coexact target advantage |
| affordance5 | coexact_minus_presence | 0.2000 | 0.2000 | 0.0486 | weak positive target advantage |
| affordance5 | presence | 0.2000 | 0.2000 | -0.0671 | one strict cell, negative mean target advantage |
| affordance5 | presence_plus_coexact | 0.2000 | 0.2000 | -0.1357 | one strict cell, not robust target-positive |
| affordance01_03_seed | coexact | 1.0000 | 0.7000 | 0.0830 | seed-stable coexact branch |
| affordance01_03_seed | coexact_minus_presence | 0.5000 | 0.5000 | 0.0333 | seed-stable on `affordance_03` |
| affordance01_03_seed | presence | 0.5000 | 0.5000 | 0.0118 | seed-stable on `affordance_01` |
| affordance01_03_seed | presence_plus_coexact | 0.5000 | 0.5000 | 0.0016 | seed-stable on `affordance_01` |
| sign_control_identity5 | negative_coexact | 0.2000 | 0.2000 | 0.0406 | prompt-specific reverse-branch exception at `identity_04` |
| sign_control_ontology5 | negative_coexact | 0.2000 | 0.0800 | 0.0692 | prompt-specific reverse-branch exception at `ontology_05` |
| sign_control_affordance01_03 | negative_coexact | 0.0000 | 0.0000 | -0.0775 | clean sign-control |

This closes the loop between the structural and autoregressive reads: the
hidden-state field is structurally coexact-dominant in the ontology and
identity families, but closed-loop branch specificity appears only in selected
prompt cells. The affordance add-on becomes clearer after seed probing:
`affordance_01` is presence/combined-specific, while `affordance_03` is
coexact/coexact-minus-specific. The `negative_coexact` sign-control is mostly
suppressive: affordance is clean, identity has a single reverse-branch exception
at `identity_04`, and ontology has a lower-rate exception at `ontology_05`.

## Branch Role Summary

The ledger now writes:

```text
spiral_out_hltd_branch_hodge/branch_role_summary.csv
```

This table averages the causal k-sweep over `k=12/16/24` for the `middle`
selector, then joins the closed-loop strict gate. The role labels are only a
compact readout; the numeric columns are the evidence.

| component | role | Hodge coexact | mean next | ontology probe | closed-loop specific | target-random |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coexact | traversal | 0.8713 | 0.3634 | -0.0240 | 0.1333 | -0.2104 |
| coexact_minus_presence | closed_loop_traversal | 0.8713 | 0.2135 | -0.5629 | 0.4667 | -0.0376 |
| negative_coexact | weak_or_control | 0.8713 | -0.5530 | -0.5731 | 0.0800 | 0.0692 |
| presence | stabilization | 0.8713 | -0.1112 | 1.0441 | 0.0000 | -0.3352 |
| presence_plus_coexact | combined | 0.8713 | 0.3452 | 0.4941 | 0.0667 | -0.3589 |

Identity-stress rows in the same file add a useful family contrast:

| component | role | mean next | identity probe | closed-loop specific | target-random |
| --- | --- | ---: | ---: | ---: | ---: |
| coexact | closed_loop_traversal | 0.3634 | -0.6972 | 0.4000 | 0.1549 |
| coexact_minus_presence | traversal | 0.2135 | -1.3045 | 0.2000 | 0.2709 |
| presence | stabilization | -0.1112 | 1.1195 | 0.0000 | -0.0593 |
| presence_plus_coexact | traversal | 0.3452 | -0.0428 | 0.2000 | 0.0073 |
| negative_coexact | stabilization | -0.5530 | 0.2337 | 0.2000 | 0.0406 |

Affordance-stress rows fill the remaining active closed-loop row:

| component | role | mean next | affordance probe | closed-loop specific | target-random |
| --- | --- | ---: | ---: | ---: | ---: |
| coexact | closed_loop_traversal | 0.3634 | -0.6245 | 0.5333 | 0.0618 |
| coexact_minus_presence | closed_loop_traversal | 0.2135 | -0.7826 | 0.4000 | 0.0384 |
| presence | stabilization | -0.1112 | 0.9008 | 0.4000 | -0.0148 |
| presence_plus_coexact | combined_closed_loop | 0.3452 | 0.0624 | 0.4000 | -0.0442 |
| negative_coexact | stabilization | -0.5530 | 0.4886 | 0.0000 | -0.0775 |

The sign-control rows complete the branch matrix:

| probe | negative-coexact role | next | probe margin | closed-loop specific | target-random |
| --- | --- | ---: | ---: | ---: | ---: |
| affordance_stress | stabilization/control | -0.5530 | 0.4886 | 0.0000 | -0.0775 |
| identity_stress | prompt-local exception | -0.5530 | 0.2337 | 0.2000 | 0.0406 |
| ontology_collapse | low-rate prompt-local exception | -0.5530 | -0.5731 | 0.0800 | 0.0692 |

The resulting branch map is now:

- `coexact`: clean next-token traversal, but weak closed-loop specificity in
  the current ontology closed-loop set; strongest strict closed-loop branch in
  the identity-stress set.
- `coexact_minus_presence`: traversal with presence removed; weaker one-step
  probe stabilization but strongest closed-loop branch-specific support.
- `presence`: stabilization/probe branch, not an autoregressive traversal
  branch.
- `presence_plus_coexact`: combined one-step traversal/stabilization; the
  affordance seed-probe gives it prompt-specific closed-loop support.
- `negative_coexact`: sign-control branch overall, but not a universal null.
  Reverse-branch support is prompt-local: `identity_04` and `ontology_05`, not
  the full identity or ontology families.

## All-Interior Branch Heatmap

The first all-interior structural localization pass is written up in:

```text
docs/hltd_branch_heatmap.md
```

It maps reconstructed branch vector norms across every centered token node.
Short read: coexact peaks are mid-to-late and family-specific, presence shifts
earlier as k increases, and `presence_plus_coexact` sits between them with the
largest peak/full ratios. This supports treating selector choice as a branch
localization question rather than a harmless sampling detail.

The position-binned all-interior causal/probe gate is in:

```text
docs/hltd_all_interior_position_gate.md
```

It now includes the full 20-prompt bundled suite at k=16 and the
`k=12/16/24` all-interior k-sweep with random-tangent seeds 0/1. The
cross-family peak read strengthens the dissociation: coexact-like branches
dominate early-phase next-token traversal, while presence dominates
ontology-probe margin at a different position phase.
The same note also renders branch-position heatmaps and peak plots for the
ontology-collapse probe.
It now also includes a selected-bin k16 seed sweep over bins 0/1/2/4 with
random-tangent seeds 0-7. That gate keeps `coexact_minus_presence` as the
cleanest traversal branch and `presence` as the strongest ontology-probe
stabilization branch.

Topology contrast is still essential:

| Topology | coexact L5-L8 | harmonic max | read |
| --- | ---: | ---: | --- |
| triangles | 0.8574-0.8887 | 0.0000 | local coexact branch is available |
| no_triangles | 0.0000 | 0.8937-0.9056 | residual is forced into harmonic |

The full radius-filtration follow-up in
[hltd_topology_filtration.md](hltd_topology_filtration.md) resolves the space
between those endpoints while holding each graph and edge flow fixed. It uses
all 20 prompts, L4/L5/L7, `k=12/16/24`, and eight seeds for each of two nulls:

| Matched Betti-1 | exact/presence | coexact | harmonic | non-exact |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.1164 | 0.0000 | 0.8836 | 0.8836 |
| 0.75 | 0.1164 | 0.1954 | 0.6904 | 0.8836 |
| 0.50 | 0.1164 | 0.5179 | 0.3751 | 0.8836 |
| 0.25 | 0.1164 | 0.7348 | 0.1434 | 0.8836 |
| 0.00 | 0.1164 | 0.8836 | 0.0000 | 0.8836 |

![Hodge topology filtration](../spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7/plots/topology_filtration_branch_persistence.png)

The exact/non-exact split is invariant to numerical precision, while coexact
and harmonic exchange the same non-exact field as cycles become boundaries
(`r=0.9794` between real harmonic ratio and normalized Betti-1). At matched
Betti-1 `0.5`, both finer branches separate from both null constructions under
a 20-prompt paired bootstrap. The harmonic gap nevertheless weakens with layer
and k; at `k=24` its vector-shuffle interval crosses zero. Thus the full-clique
result supports a useful **full-clique coexact** intervention branch, while the
interior harmonic remainder is a scale-localized causal candidate rather than
evidence of a complex-independent concept ring.

## Matched-Betti Causal Gate

The pre-registered `L5`, `k=16`, Betti-1 `0.5` follow-up is recorded in
[hltd_matched_betti_causal_gate.md](hltd_matched_betti_causal_gate.md).
It applies exact, coexact, and harmonic directions at the middle token for all
20 prompts and compares them with eight norm-matched random-tangent seeds.

At alpha 1, coexact has a positive observed-next-token gap of `+0.3532`, but
its 95% prompt-bootstrap interval is `[-0.0297, +0.7639]`. Its semantic
target-control gap is `-0.3519 [-0.7288, -0.0219]`. Harmonic has no positive
interval on KL, next-token support, or semantic margin. Thus matched topology
selects meaningful structural branches, but the current one-step middle-token
gate does not license semantic-circulation or concept-ring language. The next
test is a paired `+alpha/-alpha` gate that separates oriented and sign-even
effects.

## Layer Spine

Across families, the structural full-clique coexact branch is strongest at L5
and remains high through L8:

| Layer | exact/presence | coexact | coexact-shuffle | coexact-random |
| ---: | ---: | ---: | ---: | ---: |
| L4 | 0.1196 | 0.8804 | 0.0966 | 0.1743 |
| L5 | 0.1055 | 0.8945 | 0.0934 | 0.1828 |
| L6 | 0.1143 | 0.8857 | 0.0823 | 0.1792 |
| L7 | 0.1282 | 0.8718 | 0.0658 | 0.1421 |
| L8 | 0.1509 | 0.8491 | 0.0534 | 0.1260 |

## Causal Branches

The branch ledger makes the earlier dissociation sharper:

| Component | mean next | ontology probe mean | best ontology layer |
| --- | ---: | ---: | ---: |
| coexact | 0.3593 | -0.0230 | L8 |
| presence | -0.1076 | 1.1151 | L5 |
| presence_plus_coexact | 0.3735 | 0.6041 | L7 |
| coexact_minus_presence | 0.1924 | -0.6742 | L7 |
| negative_coexact | -0.5559 | -0.5740 | L6 |

Read:

> Structural Hodge under the full clique complex says the model's middle-layer
> token field is mostly coexact. Causal gates say coexact and coexact-derived
> directions carry
> next-token traversal, while presence-derived directions carry learned-probe
> stabilization. The sum branch can combine both effects.

This is now the working branch map:

- exact/presence: current-regime stabilization branch
- full-clique coexact: local traversal / next-token transport branch that can
  rotate away from a learned identity-transfer axis
- harmonic: topology-conditioned open-cycle residual unless an independently
  chosen complex preserves persistent holes
- presence plus coexact: combined traversal and stabilization branch
- coexact minus presence: traversal with stabilization removed

## Disjoint Counterfactual Identity Probe

The first prompt-disjoint learned-axis follow-up is recorded in
[hltd_counterfactual_identity_probe.md](hltd_counterfactual_identity_probe.md).
It trains on 12 matched artifact/creator counterfactual pairs and evaluates the
held-out `identity_02` statue/sculptor prompt.

![Disjoint counterfactual identity-probe branch surface](../spiral_out_hltd_identity02_counterfactual_probe_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/counterfactual_identity_stress_branch_surface.png)

Leave-pair-out accuracy is `0.771`, `0.782`, and `0.808` at L4, L5, and L7.
At L4-L5 and k=12/16, presence moves toward the learned identity-transfer
regime while coexact and coexact-minus-presence move away from it. The signs
weaken or reverse at k=24. L7 has the best held-out decodability but no branch
effect outside the matched random-tangent range, separating decodability from
causal alignment. Harmonic remains inactive under the current clique complex.

## Next Hodge Questions

1. Extend the disjoint counterfactual probe from `identity_02` to all held-out
   identity prompts, then score the learned axis after every closed-loop step.
   The current one-step gate supports oriented transport but not identity
   collapse.
2. Extend the prompt-held-out carrier beyond four generated tokens to test
   whether the branch advantage persists through a readable continuation.
3. Bootstrap the graph or resample trajectories so deterministic Hodge-branch
   uncertainty is measured separately from matched random-tangent uncertainty.
4. Run paired positive and negative steering at the pre-registered matched
   topology. The first one-step gate shows branch transfer and heterogeneous
   logit effects, not a concept ring.
5. Add selected-bin runs for a second model or a second prompt suite to check
   whether the branch split is GPT-2-specific.
6. Move the reverse-specificity gate beyond GPT-2 small, because the current
   `identity_04` and `ontology_05` evidence is now target-set selective but
   still model-local.
