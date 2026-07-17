# HLTD All-Interior Position Gate

This note records all-interior causal/probe gates using the branch heatmap bins.
The smoke gate used one prompt per family. The full gate uses all 20 bundled
prompts, GPT-2 layers 5/7/8, k=16, seed 0, and all centered interior token
nodes.

## Smoke Commands

Steering:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_steering_smoke_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 \
  --token-selectors all_interior \
  --max-prompts-per-family 1 \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Probe:

```bash
python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_probe_smoke_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 \
  --token-selectors all_interior \
  --max-prompts-per-family 1 \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent
```

Position-bin summary:

```bash
python3 scripts/summarize_hltd_position_gate.py \
  --steering-root spiral_out_hltd_all_interior_steering_smoke_mps \
  --probe-root spiral_out_hltd_all_interior_probe_smoke_mps \
  --output-root spiral_out_hltd_all_interior_position_smoke \
  --bins 12 \
  --token-selector all_interior
```

Outputs:

- `spiral_out_hltd_all_interior_position_smoke/steering_position_pairwise.csv`
- `spiral_out_hltd_all_interior_position_smoke/probe_position_pairwise.csv`
- `spiral_out_hltd_all_interior_position_smoke/steering_position_summary.csv`
- `spiral_out_hltd_all_interior_position_smoke/probe_position_summary.csv`
- `spiral_out_hltd_all_interior_position_smoke/joined_position_summary.csv`
- `spiral_out_hltd_all_interior_position_smoke/position_peak_summary.csv`
- `spiral_out_hltd_all_interior_position_smoke/position_cross_family_peak_summary.csv`
- `spiral_out_hltd_all_interior_position_smoke/summary_report.md`

Run sizes:

- steering: 12 prompt/layer/k runs, 43.2 seconds
- probe: 6,912 probe rows, 5.4 seconds
- position summary: 1,920 steering pairwise rows, 5,760 probe pairwise rows,
  2,160 joined position rows

## Full 20-Prompt Commands

The full gate drops `--max-prompts-per-family 1` and writes to full output
roots:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_steering_full_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 \
  --token-selectors all_interior \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_probe_full_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 \
  --token-selectors all_interior \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent

python3 scripts/summarize_hltd_position_gate.py \
  --steering-root spiral_out_hltd_all_interior_steering_full_mps \
  --probe-root spiral_out_hltd_all_interior_probe_full_mps \
  --output-root spiral_out_hltd_all_interior_position_full \
  --bins 12 \
  --token-selector all_interior
```

Full run sizes:

- steering: 60 prompt/layer/k runs, 132.0 seconds
- probe: 30,456 probe rows, 10.7 seconds
- position summary: 8,460 steering pairwise rows, 25,380 probe pairwise rows,
  2,160 joined position rows

## K-Sweep Commands

The k-sweep gate repeats the full all-interior run across `k=12/16/24` and
random-tangent seeds 0/1:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_steering_ksweep_mps \
  --layers 5 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors all_interior \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_all_interior_probe_ksweep_mps \
  --layers 5 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors all_interior \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent

python3 scripts/summarize_hltd_position_gate.py \
  --steering-root spiral_out_hltd_all_interior_steering_ksweep_mps \
  --probe-root spiral_out_hltd_all_interior_probe_ksweep_mps \
  --output-root spiral_out_hltd_all_interior_position_ksweep \
  --bins 12 \
  --token-selector all_interior
```

K-sweep run sizes:

- steering: 180 prompt/layer/k runs, 852.6 seconds
- probe: 182,736 probe rows, 55.9 seconds
- position summary: 50,760 steering pairwise rows, 152,280 probe pairwise
  rows, 6,480 joined position rows

## Plot Commands

After any position summary is written, render branch-position plots with:

```bash
python3 scripts/plot_hltd_position_gate.py \
  --summary-root spiral_out_hltd_all_interior_position_ksweep \
  --output-dir spiral_out_hltd_all_interior_position_ksweep/plots \
  --probe ontology_collapse \
  --components coexact coexact_minus_presence presence presence_plus_coexact
```

The k-sweep plots are:

- `spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_next_token_heatmap.png`
- `spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_probe_margin_heatmap.png`
- `spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_peak_bars.png`
- `spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_peak_phase.png`

![Ontology next-token heatmap](../spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_next_token_heatmap.png)

![Ontology probe-margin heatmap](../spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_probe_margin_heatmap.png)

![Ontology peak bars](../spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_peak_bars.png)

![Ontology peak phase](../spiral_out_hltd_all_interior_position_ksweep/plots/ontology_collapse_peak_phase.png)

## Smoke Peak Read

For the ontology probe, the best position bins in this small gate are:

| Family | Component | next peak | ontology-probe peak |
| --- | --- | --- | --- |
| literal_stable | coexact | L5 bin 2, +1.5095 | L7 bin 6, +3.1081 |
| literal_stable | presence | L5 bin 2, +1.4274 | L5 bin 7, +2.2116 |
| literal_stable | presence_plus_coexact | L5 bin 2, +1.4657 | L5 bin 3, +1.3794 |
| metaphor_shift | coexact | L8 bin 11, +1.3785 | L8 bin 0, +3.1079 |
| metaphor_shift | presence | L8 bin 2, +1.1895 | L7 bin 11, +5.4845 |
| metaphor_shift | presence_plus_coexact | L8 bin 2, +1.4875 | L7 bin 11, +4.2960 |
| identity_stress | coexact | L7 bin 3, +1.2746 | L8 bin 6, +2.3303 |
| identity_stress | presence | L8 bin 10, +1.1634 | L8 bin 6, +3.8121 |
| identity_stress | presence_plus_coexact | L7 bin 3, +1.4255 | L8 bin 6, +2.9321 |
| ontology_collapse | coexact | L8 bin 4, +1.8114 | L5 bin 4, +5.0332 |
| ontology_collapse | presence | L7 bin 0, +1.0895 | L5 bin 5, +8.5661 |
| ontology_collapse | presence_plus_coexact | L7 bin 0, +1.7763 | L5 bin 4, +5.5083 |

## Interpretation

The all-interior gate confirms that token position matters. The best
next-token bin and best ontology-probe bin often differ, even within the same
family/component. This means the earlier single-token selectors were sampling
different parts of a richer position field:

- coexact next-token effects can peak away from the structural coexact norm
  peak, especially in metaphor and ontology prompts.
- presence probe stabilization can be highly localized, with large positive
  bins that do not always coincide with next-token support.
- `presence_plus_coexact` remains useful, but the best hybrid position depends
  on whether the objective is traversal or probe stabilization.

This smoke gate motivated the full 20-prompt gate below. For larger future
sweeps, use these full-gate peaks as token-position priors instead of scoring
every interior token for every k/seed combination.

## Full Peak Read

The full gate changes the read from "interesting individual bins" to a clearer
branch dissociation:

| Component | cross-family next peak | cross-family ontology-probe peak |
| --- | --- | --- |
| coexact | L7 bin 0, +0.8761 | L7 bin 1, +1.3578 |
| coexact_minus_presence | L7 bin 0, +0.9304 | L7 bin 1, +0.6641 |
| presence | L8 bin 2, +0.2574 | L5 bin 4, +3.1477 |
| presence_plus_coexact | L7 bin 0, +0.7985 | L7 bin 1, +2.1565 |

Family-level ontology peaks still vary:

| Family | Component | next peak | ontology-probe peak |
| --- | --- | --- | --- |
| literal_stable | coexact | L5 bin 0, +1.3051 | L7 bin 6, +2.9826 |
| metaphor_shift | coexact | L8 bin 1, +1.3643 | L7 bin 1, +2.5519 |
| identity_stress | coexact | L7 bin 0, +1.3494 | L5 bin 1, +2.6379 |
| ontology_collapse | coexact | L7 bin 0, +1.7585 | L5 bin 0, +4.6852 |
| literal_stable | presence | L5 bin 2, +0.8405 | L5 bin 8, +3.2025 |
| metaphor_shift | presence | L8 bin 8, +0.5153 | L7 bin 1, +2.8728 |
| identity_stress | presence | L7 bin 0, +0.5469 | L7 bin 4, +3.7368 |
| ontology_collapse | presence | L7 bin 0, +0.9897 | L5 bin 0, +8.1657 |

## Full Interpretation

The all-interior gate now supports three working claims:

- `coexact` and `coexact_minus_presence` are the strongest cross-family
  next-token traversal branches. Their best average next-token bin is early
  L7/bin 0, and `coexact_minus_presence` is positive in every family at that
  bin.
- `presence` is a weak next-token traversal branch but a strong ontology-probe
  branch. Its best cross-family ontology probe bin is L5/bin 4, well after its
  best next-token bin.
- `presence_plus_coexact` behaves like a hybrid: next-token support follows
  the early coexact peak, while ontology-probe support follows the early probe
  peak more strongly than coexact alone.

The branch heatmap found coexact structural mass tending mid-to-late, while
this causal gate finds the strongest average next-token effect early. That
does not falsify the heatmap; it suggests the structural field and the causal
readout are not the same observable. The k-sweep below keeps that early-phase
coexact traversal read, but it softens the single-bin claim because the peak
migrates between L7/bin0 and L8/bin1.

## K-Sweep Read

The k-sweep keeps the branch dissociation but softens the exact location claim.
The safe statement is now early-phase coexact traversal, not a single fixed
L7/bin0 peak.

Cross-family ontology-gate peaks:

| Component | k | next peak | ontology-probe peak |
| --- | ---: | --- | --- |
| coexact | 12 | L8 bin 1, +0.9026 | L5 bin 8, +0.7833 |
| coexact | 16 | L8 bin 1, +0.8444 | L5 bin 8, +0.6833 |
| coexact | 24 | L7 bin 0, +0.7833 | L7 bin 6, +1.0769 |
| coexact_minus_presence | 12 | L8 bin 1, +0.7614 | L7 bin 6, +0.4648 |
| coexact_minus_presence | 16 | L7 bin 0, +0.8302 | L7 bin 6, +0.2790 |
| coexact_minus_presence | 24 | L8 bin 0, +0.6158 | L7 bin 6, +0.4074 |
| presence | 12 | L8 bin 0, +0.4062 | L5 bin 4, +2.4084 |
| presence | 16 | L7 bin 9, +0.2702 | L5 bin 4, +2.9042 |
| presence | 24 | L7 bin 9, +0.2807 | L8 bin 2, +2.9383 |
| presence_plus_coexact | 12 | L8 bin 1, +0.9021 | L5 bin 8, +1.3524 |
| presence_plus_coexact | 16 | L8 bin 1, +0.7653 | L5 bin 3, +1.4935 |
| presence_plus_coexact | 24 | L7 bin 0, +0.7189 | L5 bin 3, +1.4860 |

Seed-specific checks show the same broad pattern with some peak migration:

- `coexact` next-token peaks stay early across k, but seed 0 often prefers
  L7/bin0 while seed 1 often prefers L8/bin1.
- `coexact_minus_presence` has the cleanest positive next-token family range:
  k16 L7/bin0 is positive in all families, and k24 early peaks are also
  positive in all families.
- `presence` remains weaker for next-token traversal. Its best next peaks are
  later for k16/k24, while ontology-probe peaks stay much larger than coexact
  peaks.
- `presence_plus_coexact` follows the early coexact next-token peak and keeps
  a stronger probe margin than coexact alone.

So the robust k-sweep claim is:

> Coexact-like branches provide early-phase traversal under matched random
> tangent baselines across k, while presence provides stronger ontology-probe
> stabilization at a different position phase.

The next robustness gate should add a selected-bin rerun with more seeds
instead of all interior tokens. Good bins are early coexact bins
`L7/bin0`, `L8/bin1`, and presence probe bins `L5/bin4`, `L8/bin2`.

## Selected-Bin Seed Gate

The selected-bin gate adds `token_selector=position_bin` so promising
position bins can be rerun with more random-tangent seeds without scoring every
interior token. The `v2` outputs should be used for this gate because they
include explicit `token_count` metadata; selected-bin summaries need the true
prompt token count to reconstruct normalized bins correctly.

Steering:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_selected_bins_steering_k16_seeds_v2_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 2 3 4 5 6 7 \
  --token-selectors position_bin \
  --position-bins 0 1 2 4 \
  --position-bin-count 12 \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Probe:

```bash
python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_selected_bins_probe_k16_seeds_v2_mps \
  --layers 5 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 2 3 4 5 6 7 \
  --token-selectors position_bin \
  --position-bins 0 1 2 4 \
  --position-bin-count 12 \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent
```

Summary and plots:

```bash
python3 scripts/summarize_hltd_position_gate.py \
  --steering-root spiral_out_hltd_selected_bins_steering_k16_seeds_v2_mps \
  --probe-root spiral_out_hltd_selected_bins_probe_k16_seeds_v2_mps \
  --output-root spiral_out_hltd_selected_bins_position_k16_seeds_v2 \
  --bins 12 \
  --token-selector position_bin

python3 scripts/plot_hltd_position_gate.py \
  --summary-root spiral_out_hltd_selected_bins_position_k16_seeds_v2 \
  --output-dir spiral_out_hltd_selected_bins_position_k16_seeds_v2/plots \
  --probe ontology_collapse \
  --components coexact coexact_minus_presence presence presence_plus_coexact
```

Run sizes:

- steering: 60 prompt/layer/k runs, 517.3 seconds
- probe: 78,624 probe rows, 31.7 seconds
- position summary: 21,840 steering pairwise rows, 65,520 probe pairwise
  rows, 720 joined position rows

Cross-family ontology-gate peaks:

| Component | next peak | ontology-probe peak |
| --- | --- | --- |
| coexact | L7 bin 0, +0.8312 | L7 bin 1, +0.1956 |
| coexact_minus_presence | L7 bin 0, +0.8855 | L7 bin 1, -0.4982 |
| presence | L7 bin 0, +0.2014 | L5 bin 4, +2.3971 |
| presence_plus_coexact | L7 bin 0, +0.7536 | L8 bin 0, +1.0464 |

The selected-bin gate strengthens the dissociation. Under eight random-tangent
seeds, `coexact_minus_presence` is the cleanest traversal direction: its
L7/bin0 next-token peak is positive in every family. `presence` remains weak
for next-token support but keeps a much larger ontology-probe margin at
L5/bin4. The hybrid branch preserves much of the traversal signal and recovers
some probe margin, but it does not beat pure presence on probe stabilization.

![Selected-bin ontology peak bars](../spiral_out_hltd_selected_bins_position_k16_seeds_v2/plots/ontology_collapse_peak_bars.png)
