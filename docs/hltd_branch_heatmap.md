# HLTD Branch Position Heatmap

This note records the first all-interior structural branch-localization pass.
Unlike the causal steering gates, this does not edit activations. It maps
reconstructed HLTD component vector norms across every centered token node.

## Command

```bash
python3 scripts/run_hltd_branch_heatmap.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_branch_heatmap \
  --layers 4 5 6 7 8 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --bins 12 \
  --device mps
```

Outputs:

- `spiral_out_hltd_branch_heatmap/node_branch_metrics.csv`
- `spiral_out_hltd_branch_heatmap/summary_position.csv`
- `spiral_out_hltd_branch_heatmap/summary_peaks.csv`
- `spiral_out_hltd_branch_heatmap/summary_global_peaks.csv`
- `spiral_out_hltd_branch_heatmap/summary_report.md`

The full run produced 50,760 node/component rows in 53.1 seconds.

## Global Peak Read

Peak positions are normalized over each prompt and binned into 12 bins.

| k | component | mean peak bin | mean peak position | mean peak/full |
| ---: | --- | ---: | ---: | ---: |
| 12 | presence | 6.50 | 0.5806 | 0.9602 |
| 12 | coexact | 6.25 | 0.5593 | 2.0754 |
| 12 | presence_plus_coexact | 6.35 | 0.5673 | 2.3438 |
| 16 | presence | 5.10 | 0.4636 | 1.2950 |
| 16 | coexact | 6.70 | 0.5964 | 2.5592 |
| 16 | presence_plus_coexact | 6.35 | 0.5669 | 3.0332 |
| 24 | presence | 4.50 | 0.4114 | 2.0158 |
| 24 | coexact | 6.65 | 0.5918 | 3.4813 |
| 24 | presence_plus_coexact | 6.05 | 0.5417 | 4.2368 |

The useful pattern is:

- `coexact` and `semantic_flow` peak later, around normalized position
  0.56-0.60.
- `presence` shifts earlier as k increases, from position 0.58 at k=12 to
  0.41 at k=24.
- `presence_plus_coexact` sits between them and is the strongest norm branch by
  peak/full ratio.
- `harmonic` remains numerically near zero, so its peak location is not
  interpretable in this complex.

## Family Localization

At `k=16`, coexact peaks are family-specific:

| Family | coexact peak bins across L4-L8 | read |
| --- | --- | --- |
| literal_stable | 5, 5, 5, 5, 5 | stable mid-prompt transport |
| metaphor_shift | 5, 8, 5, 8, 8 | split between mid and later metaphor movement |
| identity_stress | 8, 8, 8, 8, 8 | late identity traversal |
| ontology_collapse | 7, 7, 7, 7, 7 | late but slightly earlier than identity stress |

Presence is less locked to one bin. In `literal_stable`, it tends to stay near
the middle. In `identity_stress`, it can jump earlier at late layers. This fits
the branch story: coexact is a smoother traversal branch, while presence is a
more local basin/stabilization branch.

## Interpretation

The all-interior heatmap strengthens the selector read. `middle` worked well
for coexact steering because coexact peaks are broad, stable, and often
mid-to-late rather than isolated at a single maximum-norm node. `max_component`
can still be useful, but it emphasizes local extrema and can shift the gate
toward presence/probe stabilization.

This is still structural evidence. The paired causal version now runs an
all-interior steering/probe gate and aggregates by these same position bins.

## Position-Binned Causal Gate

The all-interior causal/probe position gate is written up in:

```text
docs/hltd_all_interior_position_gate.md
```

Short read: structural coexact mass tends mid-to-late, but the strongest
cross-family next-token causal peak appears in an early phase for
coexact-like branches across the k-sweep. Presence is weaker for next-token
traversal but stronger for ontology-probe margin at a different token-position
phase. Token position is not just a sampling nuisance; it is part of the
branch mechanism.
