# Spiral Hodge

Fourier, graph, Hodge, and signed-circulation probes for transformer hidden-state trajectories.

Spiral Hodge is an experimental analysis script for asking a geometric question:

> If a language model's hidden states are projected into a low-dimensional semantic plane, does the token path behave like a field with gradients, curls, harmonics, and handedness?

The project is intentionally small and research-prototype shaped. It loads hidden states from a Hugging Face causal language model, reduces them into a 2D semantic coordinate system, treats token-to-token motion as a vector field, and exports layer-wise metrics and plots.

## What It Does

For an input text and a causal LM, `spiral_hodge.py` runs this pipeline:

1. Extract hidden states shaped `[layers, tokens, dim]`.
2. Reduce all layer-token hidden states into 2D coordinates with PCA or UMAP.
3. Build a token trajectory vector field from midpoint samples and token-to-token displacements.
4. Compute a nonuniform vector Fourier spectrum.
5. Project Fourier coefficients into Helmholtz-like gradient, curl, and harmonic components.
6. Build a graph Fourier spectrum over sampled trajectory points.
7. Build a Delaunay complex over token coordinates and run a discrete Hodge decomposition over edge flows.
8. Optionally run Hodge-Latent Traversal Dynamics (HLTD): a kNN graph Hodge decomposition over PCA/UMAP token-step node vectors.
9. Compute signed circulation and signed curl metrics, so reversed token order can be distinguished from the original direction.
10. Split spectral curl into low/mid/high frequency bands to separate coherent transport from high-frequency rotational clutter.
11. Add Hodge-independent vortex proxies: intrinsic trajectory turning and local Jacobian vorticity.
12. Export CSV metrics and diagnostic plots across every layer and optional null models.

## Hodge-Latent Traversal Dynamics

HLTD treats token generation as a sampled vector field on a hidden-state chart.
For each layer, token-step vectors are projected onto a kNN graph as scalar
edge flow, then decomposed into:

```text
exact      -> presence / gradient / source-sink dynamics
coexact    -> local semantic circulation over 3-cliques
harmonic   -> topology-conditioned open-cycle residual
flow       -> coexact + harmonic
```

This is intentionally separate from the older 2D Delaunay Hodge metric.
The Delaunay metric is a compact local-curl diagnostic for projected token
paths. HLTD can run on higher-dimensional charts, such as PCA-32, and logs
`hltd_exact_ratio`, `hltd_coexact_ratio`, `hltd_harmonic_ratio`, and
`hltd_semantic_flow_ratio`.

Example:

```bash
python3 spiral_hodge.py \
  --synthetic \
  --all-layers \
  --components 8 \
  --hltd \
  --hltd-k 12 \
  --hltd-vector-mode centered \
  --null-models all \
  --output-dir spiral_out_hltd \
  --csv-output layer_metrics.csv \
  --save-plots
```

For model runs, start with PCA rather than UMAP for decomposition:

```bash
python3 spiral_hodge.py \
  --model-path ./model/gpt2 \
  --text "The map drank the road and called it home." \
  --all-layers \
  --components 32 \
  --hltd \
  --hltd-k 16 \
  --hltd-vector-mode centered \
  --fourier-backend direct \
  --output-dir spiral_out_hltd_gpt2
```

The safest interpretation is comparative: look for layer/prompt regions where
HLTD non-exact energy beats matched nulls while fluency or downstream semantic
probes remain stable. Harmonic energy is topology-sensitive; a global-loop
interpretation requires persistence under an independently chosen filtration,
matched nulls, and a semantic or causal readout.

Use `--hltd-vector-mode centered` for reversal-sensitive experiments. The
original `forward` mode anchors each vector at `z_t`; centered mode anchors
`(z_{t+1} - z_{t-1}) / 2` at `z_t`, so real and reversed trajectories share the
same interior node set before edge-flow projection.

### Prompt-family HLTD suite

The repository includes a small prompt suite for comparing literal, metaphor,
identity-stress, and ontology-collapse text families:

```bash
python3 scripts/run_hltd_prompt_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_suite \
  --k 16 \
  --components 32 \
  --max-length 128 \
  --null-models all

python3 scripts/summarize_hltd_suite.py \
  --run-root spiral_out_hltd_suite \
  --output spiral_out_hltd_suite/summary.csv
```

For k-sweeps, pass multiple `--k` values:

```bash
python3 scripts/run_hltd_prompt_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_ksweep \
  --k 12 16 24 \
  --components 32 \
  --max-length 128 \
  --null-models all \
  --hltd-same-graph-reverse

python3 scripts/summarize_hltd_suite.py \
  --run-root spiral_out_hltd_ksweep \
  --output spiral_out_hltd_ksweep/summary.csv
```

The output root is ignored by git because per-prompt reports and plots can grow
quickly. The summary script reads each `layer_metrics.csv` and now emits:

```text
summary.csv              # one row per prompt/k/topology run
summary_family_k.csv     # family x k aggregate table
summary_layer.csv        # layer-wise aggregate curves and real-minus-null deltas
summary_prompt.csv       # prompt-level means across k
summary_bootstrap.csv    # prompt-level bootstrap confidence intervals
summary_family_gaps.csv  # pairwise family-gap bootstrap intervals
summary_report.md        # compact Markdown report for research notes
```

For triangle/topology ablations, keep run directories distinct:

```bash
python3 scripts/run_hltd_prompt_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_topology \
  --k 12 16 24 \
  --components 32 \
  --max-length 128 \
  --null-models all \
  --no-hltd-triangles
```

`--no-hltd-triangles` automatically appends `__no_triangles` to each run
directory, and the summarizer records that suffix as a `topology` column.

Use `--hltd-same-graph-reverse` for the reversal-invariance gate. It fixes the
real trajectory's kNN graph and triangle complex, reverses only the node vector
field, and writes `hltd_same_graph_reverse_*` diagnostics. The usual
`reverse_tokens` null still rebuilds the chart/graph, so comparing both gaps
helps separate graph-construction jitter from true Hodge decomposition issues.

### One-step causal steering

After a robust coexact layer has been identified, run a small causal gate by
adding reconstructed HLTD component directions to one hidden state and comparing
the next-token logits:

```bash
python3 scripts/run_hltd_steering.py \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --text "The map drank the road and called it home." \
  --layer 5 \
  --k 16 \
  --components 32 \
  --alphas 0.25 0.5 1.0 \
  --seeds 0 1 2 \
  --token-selectors max_component middle \
  --output-dir spiral_out_hltd_steering
```

The script writes `steering_metrics.csv` and `steering_report.md`. It compares
presence, coexact, semantic-flow, harmonic, and matched random-tangent
directions using KL divergence, entropy shift, top-token movement, and the
teacher-forced next-token log-probability shift. Token selectors can be
`max_component`, `middle`, `fixed`, or `all_interior`; use `--token-index` or
`--token-indices` with `fixed`.

To run the same gate across the prompt-family suite:

```bash
python3 scripts/run_hltd_steering_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_steering_suite \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle
```

The suite runner writes one `steering_metrics.csv` per prompt/layer/k and then
calls `scripts/summarize_hltd_steering.py`, producing:

```text
summary.csv                 # one row per component/alpha steering intervention
summary_component.csv       # family x layer x selector x component means
summary_pairwise.csv        # component-minus-random-tangent contrasts
summary_layer_pairwise.csv  # layer x selector contrasts across families
summary_report.md           # compact Markdown readout
```

For larger sweeps, use the fast in-process runner. It loads the model once,
reuses prompt hidden states and PCA coordinates across the layer sweep, and
writes the same output layout as the subprocess suite runner:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_fast_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps
```

Add lexical concept-cluster targets when the gate should measure whether an
intervention moves probability mass toward a family-specific semantic set
rather than only supporting the teacher-forced next token:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_semantic_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --target-set-file data/hltd_semantic_targets.json
```

The semantic target gate writes the same summaries, with additional target
mass and target-minus-control semantic-margin columns. Treat this as a coarse
lexical gate, not a learned identity or affordance probe.

To score the same HLTD directions with lightweight learned hidden-state probes,
run:

```bash
python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_probe_gate_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps
```

This trains simple layer-wise linear probes from
`data/hltd_probe_labels.json`, then evaluates whether each HLTD component
direction moves the selected hidden state toward identity, ontology, or
affordance-stress labels relative to matched random tangent.

For the stricter prompt-disjoint identity gate, train on matched
artifact/creator counterfactual pairs and hold `identity_02` out completely:

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

Training/evaluation prompt overlap is rejected by default, and CV holds out
both members of each counterfactual pair together. See
[docs/hltd_counterfactual_identity_probe.md](docs/hltd_counterfactual_identity_probe.md)
for the branch surface and construct-validity read.

For a presence/coexact dissociation gate, include derived component directions:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_dissociation_steering_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
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
  --output-root spiral_out_hltd_dissociation_probe_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent
```

The first dissociation read is in `docs/hltd_dissociation_gate.md`: presence
acts more like a learned-probe stabilizing direction, while coexact acts more
like a next-token traversal direction.

To reconnect structural graph-Hodge evidence with the steering and probe
branches, build the branch ledger:

```bash
python3 scripts/summarize_hltd_branch_hodge.py \
  --hodge-root spiral_out_hltd_invariance \
  --steering-root spiral_out_hltd_dissociation_steering_ksweep_mps \
  --probe-root spiral_out_hltd_dissociation_probe_ksweep_mps \
  --closed-loop-roots \
    spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08 \
    spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08 \
    spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08 \
    spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08 \
  --output-root spiral_out_hltd_branch_hodge \
  --topology triangles \
  --k 16 \
  --structural-ks 12 16 24 \
  --causal-ks 12 16 24 \
  --layers 4 5 6 7 8 \
  --selector middle \
  --compare-selectors middle max_component \
  --reverse-specificity-csv spiral_out_hltd_closed_loop_target_sensitivity_summary/reverse_exception_specificity_identity_ontology.csv
```

The branch read is in `docs/hltd_branch_hodge.md`: structural Hodge remains
coexact-dominant across the structural k-sweep under the full clique complex,
while causal k-sweeps separate coexact traversal from presence stabilization.
The same ledger also compares `middle` and `max_component` token selectors and
joins closed-loop `branch_specific_gate_rate` back to the family-level Hodge
branch. `docs/hltd_topology_filtration.md` then holds each graph and flow fixed
under a geometric triangle-radius filtration. Its 20-prompt, eight-null-seed
gate shows that exact/non-exact is invariant while coexact and harmonic exchange
the non-exact energy, then tests the interior branches at matched Betti-1 with
prompt-paired bootstrap intervals.

To render the branch ledger as structural and causal plots:

```bash
python3 scripts/plot_hltd_branch_hodge.py \
  --summary-root spiral_out_hltd_branch_hodge \
  --output-dir spiral_out_hltd_branch_hodge/plots \
  --probe ontology_collapse \
  --selector middle \
  --components coexact coexact_minus_presence presence presence_plus_coexact negative_coexact
```

The plot set shows the layer spine, structural k-sweep, topology contrast,
causal branch split, and traversal/stabilization phase map for the selected
probe. When closed-loop scoreboards are present, it also renders
`closed_loop_branch_specific_scoreboard.png` and `branch_role_summary.png`,
which compress structural Hodge, one-step causal k-sweep, and closed-loop
specificity into branch role maps. The plotter also renders
`branch_role_matrix.png` to compare traversal, probe margin, and closed-loop
specificity across all probes and branches, plus
`closed_loop_prompt_branch_heatmap.png` to show prompt-local branch exceptions.
When reverse-specificity rows are present, it also renders
`reverse_exception_specificity.png`, keeping the target-set control read inside
the branch ledger.
The current matrix has strict closed-loop rows for ontology-collapse,
identity-stress, and an affordance add-on suite. Ontology favors
`coexact_minus_presence`, identity-stress favors pure `coexact`, and the
affordance add-on splits by prompt after seed probing: presence-like branches
win on `affordance_01`, while coexact-derived branches win on `affordance_03`.
The `negative_coexact` sign-control column is now measured too: it is cleanly
suppressive for the measured affordance prompts, with prompt-local reverse
exceptions at `identity_04` and `ontology_05` rather than family-wide reverse
support.
`docs/hltd_reverse_exception_localization.md` then localizes those exceptions:
`identity_04` is a middle-layer k>=16 effect, while `ontology_05` strengthens
late and passes across all tested k at L8. A five-seed passing-band follow-up
keeps that split: identity is strongest at L7/k16-24, ontology at L8 across k.

To map the structural branches across every interior token position, run:

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

The heatmap read is in `docs/hltd_branch_heatmap.md`: coexact peaks tend to be
mid-to-late, presence shifts earlier as k increases, and the hybrid branch sits
between them.

For an all-interior causal/probe position gate, run the paired steering and
probe commands in `docs/hltd_all_interior_position_gate.md`, then summarize
them with:

```bash
python3 scripts/summarize_hltd_position_gate.py \
  --steering-root spiral_out_hltd_all_interior_steering_full_mps \
  --probe-root spiral_out_hltd_all_interior_probe_full_mps \
  --output-root spiral_out_hltd_all_interior_position_full \
  --bins 12 \
  --token-selector all_interior
```

The position gate writes per-token component-minus-random contrasts,
family/layer/bin summaries, family-level peaks, cross-family peaks, and a
Markdown report. It can also summarize `k=12/16/24` sweeps because peak tables
keep `k` separated. The current k-sweep read is that coexact-like branches
dominate early-phase cross-family next-token traversal, while presence
dominates ontology-probe margin at a different token-position phase.

To render the position gate as branch heatmaps and peak plots:

```bash
python3 scripts/plot_hltd_position_gate.py \
  --summary-root spiral_out_hltd_all_interior_position_ksweep \
  --output-dir spiral_out_hltd_all_interior_position_ksweep/plots \
  --probe ontology_collapse \
  --components coexact coexact_minus_presence presence presence_plus_coexact
```

For cheaper robustness checks, use the `position_bin` token selector with
explicit prompt token counts recorded in the metrics:

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

The selected-bin seed gate keeps the same branch split while reducing the
number of scored token positions: `coexact_minus_presence` is the cleanest
next-token traversal direction, and `presence` remains the stronger
ontology-probe stabilization direction.

For the first closed-loop branch steering gate, use nearest-node lookup against
the initial prompt Hodge field during greedy generation:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_smoke_tiny \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 2 \
  --alphas 1.0 \
  --seeds 0 \
  --limit 1 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence \
  --target-set-file data/hltd_semantic_targets.json
```

Summarize a closed-loop run with:

```bash
python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_smoke_tiny
```

Render the closed-loop branch summary as plots with:

```bash
python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_smoke_tiny \
  --output-dir spiral_out_hltd_closed_loop_smoke_tiny/plots \
  --components presence_plus_coexact coexact_minus_presence presence random_tangent
```

Render target-vocabulary sensitivity across multiple closed-loop roots with:

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

Combine multiple target-sensitivity CSVs into a reverse-exception specificity
figure with:

```bash
python3 scripts/plot_hltd_reverse_specificity.py \
  --panel 'identity_04 L7/k16=spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_identity04_l7_k16_full.csv' \
  --panel 'ontology_05 L8/k16=spiral_out_hltd_closed_loop_target_sensitivity_summary/target_sensitivity_ontology05_l8_k16_full.csv' \
  --output-root spiral_out_hltd_closed_loop_target_sensitivity_summary \
  --component negative_coexact \
  --output-prefix reverse_exception_specificity_identity_ontology
```

The closed-loop read is in `docs/hltd_closed_loop_gate.md`. The tiny smoke
proves the generation hook and field-following nearest-node lookup; larger
ontology-collapse runs should use the phase-map priors from the branch ledger.
The same note includes an `ontology_05` alpha sweep where
`coexact_minus_presence` stays baseline-locked at `alpha=0.25/0.5` but breaks
away at `alpha=1.0`, a narrow sweep that pins the first observed break between
`alpha=0.7` and `alpha=0.8`, and a three-prompt branch panel at `alpha=0.8`.
In that panel, `coexact_minus_presence` has the strongest drift and the only
positive mean semantic-target margin, while `presence` and
`presence_plus_coexact` mostly preserve the greedy baseline. An eight-step
persistence panel keeps the same branch ordering: `coexact_minus_presence`
continues to have the highest drift and the only positive mean target margin,
while longer horizons make some non-semantic surface drift appear in the other
branches. A first L5-L8 layer pilot on the two branch-sensitive ontology
prompts shows the same branch has positive mean target margin at every tested
layer, with L7 giving the cleanest branch split. A first L7 k-sweep keeps
`coexact_minus_presence` positive in target margin for k=12/16/24, with k=16
and k=24 producing the clearest token branches. The plotter also renders
`closed_loop_step_traces.png` from `closed_loop_steps.csv`,
`closed_loop_layer_response.png` from `closed_loop_layer_summary.csv`, and
`closed_loop_k_response.png` from `closed_loop_k_summary.csv`, so branch effects
can be read over generated decoding steps, transformer layers, and graph
neighborhood sizes rather than only as run averages. When a run summarizes
multiple k values and multiple alphas, it also renders
`closed_loop_alpha_k_threshold.png`; the first L7 alpha-k grid shows
`coexact_minus_presence` becoming a reliable closed-loop breaker at k=16/24 and
alpha >= 0.8, while k=12 preserves positive target pressure without full token
drift. Multi-component alpha-k grids additionally render
`closed_loop_alpha_k_branch_map.png`, which compares each branch's token drift
and target margin at every tested k. The summarizer also writes
`closed_loop_prompt_summary.csv`, and multi-prompt runs render
`closed_loop_prompt_branch_gate.png`; when matched random-tangent columns are
available, the plotter also renders `closed_loop_prompt_random_advantage.png`.
The five-prompt ontology robustness panel keeps `coexact_minus_presence` as the
strongest prompt-level closed-loop branch, with four of five branch gates at
L7/k16/alpha=0.8 and three of five stricter branch-specific gates. A follow-up
five-seed probe shows that `ontology_05` is branch-specific for
`coexact_minus_presence`, while `ontology_01` is control-sensitive. The stricter
`branch_specific_gate_rate` requires a branch gate plus matched random-tangent
drift/target advantage, and isolates this split better than raw gate rate.
The matched five-prompt identity-stress panel fills a second row in the branch
role matrix: pure `coexact` has the strongest strict identity gate, while
`coexact_minus_presence` has larger target advantage but fewer passing prompt
cells.
The add-on affordance panel uses `data/hltd_affordance_prompt_suite.jsonl` to
fill the third active row without changing the original 20-prompt baseline.
The `affordance_01`/`affordance_03` seed probe strengthens that row and shows a
prompt-level branch split rather than a single family-wide winner.
The negative-coexact sign-control panel fills the last branch-role matrix
column; it behaves as a control for the measured affordance prompts and most
identity/ontology prompts, with prompt-local reverse sensitivity at
`identity_04` and `ontology_05`.

## Why Signed Curl Matters

The first version of this experiment measured curl energy ratios. Those ratios are useful, but they are unsigned: reversing a trajectory can preserve the same amount of curl energy.

That means:

```text
real token order    -> curl ratio = 0.6725
reversed token order -> curl ratio = 0.6725
```

The signed metrics add orientation. They measure whether the projected motion has a preferred clockwise/counterclockwise handedness in the semantic plane.

For the included GPT-2 example, the final layer behaves like this:

```text
real layer 12
trajectory_signed_circulation_alignment = -0.2652
spectral_signed_curl_alignment          = -0.3743
spectral_signed_vorticity_ratio         = -0.6479

reversed layer 12
trajectory_signed_circulation_alignment = +0.2652
spectral_signed_curl_alignment          = +0.3743
spectral_signed_vorticity_ratio         = +0.6479
```

That sign flip is the point: the unsigned energy says "there is curl-like structure"; the signed metrics say "the structure has a direction."

## Current Research Reading

The first live JAX run shifted the working hypothesis. The early intuition was:

> meaning formation creates vortex-like structure.

The more useful version now looks like:

> coherent autoregressive representation may suppress high-frequency rotational disorder while preserving or reorganizing larger-scale transport.

In the short GPT-2 prompt:

```text
The serpent coils not around the tree, but around cognition.
```

the `reverse_tokens` baseline behaves exactly as a signed-orientation sanity check should: unsigned energy stays the same, while signed trajectory, spectral curl, spectral vorticity, Hodge curl, and local Jacobian vorticity flip sign.

The stronger separation is not in total spectral curl. It is in smoothness and local rotational clutter:

```text
graph_high_freq_ratio mean
real:          0.4063
shuffle:       0.7703
random_hidden: 0.6968

hodge_curl_ratio mean
real:          0.1109
shuffle:       0.4928
random_hidden: 0.2647
```

This suggests that `graph_*` and `hodge_*` are currently better detectors of local disorder, while `spectral_*` may be closer to a mixture of global transport and generic curl induced by projection. The research question is therefore not just "is there a vortex?" but:

> which scales of rotational structure are suppressed, preserved, or amplified across layers and null models?

See [docs/research_notes.md](docs/research_notes.md) for the fuller interpretation, caveats, and next experiments.
See [docs/hltd_prompt_family_observations.md](docs/hltd_prompt_family_observations.md)
for the first 20-prompt HLTD family sweep.
See [docs/hltd_counterfactual_identity_probe.md](docs/hltd_counterfactual_identity_probe.md)
for the prompt-disjoint learned identity-axis branch surface.
See [docs/hltd_branch_hodge.md](docs/hltd_branch_hodge.md) for the current
structural/causal branch ledger.
See [docs/hltd_topology_filtration.md](docs/hltd_topology_filtration.md) for the
radius-filtration, matched-topology, and prompt-bootstrap branch-persistence
gate.
See [docs/hltd_matched_betti_causal_gate.md](docs/hltd_matched_betti_causal_gate.md)
for the pre-registered L5/k16 interior-complex one-step causal result.
See [docs/hltd_branch_heatmap.md](docs/hltd_branch_heatmap.md) for all-interior
branch localization.
See [docs/hltd_all_interior_position_gate.md](docs/hltd_all_interior_position_gate.md)
for the position-binned all-interior causal/probe gate and k-sweep.
See [docs/hltd_closed_loop_gate.md](docs/hltd_closed_loop_gate.md) for the
first closed-loop branch-steering harness and smoke result.

## Repository Layout

```text
.
├── spiral_hodge.py                  # CLI and analysis implementation
├── spiral_hodge_report.py           # static interactive HTML report generator
├── data/hltd_prompt_suite.jsonl      # prompt-family suite for HLTD sweeps
├── data/hltd_semantic_targets.json   # lexical target/control sets for steering
├── data/hltd_probe_labels.json       # binary probe labels for hidden-state gates
├── data/hltd_identity_counterfactual_probe_suite.jsonl
│                                      # matched disjoint probe-training pairs
├── scripts/                          # prompt-suite run and summary helpers
├── docs/research_notes.md            # current hypotheses and live-run interpretation
├── docs/hltd_prompt_family_observations.md
├── docs/hltd_dissociation_gate.md
├── docs/hltd_counterfactual_identity_probe.md
├── docs/hltd_branch_hodge.md
├── docs/hltd_topology_filtration.md
├── docs/hltd_matched_betti_causal_gate.md
├── docs/hltd_branch_heatmap.md
├── docs/hltd_all_interior_position_gate.md
├── docs/hltd_closed_loop_gate.md
├── tests/test_spiral_hodge.py        # path resolution and signed-orientation tests
├── examples/izumi-gpt2/              # sample CSV and plots from a GPT-2 run
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

Create and activate a virtual environment if you want to keep dependencies local:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Or install directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install finufft      # optional faster native Fourier backend
pip install jax          # optional JAX/XLA direct Fourier backend
pip install umap-learn   # optional UMAP reducer
```

The default Fourier backend is `direct`, which avoids native FINUFFT crashes and is safer for small to medium token sequences. Use `--fourier-backend jax` to run the direct nonuniform Fourier and signed-curl evaluation kernels through JAX/XLA when JAX is installed.

## Quick Start Without a Model

Run the synthetic smoke example:

```bash
python3 spiral_hodge.py \
  --synthetic \
  --all-layers \
  --null-models all \
  --fourier-backend direct \
  --fourier-modes 16 \
  --output-dir spiral_out_synthetic \
  --csv-output layer_metrics.csv \
  --save-plots
```

This creates:

```text
spiral_out_synthetic/layer_metrics.csv
spiral_out_synthetic/null_model_curl_spectral.png
spiral_out_synthetic/null_model_signed_spectral_curl.png
...
```

Generate a static HTML report from the CSV:

```bash
python3 spiral_hodge_report.py \
  --run-dir spiral_out_synthetic \
  --output report.html
```

## Running With a Local Hugging Face Model

If your model is available as a local Hugging Face directory:

```bash
python3 spiral_hodge.py \
  --model-path ./model/gpt2 \
  --text "The serpent coils not around the tree, but around cognition." \
  --all-layers \
  --null-models all \
  --fourier-backend direct \
  --fourier-modes 32 \
  --output-dir spiral_out \
  --csv-output layer_metrics.csv \
  --save-plots
```

`--model-path` implies local-only loading. You can also use:

```bash
python3 spiral_hodge.py --model ./model/gpt2 ...
```

For long text files, GPT-2 is normally limited to 1024 tokens, so use `--max-length 1024`:

```bash
python3 spiral_hodge.py \
  --model-path ./model/gpt2 \
  --text-file ./data/my_text.txt \
  --max-length 1024 \
  --all-layers \
  --null-models all \
  --fourier-backend direct \
  --fourier-modes 32 \
  --output-dir spiral_out_long \
  --csv-output layer_metrics.csv \
  --save-plots
```

## Offline Mode

Spiral Hodge supports offline/local model loading. Use a local model path:

```bash
python3 spiral_hodge.py --model-path ./model/gpt2 --local-files-only --text "..."
```

The script also respects Hugging Face offline environment variables:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

If a local model cannot be found, the CLI reports a short actionable error instead of a long Transformers traceback.

## Null Models

`--null-models` controls comparison baselines:

```text
real
shuffle_tokens
reverse_tokens
random_hidden
all
```

The controls are intentionally simple:

- `real`: unchanged hidden states.
- `shuffle_tokens`: same hidden vectors, but token order is randomly permuted.
- `reverse_tokens`: same path traversed backwards.
- `random_hidden`: Gaussian hidden states with matched per-dimension mean and standard deviation.

The `reverse_tokens` baseline is especially important for signed metrics. Unsigned curl energy should often stay similar, while signed circulation should flip.

## Main CSV Metrics

The generated `layer_metrics.csv` contains one row per variant and layer.

Important column groups:

- `spectral_*`: Fourier-domain Helmholtz energy totals and ratios.
- `spectral_curl_low_*`, `spectral_curl_mid_*`, `spectral_curl_high_*`: Fourier curl split into radial frequency bands.
- `hodge_*`: discrete Hodge energy totals and ratios over triangulated edge flows.
- `graph_*`: graph Fourier low/high frequency summaries.
- `trajectory_signed_*`: signed circulation of the raw token trajectory.
- `turning_*`: intrinsic path-turning angles between consecutive token-step vectors.
- `local_*`: local affine-Jacobian vorticity estimates that do not use Hodge or Fourier.
- `spectral_signed_*`: signed circulation and vorticity of the Fourier curl component.
- `hodge_signed_*`: signed face-circulation metrics from the discrete Hodge curl component.

The most immediately useful signed columns are:

```text
trajectory_signed_circulation_alignment
turning_alignment
spectral_signed_curl_alignment
local_signed_vorticity_ratio
spectral_signed_vorticity_ratio
hodge_signed_curl_alignment
```

These values are normalized to a rough `[-1, 1]` orientation scale:

- near `0`: weak or mixed orientation
- positive: one handedness
- negative: opposite handedness
- sign flip under `reverse_tokens`: expected and useful

## Example: GPT-2 Layer-12 Signed Curl

The `examples/izumi-gpt2/` folder contains a reference output from a local GPT-2 run over 1024 tokens.

### Curl Energy Ratio

![Curl comparison for real GPT-2 run](examples/izumi-gpt2/curl_comparison_real.png)

The real run has a final-layer spectral curl-energy spike:

```text
spectral curl peak: layer=12, ratio=0.6725
hodge curl peak:    layer=10, ratio=0.5427
```

This suggests that the final-layer spectral field has a strong curl-like component, while the local Delaunay Hodge view peaks earlier.

### Signed Orientation

![Signed circulation comparison for real GPT-2 run](examples/izumi-gpt2/signed_circulation_comparison_real.png)

The signed metrics show that the final-layer curl spike is not merely large; it is oriented:

```text
real layer 12
trajectory_signed_circulation_alignment = -0.2652
spectral_signed_curl_alignment          = -0.3743
spectral_signed_vorticity_ratio         = -0.6479
hodge_signed_curl_alignment             = +0.0484
```

### Null Model Comparison

![Spectral curl null comparison](examples/izumi-gpt2/null_model_curl_spectral.png)

![Signed spectral curl null comparison](examples/izumi-gpt2/null_model_signed_spectral_curl.png)

![Signed spectral vorticity null comparison](examples/izumi-gpt2/null_model_signed_spectral_vorticity.png)

In this run, `reverse_tokens` mirrors the real signed orientation, while `shuffle_tokens` and `random_hidden` do not reproduce the final-layer signed vorticity spike with the same strength.

## CLI Reference

Common options:

```text
--model MODEL                 Hugging Face model name or local directory
--model-path PATH             local Hugging Face model directory
--text TEXT                   inline text
--text-file PATH              read text from a file
--max-length N                tokenizer truncation length
--all-layers                  run every layer
--layer N                     single-layer mode
--null-models LIST            real, shuffle_tokens, reverse_tokens, random_hidden, all
--reducer pca|umap            semantic coordinate reducer
--fourier-backend direct|finufft|jax
--fourier-modes N
--graph-eigs N
--k-neighbors N
--local-files-only            disable Hugging Face downloads/lookups
--save-plots                  write PNG diagnostics
--quiet                       reduce progress logs
```

## Development

Run tests:

```bash
python3 -m unittest discover
```

Run a quick syntax check:

```bash
python3 -m py_compile spiral_hodge.py
```

The tests include explicit reversal checks for signed orientation. A reversed trajectory should invert the signed circulation and signed spectral curl metrics.

## Caveats

This is a research probe, not a settled interpretation framework.

Important limitations:

- PCA and UMAP projections can introduce artifacts.
- Curl in a reduced semantic plane is not the same as curl in the full hidden-state space.
- Delaunay triangulations can be sensitive to degenerate or clustered projected points.
- Energy ratios near `0.5` require null-model comparison before interpretation.
- Signed orientation is meaningful primarily in comparative settings: real vs reverse, real vs shuffle, layer vs layer, or model vs model.

The safest reading is not "the model literally thinks in spirals." The safer claim is:

> Under this projection and decomposition, some layers produce vector-field structure with measurable curl energy and signed handedness that can be compared against simple controls.

That is already interesting enough.

## Interactive HTML Report

`spiral_hodge_report.py` turns an existing `layer_metrics.csv` into a standalone
HTML dashboard. It does not rerun the model and it does not require a web server.

```bash
python3 spiral_hodge_report.py \
  --run-dir examples/izumi-gpt2 \
  --output report.html \
  --title "Izumi / GPT-2 Spiral Hodge Report"
```

The report includes:

- selected-variant curl energy charts
- selected-variant signed orientation charts
- real vs null-model comparison for any key metric
- peak layer summaries
- reverse-direction cancellation diagnostics
- a sortable-by-eye layer table for the most useful metrics

The committed example report is here:

[examples/izumi-gpt2/report.html](examples/izumi-gpt2/report.html)
