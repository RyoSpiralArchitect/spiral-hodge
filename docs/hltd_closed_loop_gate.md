# HLTD Closed-Loop Branch Gate

This note records the first closed-loop steering harness for the branch-Hodge
program. Unlike the one-step gate, this applies a branch direction during greedy
generation.

## Method

For each prompt, layer, and k:

1. Extract the teacher-forced prompt hidden states.
2. Fit the usual PCA chart and graph-Hodge branch field.
3. During generation, run a base forward pass for the current prefix.
4. Project the current last-token hidden state into the initial PCA chart.
5. Select the nearest original branch-field node.
6. Add the requested branch direction at the chosen layer and last-token
   position.
7. Greedily decode the next token from the steered logits.

This is a nearest-node closed-loop gate. It follows the initial prompt branch
field instead of using a fixed vector for every step.

## Command

Tiny smoke command:

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

Outputs:

- `spiral_out_hltd_closed_loop_smoke_tiny/closed_loop_metrics.csv`
- `spiral_out_hltd_closed_loop_smoke_tiny/closed_loop_steps.csv`
- `spiral_out_hltd_closed_loop_smoke_tiny/closed_loop_report.md`
- `spiral_out_hltd_closed_loop_smoke_tiny/closed_loop_manifest.json`

Summarize a closed-loop run with:

```bash
python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_smoke_tiny
```

Summary outputs:

- `closed_loop_contrasts.csv`
- `closed_loop_component_summary.csv`
- `closed_loop_family_summary.csv`
- `closed_loop_summary_report.md`
- `closed_loop_prompt_summary.csv`

Render closed-loop branch plots with:

```bash
python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_smoke_tiny \
  --output-dir spiral_out_hltd_closed_loop_smoke_tiny/plots \
  --components presence_plus_coexact coexact_minus_presence presence random_tangent
```

Plot outputs:

- `plots/closed_loop_component_bars.png`
- `plots/closed_loop_drift_support_phase.png`
- `plots/closed_loop_alpha_response.png`
- `plots/closed_loop_step_traces.png`
- `plots/closed_loop_layer_response.png` when multiple layers are summarized
- `plots/closed_loop_k_response.png` when multiple k values are summarized
- `plots/closed_loop_alpha_k_threshold.png` when multiple k values and alphas
  are summarized
- `plots/closed_loop_alpha_k_branch_map.png` when an alpha-k grid has multiple
  branch components
- `plots/closed_loop_prompt_branch_gate.png` when multiple prompts are
  summarized
- `plots/closed_loop_prompt_random_advantage.png` when prompt summaries include
  matched random-tangent difference columns
- `plots/plot_manifest.json`

Smoke result:

- prompt: `literal_stable/literal_01`
- layer/k: `L7/k16`
- generated steps: 2
- runtime: 53.1 seconds on CPU

| component | mean base logp | mean gain | mean KL | target margin | overlap | generated |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | -0.3426 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | `\n\n` |
| presence_plus_coexact | -0.3426 | -0.0122 | 0.0158 | 0.3825 | 1.0000 | `\n\n` |
| coexact_minus_presence | -0.3426 | -0.0808 | 0.0762 | -0.5543 | 1.0000 | `\n\n` |

This first smoke proves the hook/generation path and the nearest-node branch
lookup. It is not yet a semantic result because the literal prompt greedily
continues with newlines under both baseline and steered branches.

## Ontology Tiny Run

The first ontology-collapse closed-loop smoke used the same tiny 2-token gate
on `ontology_05`:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_ontology_tiny \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 2 \
  --alphas 1.0 \
  --seeds 0 \
  --prompt-ids ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_ontology_tiny

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_ontology_tiny \
  --output-dir spiral_out_hltd_closed_loop_ontology_tiny/plots \
  --components presence_plus_coexact coexact_minus_presence presence random_tangent
```

Run result:

- prompt: `ontology_collapse/ontology_05`
- layer/k: `L7/k16`
- generated steps: 2
- runtime: 44.9 seconds on CPU

| component | drift | overlap | base logp | gain | KL | target margin | generated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| presence_plus_coexact | 0.0000 | 1.0000 | -0.7651 | -0.1908 | 0.0531 | -0.1662 | `\n\n` |
| coexact_minus_presence | 1.0000 | 0.0000 | -2.0500 | 0.1985 | 0.1294 | 0.6058 | ` The moon` |

![Ontology closed-loop component bars](../spiral_out_hltd_closed_loop_ontology_tiny/plots/closed_loop_component_bars.png)

![Ontology closed-loop drift/support phase](../spiral_out_hltd_closed_loop_ontology_tiny/plots/closed_loop_drift_support_phase.png)

Step-level read:

- baseline generated `\n\n`.
- `presence_plus_coexact` kept the same `\n\n` continuation while slightly
  lowering support for those tokens.
- `coexact_minus_presence` changed step 0 from newline to ` The`, then
  supported ` moon` at step 1.
- nearest-node lookup stayed bounded: node 11 at step 0, node 9 at step 1 for
  the drifting branch.

This is the first closed-loop branch drift signal. It is still tiny and should
not be read as semantic control yet, but it matches the branch-phase prior:
`coexact_minus_presence` is the branch that can break away from the stabilizing
baseline.

## Ontology Alpha Sweep

To look for a less abrupt closed-loop branch threshold, rerun `ontology_05` with
`alpha = 0.25/0.5/1.0`:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_ontology_alpha_sweep \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 2 \
  --alphas 0.25 0.5 1.0 \
  --seeds 0 \
  --prompt-ids ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_ontology_alpha_sweep

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_ontology_alpha_sweep \
  --output-dir spiral_out_hltd_closed_loop_ontology_alpha_sweep/plots \
  --components presence_plus_coexact coexact_minus_presence presence random_tangent
```

Runtime: 111.2 seconds on CPU.

| component | alpha | drift | overlap | gain | KL | target margin | generated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| presence_plus_coexact | 0.25 | 0.0000 | 1.0000 | -0.0200 | 0.0033 | -0.0934 | `\n\n` |
| presence_plus_coexact | 0.50 | 0.0000 | 1.0000 | -0.0698 | 0.0127 | -0.1114 | `\n\n` |
| presence_plus_coexact | 1.00 | 0.0000 | 1.0000 | -0.1908 | 0.0531 | -0.1662 | `\n\n` |
| coexact_minus_presence | 0.25 | 0.0000 | 1.0000 | -0.0308 | 0.0031 | 0.0259 | `\n\n` |
| coexact_minus_presence | 0.50 | 0.0000 | 1.0000 | -0.1030 | 0.0131 | 0.1559 | `\n\n` |
| coexact_minus_presence | 1.00 | 1.0000 | 0.0000 | 0.1985 | 0.1294 | 0.6058 | ` The moon` |

![Ontology alpha sweep bars](../spiral_out_hltd_closed_loop_ontology_alpha_sweep/plots/closed_loop_component_bars.png)

![Ontology alpha sweep phase](../spiral_out_hltd_closed_loop_ontology_alpha_sweep/plots/closed_loop_drift_support_phase.png)

![Ontology alpha response](../spiral_out_hltd_closed_loop_ontology_alpha_sweep/plots/closed_loop_alpha_response.png)

Read:

- `presence_plus_coexact` remains a stabilization branch across the tested
  alphas. It keeps the newline continuation and increasingly suppresses the
  selected token as alpha rises.
- `coexact_minus_presence` shows a threshold-like response. At 0.25 and 0.5 it
  remains locked to the baseline, but its target margin is already positive.
  At 1.0 it breaks away to ` The moon`.
- The current sweet-spot question is therefore not 0.25/0.5/1.0, but the
  transition band between 0.5 and 1.0.

Next narrow sweep:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_ontology_alpha_narrow \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 2 \
  --alphas 0.6 0.7 0.8 0.9 \
  --seeds 0 \
  --prompt-ids ontology_05 \
  --device cpu \
  --steering-components coexact_minus_presence \
  --target-set-file data/hltd_semantic_targets.json
```

Then summarize and render a combined broad+narrow transition plot:

```bash
python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_ontology_alpha_narrow

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_ontology_alpha_narrow \
  --comparison-summary-roots spiral_out_hltd_closed_loop_ontology_alpha_sweep \
  --output-dir spiral_out_hltd_closed_loop_ontology_alpha_narrow/plots \
  --components coexact_minus_presence presence_plus_coexact
```

Runtime: 75.4 seconds on CPU.

| alpha | drift | overlap | base logp | gain | KL | target margin | generated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.6 | 0.0000 | 1.0000 | -0.7651 | -0.1213 | 0.0194 | 0.2709 | `\n\n` |
| 0.7 | 0.0000 | 1.0000 | -0.7651 | -0.1759 | 0.0310 | 0.4394 | `\n\n` |
| 0.8 | 1.0000 | 0.0000 | -2.0502 | 0.2011 | 0.0654 | 0.4953 | ` The moon` |
| 0.9 | 1.0000 | 0.0000 | -2.0502 | 0.2159 | 0.0916 | 0.6402 | ` The moon` |

![Ontology narrow alpha response](../spiral_out_hltd_closed_loop_ontology_alpha_narrow/plots/closed_loop_alpha_response.png)

![Ontology closed-loop alpha transition band](../spiral_out_hltd_closed_loop_ontology_alpha_narrow/plots/closed_loop_alpha_transition.png)

Narrow-sweep read:

- The first observed branch break is between `alpha=0.7` and `alpha=0.8`.
- `alpha=0.8` is the current sweet-spot candidate: it reaches the same
  two-token drift as `0.9` and `1.0`, but with lower KL than both.
- `0.6` and `0.7` are pre-transition: target margin rises while the decoded
  tokens remain baseline-locked.
- `presence_plus_coexact` remains a stabilization branch in the broad sweep.

## Branch Panel at Alpha 0.8

With the first branch break pinned between `0.7` and `0.8`, run the same
`alpha=0.8` gate across three ontology-collapse prompts and all closed-loop
branches:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_01 ontology_03 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 669.8 seconds on CPU.

| component | drift | overlap | gain | KL | target margin | nearest dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.8333 | 0.1667 | 0.0581 | 0.0614 | 0.0578 | 0.3188 |
| coexact | 0.5000 | 0.5000 | -0.0210 | 0.0392 | -0.0981 | 0.3151 |
| presence | 0.1667 | 0.8333 | 0.0608 | 0.0486 | -0.3400 | 0.2998 |
| presence_plus_coexact | 0.1667 | 0.8333 | 0.0123 | 0.0346 | -0.3310 | 0.2998 |
| random_tangent | 0.1667 | 0.8333 | -0.0341 | 0.0311 | -0.0042 | 0.2998 |

![Branch panel component bars](../spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08/plots/closed_loop_component_bars.png)

![Branch panel step traces](../spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08/plots/closed_loop_step_traces.png)

Prompt-level generated-text read:

| prompt | branch | overlap | target margin | generated |
| --- | --- | ---: | ---: | --- |
| ontology_01 | all tested branches | 0.5 | mixed | `\n\nThe map` |
| ontology_03 | coexact_minus_presence | 0.0 | 0.2166 | ` The room was a` |
| ontology_03 | coexact | 0.0 | 0.1779 | ` The room was a` |
| ontology_03 | presence / combined / random | 1.0 | mixed negative | `\n\n"I` |
| ontology_05 | coexact_minus_presence | 0.0 | 0.2497 | ` The moon was in` |
| ontology_05 | presence / combined / coexact / random | 1.0 | mostly negative | `\n\n"I` |

Branch-panel read:

- `coexact_minus_presence` is the strongest branch breaker. It has the highest
  drift, the only positive mean target margin, and bounded nearest-node
  distance.
- `coexact` can also break a prompt, but less reliably: it matches
  `coexact_minus_presence` on `ontology_03` and stays baseline-locked on
  `ontology_05`.
- `presence` and `presence_plus_coexact` look stabilizing in this gate. They
  mostly preserve the baseline continuation and push target margin negative.
- `random_tangent` stays close to baseline on two of three prompts. Its
  `ontology_01` drift is not branch-specific because every tested branch lands
  on the same `\n\nThe map` continuation there.
- Step traces show the first branch break is front-loaded: `coexact_minus_presence`
  has the largest step-0 top-token changed rate and positive step-0 target
  margin, then the generated prefix carries the trajectory forward.

## Eight-Step Persistence Panel

To check whether the branch effect is only a first-token flip, extend the same
panel to eight generated steps:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 8 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_01 ontology_03 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8 \
  --output-dir spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 875.4 seconds on CPU.

| component | drift | overlap | gain | KL | target margin | nearest dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.9167 | 0.0833 | 0.0587 | 0.0762 | 0.0560 | 0.3409 |
| coexact | 0.6250 | 0.3750 | -0.0135 | 0.0559 | -0.0460 | 0.3480 |
| presence | 0.4167 | 0.5833 | 0.0796 | 0.0687 | -0.3276 | 0.3656 |
| presence_plus_coexact | 0.4167 | 0.5833 | 0.0423 | 0.0455 | -0.1363 | 0.3650 |
| random_tangent | 0.3333 | 0.6667 | -0.0024 | 0.0689 | -0.0491 | 0.3602 |

![Eight-step branch panel component bars](../spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8/plots/closed_loop_component_bars.png)

![Eight-step branch panel step traces](../spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8/plots/closed_loop_step_traces.png)

Four-step versus eight-step comparison:

| component | drift 4-step | drift 8-step | target 4-step | target 8-step |
| --- | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.8333 | 0.9167 | 0.0578 | 0.0560 |
| coexact | 0.5000 | 0.6250 | -0.0981 | -0.0460 |
| presence | 0.1667 | 0.4167 | -0.3400 | -0.3276 |
| presence_plus_coexact | 0.1667 | 0.4167 | -0.3310 | -0.1363 |
| random_tangent | 0.1667 | 0.3333 | -0.0042 | -0.0491 |

Eight-step generated-text read:

| prompt | branch | overlap | target margin | generated |
| --- | --- | ---: | ---: | --- |
| ontology_01 | coexact_minus_presence | 0.25 | -0.1332 | `\n\nThe map was a map of` |
| ontology_01 | most other branches | 0.25 | mostly negative | `\n\nThe map was a reminder of` |
| ontology_03 | coexact_minus_presence | 0.00 | 0.0745 | ` The room was a small room with a` |
| ontology_03 | coexact | 0.00 | -0.0177 | ` The room was a room of the same` |
| ontology_03 | presence / combined / random | 0.625-0.875 | mixed | baseline-like refusal preamble |
| ontology_05 | coexact_minus_presence | 0.00 | 0.2267 | ` The moon was in the middle of the` |
| ontology_05 | presence / combined / coexact / random | 0.875 | negative | baseline-like or weak moon/car variants |

Persistence read:

- `coexact_minus_presence` remains the strongest branch over eight steps. Its
  drift rises from `0.8333` to `0.9167` while target margin stays positive.
- `coexact` remains a weaker traversal branch. It breaks `ontology_03`, but the
  eight-step target margin is near zero or negative and it does not reproduce
  the `ontology_05` moon trajectory.
- `presence` and `presence_plus_coexact` accumulate more surface token drift as
  the horizon lengthens, but their target margins stay negative. This supports
  the read that these branches can preserve local fluency/support without
  pushing the ontology target direction.
- `random_tangent` also drifts more at eight steps, but it does not become a
  semantic branch: its target margin remains negative and its generated text is
  mostly baseline-like.
- Nearest-node distance rises for all branches over later steps, including the
  baseline. That is expected because the generated prefix leaves the initial
  prompt chart; it is not yet evidence of branch-specific manifold collapse.

## Layer Pilot L5-L8

To test whether the `L7` branch split is layer-specific or part of a broader
middle-layer pattern, run a closed-loop layer pilot on the two branch-sensitive
ontology prompts:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08 \
  --layers 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_03 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08 \
  --output-dir spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 1149.9 seconds on CPU.

The summarizer now writes `closed_loop_layer_summary.csv`, and the plotter
renders `plots/closed_loop_layer_response.png`.

![Layer pilot response](../spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08/plots/closed_loop_layer_response.png)

Layer summary:

| layer | coexact-minus drift | coexact-minus target | coexact drift | coexact target | presence target | random target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L5 | 1.0000 | 0.3115 | 0.5000 | 0.3387 | -0.0628 | -0.0714 |
| L6 | 0.5000 | 0.3181 | 0.5000 | 0.0889 | 0.0824 | -0.0818 |
| L7 | 1.0000 | 0.2332 | 0.5000 | 0.0293 | -0.4265 | -0.0858 |
| L8 | 0.5000 | 0.1163 | 0.5000 | -0.1014 | -0.2799 | -0.0267 |

Overall component summary across L5-L8:

| component | drift | overlap | gain | KL | target margin | nearest dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.7500 | 0.2500 | 0.0402 | 0.0575 | 0.2448 | 0.3119 |
| coexact | 0.5000 | 0.5000 | -0.0053 | 0.0453 | 0.0888 | 0.3113 |
| presence_plus_coexact | 0.3750 | 0.6250 | -0.0046 | 0.0357 | -0.1611 | 0.3022 |
| presence | 0.2500 | 0.7500 | 0.0350 | 0.0378 | -0.1717 | 0.2992 |
| random_tangent | 0.0625 | 0.9375 | 0.0408 | 0.0254 | -0.0664 | 0.2986 |

Prompt-level read:

- `ontology_03`: `coexact_minus_presence` breaks at every tested layer,
  producing ` The room was a` from L5 through L8. `coexact` also breaks on this
  prompt, but with lower target margin at later layers.
- `ontology_05`: `coexact_minus_presence` breaks at L5 (` The moon was a`) and
  L7 (` The moon was in`). At L6 and L8 it stays token-locked while still
  increasing the ontology target margin.
- `presence` usually preserves the baseline continuation and has negative
  target margin at L5/L7/L8, even when its selected-token gain is positive.
- `random_tangent` remains mostly baseline-locked and negative in target margin,
  except for one `ontology_03/L8` surface drift.

Layer-pilot read:

- The closed-loop effect is not only an L7 artifact. `coexact_minus_presence`
  has positive mean target margin at every tested layer and is the strongest
  overall branch across L5-L8.
- L7 is still the cleanest closed-loop branch-separation point: drift is high
  for `coexact_minus_presence`, while presence, combined, and random remain
  baseline-locked with negative target margins.
- L5 also has strong traversal, but `coexact` and `coexact_minus_presence` are
  less separated there. This fits a broader middle-layer circulation story,
  with L7 providing the clearest causal branch split in this pilot.
- L8 weakens token drift for `coexact_minus_presence` while preserving positive
  target margin, suggesting a logits-facing stabilization phase where semantic
  pressure can remain sub-threshold.

## k Pilot at L7

To test whether the `L7` closed-loop branch split is sensitive to the kNN graph
neighborhood size, run a k-sweep at the cleanest layer:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_k_pilot_l7_a08 \
  --layers 7 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_03 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_k_pilot_l7_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_k_pilot_l7_a08 \
  --output-dir spiral_out_hltd_closed_loop_k_pilot_l7_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 579.5 seconds on CPU.

The summarizer now writes `closed_loop_k_summary.csv`, and the plotter renders
`plots/closed_loop_k_response.png`.

![k pilot response](../spiral_out_hltd_closed_loop_k_pilot_l7_a08/plots/closed_loop_k_response.png)

k summary:

| k | coexact-minus drift | coexact-minus target | coexact drift | coexact target | presence target | random target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.5000 | 0.2198 | 0.5000 | 0.0752 | -0.3445 | -0.0858 |
| 16 | 1.0000 | 0.2332 | 0.5000 | 0.0293 | -0.4265 | -0.0858 |
| 24 | 1.0000 | 0.2487 | 0.7500 | 0.2046 | -0.4203 | -0.0858 |

Overall component summary across k=12/16/24:

| component | drift | overlap | gain | KL | target margin | nearest dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.8333 | 0.1667 | 0.0555 | 0.0736 | 0.2339 | 0.3329 |
| coexact | 0.5833 | 0.4167 | 0.0286 | 0.0485 | 0.1030 | 0.3288 |
| presence | 0.0833 | 0.9167 | 0.1359 | 0.0531 | -0.3971 | 0.3111 |
| presence_plus_coexact | 0.0000 | 1.0000 | 0.0456 | 0.0353 | -0.2555 | 0.3062 |
| random_tangent | 0.0000 | 1.0000 | 0.0541 | 0.0249 | -0.0858 | 0.3062 |

Prompt-level read:

- `ontology_03`: `coexact_minus_presence` breaks at every tested k, always
  producing ` The room was a`. `coexact` also breaks for this prompt at every
  k, but its mean target margin is lower than `coexact_minus_presence` at
  k=12/16 and only catches up at k=24.
- `ontology_05`: `coexact_minus_presence` is sub-threshold at k=12, breaks at
  k=16 (` The moon was in`), and breaks at k=24 (` The moon was a`). Its
  target margin is positive at all three k values.
- `presence`, `presence_plus_coexact`, and `random_tangent` remain mostly
  baseline-locked and have negative mean target margins across the k-sweep.

k-pilot read:

- The branch split is not a single-k artifact. `coexact_minus_presence` is the
  only branch with positive target margin at every tested k and has the highest
  overall token drift.
- k=12 preserves semantic pressure but can leave token drift sub-threshold on
  `ontology_05`; k=16 and k=24 turn that pressure into a closed-loop token
  branch.
- Increasing k strengthens coexact-like traversal in this pilot. This is useful
  but also a caution: graph density changes branch strength, so k-sweeps should
  stay part of the robustness gate.

## Alpha-k Threshold Grid at L7

To see whether the observed branch break is an alpha-only threshold or an
alpha-by-graph-density surface, run a small grid over the two branch-sensitive
ontology prompts:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_alpha_k_grid_l7 \
  --layers 7 \
  --k 12 16 24 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.7 0.8 0.9 \
  --seeds 0 \
  --prompt-ids ontology_03 ontology_05 \
  --device cpu \
  --steering-components coexact_minus_presence coexact presence random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_alpha_k_grid_l7

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_alpha_k_grid_l7 \
  --output-dir spiral_out_hltd_closed_loop_alpha_k_grid_l7/plots \
  --components coexact_minus_presence coexact presence random_tangent
```

Runtime: 897.7 seconds on CPU.

The plotter now renders an alpha-by-k threshold surface for the preferred
`coexact_minus_presence` branch, plus a branch map that compares all requested
components at each k.

![Alpha-k threshold surface](../spiral_out_hltd_closed_loop_alpha_k_grid_l7/plots/closed_loop_alpha_k_threshold.png)

![Alpha-k branch map](../spiral_out_hltd_closed_loop_alpha_k_grid_l7/plots/closed_loop_alpha_k_branch_map.png)

`coexact_minus_presence` grid:

| k | drift a=0.7 | drift a=0.8 | drift a=0.9 | target a=0.7 | target a=0.8 | target a=0.9 | KL a=0.8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.5000 | 0.5000 | 0.5000 | 0.1415 | 0.2198 | 0.2995 | 0.0418 |
| 16 | 0.7500 | 1.0000 | 1.0000 | 0.2230 | 0.2332 | 0.2946 | 0.0806 |
| 24 | 0.7500 | 1.0000 | 1.0000 | 0.2853 | 0.2487 | 0.2919 | 0.0983 |

Component summary across the full alpha-k grid:

| component | alpha | drift | gain | KL | target margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.7 | 0.6667 | 0.0149 | 0.0412 | 0.2166 |
| coexact_minus_presence | 0.8 | 0.8333 | 0.0555 | 0.0736 | 0.2339 |
| coexact_minus_presence | 0.9 | 0.8333 | 0.0354 | 0.0978 | 0.2953 |
| coexact | 0.8 | 0.5833 | 0.0286 | 0.0485 | 0.1030 |
| presence | 0.8 | 0.0833 | 0.1359 | 0.0531 | -0.3971 |
| random_tangent | 0.8 | 0.0000 | 0.0541 | 0.0249 | -0.0858 |

Alpha-k read:

- The branch break is a surface, not just a scalar alpha cutoff. At k=12,
  `coexact_minus_presence` stays at mean drift `0.5` for all tested alphas,
  while k=16 and k=24 reach full mean drift at `alpha=0.8`.
- Target margin is positive for `coexact_minus_presence` in every grid cell.
  At k=12 it grows while token drift stays sub-threshold, matching the earlier
  read that semantic pressure can precede decoded-token movement.
- KL rises with alpha and k. The current sweet spot is still `alpha=0.8`,
  especially at k=16: it reaches full drift with lower KL than `0.9`, while
  keeping target margin positive.
- `presence` is the clean counter-branch in this grid: it has positive
  selected-token gain but strongly negative target margin and little token
  drift. `random_tangent` stays baseline-locked across all tested k and alpha
  values.
- k=24 strengthens traversal but also increases KL and lets pure `coexact`
  drift more. This makes k=24 useful for robustness, while k=16 remains the
  cleaner causal branch-separation setting.
- The branch map sharpens the dissociation: `coexact_minus_presence` is the
  only branch that combines high token drift with positive target margin at
  k=16/24. Pure `coexact` becomes drift-positive, especially at k=24, but its
  target margin stays lower. `presence` has almost the opposite signature:
  little decoded drift, positive selected-token gain, and strongly negative
  semantic target margin. `random_tangent` remains the matched baseline-locking
  control.

## Five-Prompt Ontology Robustness

To test whether the `L7/k16/alpha=0.8` branch split survives beyond the three
prompt panel, run all five ontology-collapse prompts:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_01 ontology_02 ontology_03 ontology_04 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 338.4 seconds on CPU.

The summarizer now writes `closed_loop_prompt_summary.csv`. It defines
`branch_gate_rate` as the fraction of prompt/layer/k/seed cells where token
drift is at least `0.5` and target margin is positive. It also reports
`branch_specific_gate_rate`, which requires the branch to satisfy that gate
while matching or exceeding the random tangent's token drift and exceeding its
target margin at the same prompt/layer/k/seed/alpha. The plotter renders
`plots/closed_loop_prompt_branch_gate.png`; when matched random-tangent columns
are present, it also renders `plots/closed_loop_prompt_random_advantage.png`.

![Five-prompt branch gate](../spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08/plots/closed_loop_prompt_branch_gate.png)

![Five-prompt random advantage](../spiral_out_hltd_closed_loop_ontology5_prompt_robust_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Component summary:

| component | drift | gain | KL | target margin | raw gates | branch-specific gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexact_minus_presence | 0.7000 | 0.0511 | 0.0516 | 0.1397 | 4/5 | 3/5 |
| coexact | 0.5000 | 0.0198 | 0.0397 | 0.0503 | 3/5 | 2/5 |
| presence_plus_coexact | 0.2000 | 0.0313 | 0.0315 | -0.1266 | 1/5 | 1/5 |
| presence | 0.1000 | 0.0136 | 0.0354 | -0.1670 | 0/5 | 0/5 |
| random_tangent | 0.1500 | -0.0414 | 0.0272 | 0.0464 | 1/5 | 0/5 |

Prompt-level branch gates:

| prompt | coexact-minus | coexact | presence+coexact | presence | random | main generated branch |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ontology_01 | 0 | 0 | 0 | 0 | 1 | all branches drift to `\n\nThe map`; only random has positive target margin |
| ontology_02 | 1 | 1 | 0 | 0 | 0 | coexact branches drift to `\n\nThe bird` |
| ontology_03 | 1 | 1 | 0 | 0 | 0 | coexact branches drift to ` The room was a` |
| ontology_04 | 1 | 1 | 1 | 0 | 0 | coexact branches drift to `\n\nThe door`; combined branch to `\n\nThe gate` |
| ontology_05 | 1 | 0 | 0 | 0 | 0 | coexact-minus drifts to ` The moon was in` |

Five-prompt read:

- `coexact_minus_presence` is the most robust closed-loop branch in this gate:
  it has the highest mean drift, highest mean target margin among non-control
  branches, raw gate success on four of five ontology prompts, and stricter
  branch-specific success on three of five.
- Pure `coexact` remains a real traversal branch but is less specific. It
  gates on `ontology_02/03/04`, fails on the moon prompt, and has lower mean
  target margin than `coexact_minus_presence`.
- `presence` is not a closed-loop traversal branch here. It can raise target
  margin without token drift on `ontology_02/04`, but its mean target margin is
  negative and its gate rate is zero.
- `presence_plus_coexact` is prompt-specific rather than robust: it gates only
  on `ontology_04`, where the generated target changes to `The gate`.
- `random_tangent` produces one false-positive gate on `ontology_01`, where all
  tested branches drift to a similar `The map` continuation. The stricter
  branch-specific gate removes that control-sensitive prompt and also drops
  `ontology_02` for `coexact_minus_presence`, where the branch drifts but does
  not beat random tangent on target margin.

## Seed Probe for Random-Tangent False Positives

The five-prompt panel showed one random-tangent false-positive on
`ontology_01`. To test whether that is just a single seed accident, rerun the
false-positive prompt and the strongest positive prompt over five random
tangent seeds:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids ontology_01 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 682.3 seconds on CPU.

![Seed probe branch gate](../spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08/plots/closed_loop_prompt_branch_gate.png)

![Seed probe random advantage](../spiral_out_hltd_closed_loop_seed_probe_ontology01_05_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Seed-probe summary:

| prompt | branch | gate rate | target+ rate | drift>=0.5 | target margin | generated pattern |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ontology_01 | coexact_minus_presence | 0.0 | 0.0 | 1.0 | -0.2930 | deterministic `\n\nThe map` drift, target-negative |
| ontology_01 | random_tangent | 0.6 | 0.8 | 0.8 | 0.1350 | mostly `\n\nThe map`, one `\n\n"The` |
| ontology_05 | coexact_minus_presence | 1.0 | 1.0 | 1.0 | 0.2497 | deterministic ` The moon was in` |
| ontology_05 | random_tangent | 0.2 | 0.6 | 0.2 | 0.0277 | mostly baseline, one ` The moon was a` |

Component-minus-random summary:

| prompt | branch | branch-specific gate | gate-random | drift-random | target-random | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ontology_01 | coexact_minus_presence | 0.0 | -0.6 | 0.05 | -0.4280 | not branch-specific; random tangent carries the positive gate |
| ontology_05 | coexact_minus_presence | 0.8 | 0.8 | 0.8 | 0.2221 | clean branch-specific advantage |

Seed-probe read:

- `ontology_05` is the cleaner branch-specific prompt. `coexact_minus_presence`
  gates in every seed-repeated cell; after subtracting the one random-tangent
  gate, the stricter branch-specific gate remains `4/5`.
- `ontology_01` is not a clean branch-specific prompt. Every deterministic
  branch drifts to `The map` but stays target-negative, while random tangent
  gets a positive target margin in four of five seeds and satisfies the current
  branch gate in three of five.
- This means the prompt-level gate is useful but not sufficient on its own.
  The component-minus-random advantage at the same prompt/layer/k/seed/alpha
  separates branch-specific effects from prompt/control sensitivity.
- The positive result therefore strengthens for `ontology_05` and weakens for
  `ontology_01`: the latter is better treated as a control-sensitive prompt,
  not as evidence for a Hodge branch effect.

## Identity Five-Prompt Robustness

The same `L7/k16/alpha=0.8` gate can be run on the five identity-stress
prompts to test whether branch specificity is ontology-only:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --families identity_stress \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 365.9 seconds on CPU.

![Identity prompt random advantage](../spiral_out_hltd_closed_loop_identity5_prompt_robust_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Component-minus-random summary:

| component | raw gates | branch-specific gates | target-random | read |
| --- | ---: | ---: | ---: | --- |
| coexact | 2/5 | 2/5 | 0.1549 | strongest identity closed-loop branch |
| coexact_minus_presence | 1/5 | 1/5 | 0.2709 | larger target advantage, fewer gate cells |
| presence_plus_coexact | 1/5 | 1/5 | 0.0073 | prompt-specific branch |
| presence | 0/5 | 0/5 | -0.0593 | stabilization/control branch |

Identity read:

- `identity_02` is the clean positive prompt: coexact, coexact-minus-presence,
  and presence-plus-coexact all beat matched random tangent.
- `identity_03` is more selective: pure `coexact` is the only strict
  branch-specific gate.
- `presence` again stays at zero strict gates, even though one-step probe
  summaries treat it as the stabilizing branch.
- The identity run therefore fills a second family in the branch-role matrix:
  closed-loop specificity is not limited to ontology-collapse prompts, but the
  winning branch changes from `coexact_minus_presence` for ontology to pure
  `coexact` for identity stress.

## Affordance Add-On Robustness

`affordance_stress` is a learned-probe label in the 20-prompt suite, but it is
not a prompt family there. To fill the closed-loop branch-role row without
changing the original 20-prompt baseline, use the add-on suite:

```text
data/hltd_affordance_prompt_suite.jsonl
```

Run it with the same `L7/k16/alpha=0.8` closed-loop gate:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_affordance_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 \
  --families affordance_stress \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 366.6 seconds on CPU.

![Affordance prompt random advantage](../spiral_out_hltd_closed_loop_affordance5_prompt_robust_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Component-minus-random summary:

| component | raw gates | branch-specific gates | target-random | read |
| --- | ---: | ---: | ---: | --- |
| coexact | 2/5 | 1/5 | 0.0193 | weak positive coexact target advantage |
| coexact_minus_presence | 1/5 | 1/5 | 0.0486 | weak positive target advantage, one strict cell |
| presence | 1/5 | 1/5 | -0.0671 | one strict cell but negative mean target advantage |
| presence_plus_coexact | 1/5 | 1/5 | -0.1357 | one strict cell, not robustly target-positive |

Affordance read:

- `affordance_01` is a presence/combined positive cell: `presence` and
  `presence_plus_coexact` both beat matched random tangent.
- `affordance_03` is control-sensitive. Coexact-derived branches pass the
  strict gate, but random tangent also satisfies the raw gate, so this prompt
  should not be over-read.
- Coexact-derived branches have the only positive mean target-random
  advantage, but the branch-specific gate support is only `1/5`.
- This fills the third branch-role matrix row but keeps the interpretation
  deliberately weak: affordance closed-loop support is present, not yet robust.

## Affordance Seed Probe

The first affordance panel had two important cells: `affordance_01`, where
presence-like branches passed, and `affordance_03`, where coexact-like branches
passed while random tangent also produced one raw gate. Rerun those prompts
over five random-tangent seeds:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_affordance_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids affordance_01 affordance_03 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08/plots \
  --components presence_plus_coexact coexact_minus_presence presence coexact random_tangent
```

Runtime: 824.3 seconds on CPU.

![Affordance seed-probe random advantage](../spiral_out_hltd_closed_loop_seed_probe_affordance01_03_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Seed-probe summary:

| prompt | branch | branch-specific gate | random gate | gate-random | target-random | generated pattern |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affordance_01 | presence | 1.0 | 0.2 | 0.8 | 0.1891 | deterministic `The next` drift |
| affordance_01 | presence_plus_coexact | 1.0 | 0.2 | 0.8 | 0.1648 | deterministic `The silence` drift |
| affordance_01 | coexact | 0.4 | 0.2 | 0.8 | 0.0273 | `The room` drift, weaker target advantage |
| affordance_03 | coexact_minus_presence | 1.0 | 0.2 | 0.8 | 0.3051 | deterministic `The next` drift |
| affordance_03 | coexact | 1.0 | 0.2 | 0.8 | 0.1386 | deterministic `The next` drift |
| affordance_03 | presence / combined | 0.0 | 0.2 | -0.2 | negative | `The river` drift, target-negative |

Seed-probe read:

- `affordance_01` is a presence/combined branch cell. The effect survives all
  five random-tangent seeds with positive target-random margin.
- `affordance_03` is a coexact-derived branch cell. `coexact` and
  `coexact_minus_presence` survive all five seeds, while random tangent drops
  to one raw gate in five.
- This improves the affordance read: it is no longer just weak support spread
  across branches. It splits by prompt into presence-like stabilization/control
  for `affordance_01` and coexact-like traversal for `affordance_03`.
- The family-level target-random advantage still favors coexact-derived
  directions, while the learned-probe margin still favors `presence`. That is a
  useful dissociation rather than a contradiction.

## Negative-Coexact Sign Control

The branch-role matrix still had one gray column after the affordance seed
probe: `negative_coexact`. To make that an explicit sign-control measurement,
first rerun representative positive prompts with only `negative_coexact` and
matched `random_tangent`, then expand the identity and ontology controls to the
full five-prompt families.

Identity/ontology control:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_sign_control_identity02_ontology05_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids identity_02 ontology_05 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Full-family identity control:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --families identity_stress \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Full-family ontology control:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --families ontology_collapse \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Affordance control:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_affordance_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 4 \
  --alphas 0.8 \
  --seeds 0 1 2 3 4 \
  --prompt-ids affordance_01 affordance_03 \
  --device cpu \
  --steering-components negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Then summarize and render each root with the usual closed-loop tools:

```bash
python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_sign_control_identity02_ontology05_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_sign_control_identity02_ontology05_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_sign_control_identity02_ontology05_l7_k16_a08/plots \
  --components negative_coexact random_tangent

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08/plots \
  --components negative_coexact random_tangent

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08/plots \
  --components negative_coexact random_tangent

python3 scripts/summarize_hltd_closed_loop.py \
  --run-root spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08

python3 scripts/plot_hltd_closed_loop.py \
  --summary-root spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08 \
  --output-dir spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08/plots \
  --components negative_coexact random_tangent
```

Runtimes:

- identity/ontology pilot: 324.2 seconds on CPU
- identity full-family: 818.4 seconds on CPU
- ontology full-family: 797.8 seconds on CPU
- affordance: 366.5 seconds on CPU

![Negative coexact identity full-family control](../spiral_out_hltd_closed_loop_sign_control_identity5_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

![Negative coexact ontology full-family control](../spiral_out_hltd_closed_loop_sign_control_ontology5_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

![Negative coexact affordance control](../spiral_out_hltd_closed_loop_sign_control_affordance01_03_l7_k16_a08/plots/closed_loop_prompt_random_advantage.png)

Sign-control summary:

| prompt | branch-specific gate | random gate | gate-random | target-random | read |
| --- | ---: | ---: | ---: | ---: | --- |
| identity_01 | 0.0 | 0.0 | 0.0 | 0.0182 | clean strict control |
| identity_02 | 0.0 | 0.2 | -0.2 | 0.0599 | target-positive but no drift gate |
| identity_03 | 0.0 | 0.2 | -0.2 | -0.3943 | clean strict control |
| identity_04 | 1.0 | 0.0 | 1.0 | 0.3630 | exception: reverse coexact passes all seeds |
| identity_05 | 0.0 | 0.0 | 0.0 | 0.1562 | target-positive but no drift gate |
| ontology_01 | 0.0 | 0.6 | -0.6 | 0.3721 | random-sensitive, no strict reverse branch |
| ontology_02 | 0.0 | 0.4 | -0.4 | -0.0192 | clean strict control |
| ontology_03 | 0.0 | 0.4 | -0.4 | -0.0497 | clean strict control |
| ontology_04 | 0.0 | 0.0 | 0.0 | -0.0551 | clean strict control |
| ontology_05 | 0.4 | 0.2 | 0.8 | 0.0977 | exception: reverse coexact can still move ontology target |
| affordance_01 | 0.0 | 0.2 | -0.2 | -0.0379 | clean sign-control |
| affordance_03 | 0.0 | 0.2 | -0.2 | -0.1170 | clean sign-control |

Sign-control read:

- `negative_coexact` behaves as a clean negative control for the measured
  affordance prompts and for most identity/ontology prompts.
- It is not a universal null: `identity_04` and `ontology_05` are prompt-local
  reverse-branch exceptions. The full-family rates stay low, so the exception
  is not evidence for a family-wide reverse traversal branch.
- The final matrix column is therefore measured, not missing. Its pattern is
  mostly suppressive with two prompt-local exceptions.

## Reverse-Exception Layer-k Localization

The follow-up localization run is documented in:

```text
docs/hltd_reverse_exception_localization.md
```

Short read:

- `identity_04` is a middle-layer, medium/high-k exception: L5 and L7 pass at
  k=16/24, while k=12 and all tested L8 cells fail. In the five-seed
  passing-band follow-up, L7 reaches branch-specific gate 1.00 at k=16/24.
- `ontology_05` is a late-layer exception: L5 fails the strict gate, L7 is
  partial, and all tested L8 cells pass. In the five-seed follow-up, L8 remains
  branch-specific at 0.80 across k=12/16/24.
- These targeted rows are kept out of the family-level branch ledger because
  they intentionally oversample exception prompts.

## Identity-02 Branch Robustness Surface

The focused follow-up expands the earlier `identity_02` carrier across
`layer=4/5/7`, `k=12/16/24`, `alpha=0.4/0.8/1.2`, all five deterministic
Hodge-derived branches, and 20 matched random-tangent seeds:

```text
spiral_out_hltd_closed_loop_identity02_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19
```

![Identity-02 branch robustness surface](../spiral_out_hltd_closed_loop_identity02_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/closed_loop_prompt_layer_alpha_k_surface.png)

Layer-level read over the nine `k x alpha` cells:

| layer | branch | mean specific gate | cells >= 0.5 | mean target-random |
| ---: | --- | ---: | ---: | ---: |
| 4 | presence | 0.672 | 7/9 | 0.125 |
| 4 | coexact | 0.556 | 6/9 | 0.119 |
| 5 | coexact | 0.711 | 9/9 | 0.066 |
| 5 | presence + coexact | 0.689 | 8/9 | 0.089 |
| 7 | coexact | 0.761 | 8/9 | 0.151 |
| 7 | coexact - presence | 0.922 | 9/9 | 0.385 |
| 7 | presence | 0.139 | 2/9 | -0.194 |
| 7 | negative coexact | 0.000 | 0/9 | -0.147 |

The surface resolves three effects that the fixed `L7/k16/alpha=0.8` run
could not separate:

- The active branch changes with depth. Presence is strongest at L4, plain
  coexact is the most uniform branch at L5, and `coexact - presence` takes over
  at L7.
- Orientation matters. Negative coexact has zero strict support at L4 and L7,
  with target-minus-random negative in every tested cell at both layers.
- Presence subtraction is specifically late-layer enhancing. At L7,
  `coexact - presence` passes all nine hyperparameter cells, while presence
  alone becomes target-negative on average. At `L7/k16/alpha=1.2`, the former
  reaches target-minus-random `+0.730`.

There is also a useful boundary: plain coexact fails at
`L7/k24/alpha=1.2` (specific gate `0.00`, target-minus-random `-0.19`), while
`coexact - presence` remains active there (specific gate `0.85`,
target-minus-random `+0.41`). This argues against a norm-only explanation and
for a layer-conditioned interaction between the two Hodge branches.

The 20 seeds are matched random-tangent strata. Greedy baseline and
deterministic Hodge branches are computed once and copied into each stratum,
so fractional branch-specific rates estimate how often a fixed branch beats a
random direction. They are not 20 independent realizations of the branch.

Construct-validity limit at this stage: `identity_02` contains words such as
`statue` and `sculptor`, and the family target set contains the same words. The
full-target result is therefore evidence for oriented, branch-specific semantic
pressure, not by itself evidence for identity or ontology drift. The next
section directly tests how much of the surface depends on those two terms.

### Prompt-held-out target robustness

The exact surface was rescored after removing target terms that occur as whole
lexeme phrases in the prompt. For `identity_02`, this excludes `statue` and
`sculptor`, leaving 12 target terms and 24 first-token IDs. The runner records
the source set, effective terms, excluded terms, and token counts in
`closed_loop_manifest.json`.

```text
spiral_out_hltd_closed_loop_identity02_prompt_heldout_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19
```

![Identity-02 prompt-held-out branch surface](../spiral_out_hltd_closed_loop_identity02_prompt_heldout_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/closed_loop_prompt_layer_alpha_k_surface.png)

Layer-level read over the nine `k x alpha` cells:

| layer | branch | mean specific gate | cells >= 0.5 | mean target-random | change from full target |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | presence | 0.756 | 8/9 | 0.106 | -0.019 |
| 4 | coexact | 0.589 | 6/9 | 0.161 | +0.043 |
| 5 | coexact | 0.744 | 9/9 | 0.098 | +0.032 |
| 5 | presence + coexact | 0.639 | 8/9 | 0.094 | +0.006 |
| 7 | coexact | 0.844 | 9/9 | 0.218 | +0.068 |
| 7 | coexact - presence | 0.894 | 9/9 | 0.306 | -0.079 |
| 7 | presence | 0.178 | 2/9 | -0.147 | +0.047 |
| 7 | negative coexact | 0.000 | 0/9 | -0.293 | -0.146 |

The paired delta surface makes the measurement change explicit:

![Identity-02 prompt-target overlap delta](../spiral_out_hltd_closed_loop_identity02_prompt_heldout_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/target_overlap_robustness_identity_02.png)

The late-layer carrier survives the stricter vocabulary gate. L7
`coexact - presence` remains above the 0.5 specific-gate threshold in all nine
cells and remains above matched random in all nine cells. Its continuous
target-minus-random advantage falls from `+0.385` to `+0.306`, so the two
prompt words contributed signal but did not create the branch localization. At
`L7/k16/alpha=1.2`, the advantage changes from `+0.730` to `+0.526`; at
`L7/k24/alpha=1.2`, it changes from `+0.415` to `+0.228` and still passes.

The sign control becomes sharper: L7 negative coexact remains zero-gate in all
nine cells, and its mean target-minus-random value moves from `-0.147` to
`-0.293`. Plain L7 coexact also strengthens on average. Together these results
support an oriented branch effect that is not reducible to recovering the
prompt's `statue/sculptor` tokens.

This is a scorer-only comparison. All 3,420 closed-loop metric rows match on
generated text, token IDs, active steps, visited nodes, intervention norm, KL,
entropy change, top-token change, and baseline overlap. Only the semantic
target scoring changes. The strict gate is thresholded and can jump sharply
near zero, so the continuous target-minus-random values remain the primary
robustness read.

Remaining limit: prompt-overlap exclusion rules out direct lexical reuse, but
the retained family vocabulary is still a hand-built semantic proxy. A learned
ontology/identity probe or prompt-specific counterfactual target set is needed
before calling the effect identity drift.

### Disjoint Learned Identity-Probe Follow-Up

The first prompt-disjoint probe result is now recorded in
[hltd_counterfactual_identity_probe.md](hltd_counterfactual_identity_probe.md).
It trains on 12 matched artifact/creator counterfactual pairs, evaluates the
held-out `identity_02` prompt, and compares every deterministic branch with 20
matched random tangents.

![Disjoint counterfactual identity-probe branch surface](../spiral_out_hltd_identity02_counterfactual_probe_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19/plots/counterfactual_identity_stress_branch_surface.png)

At L4-L5 and k=12/16, presence moves toward the learned identity-transfer
regime while coexact and coexact-minus-presence move away from it. At L7, the
earlier closed-loop lexical carrier remains positive but the learned-axis
effect stays inside the random-tangent range. The two measurements therefore
separate lexical semantic transport from movement along this identity-transfer
classifier. The closed-loop claim remains narrow: identity drift has not yet
been established.

## Current Read

The useful signal in this smoke is mechanical:

- `closed_loop_steps.csv` records the nearest Hodge-field node at each generated
  step.
- `closed_loop_metrics.csv` records branch-level generated text, base-model
  logprob, KL, semantic-target margin, and baseline token overlap.
- `closed_loop_prompt_summary.csv` records prompt-level branch gate rates,
  branch-specific gate rates, target-positive rates, and drift rates.
- The branch direction is recomputed by nearest-node lookup at each prefix, so
  this is a real field-following gate rather than a one-vector ablation.

The current scientific read is branch-specific and still small-N:

- `presence_plus_coexact`: combined traversal/stabilization branch.
- `coexact_minus_presence`: traversal with presence removed; strongest current
  closed-loop breaker.
- `coexact`: local circulation branch; prompt-sensitive breaker.
- `presence`: stabilization branch.
- `random_tangent`: matched control.
- The alpha-k grid adds a threshold-surface read: `coexact_minus_presence`
  becomes a reliable closed-loop breaker at k=16/24 and alpha >= 0.8, while
  k=12 can keep semantic target pressure sub-threshold.
- The identity-stress five-prompt panel adds a family contrast: pure `coexact`
  is the strongest strict identity branch, while `presence` remains a
  non-traversal stabilizing branch.
- The affordance add-on plus seed probe fills the remaining probe-family row.
  It splits into two prompt-level modes: presence-like branches for
  `affordance_01`, and coexact-derived branches for `affordance_03`.
- The `negative_coexact` sign-control fills the final matrix column. It is a
  clean closed-loop control for the measured affordance prompts and most
  identity/ontology prompts, with prompt-local exceptions at `identity_04` and
  `ontology_05`.

Recommended next command, if extending the panel:

```bash
python3 scripts/run_hltd_closed_loop.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8 \
  --layers 7 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --generate-steps 8 \
  --alphas 0.8 \
  --seeds 0 \
  --prompt-ids ontology_01 ontology_03 ontology_05 \
  --device cpu \
  --steering-components presence_plus_coexact coexact_minus_presence presence coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

This run is intentionally CPU-oriented for now. In the first smoke attempt,
`--device mps` stalled during model loading on this machine, while CPU loaded
and completed the tiny gate. Revisit MPS once the harness is stable.
