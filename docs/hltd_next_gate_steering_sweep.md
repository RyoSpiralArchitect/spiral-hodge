# HLTD Next-Gate Steering Sweep

This note records a small next-gate expansion of the one-step HLTD causal
steering experiment. The goal is not to make a final semantic claim, but to
make the previous steering gate harder to pass by varying layer, random-tangent
seed, and token-selection rule.

## Run

The lite gate used local GPT-2, one prompt from each prompt family, layers 4-8,
two random-tangent seeds, and two token selectors:

```bash
python3 scripts/run_hltd_steering_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_next_gate_lite \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --max-prompts-per-family 1
```

Generated summaries:

- `spiral_out_hltd_next_gate_lite/summary.csv`
- `spiral_out_hltd_next_gate_lite/summary_component.csv`
- `spiral_out_hltd_next_gate_lite/summary_pairwise.csv`
- `spiral_out_hltd_next_gate_lite/summary_layer_pairwise.csv`
- `spiral_out_hltd_next_gate_lite/summary_report.md`

The run produced 20 prompt/layer/k steering runs and 400 steering rows.

## What Changed

The steering harness now supports:

- multiple random-tangent seeds in a single run with `--seeds`
- token-selection sweeps with `--token-selectors`
- `max_component`, `middle`, `fixed`, and `all_interior` token selectors
- `--token-indices` for multi-token fixed steering
- `summary_layer_pairwise.csv`, which aggregates component-minus-random
  contrasts by layer and token selector

Pairwise comparisons are keyed by family, prompt, layer, k, seed,
token-selector, token index, and alpha. This prevents a random tangent from one
seed or token position being compared to a component direction from another.

## Layer Pairwise Read

The table below shows coexact steering minus matched random-tangent steering.
Positive next-token delta means coexact increased the teacher-forced next-token
log probability more than random tangent did.

| Layer | Selector | n | Next-token delta | KL delta | Entropy delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| L4 | max_component | 8 | -0.1253 | -0.0267 | 0.1262 |
| L4 | middle | 8 | 0.1736 | 0.0117 | -0.0452 |
| L5 | max_component | 8 | 0.1373 | 0.0928 | 0.1982 |
| L5 | middle | 8 | 0.7084 | 0.0098 | -0.0785 |
| L6 | max_component | 8 | -0.2183 | 0.1055 | 0.5868 |
| L6 | middle | 8 | 0.6380 | 0.0056 | -0.0451 |
| L7 | max_component | 8 | -0.2087 | -0.0236 | 0.2943 |
| L7 | middle | 8 | 0.8038 | 0.0125 | -0.0794 |
| L8 | max_component | 8 | 0.5827 | -0.0656 | -0.2109 |
| L8 | middle | 8 | 1.0693 | 0.0214 | 0.0103 |

## Interpretation

The strongest pattern in this lite run is selector-dependent. The middle-token
selector shows a stable positive coexact-minus-random next-token delta from L5
through L8, while the max-component selector is less stable and changes sign in
L6 and L7.

That suggests the previous "pick the maximum coexact norm" rule may be too
local and too sensitive to the selected token. The coexact direction still moves
logits causally, but the more robust next-gate object may be:

> coexact steering under matched random-tangent controls, evaluated across
> token-selection rules rather than only at the maximum coexact node.

The lite run also keeps the earlier warning alive: KL movement and
teacher-forced next-token support are not the same quantity. Max-component
coexact has the largest positive KL delta at L5-L6, while middle-token coexact
has the cleaner next-token support pattern.

## Family Notes

The family-level coexact-minus-random next-token deltas show why selector
robustness matters:

- `ontology_collapse` with `middle` is consistently positive from L4-L8,
  peaking at L8.
- `metaphor_shift` with `middle` becomes strongly positive from L5-L8, but
  `max_component` is negative until L8.
- `identity_stress` is positive at L5 and L7-L8, especially with the middle
  selector.
- `literal_stable` is mildly positive for middle-token steering, but this is a
  small one-prompt sample and should not be overread.

## Not Yet Claimed

This gate still does not establish:

- multi-step semantic drift
- fluency preservation under closed-loop steering
- identity or affordance drift
- that coexact is uniquely semantic rather than one causal direction among
  several
- harmonic/global concept-ring behavior

## Next Gate

The next implementation gate should reduce runtime and widen the sample:

1. Load GPT-2 once per suite, not once per prompt/layer subprocess.
2. Reuse hidden states and PCA coordinates across the layer sweep for a prompt.
3. Run the same L4-L8 seed/selector sweep on all 20 current prompts.
4. Add `all_interior` token-selection summaries with per-token quantiles rather
   than only means.
5. Add semantic target scores or probe scores so the gate is not only
   teacher-forced next-token support.

## Fast Full-Suite Update

The next implementation gate is now in place:

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

The fast runner loads GPT-2 once, extracts hidden states once per prompt, reuses
the PCA chart across layers, and batches same-token steering deltas. On the MPS
device, the full 20-prompt L4-L8 suite completed 100 prompt/layer/k runs and
2000 steering rows in 38.8 seconds.

Generated summaries:

- `spiral_out_hltd_fast_full_mps/summary.csv`
- `spiral_out_hltd_fast_full_mps/summary_component.csv`
- `spiral_out_hltd_fast_full_mps/summary_pairwise.csv`
- `spiral_out_hltd_fast_full_mps/summary_layer_pairwise.csv`
- `spiral_out_hltd_fast_full_mps/summary_report.md`

Because this run uses MPS, it should be treated as a fast exploratory gate.
A one-layer smoke comparison against the CPU run preserved row keys and kept
the main scalar differences small, but it was not bit-identical.

## Full-Suite Layer Read

Across all 20 prompts, coexact-minus-random next-token support is positive for
the middle-token selector at every tested layer and strongest from L5-L8:

| Layer | Selector | n | Next-token delta | KL delta | Entropy delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| L4 | max_component | 40 | -0.0040 | 0.0590 | -0.2372 |
| L4 | middle | 40 | 0.1038 | -0.0711 | -0.1627 |
| L5 | max_component | 40 | 0.0718 | 0.0112 | -0.0356 |
| L5 | middle | 40 | 0.4372 | -0.0418 | -0.1774 |
| L6 | max_component | 40 | 0.1014 | -0.0788 | -0.0511 |
| L6 | middle | 40 | 0.3100 | -0.0651 | -0.0458 |
| L7 | max_component | 40 | -0.0950 | -0.0267 | -0.0538 |
| L7 | middle | 40 | 0.4516 | -0.0104 | 0.0616 |
| L8 | max_component | 40 | 0.1702 | -0.0557 | -0.2462 |
| L8 | middle | 40 | 0.4940 | 0.0151 | 0.1530 |

This strengthens the selector-sensitivity conclusion from the lite run. The
middle selector remains cleaner for the teacher-forced next-token metric, while
max-component selection is weaker and changes sign at L4 and L7.

## Full-Suite Family Read

The strongest family-level signal is `ontology_collapse` with the middle-token
selector:

| Family | Layer | Selector | n | Next-token delta | KL delta | Entropy delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ontology_collapse | L4 | middle | 10 | 0.4989 | -0.0687 | -0.0343 |
| ontology_collapse | L5 | middle | 10 | 1.2837 | -0.0182 | 0.1082 |
| ontology_collapse | L6 | middle | 10 | 0.7695 | -0.0850 | 0.0703 |
| ontology_collapse | L7 | middle | 10 | 1.7046 | -0.0500 | 0.4530 |
| ontology_collapse | L8 | middle | 10 | 1.6817 | -0.0403 | 0.3018 |

`metaphor_shift` also becomes positive in middle layers, especially L6 and L8.
`identity_stress` is mixed, and `literal_stable` is weaker and selector
dependent. That makes the cleanest current claim:

> In a fast MPS full-suite one-step gate, coexact steering under the middle-token
> selector consistently exceeds matched random-tangent steering on
> teacher-forced next-token support, with the strongest family-level effect in
> ontology-collapse prompts.

This still remains a one-step next-token result. It should not be upgraded to
semantic drift, identity drift, or fluency preservation until semantic probes
or closed-loop generation are added.

## Semantic Target Gate Update

The first coarse semantic-target gate adds family-specific lexical
target/control sets from:

```text
data/hltd_semantic_targets.json
```

Run:

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

The run completed 100 prompt/layer/k steering runs and 2000 steering rows in
43.8 seconds. The new gate tracks:

- target log-probability mass delta
- control log-probability mass delta
- semantic margin delta, defined as target mass delta minus control mass delta

All values below are coexact steering minus matched random-tangent steering.

| Layer | Selector | n | Next-token delta | Target mass delta | Semantic margin delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| L4 | max_component | 40 | -0.0040 | -0.2853 | -0.1839 |
| L4 | middle | 40 | 0.1038 | -0.2756 | -0.2122 |
| L5 | max_component | 40 | 0.0718 | -0.1812 | -0.1594 |
| L5 | middle | 40 | 0.4372 | -0.3791 | -0.2254 |
| L6 | max_component | 40 | 0.1014 | -0.2639 | -0.2078 |
| L6 | middle | 40 | 0.3100 | -0.2645 | -0.2200 |
| L7 | max_component | 40 | -0.0950 | -0.3535 | -0.3773 |
| L7 | middle | 40 | 0.4516 | -0.2258 | -0.1617 |
| L8 | max_component | 40 | 0.1702 | -0.2099 | -0.0419 |
| L8 | middle | 40 | 0.4940 | -0.1722 | 0.0786 |

This is a useful negative-pressure result. The coexact middle-token gate still
supports the teacher-forced next token across layers, but the lexical semantic
margin does not globally clear random tangent until L8. So the current claim
should not be "coexact always pushes the target concept set."

The family split is sharper:

| Family | Layer | Selector | n | Next-token delta | Target mass delta | Semantic margin delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| identity_stress | L5 | middle | 10 | 0.0634 | 0.2490 | 0.2237 |
| identity_stress | L7 | middle | 10 | 0.0753 | 0.3739 | 0.2100 |
| identity_stress | L8 | middle | 10 | -0.1936 | 0.4894 | 0.3789 |
| ontology_collapse | L7 | middle | 10 | 1.7046 | 0.0204 | 0.0980 |
| ontology_collapse | L8 | middle | 10 | 1.6817 | -0.1871 | 0.2125 |
| metaphor_shift | L8 | middle | 10 | 0.7049 | -0.2418 | 0.1224 |
| literal_stable | L8 | middle | 10 | -0.2169 | -0.7494 | -0.3992 |

The best read is family- and layer-dependent:

> Coexact steering has a robust one-step next-token effect under the middle
> selector. A coarse lexical semantic-margin effect appears mainly in
> identity-stress and late ontology-collapse prompts, while literal prompts
> act as a useful negative control.

`semantic_flow` is nearly identical to `coexact` in this run because harmonic
directions are rarely active. Presence does not explain the late aggregate
semantic-margin improvement: at L8/middle, presence is negative on the
layer-averaged semantic margin while coexact and semantic-flow are positive.

This gate is deliberately lexical. It is weaker than an ontology or affordance
probe, and should be treated as a bridge between next-token steering and the
next learned-probe gate.

## Learned Probe Gate Update

The next gate trains lightweight binary linear probes directly on hidden states
from the 20-prompt suite, then scores the same one-step HLTD component
directions without running the logits head. Probe labels are defined in:

```text
data/hltd_probe_labels.json
```

Run:

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

The full run completed 100 prompt/layer/k probe runs and 6000 probe rows in
11.6 seconds. Outputs:

- `spiral_out_hltd_probe_gate_full_mps/probe_metrics.csv`
- `spiral_out_hltd_probe_gate_full_mps/probe_training_summary.csv`
- `spiral_out_hltd_probe_gate_full_mps/summary_component.csv`
- `spiral_out_hltd_probe_gate_full_mps/summary_pairwise.csv`
- `spiral_out_hltd_probe_gate_full_mps/summary_layer_pairwise.csv`
- `spiral_out_hltd_probe_gate_full_mps/summary_report.md`

Probe prompt-CV accuracy is usable but not final:

| Probe | Best prompt-CV acc | Layer |
| --- | ---: | ---: |
| identity_stress | 0.8053 | L8 |
| ontology_collapse | 0.6880 | L7 |
| affordance_stress | 0.6404 | L6 |

Layer-averaged middle-token coexact-minus-random label-margin deltas:

| Layer | Identity probe | Ontology probe | Affordance probe |
| ---: | ---: | ---: | ---: |
| L4 | -0.8930 | -0.4323 | -0.8269 |
| L5 | -0.1762 | -0.0817 | -2.0080 |
| L6 | -1.0040 | -0.4329 | -1.6766 |
| L7 | -0.7548 | 0.3632 | -0.7619 |
| L8 | -0.4250 | 0.4686 | 0.0611 |

This is the sharpest separation so far. The earlier next-token and lexical
target gates made coexact look promising, especially under the middle-token
selector. The learned probe gate says something more constrained:

> Coexact does not globally move hidden states toward the suite's learned
> identity/ontology/affordance labels. Its clearest learned-probe signal is
> late ontology movement at L7-L8, with a small L8 affordance signal.

Presence is much stronger on label-consistent probe margin:

| Layer | Identity presence | Ontology presence | Affordance presence |
| ---: | ---: | ---: | ---: |
| L4 | 0.9423 | 0.9728 | 1.1250 |
| L5 | 1.0827 | 1.8824 | 1.9406 |
| L6 | 1.2472 | 1.2499 | 1.0815 |
| L7 | 0.7495 | 1.0265 | 1.0356 |
| L8 | 0.6155 | 0.4438 | 0.6206 |

That suggests a revised causal interpretation:

- presence is currently the stronger label-stabilizing / probe-consistent
  direction;
- coexact remains the stronger next-token support direction under the middle
  selector;
- coexact learned-probe drift is narrow, appearing mainly in the late ontology
  window rather than across all semantic probes.

Probability and logit probe deltas sometimes disagree because the binary probes
can saturate. For the current note, label-margin logit deltas are treated as
the more linear read, while probability deltas are kept in the CSV for sanity
checks.

This is still an internal-suite probe. The next robust version should train on
a larger prompt bank or an external ontology/affordance-labeled dataset, then
repeat the same component-minus-random gate.

## Dissociation Gate Update

The explicit presence/coexact dissociation gate is written up in:

```text
docs/hltd_dissociation_gate.md
```

Short read: derived directions make the role split much cleaner. `presence`
has weak or negative next-token support but strongly positive learned-probe
label margin. `coexact` has robust next-token support and only becomes
learned-probe aligned in a narrower late ontology window. The sum,
`presence_plus_coexact`, preserves next-token support while restoring positive
ontology probe margin; `coexact_minus_presence` keeps some next-token support
but drops probe stabilization. This is the strongest current evidence for the
presence-as-basin / coexact-as-traversal interpretation.

## Branch Hodge Ledger Update

The structural and causal reads are joined in:

```text
docs/hltd_branch_hodge.md
```

This ledger is the current checkpoint for comparing Hodge branches side by
side: structural coexact energy, structural exact/presence energy, one-step
next-token support, lexical semantic margin, and learned-probe label margin.
It now includes a matched `k=12/16/24` causal/probe sweep, where the main
presence/coexact dissociation survives the neighborhood scale change.
The selector comparison keeps the earlier selector warning alive: `middle`
is stronger for coexact next-token traversal, while `max_component` more often
amplifies presence/probe stabilization.
The all-interior structural heatmap in `docs/hltd_branch_heatmap.md` turns
that warning into a localization read: coexact peaks tend to be mid-to-late,
while presence shifts earlier as k increases.
