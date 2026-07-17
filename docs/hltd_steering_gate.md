# HLTD One-Step Steering Gate

This note records the first causal steering gate for Hodge-Latent Traversal
Dynamics. The goal is not to prove semantic causality yet. The goal is narrower:
test whether reconstructed HLTD components can move next-token logits, and
whether that movement differs from a matched random tangent direction.

## Run

The gate used local GPT-2 and the 20-prompt family suite:

```bash
python3 scripts/run_hltd_steering_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_steering_suite \
  --layers 5 \
  --k 16 \
  --components 32 \
  --max-length 96 \
  --alphas 0.5 1.0
```

Summary artifacts:

- `spiral_out_hltd_steering_suite/summary.csv`
- `spiral_out_hltd_steering_suite/summary_component.csv`
- `spiral_out_hltd_steering_suite/summary_pairwise.csv`
- `spiral_out_hltd_steering_suite/summary_report.md`

The run produced 20 prompt/layer/k steering runs and 200 component/alpha rows.

## Method

For each prompt, the script:

1. Extracts GPT-2 hidden states and builds a PCA-32 chart.
2. Runs centered HLTD at layer 5 with k=16.
3. Selects the interior token node with the largest coexact node-vector norm.
4. Reconstructs node-vector fields for presence, coexact, harmonic, and
   semantic flow.
5. Maps PCA-chart differential vectors back into hidden-space directions.
6. Adds a scaled direction to the selected hidden state.
7. Compares the base and steered next-token logits.

The steering components are:

- `presence`
- `coexact`
- `semantic_flow`
- `harmonic`
- `random_tangent`

`random_tangent` is norm-matched to the coexact chart vector at the selected
node. Components whose selected chart norm falls below the active threshold are
treated as no-op controls. Pairwise component-minus-random summaries exclude
inactive rows.

## Metrics

The main metrics are:

- `kl_base_to_steered`: how much the intervention moves the distribution
- `entropy_delta`: whether the distribution sharpens or diffuses
- `next_token_logprob_delta`: whether the teacher-forced next token is
  supported
- `top_shift_logprob_delta`: largest individual token log-probability increase
- `top_changed`: whether the argmax next token changes

The most useful read is comparative. Raw KL alone can reward generic
disruption, so the key table is component minus random tangent.

## Alpha 1 Component Means

At alpha 1.0:

| Family | Component | KL | Entropy delta | Next-token delta | Top changed |
| --- | --- | ---: | ---: | ---: | ---: |
| literal_stable | coexact | 0.1265 | 0.2803 | -0.2125 | 0.0 |
| literal_stable | random_tangent | 0.1211 | 0.4770 | -0.2046 | 0.0 |
| metaphor_shift | coexact | 0.4534 | 0.3858 | 0.3678 | 0.2 |
| metaphor_shift | random_tangent | 0.5262 | 0.6572 | 0.4776 | 0.4 |
| identity_stress | coexact | 0.1232 | -0.0317 | 0.1942 | 0.2 |
| identity_stress | random_tangent | 0.1030 | 0.0997 | 0.0338 | 0.0 |
| ontology_collapse | coexact | 0.0878 | 0.3541 | 0.0498 | 0.2 |
| ontology_collapse | random_tangent | 0.2193 | 0.1020 | -0.2410 | 0.4 |

This table says coexact steering is not simply "larger perturbation wins." In
ontology-collapse prompts, coexact moves the logits less than random tangent but
supports the teacher-forced next token more. In identity-stress prompts, coexact
also improves next-token support relative to random tangent.

## Component Minus Random Tangent

At alpha 1.0:

| Family | Component | KL delta | Entropy delta | Next-token delta | Top-shift delta |
| --- | --- | ---: | ---: | ---: | ---: |
| identity_stress | coexact | 0.0202 | -0.1315 | 0.1604 | -0.0271 |
| literal_stable | coexact | 0.0054 | -0.1966 | -0.0079 | -0.2517 |
| metaphor_shift | coexact | -0.0729 | -0.2713 | -0.1099 | 0.1526 |
| ontology_collapse | coexact | -0.1315 | 0.2521 | 0.2908 | -1.5780 |

The strongest positive coexact-minus-random next-token effects appear in:

- `ontology_collapse`: +0.2908
- `identity_stress`: +0.1604

Literal prompts remain effectively neutral, and metaphor prompts show a
different pattern: random tangent has stronger teacher-forced next-token support
in this one-step setup.

## Presence Comparison

Presence is not a passive baseline. At alpha 1.0, presence-minus-random
next-token effects are:

| Family | Presence minus random next-token delta |
| --- | ---: |
| identity_stress | -0.0165 |
| literal_stable | -0.0733 |
| metaphor_shift | -0.0605 |
| ontology_collapse | 0.4270 |

This is important for interpretation. Ontology-collapse prompts show a clear
coherent-steering signal for coexact, but presence is even stronger on the
teacher-forced next token in this run. That means the causal story is not simply
"coexact equals semantics, presence equals confidence." The next gate should
separate teacher-forced token support from semantic drift and identity or
affordance movement.

## Harmonic Read

Harmonic remains weak:

| Family | Harmonic active rate at alpha 1 |
| --- | ---: |
| identity_stress | 0.2 |
| literal_stable | 0.6 |
| metaphor_shift | 0.4 |
| ontology_collapse | 0.0 |

The steering gate therefore agrees with the topology robustness gate:
harmonic/global concept-ring claims remain deferred.

## Accepted v0.2 Claim

The cautious claim after this gate is:

> Reconstructed HLTD component directions causally move next-token logits in a
> one-step activation steering setup. Coexact steering is family-dependent and
> differs from matched random tangent steering: it improves teacher-forced
> next-token support in identity-stress and ontology-collapse prompts, while
> literal-stable prompts remain mostly neutral and metaphor-shift prompts do not
> yet show a coexact advantage over random tangent.

## Not Yet Claimed

This gate does not yet establish:

- multi-step semantic drift
- fluency preservation under closed-loop generation
- ontology or affordance drift
- that coexact is uniquely semantic rather than one causal direction among
  several
- that harmonic components encode global concept loops

## Next Gates

The next experiments should make the steering claim harder to pass:

1. Reuse one loaded model across the suite so larger sweeps are cheaper.
2. Sweep layers 4-8, not only layer 5.
3. Sweep random-tangent seeds.
4. Compare token selection rules: max coexact norm, fixed token positions, and
   all interior nodes.
5. Add semantic probes or embedding-target scores so the gate is not limited to
   teacher-forced next-token support.
6. Run short closed-loop generation with presence/coexact/random steering and
   measure fluency, identity drift, and affordance drift.

## Next-Gate Sweep Update

The first layer/seed/token-selector extension is summarized in:

```text
docs/hltd_next_gate_steering_sweep.md
```

Short read: in a lite run over one prompt per family, layers 4-8, seeds 0/1,
and `max_component` versus `middle` token selectors, the middle-token selector
showed a cleaner positive coexact-minus-random teacher-forced next-token signal
from L5-L8. The max-component selector was less stable, which makes token
selection itself part of the robustness gate rather than a harmless detail.

The same note now includes the fast full-suite MPS run over all 20 prompts.
That run completed 100 prompt/layer/k steering runs and 2000 rows in 38.8
seconds, and preserved the main selector read: middle-token coexact steering
exceeds matched random tangent on teacher-forced next-token support across
L4-L8, with the strongest family-level effect in ontology-collapse prompts.

The note also includes the first coarse semantic-target gate using
`data/hltd_semantic_targets.json`. That gate adds family-level lexical
target/control sets and reports target-mass and semantic-margin deltas. Short
read: coexact middle-token steering still has the cleaner next-token effect,
but lexical semantic-margin evidence is family-dependent rather than global.
It appears most clearly in `identity_stress` and late `ontology_collapse`
layers, while `literal_stable` remains a useful negative control.

The learned-probe update goes one step further by training layer-wise hidden
state probes for `identity_stress`, `ontology_collapse`, and
`affordance_stress`. It changes the causal read: presence is the stronger
label-consistent probe direction, while coexact's learned-probe advantage is
narrow and appears mainly in late ontology movement at L7-L8. Coexact remains
important, but the current evidence now separates "next-token support" from
"probe-consistent identity/ontology stabilization."

The explicit dissociation gate in `docs/hltd_dissociation_gate.md` sharpens
that separation. `presence_plus_coexact` combines the coexact next-token effect
with positive ontology probe margin, while `coexact_minus_presence` retains
some next-token support but loses much of the learned-probe stabilization.

The structural/causal branch ledger in `docs/hltd_branch_hodge.md` reconnects
that dissociation to the graph-Hodge decomposition. Its current read is that
the structural middle-layer field is mostly coexact with triangles, while the
causal directions split into coexact traversal and presence stabilization.
