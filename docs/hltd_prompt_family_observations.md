# HLTD Prompt-Family Observations

This note summarizes the first larger Hodge-Latent Traversal Dynamics sweep
after adding centered token-vector anchoring.

The run used local GPT-2 hidden states from:

```text
/Users/ryospiralarchitect/SpiralReality/model/gpt2
```

and the 20-prompt suite in:

```text
data/hltd_prompt_suite.jsonl
```

The suite has four prompt families with five prompts each:

- `literal_stable`
- `metaphor_shift`
- `identity_stress`
- `ontology_collapse`

Each prompt was analyzed over all GPT-2 layers with PCA-32, `--hltd-k 16`,
centered HLTD vectors, and the `real`, `shuffle_tokens`, `reverse_tokens`, and
`random_hidden` null models. The generated run directory is intentionally
ignored by git:

```text
spiral_out_hltd_suite/
```

## Commands

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

## Aggregate Pattern

The middle-layer window below is GPT-2 layers 5 through 8.

| Family | Real coexact L5-L8 | Real minus shuffle | Real minus random | Real exact L5-L8 | Graph high freq L5-L8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `literal_stable` | 0.8574 | 0.0348 | 0.1614 | 0.1426 | 0.2241 |
| `metaphor_shift` | 0.8748 | 0.0782 | 0.1509 | 0.1252 | 0.1926 |
| `identity_stress` | 0.8801 | 0.0816 | 0.1612 | 0.1199 | 0.2040 |
| `ontology_collapse` | 0.8889 | 0.1006 | 0.1568 | 0.1111 | 0.1744 |

The first useful read is not that coexact ratios are high in isolation. Random
hidden controls can also produce high non-exact ratios. The stronger signal is:

> real trajectories show a middle-layer coexact emphasis that is most separated
> from shuffle in the ontology-collapse family, while all families remain
> clearly above random-hidden controls in layers 5-8.

This supports the narrower HLTD claim: the current evidence points to local
semantic circulation, not to a global concept-ring component.

## Layer Localization

Real coexact peak layers by family:

| Family | Peak-layer counts |
| --- | --- |
| `literal_stable` | L4: 2, L5: 2, L7: 1 |
| `metaphor_shift` | L4: 1, L5: 1, L6: 2, L8: 1 |
| `identity_stress` | L5: 2, L6: 3 |
| `ontology_collapse` | L5: 4, L6: 1 |

The pattern is consistent with the working expectation that the interesting
HLTD action is not primarily final-layer decoding. It clusters in middle layers,
with the surreal and identity-stressed prompts concentrating most tightly around
layers 5 and 6.

## Reversal Check

Centered vector anchoring fixes the earlier concern that forward differences
made the unsigned HLTD ratios sensitive to endpoint anchoring.

Across the 20-prompt suite:

| Family | Max real-vs-reverse HLTD coexact gap | Max signed trajectory cancellation gap |
| --- | ---: | ---: |
| `literal_stable` | 0.0011 | 0.0000 |
| `metaphor_shift` | 0.0046 | 0.0000 |
| `identity_stress` | 0.0033 | 0.0000 |
| `ontology_collapse` | 0.0003 | 0.0000 |

The unsigned HLTD ratios are now nearly invariant under reversal, while the
signed trajectory metric cancels as expected. This makes centered mode the safer
default for structural Hodge comparisons.

## Harmonic Component

The harmonic/global-loop ratio stayed effectively zero in every family:

```text
real_harmonic_max = 0.0000 for all four families
```

That does not falsify the concept-ring idea. It says this particular graph
construction and k value do not reveal a stable harmonic component. The kNN
3-clique complex may be filling local holes aggressively, and these short
teacher-forced trajectories may not contain enough topology to support a
harmonic signal.

The responsible claim is therefore:

> In this v0 suite, HLTD detects coexact/local-swirl structure, while harmonic
> concept-ring structure remains unobserved.

## Interpretation

Three points survive this run:

1. **Local semantic swirl is the active component.** Coexact energy dominates
   the non-exact HLTD field, especially in middle layers.
2. **Absolute flow ratio is not enough.** Random hidden controls also have high
   non-exact ratios, so claims need null differences, layer localization, and
   reversal checks.
3. **Ontology-collapse prompts are the most promising family.** They have the
   highest L5-L8 real coexact mean and the largest real-minus-shuffle gap in
   this suite.

This is close to the refined hypothesis:

> HLTD coexact energy marks a middle-layer local circulation regime that becomes
> stronger under metaphor, identity stress, and ontology stress than under
> literal stable narration, but it must be judged against matched nulls rather
> than by raw ratio alone.

## Next Gates

The next experiment should make the claim harder to pass:

1. Sweep `k = 12, 16, 24` and require the L5-L8 pattern to remain stable.
2. Increase to 200-500 prompts per family.
3. Add bootstrap confidence intervals over prompt families.
4. Add path-integration statistics: bending energy, graph shortest-path defect,
   and manifold adherence.
5. Add one-step causal steering from PCA-chart HLTD components.
6. Treat harmonic energy as a separate topology-stability question, not as a
   required signal for the coexact finding.

## k-Sweep Update

The first `k = 12, 16, 24` sweep is summarized in the generated report:

```text
spiral_out_hltd_ksweep/summary_report.md
```

Across all families and k values, real HLTD coexact energy peaks at layer 5:

```text
L5 = 0.8906
L6 = 0.8824
L4 = 0.8773
```

The real-minus-shuffle coexact delta also peaks at layer 5:

```text
L5 = +0.0925
L4 = +0.0903
L6 = +0.0793
```

The family ordering from the first `k=16` run remains visible when averaged
over k:

```text
ontology_collapse  0.8826
identity_stress    0.8770
metaphor_shift     0.8705
literal_stable     0.8554
```

The cautious read is stronger now:

> HLTD coexact energy robustly localizes around middle layers across k=12/16/24,
> with ontology-collapse prompts still showing the strongest family-level
> signal. The family ordering remains provisional until prompt counts grow.

## Reversal-Invariance Gate

The next robustness check is implemented as `--hltd-same-graph-reverse`.
It keeps the real kNN graph, edge orientation, and triangle complex fixed, then
reverses only the node-vector field:

```text
v -> -v
```

This should preserve exact/coexact/harmonic energy ratios to numerical
precision and flip component directions. It gives a stricter reference than
the `reverse_tokens` null, which rebuilds the chart and graph after reversing
token order.

The intended read is:

```text
same-graph reverse gap near zero:
    Hodge decomposition is orientation-stable on a fixed complex.

reverse_tokens gap nonzero:
    likely graph/triangle/chart rebuild jitter, not necessarily a flow failure.
```

For the next full gate run, use both:

```bash
python3 scripts/run_hltd_prompt_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_invariance \
  --k 12 16 24 \
  --components 32 \
  --max-length 128 \
  --null-models all \
  --hltd-same-graph-reverse
```

## Topology Robustness Gate

The full triangle/no-triangle topology ablation and same-graph reversal gate is
written up in:

```text
docs/hltd_robustness_gate.md
```

Short read: with triangles, the v0.1 signal is a middle-layer coexact/local
swirl component; without triangles, coexact is structurally zero and residual
energy is forced into harmonic. The current evidence therefore supports local
semantic circulation, while harmonic/global concept-ring claims remain
deferred.

## One-Step Steering Gate

The first causal steering gate is written up in:

```text
docs/hltd_steering_gate.md
```

Short read: reconstructed HLTD component directions do move next-token logits.
Compared with matched random tangent steering, coexact improves teacher-forced
next-token support in `identity_stress` and `ontology_collapse` prompts, but
does not yet show a universal advantage across all families. Presence is also
causally active, especially in ontology-collapse prompts, so the next gate
needs semantic probes and multi-step generation rather than relying only on
teacher-forced next-token support.

## Steering Selector Sweep

The first next-gate steering expansion is written up in:

```text
docs/hltd_next_gate_steering_sweep.md
```

Short read: in a lite sweep over GPT-2 layers 4-8, random-tangent seeds 0/1,
and `max_component` versus `middle` token selectors, token selection mattered.
The middle-token selector produced a cleaner positive coexact-minus-random
next-token signal across L5-L8, while max-component selection changed sign in
some layers. Token selection is therefore part of the robustness gate, not a
harmless implementation detail.

The fast full-suite update in the same note extends this to all 20 current
prompts. The middle-token coexact-minus-random next-token delta stays positive
across L4-L8, and the strongest family-level effect appears in
`ontology_collapse` prompts.

The semantic-target update adds coarse lexical target/control sets. It keeps
the same next-token read but separates semantic mass from generic logit
movement: the layer-averaged semantic margin is not globally positive until
L8/middle, while `identity_stress` and late `ontology_collapse` prompts show
the clearest positive coexact-minus-random semantic-margin deltas. This makes
learned identity and affordance probes the next natural gate.

The learned-probe gate is now also included in the same next-gate note. It
trains internal-suite probes for `identity_stress`, `ontology_collapse`, and
`affordance_stress`. The strongest new read is that presence, not coexact, is
the more label-consistent direction under these probes. Coexact keeps a narrow
late ontology signal at L7-L8, so the project now has three separable effects:
middle-token next-token support, lexical semantic-margin hints, and learned
probe-consistent stabilization.

The dissociation gate extends this into derived directions:

```text
docs/hltd_dissociation_gate.md
```

The short read is now cleaner: presence behaves like a stabilizing basin
direction; coexact behaves like a traversal direction. Adding them can combine
next-token support with ontology-probe stabilization, while subtracting
presence preserves some next-token support but removes much of the
probe-consistent stabilization.

## Branch Hodge Ledger

The current structural/causal join is written up in:

```text
docs/hltd_branch_hodge.md
```

Short read: the structural Hodge branch remains coexact-dominant under the
triangle complex, while the causal gates split into coexact traversal and
presence stabilization. `presence_plus_coexact` is the first combined branch
that keeps next-token support while retaining positive ontology-probe margin.
The same ledger now includes the structural `k=12/16/24` branch sweep, where
coexact remains dominant and same-graph reversal gaps stay at zero.
It now also includes the causal/probe `k=12/16/24` sweep: coexact keeps
positive next-token traversal across k, presence keeps learned-probe
stabilization across k, and `presence_plus_coexact` remains the combined branch.
The ledger also compares `middle` against `max_component`; the middle selector
keeps stronger coexact traversal, while max-component selection emphasizes
presence/probe stabilization more strongly.

The all-interior structural localization pass is written up in:

```text
docs/hltd_branch_heatmap.md
```

Short read: coexact peaks are mid-to-late and family-specific, presence shifts
earlier as k increases, and the hybrid branch sits between them.

The all-interior causal/probe position gate is written up in:

```text
docs/hltd_all_interior_position_gate.md
```

The full 20-prompt k=16 gate and `k=12/16/24` k-sweep show that next-token
peaks and ontology-probe peaks can land in different position bins.
Coexact-like branches dominate early-phase cross-family next-token traversal,
while presence dominates ontology-probe margin at a different token-position
phase.
The selected-bin seed gate then makes this sharper: at k=16 over bins
0/1/2/4 and seeds 0-7, `coexact_minus_presence` is the cleanest traversal
branch, while `presence` remains the strongest ontology-probe branch.
