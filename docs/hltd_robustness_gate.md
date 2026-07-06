# HLTD Robustness Gate: Topology and Reversal

This note records the first robustness gate for Hodge-Latent Traversal
Dynamics after the prompt-family k-sweep. It is intentionally conservative:
the current evidence supports a middle-layer coexact/local-swirl signal, but
does not yet support a harmonic/global concept-ring claim.

## Run

The gate used the local GPT-2 checkpoint and the prompt-family suite:

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

python3 scripts/run_hltd_prompt_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_invariance \
  --k 12 16 24 \
  --components 32 \
  --max-length 128 \
  --null-models all \
  --hltd-same-graph-reverse \
  --no-hltd-triangles

python3 scripts/summarize_hltd_suite.py \
  --run-root spiral_out_hltd_invariance \
  --output spiral_out_hltd_invariance/summary.csv
```

Summary artifacts:

- `spiral_out_hltd_invariance/summary.csv`
- `spiral_out_hltd_invariance/summary_family_k.csv`
- `spiral_out_hltd_invariance/summary_layer.csv`
- `spiral_out_hltd_invariance/summary_bootstrap.csv`
- `spiral_out_hltd_invariance/summary_family_gaps.csv`
- `spiral_out_hltd_invariance/summary_report.md`

The summary covered 120 runs across topology, prompt family, k, and null
condition.

## Main Read

The cleanest current claim is:

> HLTD v0.1 finds a robust middle-layer coexact component when a triangle
> complex is present. The signal is strongest as a local semantic-swirl
> candidate, not as a harmonic concept-ring candidate.

This matters because the original Poisson-only residual was only
non-gradient. The triangle complex lets us ask whether the residual contains a
coexact component. In this run, the answer is yes, and it appears in the
middle layers.

## Topology Gate

The topology ablation is the key result.

| Topology | Coexact L5-L8 | Harmonic max | Interpretation |
| --- | ---: | ---: | --- |
| triangles | 0.855-0.883 by family | approximately zero | coexact/local swirl is available |
| no triangles | 0.000 | 0.891-0.905 by family | non-exact residual is forced into harmonic |

Without triangles, the coexact channel is structurally unavailable, so the
large harmonic value should not be read as a concept ring. It is residual
energy under an underspecified complex.

With triangles, harmonic energy is near zero:

| Family | Harmonic max mean |
| --- | ---: |
| identity_stress | 1.76e-12 |
| literal_stable | 2.53e-12 |
| metaphor_shift | 5.18e-4 |
| ontology_collapse | 1.22e-4 |

So the v0.1 language should be:

- accepted: coexact/local semantic circulation
- deferred: harmonic/global concept rings

## Middle-Layer Ridge

In the triangle condition, raw coexact energy peaks in the middle layers:

| Layer | Real coexact |
| --- | ---: |
| L5 | 0.8904 |
| L6 | 0.8823 |
| L4 | 0.8776 |
| L7 | 0.8675 |
| L3 | 0.8529 |
| L8 | 0.8450 |

The same region also survives a shuffle contrast:

| Layer | Real minus shuffle coexact |
| --- | ---: |
| L5 | 0.0922 |
| L4 | 0.0906 |
| L6 | 0.0793 |
| L3 | 0.0751 |
| L7 | 0.0649 |
| L1 | 0.0636 |

This supports the "middle-layer local swirl" framing better than a raw flow
ratio framing. Random hidden states can also produce high non-exact ratios, so
the robust object is the layer-localized excess over matched nulls.

## Family Signal

The strongest absolute family separation is ontology collapse versus literal
stable prompts:

| Family | Real coexact L5-L8 | Real minus shuffle | Real minus random |
| --- | ---: | ---: | ---: |
| identity_stress | 0.8770 | 0.0776 | 0.2201 |
| literal_stable | 0.8554 | 0.0341 | 0.2232 |
| metaphor_shift | 0.8704 | 0.0789 | 0.2179 |
| ontology_collapse | 0.8825 | 0.0958 | 0.2202 |

Bootstrap intervals for `real_minus_shuffle` separate from zero for
`identity_stress`, `metaphor_shift`, and `ontology_collapse`, but not for
`literal_stable`:

| Family | Mean | 95% CI |
| --- | ---: | --- |
| identity_stress | 0.0776 | [0.0060, 0.1637] |
| literal_stable | 0.0341 | [-0.0289, 0.0972] |
| metaphor_shift | 0.0789 | [0.0256, 0.1412] |
| ontology_collapse | 0.0958 | [0.0242, 0.1610] |

Pairwise family gaps are still provisional. The absolute
`real_coexact_l5_l8` gap between `literal_stable` and `ontology_collapse`
excludes zero, but null-adjusted family gaps do not yet clearly separate. That
means the family story is promising but needs more prompts before it becomes a
central claim.

## k Sweep

The coexact signal is stable across k, while its magnitude decreases as the
graph becomes denser:

| Family | k=12 | k=16 | k=24 |
| --- | ---: | ---: | ---: |
| identity_stress | 0.8945 | 0.8801 | 0.8565 |
| literal_stable | 0.8798 | 0.8574 | 0.8289 |
| metaphor_shift | 0.8907 | 0.8748 | 0.8457 |
| ontology_collapse | 0.8979 | 0.8887 | 0.8610 |

The same downward trend appears in the shuffle-adjusted gap:

| Family | k=12 | k=16 | k=24 |
| --- | ---: | ---: | ---: |
| identity_stress | 0.0812 | 0.0816 | 0.0700 |
| literal_stable | 0.0542 | 0.0348 | 0.0134 |
| metaphor_shift | 0.1088 | 0.0782 | 0.0496 |
| ontology_collapse | 0.1061 | 0.1004 | 0.0808 |

This is a useful robustness pattern: denser graphs smooth the local swirl
signal, but do not erase it.

## Reversal Gate

Earlier runs showed that unsigned HLTD ratios were not perfectly invariant
under token reversal. The likely issue was graph rebuilding plus node anchoring.

The same-graph reverse gate addresses that directly. In the triangle condition,
same-graph reverse coexact gaps were zero for all families. Standard
reverse-token rebuild gaps remained tiny:

| Family | Standard reverse max coexact gap |
| --- | ---: |
| identity_stress | 0.0024 |
| literal_stable | 0.0006 |
| metaphor_shift | 0.0019 |
| ontology_collapse | 0.0005 |

This resolves the main reversal concern for the current scalar-energy claims:
when the graph complex is held fixed, the decomposition is invariant under the
tested reversal operation.

## Accepted v0.1 Claim

The v0.1 robustness gate supports this statement:

> LLM hidden-state token dynamics, projected into a PCA chart and decomposed
> on a kNN graph with a triangle complex, show a middle-layer coexact component
> that exceeds matched shuffle and random-hidden nulls. This component is a
> plausible local semantic circulation signal and is strongest for the more
> identity- or ontology-stressing prompt families in the current sample.

## Not Yet Claimed

This run does not yet support these stronger statements:

- harmonic energy is a global concept-ring signal
- ontology collapse prompts have a fully established null-adjusted family
  separation
- coexact energy alone proves semantic causality
- UMAP geometry can be used for decomposition or steering claims

## Next Gates

The natural next steps are:

1. Increase prompt count for the family gaps.
2. Add one-step activation steering along reconstructed coexact and presence
   vectors.
3. Compare centered-difference, forward-difference, and path-edge-native
   flows.
4. Add path-integration metrics: bending energy, manifold adherence, and
   matched-speed tangent nulls.
5. Revisit harmonic claims only after testing complexes that preserve
   meaningful holes instead of filling nearly all local cycles with triangles.
