# HLTD Topology Filtration

Status: full 20-prompt GPT-2 robustness gate. The result is exploratory and
model-specific; it does not establish a harmonic concept ring.

## Question

The earlier Hodge topology contrast had two degenerate endpoints:

- no triangles: every non-exact graph cycle is assigned to harmonic
- all kNN 3-cliques filled: every supported cycle boundary is assigned to
  coexact

Those endpoints cannot tell us whether either finer branch persists at an
intermediate complex. This gate holds the PCA points, centered node field, kNN
graph, edge flow, and vertex incidence matrix fixed while adding geometrically
ordered clique triangles.

For each triangle complex `C_r`, the orthogonal decomposition is

```text
omega = exact + coexact(C_r) + harmonic(C_r)
```

The radius of a clique triangle is its longest chart-space edge divided by the
median edge length of the fixed kNN graph. A triangle enters when this scaled
radius is below the requested threshold. The `full` endpoint adds every clique
triangle. This creates a nested geometric filtration rather than filling a
fixed percentage of a different triangle set in every field.

## Full Gate

- model: local GPT-2 small
- prompts: all 20 bundled prompts, five each from `literal_stable`,
  `metaphor_shift`, `identity_stress`, and `ontology_collapse`
- field: centered token differences on a normalized PCA-32 chart
- layers: L4, L5, L7
- graphs: `k=12,16,24`
- radius scales: `0, 0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.3, full`
- nulls: speed-preserving vector shuffle and random tangent, seeds 0-7
- Hodge solver: orthogonal exact/coexact projections
- fixed real fields: 180
- rows: 30,600 (`1,800` real and `28,800` null)

Null trajectories use the same graph and filtration as their matched real
field. At the topology-matched stage, each real/null trajectory is interpolated
at Betti-1 fractions `1.0, 0.75, 0.5, 0.25, 0.0` before differences are taken.

## Numerical Contract

The implementation satisfies the decomposition contract to numerical
precision:

- maximum within-field exact-ratio range across the filtration: `0.000e+00`
- maximum within-field non-exact-ratio range: `8.882e-16`
- maximum relative energy-closure error: `7.916e-16`
- maximum edge-flow reconstruction error: `0.000e+00`
- maximum exact/harmonic alignment while both branches have ratio above
  `1e-12`: `5.093e-13`

Large normalized alignments can occur only after harmonic energy has fallen to
numerical zero, where its direction is undefined. They are excluded by the
same activity floor used by the branch diagnostics.

## Radius Filtration

![HLTD radius-filtration branch persistence](../spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7/plots/topology_filtration_branch_persistence.png)

The exact/non-exact split is unchanged while coexact and harmonic exchange the
same non-exact energy. Across real fields:

- open-complex median harmonic ratio: `0.8836`
- full-complex median harmonic ratio: `0.0000`
- median coexact/harmonic crossover radius: `1.000`
- harmonic-ratio/Betti-1 correlation: `r=0.9794`

This is primarily a calibration result. It shows that the orthogonal solver
and nested filtration behave as Hodge theory predicts. It also shows why an
open-graph harmonic value or a full-clique coexact value cannot, alone, identify
a semantic mechanism.

## Matched Topology

![HLTD matched-Betti branch persistence](../spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7/plots/topology_filtration_matched_betti.png)

Medians over the 180 real prompt/layer/k fields are:

| Betti-1 fraction | exact | coexact | harmonic | total non-exact |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.1164 | 0.0000 | 0.8836 | 0.8836 |
| 0.75 | 0.1164 | 0.1954 | 0.6904 | 0.8836 |
| 0.50 | 0.1164 | 0.5179 | 0.3751 | 0.8836 |
| 0.25 | 0.1164 | 0.7348 | 0.1434 | 0.8836 |
| 0.00 | 0.1164 | 0.8836 | 0.0000 | 0.8836 |

The pooled median real-minus-null gaps show the same transfer:

| Betti-1 fraction | exact gap | coexact gap | harmonic gap |
| ---: | ---: | ---: | ---: |
| 1.00 | -0.3951 | +0.0000 | +0.3951 |
| 0.75 | -0.3951 | +0.0734 | +0.3144 |
| 0.50 | -0.3951 | +0.2739 | +0.1263 |
| 0.25 | -0.3951 | +0.3631 | +0.0225 |
| 0.00 | -0.3951 | +0.3951 | +0.0000 |

The endpoint values are algebraically forced. Betti-1 `0.5` is the useful
interior read because both finer branches are active and neither owns the whole
non-exact residual by construction.

## Prompt Bootstrap

![HLTD prompt-paired branch inference](../spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7/plots/topology_filtration_prompt_inference.png)

For inference, null seeds are averaged within each field, layer and k are then
averaged within each prompt, and the 20 paired prompt gaps are resampled 5,000
times with seed 1729. At matched Betti-1 `0.5`:

| Null | Branch | Mean gap | 95% prompt bootstrap CI | Positive prompts |
| --- | --- | ---: | ---: | ---: |
| random tangent | harmonic | +0.1433 | [+0.1185, +0.1679] | 20/20 |
| random tangent | coexact | +0.2550 | [+0.2258, +0.2835] | 20/20 |
| vector shuffle | harmonic | +0.1121 | [+0.0863, +0.1374] | 19/20 |
| vector shuffle | coexact | +0.2680 | [+0.2423, +0.2922] | 20/20 |

Thus the interior harmonic allocation is not only an open-endpoint artifact:
it remains above each matched null after matching topological capacity. The
larger and more uniform coexact gap also says the finer branches are not
equally robust.

## Scale Localization

The interior harmonic gap weakens monotonically with denser neighborhoods:

| k | Random-tangent gap, 95% CI | Vector-shuffle gap, 95% CI |
| ---: | ---: | ---: |
| 12 | +0.2310 [+0.2051, +0.2566] | +0.1985 [+0.1733, +0.2214] |
| 16 | +0.1501 [+0.1209, +0.1781] | +0.1207 [+0.0900, +0.1511] |
| 24 | +0.0488 [+0.0253, +0.0738] | +0.0170 [-0.0087, +0.0438] |

It also declines from L4 to L7. At Betti-1 `0.5`, mean harmonic gaps against
random tangent are `0.2150`, `0.1514`, and `0.0634` for L4/L5/L7; against
vector shuffle they are `0.1758`, `0.1192`, and `0.0412`.

This is the main qualification added by the larger run: the harmonic remainder
is a scale-localized candidate at `k=12/16`, not a graph-independent global
loop. At `k=24`, its vector-shuffle interval crosses zero even after matching
Betti-1. The corresponding non-exact excess has not vanished; it has moved
mostly into coexact.

## Family View

![HLTD branch transfer by prompt family](../spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7/plots/topology_filtration_family_persistence.png)

At Betti-1 `0.5`, ontology-collapse prompts have the largest descriptive mean
harmonic gaps (`0.1665` random tangent, `0.1320` vector shuffle), while literal
prompts have the smallest (`0.1170`, `0.0839`). All five prompts are positive
in every family/null cell except literal versus vector shuffle, which is 4/5.

Each family contains only five hand-authored prompts, so these rankings are
hypothesis-generating. The full gate supports a shared branch-transfer pattern;
it does not yet support a family-specific semantic claim.

## Interpretation

The evidence now supports four scoped statements:

1. **Numerically robust:** exact versus non-exact energy is invariant to the
   triangle filtration, and real centered token flow contains substantially
   more non-exact energy than both matched null constructions.
2. **Topology-conditioned:** coexact and harmonic are allocations of that same
   non-exact field relative to a chosen complex. Their endpoint identities are
   forced, not discovered.
3. **Interior candidate:** at matched Betti-1 `0.5`, both coexact and harmonic
   real-minus-null gaps survive prompt bootstrap, especially at `k=12/16` and
   L4/L5.
4. **Not established:** no result yet ties the surviving harmonic component to
   recurring concepts, semantic return, ontology drift, or causal generation.
   A harmonic concept ring therefore remains unproven.

The full-clique coexact direction remains a valid operational intervention
branch, but its complete name is **full-clique coexact**. The interior harmonic
direction is now eligible for causal testing, not for semantic naming.

## Reproduce

```bash
python3 scripts/run_hltd_topology_filtration.py \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_topology_radius_full20_l4_5_7_k12_16_24_s0_7 \
  --layers 4 5 7 \
  --k 12 16 24 \
  --filtration-mode radius \
  --radius-scale 0 0.75 0.85 0.95 1.0 1.05 1.1 1.15 1.3 full \
  --null-variants vector_shuffle random_tangent \
  --null-seeds 0 1 2 3 4 5 6 7 \
  --components 32 \
  --max-length 64 \
  --device cpu
```

Primary artifacts:

- `topology_filtration_metrics.csv`: all field/null/filtration measurements
- `summary_matched_betti.csv`: per-field interpolation at shared Betti targets
- `summary_matched_betti_gaps_by_null.csv`: seed-averaged paired gaps for each
  null construction
- `summary_prompt_bootstrap.csv`: prompt-level overall, layer, and k inference
- `summary_report.md`: generated compact readout
- `plots/`: radius, family, matched-topology, and prompt-inference figures

## Causal Follow-Up

The pre-registered `L5`, `k=16`, matched Betti-1 `0.5` one-step gate is now
recorded in
[hltd_matched_betti_causal_gate.md](hltd_matched_betti_causal_gate.md). It
compares exact, coexact, and harmonic node directions with eight norm-matched
random-tangent seeds over all 20 prompts.

Coexact shows a positive observed-next-token trend at larger alpha, but its
prompt-bootstrap intervals cross zero. Its coarse semantic target-control
margin is negative relative to random tangent at every tested strength.
Harmonic has no positive causal interval. The structural topology result is
therefore a candidate-selection result; the present causal gate does not
establish semantic circulation or a concept ring.

The next fixed gate is signed steering at the same topology. Paired `+alpha`
and `-alpha` contrasts will separate oriented transport from sign-symmetric
perturbation before expanding to position bins or closed-loop generation. A
second-model replication should follow before treating the L5/k16 localization
as general.
