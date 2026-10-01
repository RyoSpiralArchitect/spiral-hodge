# Prefix-Only Transfer V2: Interpolated Calibration Readout

## Question and Scope

The [v1 gate](hltd_prefix_transfer_gate.md) stopped with 56/60 active
evaluation cells and no nonzero treatments. Its four misses retrieved
exact-zero node vectors. That result stays unchanged.

V2 asks whether a prospectively specified **interpolated calibration-coexact
readout** is available at all new evaluation prefixes, and, conditional on
coverage and numerical gates, has a positive signed next-token response
relative to a matched interpolated random field. It is not an orthogonal
Hodge decomposition on evaluation points, a globally smooth field, or a
semantic/fluency-control experiment.

The [design](data/hltd_prefix_transfer_v2/design.json) fixes one choice before
v2 model work: eight geometry-selected neighbors, without an activity filter
or parameter sweep. V1 coverage and the existence of 52 zero atlas vectors
were known at design time. No v2 readout was tested on the v1 evaluation
prefixes to select the configuration.

## Frozen Field and Readout

Reuse the exact v1 PCA-32, 40-text calibration field, 1,170 interior nodes,
and raw-hidden dose scale. Model bytes and recorded runtime must agree with
those that produced the cached calibration states. There is no new atlas
fit and no addition of v1 or v2 evaluation states to calibration.

For each calibration node, exclude its entire prompt and measure distance
to the eighth nearest remaining node. Fix Gaussian bandwidth `s` to the
median of these distances and the eighth-neighbor support radius to their
99th percentile (`higher`). Both depend only on calibration geometry.

For an observed prefix hidden vector, apply the unchanged normalization and
chart, choose the eight nearest atlas nodes, and set:

```text
w_j = exp(-0.5 * (distance_j / s)^2) / sum_k exp(-0.5 * (distance_k / s)^2)
u_coexact = sum_j w_j * atlas_coexact_j
```

Distances alone select the nodes and weights, with frozen-index tie breaks.
Zero and inactive nodes remain in the sum and its denominator. There is no
exact-match override, nearest-active-node substitution, or per-node unit
normalization of the coexact field. Opposing vectors may cancel; cancellation
is a legitimate reason to abstain, not a reason to select different nodes.

Support requires both the original nearest-node radius and the new
eighth-neighbor radius. The coexact aggregate must meet the original
`1e-6` activity threshold. Backproject through the frozen PCA and normalize
the resulting hidden direction, then apply the unchanged calibration-only
dose times signed alpha. This remains a candidate intervention direction,
not an exact inverse of hidden normalization.

### Matched Random Field

At each atlas node, draw a Gaussian chart vector keyed by `(seed,node_index)`,
normalize it, and multiply by that node's raw coexact norm. Interpolate those
vectors using exactly the same neighbors and weights as the coexact readout.
Exact-zero nodes contribute zero to both fields. Normalize the aggregate
in hidden space and apply the same absolute dose.

The aggregate random chart norm must also be at least `1e-6`; failure at any of
the eight seeds invalidates the cell. There is no redraw or alternate
neighbor set. The legacy labels `coexact` and `random_tangent` remain in
output tables, but both designate these interpolated calibration fields.
This random control is not a separately estimated local manifold tangent.

The random-control construction differs from v1, as do evaluation texts.
Even improved coverage or a positive endpoint would not isolate a pure
readout-only causal difference between v1 and v2.

## New Texts and Decision Rules

The [first-authored suite](../data/hltd_prefix_transfer_v2/prompts.jsonl)
contains four pilot texts and twenty evaluation texts, balanced across the
same four families. All IDs, normalized texts, and authored scenario labels
are checked against historical suites, including v1 pilot/evaluation texts.
They are new strings in the same authored style, not an external population
sample or a pretraining holdout. No response-based replacement is allowed.

Inherited conditions remain:

| Condition | Fixed value |
| --- | --- |
| Model | GPT-2, native FP32, L7, MPS SDPA, no fallback/autocast |
| Prefix lengths | 8, 16, 24; never fractions of full-text length |
| Pilot | Four texts, 12 cells, four suffix variants each; zero hooks only |
| Evaluation | Twenty new texts, 60 cells; complete coverage required |
| Treatment grid | Eight seeds x two components x six signed alphas |
| Signed magnitudes | 0.25, 0.5, 1.0, each with its negative |
| Planned raw rows | 5,760, only after all gates pass |
| Baseline | Unhooked batch-12, same prefix IDs as treatment |
| Zero gate | Maximum logit error <=1e-4 and exactly zero row spread |
| Future gate | Prefix, hidden, lookup, weights, directions and FP32 deltas byte-identical across suffix variants |

For each prompt/prefix/seed, the primary coefficient is the zero-intercept
fit of the odd coexact-minus-random log-probability difference against the
three positive magnitudes. Average seeds, then prefixes, giving each of
twenty prompts equal weight. The unchanged prompt bootstrap uses 5,000
draws, NumPy `default_rng(1729)`, and 2.5/97.5 percentiles. A strictly
positive lower endpoint and all 60 valid cells are both required.

The interval is conditional on the frozen atlas and control draws. It does
not make this authored sample a random population sample. A response is
finite, signed, and relative to the random field; it is not an infinitesimal
derivative or an absolute improvement over unsteered generation.

## Execution and Audit

[run_hltd_prefix_v2.py](../scripts/run_hltd_prefix_v2.py) pins the v1 protocol,
manifest, verdict, atlas, and source receipts before accepting the new
design and corpus. It freezes the readout scales and all token IDs before
loading model parameters. V1 files are reused read-only; the v2 corpus is
placed in its own directory, preserving the historical v1 inventory.

The actual model and steering kernel are reused from v1. Token slicing
precedes hidden extraction, and only the scorer reads the next-token ID.
Full prefix states, baseline logits and zero-hook logits are saved for all
pilot variants and evaluation cells. All seeded directions and nominal FP32
deltas are replayable from those arrays and the immutable atlas.

The [analyzer](../scripts/analyze_hltd_prefix_v2.py) checks the complete grids,
chronology, model-weight audits, every query/weight/delta, and matched raw
baselines before estimation. The existing standard-library auditor separately
reconstructs coefficients from saved probabilities and checks the bootstrap
using the saved draw indices. Neither offline audit reruns the model.

The runner records attempted nonzero forward rows as well as successfully
saved response rows, so an interrupted forward cannot be mislabeled as no
treatment. No automatic retry, threshold relaxation, partial-sample primary,
or rewriting of v1 is permitted.

Commands for the single prospective attempt are below. Existing protocol,
run and result paths are rejected; these are not retry commands.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/run_hltd_prefix_v2.py \
  --freeze docs/data/hltd_prefix_transfer_v2/protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/run_hltd_prefix_v2.py \
  --run docs/data/hltd_prefix_transfer_v2/protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/analyze_hltd_prefix_v2.py \
  --protocol docs/data/hltd_prefix_transfer_v2/protocol.json \
  --output docs/data/hltd_prefix_transfer_v2/result

python3 -S scripts/audit_hltd_prefix_transfer.py \
  --result docs/data/hltd_prefix_transfer_v2/result \
  --output docs/data/hltd_prefix_transfer_v2/result/stdlib_audit.json
```

For read-only protocol validation, use `--validate` instead of `--run`.

## Completed Attempt: 2026-10-01 JST

**Disposition: `NOT_SUPPORTED`.** The prospective run reached all 5,760
nonzero treatments, but the prespecified signed response endpoint did not
pass. Complete prefix coverage and a positive response are separate claims.

The [protocol](data/hltd_prefix_transfer_v2/protocol.json) was frozen at
`2026-09-30T20:39:14.367727+00:00`; execution ran from
`20:39:29.172340` to `20:40:25.254946` UTC. No source, readout setting, prompt,
threshold, dose, or estimator changed between freezing and analysis.
There was one model load and no retry.

| Frozen calibration quantity | Value |
| --- | ---: |
| Atlas nodes | 1,170 |
| Neighbors per query | 8 |
| Gaussian bandwidth | 0.3219910309 |
| Original nearest-node radius | 0.3542267077 |
| Eighth-neighbor radius | 0.4354238209 |
| Raw-hidden dose at alpha=1 | 40.4991076282 |

All 24 new texts passed the repository-history novelty audit against 93
historical rows; no exact duplicate was found. Pilot lengths were 38-40
tokens and evaluation lengths 39-44, so all absolute prefixes and targets
were available without truncation.

### Engineering and Coverage

| Check | Observed result |
| --- | --- |
| Suffix-invariance pilot | 12/12 cells, four variants each, byte-identical |
| Evaluation support | 60/60 cells, including every random-control seed |
| Zero-hook checks | 108/108; maximum logit error 0 and row spread 0 |
| Nonzero rows attempted / saved | 5,760 / 5,760 |
| Model weights before / after | All 148 tensors, 124,439,808 parameters, source-byte equal |
| Offline query replay | All 48 pilot variants and 60 evaluation observations passed |
| Independent probability-based estimator | Passed; maximum primary discrepancy 1.11e-16 |
| Regression tests | 479 passed, plus 194 subtests |
| Targeted lint | Ruff passed with the existing E402 import-path exception |

Unfiltered Ruff reports twelve E402 import-order findings in the two CLI
entry points, which insert the repository root before local imports. The
frozen sources retain the v1 convention; no post-run source edit was made
to silence these findings. Tests retain one existing Transformers cache
deprecation warning.

Seven evaluation cells included at least one inactive atlas neighbor; one
included seven. Those nodes were retained with their geometric weights.
The minimum coexact aggregate norm was `0.001154826`, above the unchanged
`1e-6` threshold. The maximum nearest/eighth distances were
`0.3062860511` / `0.4011225194`, within both frozen radii.
There was no active-neighbor fallback or partial-sample endpoint.

This establishes complete availability on the **new v2 sample**. It does
not demonstrate rescue of the four v1 failures on the same prefixes: the
texts and random-control construction changed, and those old queries were
not used to tune or evaluate v2.

### Primary Result

The [verdict](data/hltd_prefix_transfer_v2/result/verdict.json) and
[independent audit](data/hltd_prefix_transfer_v2/result/stdlib_audit.json)
agree:

| Prompt-weighted quantity | Value |
| --- | ---: |
| Mean odd coexact-minus-random coefficient | +0.0064649743 |
| Prompt-bootstrap 95% interval | [-0.1031005873, +0.1342704455] |
| Positive prompt coefficients | 10/20 |
| Mean coexact odd coefficient | -0.0281720484 |
| Mean random odd coefficient | -0.0346370227 |

Coefficient units are nats per unit nominal alpha in the finite signed fit.
The lower interval endpoint is not positive, so the gate fails. This is
not an equivalence test and does not establish absence of all effects.
There are 20 prompt units, not 5,760 independent samples; repeated coexact
responses across the eight control seeds were exactly equal.

### Descriptive Breakdown

These summaries use the saved
[coefficients](data/hltd_prefix_transfer_v2/result/coefficients.csv).
They are post-result descriptions, not replacement endpoints or selected
subgroup confirmations. No subgroup interval or multiplicity-adjusted
claim is made.

| Prefix length | Mean odd gap | Positive prompt means |
| --- | ---: | ---: |
| 8 | +0.075940 | 12/20 |
| 16 | -0.100660 | 7/20 |
| 24 | +0.044115 | 13/20 |

| Family | Mean odd gap | Positive prompt means |
| --- | ---: | ---: |
| Literal stable | +0.021678 | 4/5 |
| Metaphor shift | +0.042719 | 2/5 |
| Identity stress | -0.098614 | 1/5 |
| Ontology collapse | +0.060077 | 3/5 |

Even the small positive aggregate gap is not an improvement over an
unsteered baseline. At alpha=+1, the mean next-token log-probability changes
were `-0.144847` for coexact and `-0.167797` for random. These are descriptive
one-token changes, not fluency measurements.

### Interpretation and Next Gate

V1 was stopped by insufficient coverage, with no response estimate. V2
provides an executable prefix-only candidate on all sixty new cells, but
does not support the specified positive average transfer response. Neither
result should be rewritten as semantic steering evidence.

A useful next diagnostic is **leave-one-calibration-prompt-out direction
recovery**, using only the existing calibration atlas: measure neighbor
direction agreement, cancellation, and token-position mixing before any
new model intervention. This would distinguish whether cross-prompt
readout has directional coherence from whether that direction aligns with
next-token response. Neither explanation has been established by v2.
Any subsequent parameter choice or response gate needs a new protocol and
new evaluation texts; do not tune on these sixty observed prefixes.

### Evidence Locations

- [Frozen protocol](data/hltd_prefix_transfer_v2/protocol.json), SHA256
  `88666f7a5d35c4e2f7dde490a1ce3ae19fa2ef08ef5555908298adca0a16f76c`.
- [Result manifest](data/hltd_prefix_transfer_v2/result/manifest.json)
  binds the raw run and derived verdict, support, coefficients and bootstrap
  indices. The independent audit is a separate, subsequent receipt.
- Raw arrays, pre-treatment gates and the response CSV remain under the
  gitignored `spiral_out_hltd_prefix_transfer_v2/` directory in the execution
  checkout. Full replay requires those local artifacts and the v1 atlas;
  the compact documentation packet alone is not a model replay bundle.
- V1 protocol, verdict, source, atlas and raw-run receipts still validate
  byte-for-byte. The primary checkout remains untouched; v2 is local work
  on `SpiralReality/hltd-prefix-transfer`, not a published or merged result.
