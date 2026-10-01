# Calibration Readout Diagnostics

## Question and Boundary

[Prefix-transfer v2](hltd_prefix_transfer_v2.md) completed all 5,760
treatments with full prefix coverage, but its prespecified positive response
was not supported. This follow-up uses **only its existing calibration
atlas** to ask whether nearby nodes from other prompts reconstruct a
node's coexact direction, how much their vectors cancel, and whether token
positions mix.

This is post-v2 exploratory diagnosis, not a new causal gate or a
retrospective repair of v2. The target is an already reconstructed coexact
node vector, not semantic ground truth or an optimal steering direction.
The entire query prompt is removed from donor candidates. PCA, bandwidth,
distance radii and the target fields remain fixed from the shared 40-prompt
calibration. Consequently this is **donor-prompt-excluded reconstruction**,
not independently held-out PCA validation or a population generalization
estimate.

No model parameters, tokenizer, hidden extraction, evaluation-prefix
arrays or treatment-response CSV are needed. Both earlier verdicts and
their frozen sources remain unchanged.

## Fixed Method

The [method](data/hltd_prefix_calibration_readout/method.json) is specified
before inspecting the reconstruction results. There is one configuration:
the v2 eight-neighbor Gaussian readout, with the whole query prompt excluded
before selection, stable original-index tie breaks, raw vectors, and no
inactive-donor filtering or fallback. No bandwidth or neighbor sweep is
performed.

For a query target `v_i`, donor vectors `v_j`, and geometric weights `w_j`:

```text
u_i = sum_j w_j v_j
recovery_cosine = cos(v_i, u_i)
resultant_fraction = ||u_i|| / sum_j w_j ||v_j||
cancellation_fraction = 1 - resultant_fraction
q_j = w_j ||v_j||
effective_contributors = (sum_j q_j)^2 / sum_j q_j^2
```

Recovery is undefined if either target or prediction has norm below
`1e-6`. It is not scored as zero. An all-zero donor set has undefined
resultant fraction. A single nonzero contributor has resultant fraction
one but no pairwise agreement statistic, so it must not be mistaken for
multi-neighbor coherence. Pairwise agreement is the donor-cosine mean
weighted by `q_j q_k`, over distinct pairs.

Every node is retained in the output. Means first average usable nodes
within a prompt, then give each usable prompt equal weight; node and prompt
denominators are recorded. Results for all recoverable nodes and for nodes
within both original distance radii are reported separately. The latter is
a fixed descriptive stratum, not a pass/fail gate.

### Comparators

- The nearest cross-prompt vector, compared on identical defined nodes.
- The eight original v2 isotropic random-field seeds, with original node
  norms, zeros, geometric weights and seed-by-original-index recipe.
- 128 fixed-seed (`6043`) within-prompt permutations of complete donor
  vectors. These preserve each prompt's joint direction/norm inventory and
  zeros, while breaking the mapping from node position to vector. Query
  targets stay fixed. Each permutation is shared across all queries.

Actual and control means use the intersection of their valid nodes for
each comparison, with paired counts reported. No controls are redrawn.
Null-reference quantiles describe these finite draws; they are not
population confidence intervals, and no exchangeability-based significance
test is claimed. The shared atlas and overlapping donor sets couple rows.

### Token Positions

The atlas stores zero-based token indices. A query at token `t` corresponds
to prefix length `t+1`, and its relative position is `t/(token_count-1)`.
Early, middle and late bins divide this relative coordinate into thirds.
Both geometric-weighted and norm-contribution-weighted donor-bin masses
are reported, together with absolute/relative position gaps and donor-prompt
concentration. These full-text calibration bins are diagnostic labels,
not inputs available to an online prefix-only controller.

Associations with recovery are pooled-node Spearman correlations for
resultant fraction, weighted relative position gap and nearest distance.
They are descriptive, potentially confounded and dependent; no p-values
or causal mediation claim is made.

## Reproduction

[diagnose_hltd_prefix_calibration.py](../scripts/diagnose_hltd_prefix_calibration.py)
pins the actual v2 protocol, verifies the consumed atlas and source receipts,
and freezes the method, calibration membership and diagnostic code before
computing the results. Output paths are exclusive and cannot be overwritten.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/diagnose_hltd_prefix_calibration.py \
  --freeze docs/data/hltd_prefix_calibration_readout/protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/diagnose_hltd_prefix_calibration.py \
  --run docs/data/hltd_prefix_calibration_readout/protocol.json
```

The result directory contains per-node and per-prompt metrics, paired
control draws, position-mixture tables, exact neighbor/weight/permutation
arrays, a summary, and receipts. Full replay needs the original local v1
atlas. For read-only validation use `--validate` instead of `--run`.

## Observations: 2026-10-01 JST

The [frozen protocol](data/hltd_prefix_calibration_readout/protocol.json)
precedes the diagnostic calculation. There were no model loads, model
forwards, evaluation-prefix queries or new interventions. The
[summary](data/hltd_prefix_calibration_readout/result/summary.json) covers
all 1,170 atlas nodes from 40 prompts.

| Coverage quantity | Nodes |
| --- | ---: |
| Defined target/prediction cosine | 1,117 |
| Inactive target | 52 |
| Active target, inactive prediction | 1 |
| Within both distance radii | 1,149 |
| Defined cosine and within both radii | 1,098 |

The single inactive prediction is `fresh_ontology_01`, token index 11
(prefix length 12). All eight cross-prompt donors are inactive, giving an
exact-zero prediction despite target norm `0.0100994`. The eight-neighbor
rule therefore does not guarantee availability in every calibration fold;
v2's full coverage was a property of its observed new evaluation sample.

### Direction Recovery Is Weak but Structured

All means below give each of the forty prompts equal weight. Control means
are additionally averaged over their fixed draws, not extra prompt units.

| Readout / reference | Mean target cosine | Paired nodes |
| --- | ---: | ---: |
| Eight-neighbor interpolation | +0.122079 | 1,117 |
| V2 isotropic random field, eight seeds | +0.001935 | 1,117 in every seed |
| Within-prompt shuffle, 128 draws | +0.002686 | 1,117 in every draw |
| Interpolation on nearest-defined subset | +0.122076 | 1,082 |
| Single nearest donor on the same subset | +0.076608 | 1,082 |

The interpolated readout has a positive prompt mean in 40/40 prompts,
ranging from `+0.031300` to `+0.185521`. At node level, 814/1,117 cosines
are positive, with 10th/50th/90th percentiles
`[-0.125135, +0.129817, +0.365286]`. Agreement is not strong or uniform.
Restricting to distance-supported nodes gives a similar prompt-weighted
mean, `+0.121152`.

The shuffle-reference mean-cosine 10th-90th range is
`[-0.003920, +0.009500]`; the real-minus-shuffle mean gap is `+0.119394`.
These are descriptive finite-reference contrasts, not confidence intervals
or a hypothesis-test pass. Still, they make "the donor directions have no
location-dependent structure" an inadequate description of this atlas.
Improvement over a single neighbor is bounded to this shared-chart
reconstruction task, not a causal comparison of v1 and v2.

### Cancellation Does Not Identify the Failure Mechanism

| Metric across calibration nodes | Equal-prompt mean |
| --- | ---: |
| Resultant fraction | 0.463163 |
| Cancellation fraction | 0.536837 |
| Magnitude-weighted donor-pair cosine | +0.052371 |
| Effective contributors out of eight | 6.143335 |
| Largest single-node contribution share | 0.264223 |
| Distinct donor prompts out of eight | 7.176193 |
| Largest donor-prompt geometric mass | 0.215312 |

Resultant fraction is defined for 1,169 nodes, pairwise agreement for 1,159.
On the common 1,117 defined-recovery nodes, the real resultant fraction is
`0.455136`, versus `0.407719` under within-prompt shuffle and `0.404394`
under the random-field control. Averaging misaligned directions itself
reduces this ratio; a 54% cancellation fraction is **not** 54% semantic
information loss.

Pooled-node Spearman correlation between resultant fraction and recovery
is only `+0.006080`. Thus "reduce cancellation and recovery will improve"
is not established. A coherent donor sum and alignment with the excluded
query's target are different properties. The correlations are descriptive,
not evidence that cancellation has no effect in every setting.

### Position Mixing Is Present, Not Yet Explanatory

The mean geometric-weighted donor/query gap is `5.837935` tokens or
`0.186168` of full-text relative position. The mean weight remaining in the
query's own relative-position bin is `0.549318`.

The [position mixture](data/hltd_prefix_calibration_readout/result/position_mix.csv)
below gives geometric donor mass, averaging within each query prompt before
averaging forty prompts. Rows sum to one.

| Query bin | Early donors | Middle donors | Late donors |
| --- | ---: | ---: | ---: |
| Early | 0.623235 | 0.274258 | 0.102507 |
| Middle | 0.239574 | 0.480385 | 0.280042 |
| Late | 0.102931 | 0.348228 | 0.548841 |

The corresponding diagonal mass after magnitude-contribution weighting is
`0.628542`, `0.491900`, and `0.536728`. Position concentration is therefore
neither exact nor absent. Recovery correlates weakly with relative-position
gap (`rho=-0.092569`) and nearest distance (`rho=-0.100338`).

A post-result descriptive split gives recovery means `+0.153714` early,
`+0.112780` middle, and `+0.101225` late, using 367/377/373 defined nodes
and all forty prompts in each bin. These bins are not selected endpoints,
and this split does not explain the causal v2 result.

## Interpretation and Next Control

The readout captures some location-dependent directional structure within
the calibration atlas, but only weakly reconstructs individual target
directions. Neither substantial cancellation nor position mixing alone is
shown to cause v2's unsupported next-token endpoint. Calibration vector
reconstruction is not next-token response alignment, semantic control, or
transfer to unseen text.

The important remaining confound is **position-dependent structure**: the
current within-prompt shuffle destroys token-position matching as well as
any finer geometry-to-direction correspondence. The next useful control
is a separately frozen, position-preserving shuffle within each prompt and
relative-position bin, with counts and singleton bins explicitly retained.
It can test whether the real-readout advantage survives preserving coarse
position structure, still without model work or touching v2 evaluation
data. This control has not been run, and no steering setting is selected
from the current results.

## Verification and Visualization

The standalone [stdlib auditor](../scripts/audit_hltd_prefix_calibration.py)
checks frozen hashes, row inventories, missing-value masks, prompt means,
distribution summaries and the complete 272-row control table. Its
[receipt](data/hltd_prefix_calibration_readout/result/stdlib_audit.json)
records maximum arithmetic discrepancy `2.664536e-15`. This is independent
aggregation arithmetic, not an independent vector-reconstruction or model
replay. The auditor was added after the diagnostic result, separately from
the frozen estimator; its own source hash is recorded in the receipt.

```bash
python3 -S scripts/audit_hltd_prefix_calibration.py \
  --result docs/data/hltd_prefix_calibration_readout/result \
  --output docs/data/hltd_prefix_calibration_readout/result/stdlib_audit.json
```

Existing output is rejected. The pre-run suite passed 509 tests plus 194
subtests. After adding the six independent-auditor tests, the final full
suite passed 515 tests plus 194 subtests. Targeted Ruff
uses the existing CLI import-path exception for E402. V1/v2 artifacts and
their scientific verdicts remain unchanged.

The inline native scatter uses forty prompt-mean points, with cancellation
on x and recovery cosine on y. Both coordinates average the **same defined
nodes** within each prompt (1,117 nodes total); the above all-node
cancellation summary instead includes all 1,169 defined resultant ratios.
This chart is for prompt-level distribution, not the pooled-node Spearman
statistic. The [SQL aggregation](data/hltd_prefix_calibration_readout/chart_query.sql)
runs on an in-memory SQLite import of the audited `nodes.csv`, with blank
numeric cells imported as NULL. It reproduces the independently calculated
forty-point preview exactly. Palette policy is a single series with neutral
axes; prompt labels, family and node counts are retained for inspection.
The native widget accepted all forty rows without quality warnings; pixel
layout was not independently captured.
