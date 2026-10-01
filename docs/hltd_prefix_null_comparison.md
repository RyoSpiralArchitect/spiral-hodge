# Position-Matched Nulls and Saved V2 Responses

## Conclusion

The calibration field retains a weak but positive directional recovery gap
over the fixed references, including a new within-prompt, relative-position-bin
shuffle. This result does **not** convert the saved v2 response result into a
success: its original signed endpoint remains `NOT_SUPPORTED`.

The useful separation is now sharper:

- Neighbor geometry contains information about the existing calibration
  coexact targets beyond these coarse position bins.
- Reconstructing those targets is not the same as aligning a prefix-only
  intervention with the next-token objective.
- Donor cancellation, position gaps and distance do not provide a consistent
  cross-prefix explanation of v2's response. A strong prefix-24 association
  is retained as exploratory evidence, not used to choose a new treatment.

This follow-up is post-v2 and post-calibration diagnosis. It is not a new
confirmatory study, semantic validation, or retrospective repair of either
v1 or v2. No model or tokenizer was loaded; no model forward or new treatment
was performed.

## Fixed Order and Evidence

The [plan](data/hltd_prefix_null_comparison/plan.json) fixes both stages before
this follow-up calculation. The
[protocol](data/hltd_prefix_null_comparison/protocol.json) binds that plan,
scripts, tests, previous calibration artifacts and v2 manifest with SHA256
receipts. Protocol SHA256:

```text
d1fc6ed31b5a8a40c82f0bcc4e45dd461e8d58ce6a94517ed8f6d10f6246fc91
```

| Event | UTC on 2026-10-01 |
| --- | --- |
| Both stages frozen | 00:49:48.019487 |
| Calibration comparison started | 00:49:57.560071 |
| Calibration comparison completed | 00:49:59.529406 |
| Saved-v2 analysis started | 00:50:21.501182 |
| Saved-v2 analysis completed | 00:50:26.920761 |

Stage A does not open evaluation observations, treatment responses, or model
assets. Stage B requires a completed, receipt-verified Stage A manifest.
Its analysis rules do not depend on the values found in Stage A.

## Calibration Comparison

The eight-neighbor Gaussian readout, shared PCA chart, distances, radii,
activity threshold and query-prompt exclusion are unchanged from the
[previous calibration diagnostic](hltd_prefix_calibration_readout.md).
The target remains the original reconstructed coexact vector at each node.
It is not semantic ground truth or an optimal causal direction.

The new reference permutes whole vectors within `(prompt_id, position_bin)`.
Position bins are the already defined relative thirds of each complete
calibration text. The permutation retains the joint direction/norm inventory
and zero vectors within each group; query targets and geometric weights are
fixed. There are 128 draws with seed `6043`. Fixed points are allowed and no
draw is retried. The 120 strata contain 8-11 nodes each, with no singletons;
10.0781% of node assignments remain fixed across the saved draws.

All four arms use the intersection of valid-direction masks over **every**
null draw. Means average nodes within each prompt, then prompts equally,
then fixed control draws. The common mask retains all 1,117 originally
recoverable nodes from all 40 prompts. It introduces no additional omission.
The other 53 atlas nodes remain recorded with undefined recovery, not zeros.

| Readout / reference | Draws | Mean cosine | Draw q10-q90 |
| --- | ---: | ---: | ---: |
| Actual coexact interpolation | 1 | +0.122079 | Not a reference distribution |
| Isotropic random field | 8 | +0.001935 | [-0.005158, +0.009072] |
| Within-prompt shuffle | 128 | +0.002686 | [-0.003920, +0.009500] |
| Within-prompt, position-bin shuffle | 128 | +0.009008 | [+0.000779, +0.017464] |

Actual minus position-bin shuffle is `+0.113071`. Restricting to the common
1,098 distance-supported nodes gives `+0.121152` versus `+0.008872`, a gap of
`+0.112280`. All 40 prompts remain represented.

These quantiles describe the fixed null draws; they are **not confidence
intervals** or a significance test. The position-matched mean is higher
than the unconstrained shuffle mean, but this does not identify a percentage
of the signal caused by position. The intervention on donor assignments
changes more than a single causal variable.

The narrow reading is that this coarse three-bin reference does not account
for the observed recovery gap. Exact position, token identity, syntax,
local geometric structure, shared-chart fitting and trajectory dependence
remain possible explanations. Absolute cosine `0.122` is still weak
agreement. Shared PCA is not fitted out-of-fold, so this is donor-prompt-
excluded reconstruction, not fully independent cross-validation.

The old neighbor, distance, weight, prediction and within-prompt permutation
arrays replay byte-for-byte; the previous random/shuffle statistics have
maximum difference `0.0`. See the
[calibration summary](data/hltd_prefix_null_comparison/calibration/summary.json)
and [paired comparisons](data/hltd_prefix_null_comparison/calibration/draws.csv).

## Saved V2 Analysis

The replay verifies all 48 pilot-variant observations and 60 evaluation
observations, every saved nominal treatment delta and matched baseline,
the complete 5,760-row response grid, the original primary coefficients,
and the same 5,000 bootstrap index rows. A separate standard-library audit
recalculates the primary endpoint from recorded probabilities, with maximum
primary error `1.11e-16`.

| Original v2 endpoint | Unchanged result |
| --- | ---: |
| Mean coexact-minus-random odd slope | +0.006465 |
| Original 95% prompt-bootstrap interval | [-0.103101, +0.134270] |
| Positive prompt coefficients | 10/20 |
| Verdict | `NOT_SUPPORTED` |

This is lack of support for the prespecified positive effect, not evidence
of equivalence or proof of zero effect. The
[original v2 record](hltd_prefix_transfer_v2.md) remains authoritative.

### Geometry Is Not a Consistent Response Predictor

Eight random seeds are averaged before forming each of the 60 prefix cells.
The three cells per prompt are repeated observations, not 60 independent
prompts. Donor descriptors are reconstructed from the saved indices and
weights. The query token index is `prefix_length - 1`. Only calibration
lengths define donor bins; future evaluation-text length is never used.
There is no observed coexact ground truth at these v2 queries, so these
descriptors must not be called v2 reconstruction accuracy.

| Descriptor | Equal-prompt mean |
| --- | ---: |
| Resultant fraction | 0.449745 |
| Cancellation fraction | 0.550255 |
| Effective contributors out of eight | 6.152341 |
| Weighted absolute token-position gap | 5.839898 tokens |
| Nearest donor distance | 0.197579 |

Resultant fraction is a property of the unnormalized donor sum. V2 normalizes
the aggregate intervention direction before dosing; a lower resultant
fraction is not automatically a smaller injected perturbation.

All predefined Spearman correlations with the `coexact - random` odd slope:

| Predictor | Pooled 60 cells | Prefix 8 | Prefix 16 | Prefix 24 | 20 prompt means |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resultant fraction | -0.154 | -0.083 | +0.308 | -0.746 | +0.108 |
| Effective contributors | -0.036 | -0.268 | +0.011 | +0.180 | +0.191 |
| Weighted token gap | -0.191 | -0.008 | -0.164 | -0.489 | +0.096 |
| Nearest distance | -0.044 | -0.033 | -0.041 | +0.047 | +0.042 |

Prefix 24's resultant/gap association is not small (`-0.745865`), but it
changes sign relative to prefix 16 and is absent as a strong prompt-average
relationship. At prefix 24, the corresponding correlation with **coexact
alone** is only `-0.275188`; a contrast involving random response is not a
pure measure of coexact failure. Do not discard this result, but do not
select prefix 24, an intervention sign, or a new coherence threshold from it.
All 40 predictor/outcome/scope combinations, including coexact-only outcomes,
are retained in [associations.csv](data/hltd_prefix_null_comparison/v2/associations.csv).
No p-values or causal mediation claims are made.

### Signed and Symmetric Finite-Dose Responses

For each component and the three fixed positive magnitudes:

```text
delta_plus  = logp(+a) - logp(baseline)
delta_minus = logp(-a) - logp(baseline)
odd(a)     = (delta_plus - delta_minus) / 2
even(a)    = (delta_plus + delta_minus) / 2
```

These are finite-dose responses, not derivatives. Each mean first averages
eight seeds, then three prefixes, then 20 prompts equally.

| Magnitude | Coexact odd | Random odd | Coexact even | Random even |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | -0.001460 | -0.008393 | -0.004460 | -0.004409 |
| 0.50 | -0.008002 | -0.017640 | -0.018769 | -0.020525 |
| 1.00 | -0.032610 | -0.034543 | -0.112237 | -0.133254 |

At magnitude one, coexact gives mean changes `-0.144847` for the positive
sign and `-0.079628` for the negative sign; random gives `-0.167797` and
`-0.098711`. Thus simply reversing the sign does not turn either largest-dose
average into a gain. Symmetric negative response is larger in magnitude
than the signed response for both fields at this dose. That is compatible
with a shared perturbation cost, but does not establish its mechanism,
an optimal smaller dose, loss of fluency, or absence of semantic effects.

Prefix-specific values differ and are all retained in
[dose_summary.csv](data/hltd_prefix_null_comparison/v2/dose_summary.csv).
The complete saved-v2 diagnostic is in
[summary.json](data/hltd_prefix_null_comparison/v2/summary.json).

## Reproduction and Checks

The recorded run used these commands, in this order, in the managed
`SpiralReality/hltd-prefix-transfer` worktree. Freeze/run commands are
exclusive: running them against the existing artifacts intentionally fails.
Do not delete or replace the old run to make them succeed. Full replay
requires the original local atlas and saved v2 arrays.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/compare_hltd_prefix_nulls.py \
  --freeze docs/data/hltd_prefix_null_comparison/protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/compare_hltd_prefix_nulls.py \
  --run docs/data/hltd_prefix_null_comparison/protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/analyze_hltd_prefix_nulls_v2.py \
  --protocol docs/data/hltd_prefix_null_comparison/protocol.json

python3 -S scripts/export_hltd_prefix_null_charts.py
```

Read-only contract verification remains available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/compare_hltd_prefix_nulls.py \
  --validate docs/data/hltd_prefix_null_comparison/protocol.json
```

New tests cover strata, singleton/fixed-point handling, common masks,
undefined targets, unchanged old controls, source tampering, phase order,
query token indexing, seed aggregation, signed/even response formulas,
complete grids and missing/constant correlations. Chart SQL is checked
separately for common-mask selection and nested equal-prompt aggregation.
Ruff uses the repository CLI-bootstrap `E402` exception for the analysis
scripts; the stdlib-only chart exporter needs no exception.

Final verification on 2026-10-01: `536 passed, 194 subtests passed`; one
existing Transformers cache deprecation warning. Targeted Ruff and
read-only protocol validation pass. The primary checkout remains clean on
`main`; all follow-up files are in the existing managed worktree, with no
commit, push, PR, or merge in this follow-up.

The inline comparison bars use executed SQLite queries from the pinned CSVs,
not hand-entered plot values. Independent SQL aggregation agrees with the
Python summaries to `2.78e-17`. Source queries, widget tables and receipts are
in [charts/](data/hltd_prefix_null_comparison/charts/). The native widget
renderer accepted both charts without quality warnings; this is not a
desktop/mobile pixel-level rendering audit.

## Next Question, Not Executed

The next useful distinction is **direction reconstruction versus local logit
alignment**, rather than tuning the geometric readout using these v2 cells.
A new-text, prospectively specified diagnostic could compare the frozen
readout direction with the local gradient of a fixed next-token objective,
then check a predeclared small symmetric dose against that directional
derivative. Targets would be evaluation labels, never inputs used to choose
the online direction. Matched random controls and complete prompt-level
reporting would remain mandatory.

That experiment could distinguish local objective misalignment from
finite-dose nonlinearity. It would still not establish semantic control;
semantic or fluency endpoints need their own independently fixed validation.
No such model computation or v3 selection was performed in this follow-up.
