# Prefix-Only Calibration-Field Transfer Gate

## Status and Question

The first native-FP32 model execution stopped at **`INSUFFICIENT_COVERAGE`**:
56/60 evaluation cells had an active retrieved coexact direction, and four
retrieved exact-zero vectors. All pilot future-invariance and zero-hook
checks passed. **No nonzero treatments or primary response estimate were
produced.** The original tokenizer-only preparation remains unchanged;
execution is recorded separately below.

The [fresh-text L7/L8 gate](hltd_fresh_l7_l8_gate.md) retained a signed
next-token endpoint, but its PCA and centered field used the evaluation
text, including future positions. The next question is:

> Does a coexact field learned on separate calibration texts transfer to
> evaluation prefixes when neither direction nor dose can use future text?

The [canonical specification](data/hltd_prefix_transfer/spec.json) fixes the
design before model responses. The implementation is in
[hltd_prefix_transfer.py](../scripts/hltd_prefix_transfer.py); it accepts
calibration hidden arrays or an observed-prefix hidden vector and does not
load a model. The integrated
[execution runner](../scripts/run_hltd_prefix_transfer_gate.py) binds that
implementation to the [execution protocol](data/hltd_prefix_transfer/execution_protocol.json).

## Data and Fixed Positions

| Split | Texts | Role |
| --- | ---: | --- |
| Calibration | 40 | All 20 original and 20 fresh-gate texts; fit chart, field, support, and dose |
| Pilot | 4 | One new text per family; integrated zero-hook and future-invariance checks only |
| Evaluation | 20 | Five new texts per family; fixed primary sample |

The [new suite](../data/hltd_prefix_transfer_prompt_suite.jsonl) contains
all 24 first-authored pilot/evaluation texts. No alternatives were selected
using model responses. Calibration outcomes are already known; all 40
historical texts are retained, without response-based selection.

The four families remain literal, metaphor, identity stress, and ontology
collapse. IDs and case/whitespace-normalized texts must be unique across
splits. New texts are also checked against every other historical JSONL
suite. Authored scenario IDs are unique among the 24 new rows. Word-trigram
overlap is reported descriptively, not used to select favorable responses.
Different strings and scenario labels do not establish independent writing
styles, a random population sample, or pretraining holdout.

Evaluation uses **absolute prefix lengths 8, 16, and 24**, not fractions of
the full text length. For length `n`, hidden extraction receives only
`input_ids[:n]`, and only the scorer may read target `input_ids[n]`.
There are exactly 60 planned evaluation prompt/prefix cells. Tokenization
has no special tokens or truncation; new texts must have 25-96 tokens.

## Frozen Calibration Field

1. Extract raw L7 token states from the 40 calibration texts. Normalize
   each hidden vector, then fit one shared PCA-32 chart, full solver, using
   calibration token states only.
2. In that chart, construct a separate centered field and k16 graph for
   each calibration text. Apply the existing orthogonal matched-Betti 0.5
   decomposition, and reconstruct node coexact vectors with ridge `1e-4`.
   Never form increments across text boundaries.
3. Pool the interior nodes and reconstructed vectors. Freeze node order
   by sorted calibration ID, then token position. A query selects the
   single nearest node; the first frozen node wins a distance tie.
4. Freeze the support radius at the 99th percentile (`method="higher"`)
   of nearest-node distances computed while excluding the node's entire
   calibration prompt. This is a descriptive calibration threshold, not
   a guarantee of 99% coverage on evaluation texts.
5. Freeze dose scale at the median of the 40 per-prompt medians of raw
   hidden centered-step norms. Evaluation hidden norms, targets, and
   future states cannot change that scale.

Each text gets its own graph to reuse the established small-graph
decomposition and avoid artificial cross-text increments. The shared chart
allows pooled lookup without a single much larger dense-basis decomposition.
This is a changed estimator, not the old evaluation-text field with one
variable removed. There is no continuity or smooth-traversal claim for the
nearest-node lookup.

Offline calibration may use the complete calibration texts and their
centered differences. Only the held-out evaluation direction must be
prefix-available. Calibration hidden hashes, PCA arrays, node vectors,
ownership, support radius, and dose contribute to the immutable field
fingerprint. The later execution freeze must also bind that field to this
specification and the exact calibration extraction/runtime receipts.

## Query and Controls

`direction_at_prefix` truncates IDs before calling the hidden provider;
`direction_for_prefix` passes an immutable tuple of observed IDs. The
provider returns one L7 hidden vector. `query_hidden` normalizes that vector,
applies the frozen chart, and retrieves the frozen node without fitting
anything on evaluation inputs.

The coexact vector is backprojected through the PCA components and
normalized in hidden space. The random control is an isotropic Gaussian in
the same frozen PCA span, keyed only by seed and frozen node index, then
normalized in hidden space. Both receive the same signed magnitude:

```text
delta = alpha * calibration_dose_scale * hidden_unit_direction
```

The legacy component label `random_tangent` is retained in the specification,
but this is a **PCA-span random control**, not a separately measured local
manifold tangent. PCA backprojection defines a candidate intervention
direction; it is not the exact inverse derivative of hidden-state L2
normalization.

Outside the fixed support radius, the query returns `OUT_OF_SUPPORT` and
zero deltas for both arms. A chart coexact norm below `1e-6` returns
`INACTIVE_COEXACT` with zero deltas. No alternative neighbor, random redraw,
or threshold relaxation is allowed. Invalid/nonfinite inputs are rejected.

## Gates Before Any Nonzero Treatment

The execution design is native-FP32 GPT-2 on MPS SDPA, no autocast or CPU
fallback, with float64 geometry. Source parameter bytes must match before
and after execution. Baselines use the same prefix IDs and batch-12 shape
as treatments. Zero hooks must have maximum logit error at most `1e-4`,
with exactly zero baseline-row spread. Mutable KV caches must not be shared
between prefixes or treatment arms.

The four pilot texts permit only integrated zero-hook and future-invariance
checks, not nonzero treatments or response-based parameter tuning. With
the observed prefix fixed, replacing, appending, or removing its suffix
must leave hidden extraction, chart coordinates, nearest node, support,
directions, dose, and actual deltas byte-identical within the fixed runtime,
seed, alpha, and batch shape. A failure is `INVALID_FUTURE_DEPENDENCE` and
stops execution before nonzero treatment.

The synthetic tests exercise the real truncation/query adapter with a
controlled hidden provider and forbid PCA/Hodge refitting. Those tests alone
do not verify the model pipeline; the separate integrated pilot result below
does check this model/runtime on the fixed pilot prefixes.

After the pilot, preflight all 60 evaluation cells before any nonzero
treatment. An unsupported or inactive cell yields `INSUFFICIENT_COVERAGE`:
retain the complete support table and report no reduced-sample primary
estimate. Invalid input/numerics must also stop execution. No favorable
subset, replacement text, or result-driven extension is allowed.

## Primary Endpoint

The planned grid is 20 prompts x 3 prefixes x 8 seeds x 2 components x
6 signed alphas: **5,760 nonzero treatment rows**, conditional on all gates.
Positive magnitudes are `0.25, 0.5, 1.0`, each with its negative counterpart.

For each prompt, prefix, and seed, use natural-log target probabilities:

```text
odd_gap(a) = ((lp_coexact(+a) - lp_coexact(-a))
             - (lp_random(+a) - lp_random(-a))) / 2
coefficient = sum(a * odd_gap(a)) / sum(a * a)
```

Average over eight seeds, then all three prefixes. Give each of the 20
prompts equal weight. Bootstrap prompts with 5,000 draws, seed 1729; use
the 2.5th and 97.5th percentiles. Complete 60-cell coverage and a strictly
positive lower endpoint are both required for
`SUPPORTED_WITHIN_SAMPLE_PREFIX_TRANSFER`. Otherwise a valid complete run
is `NOT_SUPPORTED`; coverage and validity failures remain separate.

This is a finite signed coexact-minus-random coefficient in nats per alpha,
not an infinitesimal derivative or an absolute gain over unsteered decoding.
Raw responses versus the matched baseline, per-family/per-prefix results,
support distances, and component activity are descriptive secondary outputs.
The author-selected sample and shared random-node controls limit broader
statistical interpretation; prompt bootstrap does not remove those limits.

## Preparation and Reproduction

[prepare_hltd_prefix_gate.py](../scripts/prepare_hltd_prefix_gate.py) verifies
the canonical spec, split/text inventory, model asset bytes, and numerical
source receipts, then loads **only the local tokenizer**. Model weights are
hashed as files, not loaded into a model. It records token IDs and exact
preparation package versions. It refuses an existing output or changed
canonical inputs.

The initial tokenizer-only preparation was created with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/prepare_hltd_prefix_gate.py \
  --prepare docs/data/hltd_prefix_transfer/preparation.json
```

The original preparation is now a pinned historical input. Its original
63-file inventory is verified byte-for-byte by the execution stage; four
new execution/analysis/test sources and the preparation artifact extend the
execution inventory to 68 receipts. The original preparation validator's
live script-directory inventory has intentionally not been weakened to
accept new stage files. Use execution validation on the current checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/run_hltd_prefix_transfer_gate.py \
  --validate docs/data/hltd_prefix_transfer/execution_protocol.json

JAX_PLATFORMS=cpu python3 -m pytest \
  tests/test_hltd_prefix_transfer.py tests/test_hltd_prefix_protocol.py \
  tests/test_hltd_prefix_execution.py tests/test_hltd_prefix_diagnostics.py -q
```

Preparation status must remain `PREPARED_NOT_EXECUTED`, with
`execution_allowed=false`, `model_parameters_loaded=false`, and null
`atlas`/`results`. Source or package changes invalidate that receipt; a new
stage must preserve the earlier artifact and explicitly bind its own inputs.
Preparation is neither execution authorization nor a passed scientific gate.

### Completed Preparation Checks

The [saved preparation](data/hltd_prefix_transfer/preparation.json) was
created and independently revalidated through the CLI on 2026-10-01 JST,
using the local GPT-2 tokenizer with network access disabled by offline
settings. Calibration texts have 26-35 tokens, pilot texts 37-40, and
evaluation texts 37-43; no text was truncated or replaced.

All 24 new rows are exact-duplicate-free against 69 historical rows. The
maximum historical word-trigram Jaccard is 0.053571, a descriptive overlap
measure only. All 63 frozen file receipts validate, including the seven
canonical model/tokenizer assets; during this initial preparation checkpoint
bytes were hashed, not loaded as model parameters.

At preparation time the two new test modules passed 72 tests, and the
complete suite passed **411 tests and 194 subtests**, with the existing
`TRANSFORMERS_CACHE` deprecation
warning. Targeted Ruff checks pass with the existing E402 import-path
exception. These are preparation and implementation checks, not evidence
that the model-level prefix-transfer endpoint passes.

## Execution Outcome

The [verdict](data/hltd_prefix_transfer/result/verdict.json) is
**`INSUFFICIENT_COVERAGE`**, not `NOT_SUPPORTED` and not a positive response
result. The primary is null because the all-cell coverage gate failed.

| Stage | Observed result |
| --- | --- |
| Model | One native-FP32 GPT-2 load, MPS SDPA, no fallback/autocast |
| Calibration | 40 texts, 1,170 pooled interior nodes, frozen PCA-32 |
| Dose | 40.4991076282, raw hidden units per alpha, calibration-only |
| Support radius | 0.3542267077, calibration-only leave-one-prompt-out quantile |
| Pilot | 12 cells x 4 suffix variants; byte-identical query/delta receipts |
| Pilot activity | 11 active cells and one inactive cell; no nonzero pilot treatment |
| Zero controls | 108 checks: 48 pilot variants + 60 evaluation cells; all errors and row spreads exactly zero |
| Evaluation distance support | 60/60 inside the fixed radius; distances 0.049183-0.339808 |
| Active coexact directions | 56/60; four exact-zero vectors |
| Nonzero treatments | 0/5,760 planned; stopped before treatment |
| Primary effect/CI | Not estimated; no reduced-sample result |

The [support table](data/hltd_prefix_transfer/result/support.csv) retains all
60 planned cells. These four failed on component activity, not distance:

| Evaluation prompt | Prefix length | Retrieved calibration text | Calibration token index (zero-based) | Distance | Coexact norm |
| --- | ---: | --- | ---: | ---: | ---: |
| `prefix_eval_metaphor_01` | 16 | `metaphor_01` | 17 | 0.092371 | 0 |
| `prefix_eval_identity_01` | 16 | `fresh_identity_01` | 11 | 0.063768 | 0 |
| `prefix_eval_identity_02` | 16 | `fresh_identity_02` | 13 | 0.075448 | 0 |
| `prefix_eval_ontology_01` | 24 | `fresh_ontology_03` | 17 | 0.156114 | 0 |

No nearest-active-node fallback, threshold change, prompt replacement,
partial-sample estimate, or second model attempt was used. In particular,
these vectors are exactly zero, not merely just below the activity cutoff.

### Structural Diagnostic

A separately labeled [offline post-outcome diagnostic](data/hltd_prefix_transfer/result/coverage_diagnostic.json)
rebuilt the entire calibration atlas from saved hidden arrays. Its fingerprint
was byte-identical to the frozen field. Reconstructing each of the four local
graphs also reproduced the stored zero vector exactly, without loading a model.

The four retrieved nodes have 16, 17, 16, and 17 incident graph edges, but
**zero incident selected triangles**. The corresponding incident-edge rows
of the triangle boundary matrix are zero, as are all incident coexact edge
values. Thus the local reconstruction has no coexact signal at those nodes
under this particular selected complex. This is a structural consequence of
the frozen estimator, not evidence that the underlying concepts have no
circulation in another representation or complex.

The atlas has **52/1,170 exact-zero node vectors**; all 52 fail the activity
threshold. The four evaluation misses select such nodes. Triangle-incidence
replay was performed for those four misses, not for every one of the 52
atlas zeros. No alternative interpolation, graph, or threshold was measured.

The diagnostic's first attempt exposed a missing scikit-learn PCA attribute
in a manually reconstructed helper object. Only the post-outcome diagnostic
was repaired to refit the unchanged calibration PCA via its public API and
require byte-identical components. The model runner, frozen sources, atlas,
support table, and gate verdict were not changed or rerun.

### Evidence and Reproduction

Execution was frozen at **2026-09-30 20:03:32 UTC** (2026-10-01 JST), began at
20:03:56, and ended at 20:04:12. The 148 parameter tensors, comprising
124,439,808 parameters, match native-FP32 checkpoint bytes before and after
the run. All 68 execution input receipts remain intact. These timings are
receipts, not a throughput benchmark.

Before model execution, the complete suite passed 440 tests and 194 subtests.
After adding the separately scoped offline diagnostic and its regression
test, the final suite passes **442 tests and 194 subtests**, with the one
existing `TRANSFORMERS_CACHE` deprecation warning. Targeted Ruff checks pass
with the existing E402 exception, and execution-protocol validation and
`git diff --check` pass. No frozen execution source was edited after launch.

The analyzer replays all 60 field queries and all nominal float32 delta
grids from the frozen atlas and saved prefix states. The
[standard-library audit](data/hltd_prefix_transfer/result/stdlib_audit.json)
independently checks artifact hashes, the complete support grid, and absence
of treatment or partial-sample inference. There are no response coefficients
to recompute. The [manifest](data/hltd_prefix_transfer/result/manifest.json)
binds the compact verdict/support output and the local raw execution files;
the post-outcome diagnostic separately records its own source/input receipts.

The following commands were used for the one execution, followed by offline
analysis. Existing protocol, run, and output paths are rejected, so they must
not be treated as retry commands:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/run_hltd_prefix_transfer_gate.py \
  --freeze docs/data/hltd_prefix_transfer/execution_protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/run_hltd_prefix_transfer_gate.py \
  --run docs/data/hltd_prefix_transfer/execution_protocol.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/analyze_hltd_prefix_transfer_gate.py \
  --protocol docs/data/hltd_prefix_transfer/execution_protocol.json \
  --output docs/data/hltd_prefix_transfer/result

python3 -S scripts/audit_hltd_prefix_transfer.py \
  --result docs/data/hltd_prefix_transfer/result \
  --output docs/data/hltd_prefix_transfer/result/stdlib_audit.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
python3 scripts/diagnose_hltd_prefix_coverage.py \
  --protocol docs/data/hltd_prefix_transfer/execution_protocol.json \
  --output docs/data/hltd_prefix_transfer/result/coverage_diagnostic.json
```

## Interpretation and Next Stage

A positive result would establish only a within-sample prefix-available
transfer endpoint under this calibration construction. A negative result
could reflect transfer failure, field estimation, support, or other changed
design choices; it would not isolate future-token dependence by itself.
The older L7/L8 results and all of their caveats remain unchanged.

The measured result establishes an operational prefix-only lookup path on
this pilot, but not a nonzero intervention effect. It exposes an availability
limit of one-nearest-node coexact readout: being close to calibration geometry
does not imply an active retrieved direction.

There is no generated-text fluency, semantic, identity/affordance, cross-model,
L7/L8 common-dose, or concept-ring claim. A possible v2 would define a
calibration-only interpolation and its zero/support handling prospectively,
including inactive neighbors rather than choosing directions by evaluation
success. It would need a new protocol and genuinely new primary evaluation
texts: this v1 evaluation coverage is now observed. No v2 intervention or
parameter search has been performed, and the v1 failure is retained.
