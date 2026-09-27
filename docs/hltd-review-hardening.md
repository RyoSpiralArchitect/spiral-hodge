# HLTD review hardening

The initial publication commit is
`f6dd7a941624f0a21a2b6d7b8314aef38a85c5e1`. External review of PR #7 identified
three prospective gate failures: L8 freeze could omit missing required files;
precision analysis could attempt a fit before reporting insufficient early
coverage; and the pilot runner did not compare its design with the canonical
pilot contract before execution.

The fixes reject missing required inputs before saving a protocol, save an
explicit `INSUFFICIENT_COVERAGE` verdict and all-arm coverage/activity tables
before any fit, and validate the pilot's complete scientific contract before
creating a run directory or loading a model. The pilot reference is pinned by
SHA256, so editing both the supplied and canonical protocol cannot silently
change the contract. A new top-level output name is allowed; revised hypotheses
require a separately defined protocol, not relaxed validation.

A second review found that directory discovery alone still allowed a partial
model cache to omit configuration/tokenizer files. L8 freeze now requires the
reference's model-file inventory (including the six frozen GPT-2 weight,
configuration and tokenizer assets) at the requested model path, and compares
their bytes with the reference before writing anything. Optional loader files
are also recorded when present. Tests remove each required asset individually,
retain the complete original reference directory, and verify that the relocated
partial cache still fails before runtime inspection or loading; changed config
bytes are rejected as well.

A third review checked the opposite relocation case: the requested cache is
complete, but the original model directory is gone. All original model receipts
are now excluded from the historical live-file audit and new runtime inventory;
the verified relocated copies replace them. Other historical inputs remain
mandatory. Positive fixtures remove the entire original model directory both
before and after freeze, then check the new execution receipts strictly.

A fourth review identified the missing MPS fallback guard in the precision
pilot/full shared runner. It now rejects `PYTORCH_ENABLE_MPS_FALLBACK=1` before
model work and records `device_fallback: false` in future load audits. Both
runner entrypoints are tested to retain a failed receipt without calling the
model loader. Historical precision load audits did **not** record this flag;
MPS parameter placement alone cannot retroactively establish that fallback was
disabled. The original evidence is retained with that verification limitation,
not reclassified as having passed the new guard.

A fifth review found that the full runner could compare a candidate against
candidate-selected edited companion protocols. Both precision runners now use
the same canonical-contract validator, with separately pinned canonical JSON
hashes. The full comparison binds the reference paths, scientific fields,
frozen input inventory and non-code input hashes before any output or model
work. Tests cover coupled edited companions, a rewritten canonical full
protocol, design/input changes and output-path escapes.

The related L8/fresh model contracts now require receipts for the actual selected
model directory, matching every referenced model/config/tokenizer asset's bytes.
An unchanged declared checkpoint hash cannot cover a different loaded model or
an omitted/modified model receipt. Fresh-text validation also binds the model
path and runtime to its reference. These checks run before model work and have
negative entrypoint tests; normal freezes record the required assets first.

A sixth review identified candidate-selected bridge CSVs. L8 bridge paths and
receipts are now pinned to the original full-precision figure manifest, whose
SHA256 is fixed in code. The manifest also anchors the full reference protocol.
Both the aggregate input and the explicitly amended five-shard recovery are
checked against original paths, hashes and byte counts; candidate-selected
hashes cannot substitute for the original evidence. Tests cover altered or
missing receipts, edited CSV pairs, changed bytes during freeze, and manifest
replacement. Recovery protocols remain valid only with the pinned shards and
must use the continuation entrypoint; the ordinary runner rejects them before
model work. The original failed aggregate attempt is preserved, not made valid
by inventing a retrospective receipt.

A seventh review found that family counts alone allowed response-selected
replacement texts in the fresh gate. Validation now requires the exact recorded
20-prompt inventory, including text, family, order and token IDs, plus the
canonical suite's byte count and hash. Both the original fresh protocol and its
L8 parent are hash-pinned. Changing the suite and its candidate-controlled receipt
together cannot satisfy this check. Tests mutate individual prompt fields,
same-family text sets, suite receipts and either recorded protocol, and confirm
rejection before output/model work. Prospective freezes describe these fixed
texts as a rerun with known prior outcomes, not a new held-out replication.

## Historical evidence

No prior protocol, execution receipt, coefficient, figure, result hash, or failed
run has been rewritten. The twelve source files affected by these fixes and
their historical-code tests are preserved byte-for-byte in
[`source_snapshots/f6dd7a941624f0a21a2b6d7b8314aef38a85c5e1`](source_snapshots/f6dd7a941624f0a21a2b6d7b8314aef38a85c5e1/manifest.json).
The snapshot manifest records the original repository path, byte count, SHA256,
and source commit. An added `.txt` suffix prevents accidental Python import or
test discovery; file contents are unchanged. These files are audit material,
not executable replacements for the active runners.

Artifact tests resolve historical `verification_code` receipts against matching
live code or this explicit snapshot. Data and figures must still match their
live bytes. Prospective L8/fresh freezes can audit historical source against the
snapshot, disclose the snapshot paths, then bind **current live source** in the
new protocol. Missing or changed data, checkpoints and external library code
cannot use this fallback.

Execution preflights remain strict: `verify_frozen_files` never consults the
snapshot. Consequently, running a historical protocol under patched code must
fail. To reproduce the original command, use its historical source revision and
the separately recorded model, raw inputs and environment; a source checkout
alone does not provide those ignored/external inputs. A prospective run needs a
new freeze and output directory. These changes do not retroactively claim that
past measurements ran under the new checks.

Regression tests cover missing raw/source/third-party inputs at freeze time,
all-inactive and sparse early coverage through both analyzer entrypoints,
contract mutations before model work, and historical source validation without
loosening execution or data receipts. All use synthetic files or saved compact
evidence; no model experiment is rerun for this review.
