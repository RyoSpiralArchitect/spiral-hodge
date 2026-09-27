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

## Historical evidence

No prior protocol, execution receipt, coefficient, figure, result hash, or failed
run has been rewritten. The eight source files affected by these fixes and
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
