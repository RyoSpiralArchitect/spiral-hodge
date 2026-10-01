# Prefix Evidence Review Checkpoint

PR #8 preserves the original experimental sources and outcomes. Review found
that three historical convenience tools checked receipt lists but could read
different local files afterward. This is a valid path-binding defect for
arbitrary caller-supplied result trees, not evidence that the recorded results
were calculated from substituted files.

## Current Verification Entry Point

Use the new standard-library verifier in the original evidence workspace:

```bash
python3 -S scripts/verify_hltd_prefix_release.py
```

It pins six canonical manifests, requires exact output and raw-file
inventories, rejects missing/duplicate/redirected records, and binds the
actual consumed paths to their verified bytes. Only after this validation
does it reuse the unchanged historical arithmetic routines. It recomputes
both saved chart tables from verified CSV and SQL inputs, then repeats the
bindings before issuing a result. It performs no model inference or new
scientific analysis, and cannot promote a semantic claim.

The original `audit_hltd_prefix_transfer.py`,
`audit_hltd_prefix_calibration.py`, and `export_hltd_prefix_null_charts.py`
remain historical replay sources, **not recommended entry points for
validating untrusted result directories**. Their bytes and old receipts
are retained. Historical notes show the commands actually used at the time;
they are not silently rewritten to claim use of the new verifier.

The verifier is for this specific recorded release. It is not a portable
replacement for missing raw runs/model receipts, and it must fail when the
original pinned files are unavailable. It does not regenerate the old charts
or update the old audit verdicts.

## Review Disposition

- **Accepted:** calibration audit could verify files elsewhere and consume
  local summary/CSV files. Exact path/inventory binding now precedes its
  arithmetic in the current verification entry point.
- **Accepted:** transfer audit could read an unbound verdict/bootstrap table.
  Both are in the exact derived-file inventory, tied to the canonical manifest
  before the unchanged probability/interval calculation runs.
- **Accepted:** chart export did not require complete stage inventories.
  The successor verifies both complete inventories and the recorded chart
  source inventory, then queries only bound paths and compares saved tables.
- **Partly accepted:** the PR's intermediate count of 17 integration cases
  was stale; the final split is 18. The proposed removal of an unused fixture
  parameter is not applied to frozen tests: pytest still evaluates that
  fixture, even if the function body ignores the argument. The assertion also
  runs in the portable counterpart with a model-free protocol fixture, so
  ordinary runs retain its coverage.

Codex connector review was unavailable due to its account review limit.
Copilot reviewed commit `6c067feafd2b9c1b27ea60ca39f310775c9e77fe`;
its findings were evaluated individually, not automatically applied.
Engineering review is not confirmation of the HLTD semantic hypothesis.
