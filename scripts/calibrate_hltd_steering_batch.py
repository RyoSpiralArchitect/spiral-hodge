#!/usr/bin/env python3
"""Audit zero interventions at matched batch size before a signed layer run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hltd_steering import _load_model_and_tokenizer, _log_softmax, _logits_with_deltas
from scripts.run_hltd_steering_suite import read_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "data/hltd_prompt_suite.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("batch-size and finite tolerance must be positive")

    import torch

    started = datetime.now(timezone.utc).isoformat()
    model, tokenizer = _load_model_and_tokenizer(
        args.model_path, device=args.device, local_files_only=True, trust_remote_code=False,
    )
    rows = []
    for item in read_suite(args.suite):
        inputs = tokenizer(
            item["text"], return_tensors="pt", truncation=True, max_length=args.max_length,
        ).to(args.device)
        ids = inputs["input_ids"][0].cpu().tolist()
        repeated = {key: value.repeat(args.batch_size, 1) for key, value in inputs.items()}
        with torch.no_grad():
            single = model(**inputs, output_hidden_states=True, return_dict=True)
            dim = single.hidden_states[args.layer].shape[-1]
            logits = single.logits[0].detach().float().cpu().numpy()
            batch = model(**repeated, return_dict=True).logits.detach().float().cpu().numpy()
        for token in range(1, len(ids) - 1):
            zero = _logits_with_deltas(
                model, inputs, layer=args.layer, token_index=token,
                deltas=np.zeros((args.batch_size, dim)),
            )
            base = batch[:, token]
            matched_error = float(np.max(np.abs(zero - base)))
            row_error = float(np.max(np.abs(base - base[0])))
            offset = float(_log_softmax(base[0])[ids[token + 1]] - _log_softmax(logits[token])[ids[token + 1]])
            rows.append({
                "prompt_id": item["prompt_id"], "token_index": token, "token_count": len(ids),
                "zero_vs_matched_batch_logit_max_abs": matched_error,
                "batch_row_logit_max_abs": row_error,
                "batch_vs_single_logit_max_abs": float(np.max(np.abs(base - logits[token]))),
                "batch_vs_single_next_token_logprob_delta": offset,
            })
        print(f"calibrated {item['prompt_id']}: {len(ids) - 2} token positions", flush=True)

    finite = all(np.isfinite(row[key]) for row in rows for key in (
        "zero_vs_matched_batch_logit_max_abs", "batch_row_logit_max_abs",
        "batch_vs_single_logit_max_abs", "batch_vs_single_next_token_logprob_delta",
    ))
    hook_error = max(row["zero_vs_matched_batch_logit_max_abs"] for row in rows)
    row_error = max(row["batch_row_logit_max_abs"] for row in rows)
    result = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path, "model_dtype": str(next(model.parameters()).dtype),
        "layer": args.layer, "batch_size": args.batch_size, "max_length": args.max_length,
        "device": args.device, "tolerance": args.tolerance,
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "n_prompts": len({row['prompt_id'] for row in rows}), "n_token_units": len(rows),
        "zero_vs_matched_batch_logit_max_abs": hook_error,
        "batch_row_logit_max_abs": row_error,
        "batch_vs_single_logit_max_abs": max(row["batch_vs_single_logit_max_abs"] for row in rows),
        "batch_vs_single_next_token_logprob_max_abs": max(abs(row["batch_vs_single_next_token_logprob_delta"]) for row in rows),
        "passed": bool(finite and hook_error <= args.tolerance and row_error <= args.tolerance),
        "scope": "Zero-control agreement only; not nonzero steering precision equivalence.",
        "token_units": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "token_units"}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
