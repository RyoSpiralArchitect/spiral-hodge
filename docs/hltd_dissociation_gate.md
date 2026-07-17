# HLTD Presence/Coexact Dissociation Gate

This note records the first explicit dissociation gate for HLTD one-step
directions. The goal is to separate two effects that were mixed in earlier
gates:

- next-token support from the logits head
- learned-probe label stabilization in hidden state

The tested directions were:

```text
presence
coexact
presence_plus_coexact
coexact_minus_presence
negative_coexact
random_tangent
```

`coexact_minus_presence` is the same vector expression as `-presence +
coexact`. The component vector is normalized before scaling by the median
natural hidden-state step, matching the existing one-step steering convention.

## Runs

Logits/next-token gate:

```bash
python3 scripts/run_hltd_steering_fast_suite.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_dissociation_steering_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent \
  --target-set-file data/hltd_semantic_targets.json
```

Learned hidden-state probe gate:

```bash
python3 scripts/run_hltd_probe_gate.py \
  --suite data/hltd_prompt_suite.jsonl \
  --model-path /Users/ryospiralarchitect/SpiralReality/model/gpt2 \
  --output-root spiral_out_hltd_dissociation_probe_full_mps \
  --layers 4 5 6 7 8 \
  --k 16 \
  --components 32 \
  --max-length 64 \
  --alphas 1.0 \
  --seeds 0 1 \
  --token-selectors max_component middle \
  --device mps \
  --steering-components presence coexact presence_plus_coexact coexact_minus_presence negative_coexact random_tangent
```

The logits run completed 100 prompt/layer/k runs and 2400 rows in 52.0
seconds. The probe run completed 100 prompt/layer/k runs and 7200 rows in 17.2
seconds.

## Middle-Token Layer Read

The table below joins the layer-averaged middle-token summaries:

- `next`: component-minus-random teacher-forced next-token log-probability
  delta
- `semantic margin`: lexical target-minus-control semantic-margin delta
- `ontology probe`: learned `ontology_collapse` probe label-margin delta

| Layer | Component | next | semantic margin | ontology probe |
| ---: | --- | ---: | ---: | ---: |
| L4 | presence | -0.0621 | -0.2072 | 0.9728 |
| L4 | coexact | 0.1038 | -0.2122 | -0.4323 |
| L4 | presence_plus_coexact | 0.1718 | -0.1998 | 0.3926 |
| L4 | coexact_minus_presence | 0.0178 | -0.2292 | -0.9780 |
| L4 | negative_coexact | -0.4005 | 0.0036 | -0.2037 |
| L5 | presence | -0.0382 | -0.2154 | 1.8824 |
| L5 | coexact | 0.4372 | -0.2254 | -0.0817 |
| L5 | presence_plus_coexact | 0.3809 | -0.2344 | 0.7207 |
| L5 | coexact_minus_presence | 0.2222 | -0.1723 | -0.8784 |
| L5 | negative_coexact | -0.4477 | -0.0186 | -0.2210 |
| L6 | presence | -0.0864 | -0.2022 | 1.2499 |
| L6 | coexact | 0.3100 | -0.2200 | -0.4329 |
| L6 | presence_plus_coexact | 0.2997 | -0.1929 | 0.4753 |
| L6 | coexact_minus_presence | 0.2065 | -0.2117 | -0.8375 |
| L6 | negative_coexact | -0.5309 | -0.0071 | -0.1481 |
| L7 | presence | -0.2881 | -0.3007 | 1.0265 |
| L7 | coexact | 0.4516 | -0.1617 | 0.3632 |
| L7 | presence_plus_coexact | 0.4462 | -0.1214 | 0.8426 |
| L7 | coexact_minus_presence | 0.1827 | -0.1705 | -0.3331 |
| L7 | negative_coexact | -0.7183 | -0.0844 | -0.6506 |
| L8 | presence | -0.0633 | -0.2957 | 0.4438 |
| L8 | coexact | 0.4940 | 0.0786 | 0.4686 |
| L8 | presence_plus_coexact | 0.5689 | 0.0706 | 0.5895 |
| L8 | coexact_minus_presence | 0.3329 | 0.1303 | -0.3443 |
| L8 | negative_coexact | -0.6821 | -0.1213 | -1.6466 |

## Read

This gate gives the cleanest separation so far:

> Presence is the stronger learned-probe stabilizing direction, while coexact
> is the stronger next-token traversal direction.

The decisive pattern is:

- `presence` has weak or negative next-token support, but strongly positive
  ontology probe margin from L4-L8.
- `coexact` has positive next-token support at every tested middle layer, but
  its ontology probe margin is only positive at L7-L8.
- `presence_plus_coexact` preserves the coexact next-token effect and restores
  positive ontology probe margin, especially at L7-L8.
- `coexact_minus_presence` keeps some next-token support but loses ontology
  probe stabilization, especially before L8.
- `negative_coexact` strongly harms next-token support, which is a useful sign
  check that the coexact direction is not arbitrary.

The L8 point is the current best dissociation example:

| Component | next | ontology probe | affordance probe | identity probe |
| --- | ---: | ---: | ---: | ---: |
| coexact | 0.4940 | 0.4686 | 0.0611 | -0.4250 |
| presence | -0.0633 | 0.4438 | 0.6206 | 0.6155 |
| presence_plus_coexact | 0.5689 | 0.5895 | 0.2765 | -0.1252 |
| coexact_minus_presence | 0.3329 | -0.3443 | 0.2926 | -0.6077 |
| negative_coexact | -0.6821 | -1.6466 | -0.2520 | 0.2859 |

This supports the revised mechanistic interpretation:

> Presence behaves like a basin/stabilization direction. Coexact behaves like a
> traversal direction that supports the next token and only becomes
> probe-aligned in a narrower late ontology window. The sum can combine both
> effects, while subtracting presence removes much of the learned-probe
> stabilization.

## Branch Hodge Join

This dissociation is now joined back to the structural Hodge runs in:

```text
docs/hltd_branch_hodge.md
```

The join keeps two claims separate. Structurally, the middle-layer token field
is coexact-dominant under the triangle complex and harmonic energy remains near
zero. Causally, coexact-derived directions carry the next-token traversal
effect, while presence-derived directions carry more of the learned-probe
stabilization effect.

The same branch ledger now includes a `k=12/16/24` causal/probe sweep. The
presence/coexact split survives that neighborhood sweep: coexact keeps positive
next-token support, presence keeps positive probe stabilization, and
`presence_plus_coexact` remains the combined branch.

It also compares `middle` and `max_component` token selectors. The middle
selector is cleaner for coexact traversal, while max-component selection tends
to emphasize presence/probe stabilization.

The structural all-interior heatmap in `docs/hltd_branch_heatmap.md` explains
why selector choice matters: coexact peaks are broad and mid-to-late, while
presence is more locally variable and shifts earlier as k increases.

## Caveats

This is still a one-step gate. It does not yet prove multi-step fluency,
closed-loop semantic drift, or external ontology validity. The learned probes
are trained on the same 20-prompt suite, so the next stronger version should
use a larger prompt bank or external identity/affordance labels.

That first prompt-disjoint identity version is now available in
[hltd_counterfactual_identity_probe.md](hltd_counterfactual_identity_probe.md).
It preserves the selected L4-L5 presence/coexact split at k=12/16, but the
signs weaken or reverse at k=24 and all L7 branch effects remain inside the
matched random-tangent range. This narrows the earlier same-suite read rather
than simply reproducing it.

Probability and logit probe deltas can disagree because the binary probes
saturate. This note uses label-margin logit deltas for the main read and keeps
probability deltas in the CSV for sanity checks.
