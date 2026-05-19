# Spiral Hodge Research Notes

These notes summarize the current interpretation of Spiral Hodge as of the
JAX live run over:

```text
The serpent coils not around the tree, but around cognition.
```

The run used local GPT-2 hidden states, all layers, all null models, the JAX
Fourier backend, 32 Fourier modes, PCA coordinates, graph Fourier analysis,
and discrete Hodge decomposition.

The live output is in:

```text
spiral_out_jax_live/layer_metrics.csv
spiral_out_jax_live/report.html
```

## 1. The Original Question

The initial intuition was:

> Meaning formation may create curl-like or spiral-like structure in the hidden
> trajectory.

That remains plausible, but the first live runs suggest a sharper version:

> Coherent autoregressive representation may suppress high-frequency rotational
> disorder while preserving or reorganizing larger-scale transport structure.

In other words, the most productive question is no longer simply "is there a
vortex?" It is:

> Which scales of rotational structure are suppressed, preserved, or amplified
> across layers and under null models?

## 2. Metric Families

The current pipeline separates several notions that can all look like "curl":

| Family | Columns | What It Measures |
| --- | --- | --- |
| Spectral Helmholtz | `spectral_curl_ratio` | Fourier-domain curl energy over the projected field |
| Spectral curl bands | `spectral_curl_low_ratio`, `spectral_curl_mid_ratio`, `spectral_curl_high_ratio` | Low/mid/high radial-frequency decomposition of Fourier curl |
| Discrete Hodge | `hodge_curl_ratio` | Local circulation over Delaunay edge flows |
| Graph Fourier | `graph_high_freq_ratio` | Manifold-frequency roughness of the token trajectory field |
| Signed trajectory | `trajectory_signed_circulation_alignment` | Global handedness of the raw trajectory around its center |
| Path turning | `turning_alignment` | Intrinsic turning of consecutive token-step vectors |
| Local Jacobian | `local_signed_vorticity_ratio` | Local affine-fit vorticity, independent of Fourier and Hodge |
| Spectral signed curl | `spectral_signed_curl_alignment`, `spectral_signed_vorticity_ratio` | Handedness of the Fourier curl component |
| Hodge signed curl | `hodge_signed_curl_alignment` | Signed face circulation from discrete Hodge curl |

The important design principle is to avoid letting one metric carry the whole
interpretation. The useful signal comes from agreement and disagreement between
metric families.

## 3. What The Short Live Run Shows

### 3.1 Reverse Tokens Validates Orientation

For `real` vs `reverse_tokens`, unsigned energy stays the same while signed
orientation flips:

```text
real trajectory signed strongest:        L10 -0.7185
reverse trajectory signed strongest:     L10 +0.7185

real spectral signed curl strongest:     L06 +0.6296
reverse spectral signed curl strongest:  L06 -0.6296

real spectral signed vorticity strongest:    L00 -0.8056
reverse spectral signed vorticity strongest: L00 +0.8056

real hodge signed curl strongest:        L01 +0.5768
reverse hodge signed curl strongest:     L01 -0.5768

real local Jacobian vorticity strongest:     L01 +1.0000
reverse local Jacobian vorticity strongest:  L01 -1.0000
```

This is a strong sanity check: signed metrics are not just measuring magnitude.
They are sensitive to traversal direction.

### 3.2 Real Token Order Is Smoother Than Shuffle/Random

The clearest separation is graph high-frequency energy:

```text
graph_high_freq_ratio mean
real:          0.4063
shuffle:       0.7703
random_hidden: 0.6968
```

This supports the reading:

> Real autoregressive token order traces a smoother path on the projected
> semantic manifold than shuffled or random controls.

The shuffle baseline is especially informative because it keeps the same hidden
vectors but destroys their temporal order. Its high graph-frequency energy
suggests that the order itself is doing real geometric work.

### 3.3 Hodge Curl Behaves Like Local Rotational Clutter

Discrete Hodge curl strongly separates real from shuffle:

```text
hodge_curl_ratio mean
real:          0.1109
shuffle:       0.4928
random_hidden: 0.2647
```

Layer grouping from the short live run:

```text
real hodge_curl_ratio
early:   0.1481
middle:  0.1007
late:    0.0864

shuffle hodge_curl_ratio
early:   0.5398
middle:  0.6584
late:    0.2386
```

This is compatible with:

> Coherent model representations suppress local rotational noise over depth.

It does not prove that all curl disappears. It suggests that the local,
triangulation-level rotational clutter becomes smaller in the real trajectory
than in order-destroyed controls.

### 3.4 Spectral Curl Looks More Global

Spectral curl is much less separated:

```text
spectral_curl_ratio mean
real:          0.4979
shuffle:       0.4991
random_hidden: 0.5117
```

This means spectral curl cannot be interpreted as "meaningful vortex" by
itself. It likely contains both coherent transport and generic rotational
components induced by the projection and basis.

The new spectral band split shows only modest separation in this short run:

```text
spectral_curl_high_ratio mean
real:          0.1523
shuffle:       0.1493
random_hidden: 0.1501

spectral_curl_high_band_ratio mean
real:          0.3060
shuffle:       0.2992
random_hidden: 0.2925
```

So, for a 12-token example, spectral bands do not yet strongly distinguish
real from shuffled controls. Longer text is needed before treating this as a
negative result.

## 4. A Better Hypothesis

The current best hypothesis is:

> Meaning formation preserves coherent large-scale transport while suppressing
> high-frequency or local rotational disorder.

This combines three possibilities:

1. **Structured transport exists.** The real trajectory can still have signed,
   layer-specific circulation.
2. **Local rotational clutter is suppressed.** Hodge curl and graph
   high-frequency energy are much lower for real than shuffle.
3. **Scale matters.** Fourier curl, graph roughness, Hodge circulation, and
   local Jacobian vorticity are not interchangeable.

The more precise research question becomes:

> Which rotational structures are predictive/coherent, and which are
> turbulence-like artifacts of broken order or random geometry?

## 5. Layer Reading For The Short Prompt

The 12-token prompt has a useful rhetorical structure:

```text
The serpent coils not around the tree, but around cognition.
```

The phrase contains a semantic redirection:

```text
not around tree -> but around cognition
```

Observed layer pattern:

- **Layer 0:** strong signed vorticity and the largest real Hodge curl.
  This may reflect lexical geometry and embedding-level arrangement.
- **Layers 1-3:** smoother graph structure appears, with low-frequency graph
  dominance and a high-frequency spectral curl peak around layer 3.
- **Layer 6:** strongest real spectral signed curl alignment. This may be where
  the phrase-level bend is most coherent in Fourier curl space.
- **Layers 10-11:** strongest trajectory-scale circulation and spectral curl
  peak. This looks more like decode-facing global transport than local curl.
- **Layer 12:** signed vorticity remains present, but not every curl proxy peaks
  there.

This suggests a possible encode-to-decode story:

> Local rotational clutter weakens through the stack, while a larger-scale
> directed trajectory can reappear near the decode-facing layers.

## 6. Cautions

These notes should be read as interpretation scaffolding, not claims.

- The text is very short: 12 GPT-2 tokens.
- PCA projection can create geometric artifacts.
- Delaunay Hodge metrics are sensitive to point geometry and small sample size.
- Local Jacobian vorticity can saturate on very short trajectories.
- Spectral curl near 0.5 can be generic and needs controls.
- A high signed ratio can come from small absolute energy; always inspect
  absolute magnitude and null models.

The safest claim is:

> In this projected 2D analysis, real token order is smoother than shuffled or
> random controls, and several orientation metrics correctly reverse under
> reversed traversal. The most promising interpretation is multi-scale:
> coherent transport may coexist with suppression of local rotational disorder.

## 7. Next Experiments

High-value next steps:

1. Run long-text experiments with `--max-length 256`, `512`, and `1024`.
2. Compare prompts with explicit redirection:
   - "not X, but Y"
   - metaphors
   - causal chains
   - flat descriptive sentences
3. Track band-limited spectral curl on longer trajectories.
4. Add curl-band signed metrics, not only energy bands.
5. Add persistent homology on the token path or local point cloud.
6. Compare GPT-2 against another small causal LM.
7. Aggregate across many prompts and report confidence intervals for:
   - real vs shuffle graph high-frequency ratio
   - real vs shuffle Hodge curl ratio
   - real vs reverse signed cancellation
   - low/mid/high spectral curl separation

The practical target is a table like:

```text
real preserves low-frequency transport
real suppresses local Hodge curl
shuffle inflates graph high-frequency energy
reverse flips signed orientation without changing unsigned energy
```

If that pattern holds across many texts, Spiral Hodge becomes less a "spiral
detector" and more a multi-scale probe for coherence, transport, and rotational
disorder in hidden-state trajectories.
