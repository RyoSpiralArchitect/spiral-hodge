# Izumi / GPT-2 Signed-Curl Example

This folder contains a small reference run generated with local GPT-2 weights
over the first 1024 tokens of a local English text file.

The input text itself is not included. The committed artifacts are only the
derived CSV metrics and a small set of plots that demonstrate the repository's
main outputs.

## Run Shape

- model: GPT-2 loaded from a local Hugging Face directory
- tokens: 1024
- layers: 13 hidden-state slices, including embedding output
- reducer: PCA
- Fourier backend: direct nonuniform DFT
- Fourier modes: 32
- null models: real, shuffled token order, reversed token order, matched random hidden states

## Notable Observation

The real trajectory shows a final-layer spectral curl-energy spike and a
strong signed vorticity orientation:

```text
real layer 12
trajectory_signed_circulation_alignment = -0.2652
spectral_signed_curl_alignment          = -0.3743
spectral_signed_vorticity_ratio         = -0.6479
hodge_signed_curl_alignment             = +0.0484
```

The reversed-token control flips the signed orientation while preserving the
unsigned curl-energy ratios. This is the core reason the signed metrics are
included: they distinguish handedness that ordinary energy ratios erase.
