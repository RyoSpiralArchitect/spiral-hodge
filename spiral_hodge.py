"""
spiral_hodge.py

Prototype pipeline for studying transformer hidden-state trajectories:
  hidden states [layers, tokens, dim]
  -> PCA / UMAP semantic coordinates
  -> token trajectory vector increments
  -> nonuniform Fourier spectrum, graph Fourier spectrum
  -> Helmholtz / Hodge decomposition
  -> signed circulation / signed curl orientation metrics

Design notes:
  - The original used FINUFFT by default. FINUFFT is fast, but it is a native
    extension; if the local binary is mismatched it can produce a zsh
    "segmentation fault" instead of a Python exception.
  - The original auto-selected Apple's MPS device. PyTorch+MPS can be fragile
    depending on local Torch/macOS/Python builds. This version defaults to CPU.
  - --text can now be omitted or accidentally left blank; a demo text is used.

Install core deps:
  pip install numpy scipy scikit-learn matplotlib torch transformers
Optional:
  pip install finufft umap-learn jax

Examples:
  python3 spiral_hodge.py --synthetic --save-plots
  python3 spiral_hodge.py --synthetic --all-layers --null-models all --save-plots
  python3 spiral_hodge.py --model gpt2 --text "The serpent coils around cognition." --all-layers --save-plots
  python3 spiral_hodge.py --model-path ./model/gpt2 --local-files-only --text "..." --all-layers
  python3 spiral_hodge.py --model gpt2 --text "..." --fourier-backend direct
  python3 spiral_hodge.py --model gpt2 --text "..." --fourier-backend finufft
  python3 spiral_hodge.py --model gpt2 --text "..." --fourier-backend jax
"""
from __future__ import annotations

# Put conservative thread settings before importing numeric libraries.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import argparse
import csv
from pathlib import Path
import sys
import warnings

import numpy as np

Array = np.ndarray
DEFAULT_TEXT = "The serpent coils not around the tree, but around cognition."
SCRIPT_DIR = Path(__file__).resolve().parent
HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
FourierBackend = Literal["direct", "finufft", "jax"]

_JAX_VECTOR_SPECTRUM_IMPL: Optional[Callable[..., Tuple[Any, Any, Any]]] = None
_JAX_EVAL_VECTOR_IMPL: Optional[Callable[..., Any]] = None
_JAX_EVAL_VORTICITY_IMPL: Optional[Callable[..., Any]] = None


@dataclass
class HiddenStateBundle:
    """Hidden states and token metadata.

    hidden shape: [layers, tokens, dim]
    For many HF models, layer 0 is the embedding output.
    """

    hidden: Array
    tokens: List[str]
    input_ids: Optional[Array] = None
    model_name: Optional[str] = None


@dataclass
class CoordinateBundle:
    """PCA/UMAP semantic coordinates for every [layer, token]."""

    coords: Array  # [layers, tokens, coord_dim]
    reducer: Any
    explained_variance_ratio: Optional[Array]
    flattened_input_shape: Tuple[int, int, int]


class HFModelLoadError(RuntimeError):
    """Raised when Transformers cannot load the requested tokenizer/model."""


@dataclass
class VectorFieldBundle:
    """Token trajectory vector field in semantic coordinate space."""

    points: Array  # [token_steps, coord_dim], midpoint coordinates
    vectors: Array  # [token_steps, coord_dim], coordinate displacement vectors
    token_edges: List[Tuple[int, int]]
    layer: int


@dataclass
class FourierSpectrum:
    """Nonuniform vector spectrum and frequency grids."""

    coeffs: Array  # [2, modes, modes], complex vector coeffs for x/y components
    power: Array  # [modes, modes]
    kx: Array  # [modes]
    ky: Array  # [modes]
    scaled_points: Array  # [n_samples, 2]
    backend: str


@dataclass
class HelmholtzSpectrum:
    """Fourier-domain 2D Helmholtz projection result."""

    grad_coeffs: Array  # [2, modes, modes]
    curl_coeffs: Array  # [2, modes, modes]
    harmonic_coeffs: Array  # [2, modes, modes], only k=(0,0)
    energy: Dict[str, float]


@dataclass
class GraphFourierSpectrum:
    """Graph Fourier coefficients over vector-field sample points."""

    eigenvalues: Array
    coeffs: Array  # [2, n_eigs]
    power: Array  # [n_eigs]
    adjacency: Any
    laplacian: Any


@dataclass
class HodgeDecomposition:
    """Discrete Hodge decomposition for scalar edge flows."""

    edges: Array  # [n_edges, 2], oriented tail->head
    faces: Array  # [n_faces, 3], oriented CCW when possible
    flow: Array  # [n_edges]
    grad: Array  # [n_edges]
    curl: Array  # [n_edges]
    harmonic: Array  # [n_edges]
    energy: Dict[str, float]
    D0: Any  # sparse [n_edges, n_vertices]
    D1: Any  # sparse [n_faces, n_edges]


def log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


# -----------------------------------------------------------------------------
# Diagnostics / synthetic data
# -----------------------------------------------------------------------------


def print_diagnostics() -> None:
    """Print versions without doing a model forward pass."""

    print(f"python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}")
    try:
        import scipy
        print(f"scipy: {scipy.__version__}")
    except Exception as e:
        print(f"scipy: import failed: {e}")
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except Exception as e:
        print(f"scikit-learn: import failed: {e}")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        print(f"torch mps available: {mps_ok}")
    except Exception as e:
        print(f"torch: import failed: {e}")
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except Exception as e:
        print(f"transformers: import failed: {e}")
    try:
        import finufft  # type: ignore
        version = getattr(finufft, "__version__", "unknown")
        print(f"finufft: {version}")
    except Exception as e:
        print(f"finufft: import failed or not installed: {e}")
    try:
        import jax  # type: ignore
        print(f"jax: {jax.__version__}")
        print("jax devices:", ", ".join(str(d) for d in jax.devices()))
    except Exception as e:
        print(f"jax: import failed or not installed: {e}")


def synthetic_hidden_states(
    *,
    layers: int = 13,
    tokens: int = 48,
    dim: int = 128,
    seed: int = 0,
) -> HiddenStateBundle:
    """Generate smooth synthetic hidden states for pipeline testing."""

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, tokens)
    hidden = rng.normal(scale=0.05, size=(layers, tokens, dim))
    # Create a rotating low-dimensional signal embedded in hidden dim.
    basis = rng.normal(size=(4, dim))
    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    for ell in range(layers):
        phase = 0.18 * ell
        radius = 1.0 + 0.04 * ell
        components = np.stack(
            [
                radius * np.cos(t + phase),
                radius * np.sin(t + phase),
                0.35 * np.cos(2 * t - phase),
                0.25 * np.sin(3 * t + 0.5 * phase),
            ],
            axis=1,
        )
        hidden[ell] += components @ basis
    toks = [f"tok{i}" for i in range(tokens)]
    return HiddenStateBundle(hidden=hidden.astype(np.float32), tokens=toks, model_name="synthetic")


# -----------------------------------------------------------------------------
# Hidden states
# -----------------------------------------------------------------------------


def choose_device(device: Literal["auto", "cpu", "cuda", "mps"]) -> str:
    """Resolve device. In safe mode, auto does not select MPS."""

    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # Do not select MPS automatically; it is a frequent source of hard crashes.
    return "cpu"


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def hf_offline_enabled() -> bool:
    """Return True when Hugging Face/Transformers offline mode is enabled."""

    return any(_is_truthy_env(name) for name in HF_OFFLINE_ENV_VARS)


def _expand_model_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _looks_like_local_path(value: str) -> bool:
    expanded = os.path.expandvars(os.path.expanduser(value))
    parent = os.path.dirname(expanded)
    return (
        value.startswith(("~", ".", os.sep))
        or os.path.isabs(expanded)
        or bool(parent and os.path.isdir(parent))
    )


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path.resolve())
    return out


def _local_model_candidates(model_name: str) -> List[Path]:
    """Find common local model directories for bare names like ``gpt2``."""

    raw = model_name.strip()
    if not raw:
        return []

    names = [raw]
    basename = raw.rstrip("/").split("/")[-1]
    if basename and basename not in names:
        names.append(basename)

    roots = _dedupe_paths([Path.cwd(), SCRIPT_DIR])
    model_roots = ("model", "models", "Models")
    candidates: List[Path] = []
    for root in roots:
        for model_root in model_roots:
            for name in names:
                candidate = root / model_root / name
                if candidate.is_dir():
                    candidates.append(candidate)
    return _dedupe_paths(candidates)


def resolve_hf_model_ref(model_name: str, model_path: Optional[str] = None) -> Tuple[str, bool]:
    """Resolve the HF load reference and whether it is a local directory."""

    raw_ref = model_path or model_name
    if not raw_ref or not str(raw_ref).strip():
        raise ValueError("Model name/path must not be empty.")

    raw_ref = str(raw_ref).strip()
    expanded = _expand_model_path(raw_ref)
    explicit_local = model_path is not None

    if os.path.isdir(expanded):
        return str(Path(expanded).resolve()), True
    if os.path.exists(expanded):
        raise FileNotFoundError(f"Model path must be a directory, got file: {raw_ref}")

    if explicit_local or _looks_like_local_path(raw_ref):
        msg = f"Local model directory not found: {raw_ref}"
        if hf_offline_enabled():
            envs = ", ".join(name for name in HF_OFFLINE_ENV_VARS if _is_truthy_env(name))
            msg += f"\nHF offline mode is enabled ({envs}); downloads/lookups are disabled."
        raise FileNotFoundError(msg)

    candidates = _local_model_candidates(raw_ref)
    if candidates:
        return str(candidates[0]), True

    return raw_ref, False


def _hf_load_error_message(model_ref: str, *, local_files_only: bool, phase: str) -> str:
    offline_hint = ""
    if local_files_only or hf_offline_enabled():
        offline_hint = (
            "\nHF offline/local-only mode is active. If this model is not already "
            "in the HF cache, pass a local Hugging Face model directory with "
            "--model-path /path/to/model or --model /path/to/model."
        )
    return (
        f"Failed to load {phase} from '{model_ref}'."
        f"{offline_hint}\nThe local directory should contain config.json, tokenizer "
        "files, and model weights."
    )


def resolve_text_file_path(text_file: str) -> Path:
    """Resolve text input paths relative to cwd, this script, or repo-name prefix."""

    raw = str(text_file).strip()
    if not raw:
        raise ValueError("--text-file path must not be empty.")

    expanded = os.path.expandvars(os.path.expanduser(raw))
    base = Path(expanded)
    candidates = [base]
    if not base.is_absolute():
        candidates.extend([Path.cwd() / base, SCRIPT_DIR / base])
        parts = base.parts
        if parts and parts[0] == SCRIPT_DIR.name:
            candidates.append(SCRIPT_DIR.joinpath(*parts[1:]))

    for candidate in _dedupe_paths(candidates):
        if candidate.is_file():
            return candidate
        if candidate.exists() and candidate.is_dir():
            raise IsADirectoryError(f"--text-file must point to a file, got directory: {text_file}")

    hint = ""
    if not base.is_absolute() and base.parts and base.parts[0] == SCRIPT_DIR.name:
        hint = f"\nYou are already in {SCRIPT_DIR}; try --text-file ./{Path(*base.parts[1:])}"
    raise FileNotFoundError(f"Text file not found: {text_file}{hint}")


def extract_hidden_states(
    model_name: str,
    text: str,
    *,
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu",
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
    trust_remote_code: bool = False,
    max_length: Optional[int] = None,
    local_files_only: bool = False,
    verbose: bool = True,
) -> HiddenStateBundle:
    """Load a HF causal LM and return hidden states shaped [layers, tokens, dim]."""

    if not str(text).strip():
        text = DEFAULT_TEXT

    log("[stage] importing torch/transformers", verbose=verbose)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = choose_device(device)
    log(f"[stage] using device={resolved_device}", verbose=verbose)

    model_ref, model_is_local = resolve_hf_model_ref(model_name)
    resolved_local_files_only = bool(local_files_only or model_is_local or hf_offline_enabled())
    if model_is_local and model_ref != model_name:
        log(f"[stage] resolved local model path: {model_ref}", verbose=verbose)

    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32

    log(f"[stage] loading tokenizer: {model_ref}", verbose=verbose)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            local_files_only=resolved_local_files_only,
        )
    except OSError as e:
        raise HFModelLoadError(
            _hf_load_error_message(model_ref, local_files_only=resolved_local_files_only, phase="tokenizer")
        ) from e

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": resolved_local_files_only,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    log(f"[stage] loading model: {model_ref}", verbose=verbose)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    except OSError as e:
        raise HFModelLoadError(
            _hf_load_error_message(model_ref, local_files_only=resolved_local_files_only, phase="model")
        ) from e
    model = model.to(resolved_device)
    model.eval()

    tok_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        tok_kwargs.update({"truncation": True, "max_length": max_length})
    inputs = tokenizer(text, **tok_kwargs).to(resolved_device)

    log(f"[stage] forward pass, tokens={int(inputs['input_ids'].shape[1])}", verbose=verbose)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    if not getattr(outputs, "hidden_states", None):
        raise RuntimeError("Model output did not contain hidden_states. Check model compatibility.")

    hs = torch.stack(outputs.hidden_states, dim=0)[:, 0].detach().float().cpu().numpy()
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    log(f"[stage] hidden extracted: {hs.shape}", verbose=verbose)
    return HiddenStateBundle(hidden=hs, tokens=tokens, input_ids=input_ids, model_name=model_ref)


# -----------------------------------------------------------------------------
# Semantic coordinates
# -----------------------------------------------------------------------------


def _safe_l2_normalize(x: Array, eps: float = 1e-12) -> Array:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def make_semantic_coordinates(
    hidden: Array,
    *,
    method: Literal["pca", "umap"] = "pca",
    n_components: int = 2,
    normalize_hidden: bool = True,
    random_state: int = 0,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.05,
    verbose: bool = True,
) -> CoordinateBundle:
    """Reduce [layers, tokens, dim] hidden states to semantic coordinates."""

    hidden = np.asarray(hidden)
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [layers, tokens, dim], got {hidden.shape}")

    L, T, D = hidden.shape
    if L * T < n_components:
        raise ValueError(f"Need at least {n_components} samples for reduction; got {L*T}.")

    X = hidden.reshape(L * T, D).astype(np.float64, copy=False)
    if normalize_hidden:
        X = _safe_l2_normalize(X)

    log(f"[stage] reducing hidden states with {method}", verbose=verbose)
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=random_state)
        Y = reducer.fit_transform(X)
        evr = getattr(reducer, "explained_variance_ratio_", None)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError("UMAP requested; install with `pip install umap-learn`.") from e
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(umap_neighbors, max(2, L * T - 1)),
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        Y = reducer.fit_transform(X)
        evr = None
    else:
        raise ValueError(f"unknown method: {method}")

    coords = np.asarray(Y, dtype=np.float64).reshape(L, T, n_components)
    return CoordinateBundle(coords=coords, reducer=reducer, explained_variance_ratio=evr, flattened_input_shape=(L, T, D))


# -----------------------------------------------------------------------------
# Token trajectory vector field
# -----------------------------------------------------------------------------


def token_trajectory_field(
    coords: Array,
    *,
    layer: int = -1,
    step: int = 1,
) -> VectorFieldBundle:
    """Create midpoint samples and vector increments along the token path."""

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3:
        raise ValueError(f"coords must have shape [layers, tokens, coord_dim], got {coords.shape}")
    L, T, C = coords.shape
    if C < 2:
        raise ValueError("Need at least 2 coordinate dimensions for 2D spectral/Hodge analysis.")
    if step < 1:
        raise ValueError("step must be >= 1")
    if T <= step:
        raise ValueError(f"Need more than step={step} tokens; got T={T}.")
    layer_idx = layer % L
    P = coords[layer_idx, :, :2]
    points = 0.5 * (P[:-step] + P[step:])
    vectors = P[step:] - P[:-step]
    token_edges = [(i, i + step) for i in range(T - step)]
    return VectorFieldBundle(points=points, vectors=vectors, token_edges=token_edges, layer=layer_idx)


# -----------------------------------------------------------------------------
# Nonuniform Fourier + Fourier-domain Helmholtz projection
# -----------------------------------------------------------------------------


def scale_points_to_periodic_square(points: Array, margin: float = 0.95) -> Array:
    """Affine-scale 2D points to roughly [-pi*margin, pi*margin]^2."""

    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"points must be [n, >=2], got {P.shape}")
    P2 = P[:, :2]
    center = P2.mean(axis=0, keepdims=True)
    centered = P2 - center
    scale = np.max(np.abs(centered), axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    X = centered / scale * (np.pi * margin)
    return np.clip(X, -np.pi * margin, np.pi * margin)


def direct_nonuniform_vector_spectrum(
    points: Array,
    vectors: Array,
    *,
    modes: int = 32,
    isign: int = -1,
    chunk_modes: int = 4096,
) -> FourierSpectrum:
    """Direct nonuniform DFT of vector samples.

    This computes the same kind of nonuniform Fourier coefficients as a type-1
    NUFFT, but without FINUFFT. It is slower, yet very safe for short token
    sequences and avoids native FINUFFT segfaults.
    """

    P = scale_points_to_periodic_square(points)
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(P) != len(V):
        raise ValueError(f"points and vectors length mismatch: {len(P)} vs {len(V)}")
    if len(P) == 0:
        raise ValueError("Need at least one vector sample.")
    if modes < 4:
        raise ValueError("modes should be at least 4")

    k = np.fft.fftfreq(modes, d=1.0 / modes).astype(np.float64)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    flat_kx = KX.reshape(-1)
    flat_ky = KY.reshape(-1)
    M = flat_kx.size
    coeffs_flat = np.empty((2, M), dtype=np.complex128)
    x = P[:, 0]
    y = P[:, 1]
    vx = V[:, 0]
    vy = V[:, 1]

    for start in range(0, M, chunk_modes):
        end = min(start + chunk_modes, M)
        phase_arg = x[:, None] * flat_kx[None, start:end] + y[:, None] * flat_ky[None, start:end]
        phase = np.exp(1j * float(isign) * phase_arg)
        coeffs_flat[0, start:end] = vx @ phase
        coeffs_flat[1, start:end] = vy @ phase

    coeffs = coeffs_flat.reshape(2, modes, modes) / float(len(P))
    power = np.abs(coeffs[0]) ** 2 + np.abs(coeffs[1]) ** 2
    return FourierSpectrum(coeffs=coeffs, power=power, kx=k, ky=k.copy(), scaled_points=P, backend="direct")


def finufft_vector_spectrum(
    points: Array,
    vectors: Array,
    *,
    modes: int = 32,
    eps: float = 1e-6,
    isign: int = -1,
) -> FourierSpectrum:
    """Type-1 2D FINUFFT vector spectrum. Fast, but uses native code."""

    try:
        import finufft  # type: ignore
    except ImportError as e:
        raise ImportError("FINUFFT requested; install with `pip install finufft`.") from e

    P = scale_points_to_periodic_square(points)
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(P) != len(V):
        raise ValueError(f"points and vectors length mismatch: {len(P)} vs {len(V)}")
    if modes < 4:
        raise ValueError("modes should be at least 4")

    x = np.ascontiguousarray(P[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(P[:, 1], dtype=np.float64)
    c = np.ascontiguousarray(np.vstack([V[:, 0], V[:, 1]]).astype(np.complex128))
    coeffs = finufft.nufft2d1(x, y, c, (modes, modes), eps=eps, isign=isign, modeord=1)
    coeffs = np.asarray(coeffs, dtype=np.complex128) / max(len(P), 1)
    power = np.abs(coeffs[0]) ** 2 + np.abs(coeffs[1]) ** 2
    k = np.fft.fftfreq(modes, d=1.0 / modes).astype(np.float64)
    return FourierSpectrum(coeffs=coeffs, power=power, kx=k, ky=k.copy(), scaled_points=P, backend="finufft")


def _import_jax():
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except ImportError as e:
        raise ImportError(
            "JAX Fourier backend requested; install with `pip install jax` "
            "or use `--fourier-backend direct`."
        ) from e
    return jax, jnp


def _jax_float_dtype(jax: Any, jnp: Any) -> Any:
    """Use float64 only when JAX has x64 enabled on the active platform."""

    try:
        return jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    except Exception:
        return jnp.float32


def _get_jax_vector_spectrum_impl() -> Callable[..., Tuple[Any, Any, Any]]:
    global _JAX_VECTOR_SPECTRUM_IMPL
    if _JAX_VECTOR_SPECTRUM_IMPL is not None:
        return _JAX_VECTOR_SPECTRUM_IMPL

    jax, jnp = _import_jax()

    @jax.jit
    def _impl(points: Any, vectors: Any, k: Any, isign: Any) -> Tuple[Any, Any, Any]:
        kx, ky = jnp.meshgrid(k, k, indexing="ij")
        flat_kx = kx.reshape(-1)
        flat_ky = ky.reshape(-1)
        phase_arg = points[:, 0:1] * flat_kx[None, :] + points[:, 1:2] * flat_ky[None, :]
        phase = isign * phase_arg
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        coeff_x_re = vectors[:, 0] @ cos_phase
        coeff_x_im = vectors[:, 0] @ sin_phase
        coeff_y_re = vectors[:, 1] @ cos_phase
        coeff_y_im = vectors[:, 1] @ sin_phase
        denom = points.shape[0]
        coeffs_re = jnp.stack([coeff_x_re, coeff_y_re]).reshape(2, k.shape[0], k.shape[0]) / denom
        coeffs_im = jnp.stack([coeff_x_im, coeff_y_im]).reshape(2, k.shape[0], k.shape[0]) / denom
        power = coeffs_re[0] ** 2 + coeffs_im[0] ** 2 + coeffs_re[1] ** 2 + coeffs_im[1] ** 2
        return coeffs_re, coeffs_im, power

    _JAX_VECTOR_SPECTRUM_IMPL = _impl
    return _JAX_VECTOR_SPECTRUM_IMPL


def jax_nonuniform_vector_spectrum(
    points: Array,
    vectors: Array,
    *,
    modes: int = 32,
    isign: int = -1,
) -> FourierSpectrum:
    """JAX/XLA direct nonuniform DFT of vector samples."""

    jax, jnp = _import_jax()
    P = scale_points_to_periodic_square(points)
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(P) != len(V):
        raise ValueError(f"points and vectors length mismatch: {len(P)} vs {len(V)}")
    if len(P) == 0:
        raise ValueError("Need at least one vector sample.")
    if modes < 4:
        raise ValueError("modes should be at least 4")

    dtype = _jax_float_dtype(jax, jnp)
    k = np.fft.fftfreq(modes, d=1.0 / modes).astype(np.float64)
    impl = _get_jax_vector_spectrum_impl()
    coeffs_re_jax, coeffs_im_jax, power_jax = impl(
        jnp.asarray(P, dtype=dtype),
        jnp.asarray(V, dtype=dtype),
        jnp.asarray(k, dtype=dtype),
        jnp.asarray(float(isign), dtype=dtype),
    )
    coeffs = np.asarray(jax.device_get(coeffs_re_jax), dtype=np.float64) + 1j * np.asarray(
        jax.device_get(coeffs_im_jax),
        dtype=np.float64,
    )
    power = np.asarray(jax.device_get(power_jax), dtype=np.float64)
    return FourierSpectrum(coeffs=coeffs, power=power, kx=k, ky=k.copy(), scaled_points=P, backend="jax")


def vector_spectrum(
    points: Array,
    vectors: Array,
    *,
    modes: int = 32,
    backend: FourierBackend = "direct",
) -> FourierSpectrum:
    if backend == "direct":
        return direct_nonuniform_vector_spectrum(points, vectors, modes=modes)
    if backend == "finufft":
        return finufft_vector_spectrum(points, vectors, modes=modes)
    if backend == "jax":
        return jax_nonuniform_vector_spectrum(points, vectors, modes=modes)
    raise ValueError(f"unknown Fourier backend: {backend}")


def helmholtz_project_spectrum(spec: FourierSpectrum) -> HelmholtzSpectrum:
    """2D periodic Helmholtz projection in Fourier domain."""

    coeffs = np.asarray(spec.coeffs)
    if coeffs.shape[0] != 2:
        raise ValueError(f"spec.coeffs must have shape [2, modes, modes], got {coeffs.shape}")
    Fu, Fv = coeffs[0], coeffs[1]
    KX, KY = np.meshgrid(spec.kx, spec.ky, indexing="ij")
    K2 = KX**2 + KY**2
    nonzero = K2 > 0

    grad_u = np.zeros_like(Fu)
    grad_v = np.zeros_like(Fv)
    dot = KX * Fu + KY * Fv
    grad_u[nonzero] = KX[nonzero] * dot[nonzero] / K2[nonzero]
    grad_v[nonzero] = KY[nonzero] * dot[nonzero] / K2[nonzero]

    harm_u = np.zeros_like(Fu)
    harm_v = np.zeros_like(Fv)
    harm_u[0, 0] = Fu[0, 0]
    harm_v[0, 0] = Fv[0, 0]

    curl_u = Fu - grad_u - harm_u
    curl_v = Fv - grad_v - harm_v

    total = float(np.sum(np.abs(Fu) ** 2 + np.abs(Fv) ** 2))
    grad_e = float(np.sum(np.abs(grad_u) ** 2 + np.abs(grad_v) ** 2))
    curl_e = float(np.sum(np.abs(curl_u) ** 2 + np.abs(curl_v) ** 2))
    harm_e = float(np.sum(np.abs(harm_u) ** 2 + np.abs(harm_v) ** 2))
    denom = max(total, 1e-30)
    energy = {
        "total": total,
        "grad": grad_e,
        "curl": curl_e,
        "harmonic": harm_e,
        "grad_ratio": grad_e / denom,
        "curl_ratio": curl_e / denom,
        "harmonic_ratio": harm_e / denom,
    }
    return HelmholtzSpectrum(
        grad_coeffs=np.stack([grad_u, grad_v]),
        curl_coeffs=np.stack([curl_u, curl_v]),
        harmonic_coeffs=np.stack([harm_u, harm_v]),
        energy=energy,
    )


# -----------------------------------------------------------------------------
# Graph Fourier transform
# -----------------------------------------------------------------------------


def _knn_adjacency(points: Array, k: int = 8, sigma: Optional[float] = None):
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    P = np.asarray(points, dtype=np.float64)[:, :2]
    n = len(P)
    if n < 2:
        raise ValueError("Need at least two points for graph Fourier transform.")
    k_eff = min(k + 1, n)
    tree = cKDTree(P)
    dists, inds = tree.query(P, k=k_eff)
    if dists.ndim == 1:
        dists = dists[:, None]
        inds = inds[:, None]
    nbr_dists = dists[:, 1:]
    nbr_inds = inds[:, 1:]
    if nbr_inds.size == 0:
        raise ValueError("kNN graph has no neighbor edges.")
    if sigma is None:
        positive = nbr_dists[nbr_dists > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(float(sigma), 1e-12)

    rows = np.repeat(np.arange(n), nbr_inds.shape[1])
    cols = nbr_inds.reshape(-1)
    dd = nbr_dists.reshape(-1)
    weights = np.exp(-(dd**2) / (2.0 * sigma**2))
    A = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def graph_fourier_spectrum(
    points: Array,
    vectors: Array,
    *,
    k_neighbors: int = 8,
    n_eigs: int = 32,
    normalized_laplacian: bool = True,
) -> GraphFourierSpectrum:
    """Graph Fourier transform of vector components over sample points."""

    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh

    P = np.asarray(points, dtype=np.float64)[:, :2]
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    n = len(P)
    if len(V) != n:
        raise ValueError(f"points and vectors length mismatch: {n} vs {len(V)}")
    if n < 3:
        raise ValueError("Need at least 3 samples for a useful graph spectrum.")

    A = _knn_adjacency(P, k=k_neighbors)
    L = csgraph.laplacian(A, normed=normalized_laplacian)

    n_eigs_eff = max(1, min(n_eigs, n - 1))
    if n <= 96:
        vals, vecs = np.linalg.eigh(L.toarray())
        vals = vals[:n_eigs_eff]
        vecs = vecs[:, :n_eigs_eff]
    else:
        vals, vecs = eigsh(L, k=n_eigs_eff, which="SM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]

    coeff_x = vecs.T @ V[:, 0]
    coeff_y = vecs.T @ V[:, 1]
    coeffs = np.vstack([coeff_x, coeff_y])
    power = coeff_x**2 + coeff_y**2
    return GraphFourierSpectrum(eigenvalues=vals, coeffs=coeffs, power=power, adjacency=A, laplacian=L)


# -----------------------------------------------------------------------------
# Signed circulation / signed curl orientation
# -----------------------------------------------------------------------------


def signed_circulation_metrics(points: Array, vectors: Array, *, center: Optional[Array] = None) -> Dict[str, float]:
    """Signed circulation of vector samples around a center in 2D.

    The core term is cross(point - center, vector). Reversing every vector
    flips the sign, while the energy ratios elsewhere in this script do not.
    """

    P = np.asarray(points, dtype=np.float64)[:, :2]
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(P) != len(V):
        raise ValueError(f"points and vectors length mismatch: {len(P)} vs {len(V)}")
    if len(P) == 0:
        return {
            "signed_circulation": float("nan"),
            "abs_circulation": float("nan"),
            "signed_circulation_ratio": float("nan"),
            "signed_circulation_alignment": float("nan"),
        }

    c = np.asarray(center, dtype=np.float64)[:2] if center is not None else P.mean(axis=0)
    R = P - c.reshape(1, 2)
    terms = R[:, 0] * V[:, 1] - R[:, 1] * V[:, 0]
    signed = float(np.sum(terms))
    abs_total = float(np.sum(np.abs(terms)))
    geometric = float(np.sum(np.linalg.norm(R, axis=1) * np.linalg.norm(V, axis=1)))
    return {
        "signed_circulation": signed,
        "abs_circulation": abs_total,
        "signed_circulation_ratio": signed / max(geometric, 1e-30),
        "signed_circulation_alignment": signed / max(abs_total, 1e-30),
    }


def trajectory_turning_metrics(vectors: Array) -> Dict[str, float]:
    """Intrinsic signed turning angles along the token trajectory.

    This is independent of Fourier bases, graph construction, and Hodge
    triangulation. It measures whether consecutive token-step vectors keep
    turning in one preferred direction.
    """

    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(V) < 2:
        return {
            "signed_angle": float("nan"),
            "abs_angle": float("nan"),
            "alignment": float("nan"),
            "mean_abs_angle": float("nan"),
            "rms_angle": float("nan"),
            "max_abs_angle": float("nan"),
            "samples": 0.0,
        }

    norms = np.linalg.norm(V, axis=1)
    keep = norms > 1e-12
    if int(np.sum(keep)) < 2:
        return {
            "signed_angle": float("nan"),
            "abs_angle": float("nan"),
            "alignment": float("nan"),
            "mean_abs_angle": float("nan"),
            "rms_angle": float("nan"),
            "max_abs_angle": float("nan"),
            "samples": 0.0,
        }

    U = V[keep] / norms[keep, None]
    cross = U[:-1, 0] * U[1:, 1] - U[:-1, 1] * U[1:, 0]
    dot = np.sum(U[:-1] * U[1:], axis=1)
    angles = np.arctan2(cross, np.clip(dot, -1.0, 1.0))
    signed = float(np.sum(angles))
    abs_total = float(np.sum(np.abs(angles)))
    return {
        "signed_angle": signed,
        "abs_angle": abs_total,
        "alignment": signed / max(abs_total, 1e-30),
        "mean_abs_angle": float(np.mean(np.abs(angles))),
        "rms_angle": float(np.sqrt(np.mean(angles**2))),
        "max_abs_angle": float(np.max(np.abs(angles))),
        "samples": float(len(angles)),
    }


def local_jacobian_vorticity_metrics(points: Array, vectors: Array, *, k_neighbors: int = 6) -> Dict[str, float]:
    """Estimate local curl by fitting small affine Jacobians around each point.

    For each sample, this fits V(x, y) = [x, y] B + b on nearby vector samples and
    reads scalar vorticity as dVy/dx - dVx/dy. It is a non-Hodge, non-Fourier
    local curl proxy, useful for separating rotational clutter from coherent
    large-scale transport.
    """

    from scipy.spatial import cKDTree

    P = np.asarray(points, dtype=np.float64)[:, :2]
    V = np.asarray(vectors, dtype=np.float64)[:, :2]
    if len(P) != len(V):
        raise ValueError(f"points and vectors length mismatch: {len(P)} vs {len(V)}")
    if len(P) < 4:
        return {
            "signed_vorticity_mean": float("nan"),
            "abs_vorticity_mean": float("nan"),
            "signed_vorticity_ratio": float("nan"),
            "vorticity_rms": float("nan"),
            "max_abs_vorticity": float("nan"),
            "vorticity_samples": 0.0,
        }

    k_eff = min(max(4, int(k_neighbors)), len(P))
    tree = cKDTree(P)
    _, inds = tree.query(P, k=k_eff)
    if inds.ndim == 1:
        inds = inds[:, None]

    omega: List[float] = []
    for i, nbr in enumerate(inds):
        X = P[nbr] - P[i]
        Y = V[nbr] - np.mean(V[nbr], axis=0, keepdims=True)
        if np.linalg.matrix_rank(X) < 2:
            continue
        try:
            B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        omega.append(float(B[0, 1] - B[1, 0]))

    if not omega:
        return {
            "signed_vorticity_mean": float("nan"),
            "abs_vorticity_mean": float("nan"),
            "signed_vorticity_ratio": float("nan"),
            "vorticity_rms": float("nan"),
            "max_abs_vorticity": float("nan"),
            "vorticity_samples": 0.0,
        }

    W = np.asarray(omega, dtype=np.float64)
    mean = float(np.mean(W))
    abs_mean = float(np.mean(np.abs(W)))
    return {
        "signed_vorticity_mean": mean,
        "abs_vorticity_mean": abs_mean,
        "signed_vorticity_ratio": mean / max(abs_mean, 1e-30),
        "vorticity_rms": float(np.sqrt(np.mean(W**2))),
        "max_abs_vorticity": float(np.max(np.abs(W))),
        "vorticity_samples": float(len(W)),
    }


def evaluate_vector_coefficients_at_points(
    coeffs: Array,
    kx: Array,
    ky: Array,
    points: Array,
    *,
    chunk_modes: int = 4096,
) -> Array:
    """Evaluate vector Fourier coefficients at already-scaled periodic points."""

    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 3 or coeffs.shape[0] != 2:
        raise ValueError(f"coeffs must have shape [2,modes,modes], got {coeffs.shape}")
    P = np.asarray(points, dtype=np.float64)[:, :2]
    KX, KY = np.meshgrid(np.asarray(kx, dtype=np.float64), np.asarray(ky, dtype=np.float64), indexing="ij")
    flat_kx = KX.reshape(-1)
    flat_ky = KY.reshape(-1)
    coeffs_flat = coeffs.reshape(2, -1)
    out = np.zeros((len(P), 2), dtype=np.complex128)
    x = P[:, 0]
    y = P[:, 1]

    for start in range(0, flat_kx.size, chunk_modes):
        end = min(start + chunk_modes, flat_kx.size)
        phase_arg = x[:, None] * flat_kx[None, start:end] + y[:, None] * flat_ky[None, start:end]
        phase = np.exp(1j * phase_arg)
        out[:, 0] += phase @ coeffs_flat[0, start:end]
        out[:, 1] += phase @ coeffs_flat[1, start:end]
    return out.real


def _get_jax_eval_vector_impl() -> Callable[..., Any]:
    global _JAX_EVAL_VECTOR_IMPL
    if _JAX_EVAL_VECTOR_IMPL is not None:
        return _JAX_EVAL_VECTOR_IMPL

    jax, jnp = _import_jax()

    @jax.jit
    def _impl(coeffs_re: Any, coeffs_im: Any, k: Any, points: Any) -> Any:
        kx, ky = jnp.meshgrid(k, k, indexing="ij")
        flat_kx = kx.reshape(-1)
        flat_ky = ky.reshape(-1)
        coeffs_re_flat = coeffs_re.reshape(2, -1)
        coeffs_im_flat = coeffs_im.reshape(2, -1)
        phase_arg = points[:, 0:1] * flat_kx[None, :] + points[:, 1:2] * flat_ky[None, :]
        cos_phase = jnp.cos(phase_arg)
        sin_phase = jnp.sin(phase_arg)
        out_x = cos_phase @ coeffs_re_flat[0] - sin_phase @ coeffs_im_flat[0]
        out_y = cos_phase @ coeffs_re_flat[1] - sin_phase @ coeffs_im_flat[1]
        return jnp.stack([out_x, out_y], axis=1)

    _JAX_EVAL_VECTOR_IMPL = _impl
    return _JAX_EVAL_VECTOR_IMPL


def jax_evaluate_vector_coefficients_at_points(
    coeffs: Array,
    kx: Array,
    ky: Array,
    points: Array,
) -> Array:
    """Evaluate vector Fourier coefficients with JAX/XLA."""

    jax, jnp = _import_jax()
    kx_np = np.asarray(kx, dtype=np.float64)
    ky_np = np.asarray(ky, dtype=np.float64)
    if not np.array_equal(kx_np, ky_np):
        raise ValueError("JAX evaluator currently expects matching square kx/ky grids.")
    dtype = _jax_float_dtype(jax, jnp)
    coeffs_np = np.asarray(coeffs, dtype=np.complex128)
    if coeffs_np.ndim != 3 or coeffs_np.shape[0] != 2:
        raise ValueError(f"coeffs must have shape [2,modes,modes], got {coeffs_np.shape}")
    impl = _get_jax_eval_vector_impl()
    out = impl(
        jnp.asarray(coeffs_np.real, dtype=dtype),
        jnp.asarray(coeffs_np.imag, dtype=dtype),
        jnp.asarray(kx_np, dtype=dtype),
        jnp.asarray(np.asarray(points, dtype=np.float64)[:, :2], dtype=dtype),
    )
    return np.asarray(jax.device_get(out), dtype=np.float64)


def evaluate_vorticity_coefficients_at_points(
    coeffs: Array,
    kx: Array,
    ky: Array,
    points: Array,
    *,
    chunk_modes: int = 4096,
) -> Array:
    """Evaluate scalar 2D vorticity dVy/dx - dVx/dy from Fourier coefficients."""

    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 3 or coeffs.shape[0] != 2:
        raise ValueError(f"coeffs must have shape [2,modes,modes], got {coeffs.shape}")
    KX, KY = np.meshgrid(np.asarray(kx, dtype=np.float64), np.asarray(ky, dtype=np.float64), indexing="ij")
    omega_coeffs = 1j * (KX * coeffs[1] - KY * coeffs[0])
    P = np.asarray(points, dtype=np.float64)[:, :2]
    flat_kx = KX.reshape(-1)
    flat_ky = KY.reshape(-1)
    flat_omega = omega_coeffs.reshape(-1)
    out = np.zeros(len(P), dtype=np.complex128)
    x = P[:, 0]
    y = P[:, 1]

    for start in range(0, flat_kx.size, chunk_modes):
        end = min(start + chunk_modes, flat_kx.size)
        phase_arg = x[:, None] * flat_kx[None, start:end] + y[:, None] * flat_ky[None, start:end]
        phase = np.exp(1j * phase_arg)
        out += phase @ flat_omega[start:end]
    return out.real


def _get_jax_eval_vorticity_impl() -> Callable[..., Any]:
    global _JAX_EVAL_VORTICITY_IMPL
    if _JAX_EVAL_VORTICITY_IMPL is not None:
        return _JAX_EVAL_VORTICITY_IMPL

    jax, jnp = _import_jax()

    @jax.jit
    def _impl(coeffs_re: Any, coeffs_im: Any, k: Any, points: Any) -> Any:
        kx, ky = jnp.meshgrid(k, k, indexing="ij")
        base_re = kx * coeffs_re[1] - ky * coeffs_re[0]
        base_im = kx * coeffs_im[1] - ky * coeffs_im[0]
        omega_re = -base_im
        omega_im = base_re
        flat_kx = kx.reshape(-1)
        flat_ky = ky.reshape(-1)
        flat_omega_re = omega_re.reshape(-1)
        flat_omega_im = omega_im.reshape(-1)
        phase_arg = points[:, 0:1] * flat_kx[None, :] + points[:, 1:2] * flat_ky[None, :]
        cos_phase = jnp.cos(phase_arg)
        sin_phase = jnp.sin(phase_arg)
        return cos_phase @ flat_omega_re - sin_phase @ flat_omega_im

    _JAX_EVAL_VORTICITY_IMPL = _impl
    return _JAX_EVAL_VORTICITY_IMPL


def jax_evaluate_vorticity_coefficients_at_points(
    coeffs: Array,
    kx: Array,
    ky: Array,
    points: Array,
) -> Array:
    """Evaluate scalar vorticity from Fourier coefficients with JAX/XLA."""

    jax, jnp = _import_jax()
    kx_np = np.asarray(kx, dtype=np.float64)
    ky_np = np.asarray(ky, dtype=np.float64)
    if not np.array_equal(kx_np, ky_np):
        raise ValueError("JAX evaluator currently expects matching square kx/ky grids.")
    dtype = _jax_float_dtype(jax, jnp)
    coeffs_np = np.asarray(coeffs, dtype=np.complex128)
    if coeffs_np.ndim != 3 or coeffs_np.shape[0] != 2:
        raise ValueError(f"coeffs must have shape [2,modes,modes], got {coeffs_np.shape}")
    impl = _get_jax_eval_vorticity_impl()
    out = impl(
        jnp.asarray(coeffs_np.real, dtype=dtype),
        jnp.asarray(coeffs_np.imag, dtype=dtype),
        jnp.asarray(kx_np, dtype=dtype),
        jnp.asarray(np.asarray(points, dtype=np.float64)[:, :2], dtype=dtype),
    )
    return np.asarray(jax.device_get(out), dtype=np.float64)


def spectral_curl_band_metrics(
    spec: FourierSpectrum,
    hspec: HelmholtzSpectrum,
    *,
    low_cut: float = 1.0 / 3.0,
    high_cut: float = 2.0 / 3.0,
) -> Dict[str, float]:
    """Split Fourier Helmholtz curl energy into radial frequency bands.

    The ``*_ratio`` values are relative to total spectral energy and sum to the
    overall ``spectral_curl_ratio``. The ``*_band_ratio`` values are relative to
    curl energy only and sum to one when curl energy is present.
    """

    coeffs = np.asarray(hspec.curl_coeffs, dtype=np.complex128)
    if coeffs.ndim != 3 or coeffs.shape[0] != 2:
        raise ValueError(f"curl coeffs must have shape [2,modes,modes], got {coeffs.shape}")
    KX, KY = np.meshgrid(np.asarray(spec.kx, dtype=np.float64), np.asarray(spec.ky, dtype=np.float64), indexing="ij")
    radius = np.sqrt(KX**2 + KY**2)
    max_radius = float(np.max(radius)) if radius.size else 0.0
    if max_radius <= 0.0:
        return {
            "curl_low": float("nan"),
            "curl_mid": float("nan"),
            "curl_high": float("nan"),
            "curl_low_ratio": float("nan"),
            "curl_mid_ratio": float("nan"),
            "curl_high_ratio": float("nan"),
            "curl_low_band_ratio": float("nan"),
            "curl_mid_band_ratio": float("nan"),
            "curl_high_band_ratio": float("nan"),
        }

    low_edge = float(low_cut) * max_radius
    high_edge = float(high_cut) * max_radius
    curl_power = np.abs(coeffs[0]) ** 2 + np.abs(coeffs[1]) ** 2
    masks = {
        "low": radius <= low_edge,
        "mid": (radius > low_edge) & (radius <= high_edge),
        "high": radius > high_edge,
    }
    powers = {name: float(np.sum(curl_power[mask])) for name, mask in masks.items()}
    total_energy = float(hspec.energy.get("total", np.sum(curl_power)))
    curl_energy = float(hspec.energy.get("curl", np.sum(curl_power)))
    total_denom = max(total_energy, 1e-30)
    curl_denom = max(curl_energy, 1e-30)
    return {
        "curl_low": powers["low"],
        "curl_mid": powers["mid"],
        "curl_high": powers["high"],
        "curl_low_ratio": powers["low"] / total_denom,
        "curl_mid_ratio": powers["mid"] / total_denom,
        "curl_high_ratio": powers["high"] / total_denom,
        "curl_low_band_ratio": powers["low"] / curl_denom,
        "curl_mid_band_ratio": powers["mid"] / curl_denom,
        "curl_high_band_ratio": powers["high"] / curl_denom,
    }


def spectral_signed_curl_metrics(spec: FourierSpectrum, hspec: HelmholtzSpectrum) -> Dict[str, float]:
    """Signed orientation metrics for the Fourier Helmholtz curl component."""

    if spec.backend == "jax":
        curl_vectors = jax_evaluate_vector_coefficients_at_points(
            hspec.curl_coeffs,
            spec.kx,
            spec.ky,
            spec.scaled_points,
        )
        omega = jax_evaluate_vorticity_coefficients_at_points(
            hspec.curl_coeffs,
            spec.kx,
            spec.ky,
            spec.scaled_points,
        )
    else:
        curl_vectors = evaluate_vector_coefficients_at_points(hspec.curl_coeffs, spec.kx, spec.ky, spec.scaled_points)
        omega = evaluate_vorticity_coefficients_at_points(hspec.curl_coeffs, spec.kx, spec.ky, spec.scaled_points)
    circ = signed_circulation_metrics(spec.scaled_points, curl_vectors)
    omega_mean = float(np.mean(omega)) if omega.size else float("nan")
    omega_abs_mean = float(np.mean(np.abs(omega))) if omega.size else float("nan")
    omega_rms = float(np.sqrt(np.mean(omega**2))) if omega.size else float("nan")
    return {
        "signed_curl_circulation": circ["signed_circulation"],
        "abs_curl_circulation": circ["abs_circulation"],
        "signed_curl_circulation_ratio": circ["signed_circulation_ratio"],
        "signed_curl_alignment": circ["signed_circulation_alignment"],
        "signed_vorticity_mean": omega_mean,
        "abs_vorticity_mean": omega_abs_mean,
        "signed_vorticity_ratio": omega_mean / max(omega_abs_mean, 1e-30),
        "vorticity_rms": omega_rms,
    }


def hodge_signed_curl_metrics(hodge: HodgeDecomposition) -> Dict[str, float]:
    """Signed face-circulation metrics for the discrete Hodge curl component."""

    face_curl = np.asarray(hodge.D1 @ hodge.curl, dtype=np.float64).reshape(-1)
    if face_curl.size == 0:
        return {
            "signed_curl_circulation": float("nan"),
            "abs_curl_circulation": float("nan"),
            "signed_curl_circulation_ratio": float("nan"),
            "signed_curl_alignment": float("nan"),
        }
    signed = float(np.sum(face_curl))
    abs_total = float(np.sum(np.abs(face_curl)))
    l2_total = float(np.sqrt(np.sum(face_curl**2)))
    return {
        "signed_curl_circulation": signed,
        "abs_curl_circulation": abs_total,
        "signed_curl_circulation_ratio": signed / max(np.sqrt(face_curl.size) * l2_total, 1e-30),
        "signed_curl_alignment": signed / max(abs_total, 1e-30),
    }


# -----------------------------------------------------------------------------
# Discrete Hodge decomposition on Delaunay triangulation
# -----------------------------------------------------------------------------


def _orient_faces_ccw(points: Array, simplices: Array) -> Array:
    P = np.asarray(points, dtype=np.float64)[:, :2]
    faces = []
    for tri in np.asarray(simplices, dtype=int):
        a, b, c = tri.tolist()
        area2 = (P[b, 0] - P[a, 0]) * (P[c, 1] - P[a, 1]) - (P[b, 1] - P[a, 1]) * (P[c, 0] - P[a, 0])
        if area2 < 0:
            faces.append([a, c, b])
        else:
            faces.append([a, b, c])
    return np.asarray(faces, dtype=int)


def delaunay_complex(points: Array, *, qhull_jitter: float = 1e-8) -> Tuple[Array, Array]:
    """Return oriented edges and faces from a 2D Delaunay triangulation."""

    from scipy.spatial import Delaunay

    P = np.asarray(points, dtype=np.float64)[:, :2]
    if len(P) < 3:
        raise ValueError("Need at least 3 points for triangulation.")

    # Degenerate PCA paths can be nearly collinear or repeated. Tiny jitter helps QHull.
    rng = np.random.default_rng(0)
    Pj = P + rng.normal(scale=qhull_jitter, size=P.shape)
    tri = Delaunay(Pj, qhull_options="QJ Qbb Qc Qz")
    simplices = np.asarray(tri.simplices, dtype=int)
    # Delaunay with Qz can include an out-of-range point at infinity. Drop those
    # before orientation, because orientation indexes into the point array.
    simplices = simplices[np.all(simplices < len(P), axis=1)]
    if len(simplices) == 0:
        raise ValueError("Delaunay triangulation produced no finite faces.")
    faces = _orient_faces_ccw(Pj, simplices)

    edge_set = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((min(int(u), int(v)), max(int(u), int(v))))
    edges = np.asarray(sorted(edge_set), dtype=int)
    return edges, faces


def incidence_matrices(n_vertices: int, edges: Array, faces: Array):
    """Build D0 [E,V] and D1 [F,E] for a 2D simplicial complex."""

    from scipy.sparse import coo_matrix

    edges = np.asarray(edges, dtype=int)
    faces = np.asarray(faces, dtype=int)
    E = len(edges)
    F = len(faces)

    rows = np.repeat(np.arange(E), 2)
    cols = edges.reshape(-1)
    vals = np.tile(np.array([-1.0, 1.0]), E)
    D0 = coo_matrix((vals, (rows, cols)), shape=(E, n_vertices)).tocsr()

    edge_index = {tuple(e): i for i, e in enumerate(edges.tolist())}
    fr: List[int] = []
    fc: List[int] = []
    fv: List[float] = []
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            ei = edge_index[key]
            sign = 1.0 if (u, v) == key else -1.0
            fr.append(fi)
            fc.append(ei)
            fv.append(sign)
    D1 = coo_matrix((fv, (fr, fc)), shape=(F, E)).tocsr()
    return D0, D1


def _interpolate_vectors_idw(query_points: Array, sample_points: Array, sample_vectors: Array, k: int = 4) -> Array:
    """Inverse-distance weighted interpolation of vector samples."""

    from scipy.spatial import cKDTree

    Q = np.asarray(query_points, dtype=np.float64)[:, :2]
    P = np.asarray(sample_points, dtype=np.float64)[:, :2]
    V = np.asarray(sample_vectors, dtype=np.float64)[:, :2]
    if len(P) == 0:
        raise ValueError("Need at least one sample point for interpolation.")
    k_eff = min(max(1, k), len(P))
    tree = cKDTree(P)
    d, idx = tree.query(Q, k=k_eff)
    if k_eff == 1:
        return V[idx]
    w = 1.0 / np.maximum(d, 1e-12)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("qk,qkc->qc", w, V[idx])


def edge_flow_from_vector_samples(
    node_points: Array,
    edges: Array,
    sample_points: Array,
    sample_vectors: Array,
    *,
    interp_k: int = 4,
) -> Array:
    """Project interpolated vector field onto oriented triangulation edges."""

    P = np.asarray(node_points, dtype=np.float64)[:, :2]
    edges = np.asarray(edges, dtype=int)
    mids = 0.5 * (P[edges[:, 0]] + P[edges[:, 1]])
    vec_mid = _interpolate_vectors_idw(mids, sample_points, sample_vectors, k=interp_k)
    edge_vec = P[edges[:, 1]] - P[edges[:, 0]]
    lengths = np.linalg.norm(edge_vec, axis=1, keepdims=True)
    unit = edge_vec / np.maximum(lengths, 1e-12)
    return np.sum(vec_mid * unit, axis=1)


def hodge_decompose_edge_flow(
    flow: Array,
    D0: Any,
    D1: Any,
    *,
    lsqr_tol: float = 1e-10,
) -> Tuple[Array, Array, Array, Dict[str, float]]:
    """Orthogonal combinatorial Hodge decomposition of an edge flow."""

    from scipy.sparse.linalg import lsqr

    y = np.asarray(flow, dtype=np.float64).reshape(-1)
    if D0.shape[0] != len(y) or D1.shape[1] != len(y):
        raise ValueError("D0/D1 shapes are inconsistent with flow length.")

    phi = lsqr(D0, y, atol=lsqr_tol, btol=lsqr_tol)[0]
    grad = np.asarray(D0 @ phi).reshape(-1)
    r = y - grad

    if D1.shape[0] > 0:
        psi = lsqr(D1.T, r, atol=lsqr_tol, btol=lsqr_tol)[0]
        curl = np.asarray(D1.T @ psi).reshape(-1)
    else:
        curl = np.zeros_like(y)
    harmonic = y - grad - curl

    total = float(np.dot(y, y))
    grad_e = float(np.dot(grad, grad))
    curl_e = float(np.dot(curl, curl))
    harm_e = float(np.dot(harmonic, harmonic))
    denom = max(total, 1e-30)
    energy = {
        "total": total,
        "grad": grad_e,
        "curl": curl_e,
        "harmonic": harm_e,
        "grad_ratio": grad_e / denom,
        "curl_ratio": curl_e / denom,
        "harmonic_ratio": harm_e / denom,
        "reconstruction_error": float(np.linalg.norm(y - grad - curl - harmonic)),
    }
    return grad, curl, harmonic, energy


def hodge_from_token_field(
    node_points: Array,
    field: VectorFieldBundle,
    *,
    interp_k: int = 4,
) -> HodgeDecomposition:
    """Build Delaunay complex from token coordinates and decompose edge flow."""

    P = np.asarray(node_points, dtype=np.float64)[:, :2]
    edges, faces = delaunay_complex(P)
    D0, D1 = incidence_matrices(len(P), edges, faces)
    flow = edge_flow_from_vector_samples(P, edges, field.points, field.vectors, interp_k=interp_k)
    grad, curl, harmonic, energy = hodge_decompose_edge_flow(flow, D0, D1)
    return HodgeDecomposition(
        edges=edges,
        faces=faces,
        flow=flow,
        grad=grad,
        curl=curl,
        harmonic=harmonic,
        energy=energy,
        D0=D0,
        D1=D1,
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def plot_token_field(field: VectorFieldBundle, *, tokens: Optional[Sequence[str]] = None, title: str = "Token trajectory field"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    P = field.points
    V = field.vectors
    ax.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], angles="xy", scale_units="xy", scale=1)
    ax.scatter(P[:, 0], P[:, 1], s=12)
    if tokens is not None:
        for i, (x, y) in enumerate(P):
            if i < len(tokens):
                ax.text(x, y, str(tokens[i])[:12], fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("semantic coord 1")
    ax.set_ylabel("semantic coord 2")
    ax.axis("equal")
    fig.tight_layout()
    return fig, ax


def plot_fourier_power(spec: FourierSpectrum, *, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    power = np.fft.fftshift(spec.power)
    im = ax.imshow(np.log1p(power), origin="lower", aspect="auto")
    ax.set_title(title or f"Nonuniform Fourier vector power ({spec.backend})")
    ax.set_xlabel("ky mode")
    ax.set_ylabel("kx mode")
    fig.colorbar(im, ax=ax, label="log(1 + power)")
    fig.tight_layout()
    return fig, ax


def plot_graph_spectrum(gspec: GraphFourierSpectrum, *, title: str = "Graph Fourier power"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gspec.eigenvalues, gspec.power, marker="o")
    ax.set_title(title)
    ax.set_xlabel("graph Laplacian eigenvalue")
    ax.set_ylabel("vector power")
    fig.tight_layout()
    return fig, ax




# -----------------------------------------------------------------------------
# Layer sweep / null models / metric export
# -----------------------------------------------------------------------------

NullModelName = Literal["real", "shuffle_tokens", "reverse_tokens", "random_hidden"]

LAYER_SWEEP_FIELDS = [
    "variant",
    "layer",
    "layers",
    "tokens",
    "dim",
    "pca_var_0",
    "pca_var_1",
    "trajectory_signed_circulation",
    "trajectory_abs_circulation",
    "trajectory_signed_circulation_ratio",
    "trajectory_signed_circulation_alignment",
    "turning_signed_angle",
    "turning_abs_angle",
    "turning_alignment",
    "turning_mean_abs_angle",
    "turning_rms_angle",
    "turning_max_abs_angle",
    "turning_samples",
    "local_signed_vorticity_mean",
    "local_abs_vorticity_mean",
    "local_signed_vorticity_ratio",
    "local_vorticity_rms",
    "local_max_abs_vorticity",
    "local_vorticity_samples",
    "spectral_total",
    "spectral_grad_ratio",
    "spectral_curl_ratio",
    "spectral_harmonic_ratio",
    "spectral_grad",
    "spectral_curl",
    "spectral_harmonic",
    "spectral_curl_low",
    "spectral_curl_mid",
    "spectral_curl_high",
    "spectral_curl_low_ratio",
    "spectral_curl_mid_ratio",
    "spectral_curl_high_ratio",
    "spectral_curl_low_band_ratio",
    "spectral_curl_mid_band_ratio",
    "spectral_curl_high_band_ratio",
    "spectral_signed_curl_circulation",
    "spectral_abs_curl_circulation",
    "spectral_signed_curl_circulation_ratio",
    "spectral_signed_curl_alignment",
    "spectral_signed_vorticity_mean",
    "spectral_abs_vorticity_mean",
    "spectral_signed_vorticity_ratio",
    "spectral_vorticity_rms",
    "hodge_total",
    "hodge_grad_ratio",
    "hodge_curl_ratio",
    "hodge_harmonic_ratio",
    "hodge_grad",
    "hodge_curl",
    "hodge_harmonic",
    "hodge_reconstruction_error",
    "hodge_signed_curl_circulation",
    "hodge_abs_curl_circulation",
    "hodge_signed_curl_circulation_ratio",
    "hodge_signed_curl_alignment",
    "graph_total_power",
    "graph_low_freq_power",
    "graph_high_freq_power",
    "graph_low_freq_ratio",
    "graph_high_freq_ratio",
    "graph_first_nonzero_eigenvalue",
    "graph_eigs_used",
]


def parse_null_models(value: str) -> List[NullModelName]:
    """Parse a comma-separated null-model list."""

    aliases = {
        "real": "real",
        "none": "real",
        "shuffle": "shuffle_tokens",
        "shuffle_tokens": "shuffle_tokens",
        "reverse": "reverse_tokens",
        "reverse_tokens": "reverse_tokens",
        "random": "random_hidden",
        "random_hidden": "random_hidden",
    }
    if value is None:
        return ["real"]
    raw = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    if not raw:
        return ["real"]
    if any(x == "all" for x in raw):
        return ["real", "shuffle_tokens", "reverse_tokens", "random_hidden"]
    out: List[NullModelName] = []
    for item in raw:
        if item not in aliases:
            valid = "real, shuffle_tokens, reverse_tokens, random_hidden, all"
            raise ValueError(f"Unknown null model '{item}'. Valid values: {valid}")
        mapped = aliases[item]
        if mapped not in out:
            out.append(mapped)  # type: ignore[arg-type]
    return out


def make_null_hidden(hidden: Array, variant: NullModelName, *, seed: int = 0) -> Array:
    """Transform hidden states into a simple null/control variant.

    real:
      unchanged.
    shuffle_tokens:
      same hidden vectors, same layers, but token order is scrambled identically
      for every layer. This tests whether the token trajectory order matters.
    reverse_tokens:
      same path traversed backwards.
    random_hidden:
      independent Gaussian hidden states with per-dimension mean/std matched to
      the original hidden states. This tests distributional baselines.
    """

    H = np.asarray(hidden)
    if H.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim], got {H.shape}")
    rng = np.random.default_rng(seed)
    if variant == "real":
        return H.copy()
    if variant == "shuffle_tokens":
        perm = rng.permutation(H.shape[1])
        return H[:, perm, :].copy()
    if variant == "reverse_tokens":
        return H[:, ::-1, :].copy()
    if variant == "random_hidden":
        mean = H.mean(axis=(0, 1), keepdims=True)
        std = H.std(axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1e-6)
        return rng.normal(loc=mean, scale=std, size=H.shape).astype(H.dtype, copy=False)
    raise ValueError(f"unknown null variant: {variant}")


def graph_band_metrics(gspec: GraphFourierSpectrum, *, low_count: int = 5, high_count: int = 5) -> Dict[str, float]:
    """Small summary of graph spectrum power concentration."""

    power = np.asarray(gspec.power, dtype=np.float64)
    evals = np.asarray(gspec.eigenvalues, dtype=np.float64)
    total = float(np.sum(power))
    denom = max(total, 1e-30)
    n = len(power)
    lo = min(max(1, low_count), n)
    hi = min(max(1, high_count), n)
    low_power = float(np.sum(power[:lo]))
    high_power = float(np.sum(power[-hi:]))
    first_nonzero = float("nan")
    nz = evals[evals > 1e-9]
    if nz.size:
        first_nonzero = float(nz[0])
    return {
        "graph_total_power": total,
        "graph_low_freq_power": low_power,
        "graph_high_freq_power": high_power,
        "graph_low_freq_ratio": low_power / denom,
        "graph_high_freq_ratio": high_power / denom,
        "graph_first_nonzero_eigenvalue": first_nonzero,
        "graph_eigs_used": float(n),
    }


def _copy_energy(prefix: str, energy: Dict[str, float], row: Dict[str, Any]) -> None:
    for key in ["total", "grad", "curl", "harmonic", "grad_ratio", "curl_ratio", "harmonic_ratio", "reconstruction_error"]:
        out_key = f"{prefix}_{key}"
        if out_key in LAYER_SWEEP_FIELDS:
            row[out_key] = float(energy.get(key, float("nan")))


def _copy_metrics(prefix: str, metrics: Dict[str, float], row: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        out_key = f"{prefix}_{key}"
        if out_key in LAYER_SWEEP_FIELDS:
            row[out_key] = float(value)


def layer_metric_row(
    *,
    variant: str,
    layer: int,
    hidden_shape: Tuple[int, int, int],
    coord: CoordinateBundle,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten one layer analysis result into a CSV row."""

    L, T, D = hidden_shape
    row: Dict[str, Any] = {key: float("nan") for key in LAYER_SWEEP_FIELDS}
    row.update({"variant": variant, "layer": int(layer), "layers": int(L), "tokens": int(T), "dim": int(D)})

    evr = coord.explained_variance_ratio
    if evr is not None and len(evr) > 0:
        row["pca_var_0"] = float(evr[0])
        if len(evr) > 1:
            row["pca_var_1"] = float(evr[1])

    if "trajectory_signed" in result:
        _copy_metrics("trajectory", result["trajectory_signed"], row)
    if "trajectory_turning" in result:
        _copy_metrics("turning", result["trajectory_turning"], row)
    if "local_vorticity" in result:
        _copy_metrics("local", result["local_vorticity"], row)
    if "spectral_helmholtz" in result:
        _copy_energy("spectral", result["spectral_helmholtz"].energy, row)
    if "spectral_curl_bands" in result:
        _copy_metrics("spectral", result["spectral_curl_bands"], row)
    if "spectral_signed" in result:
        _copy_metrics("spectral", result["spectral_signed"], row)
    if "hodge" in result:
        _copy_energy("hodge", result["hodge"].energy, row)
    if "hodge_signed" in result:
        _copy_metrics("hodge", result["hodge_signed"], row)
    if "graph_fourier" in result:
        row.update(graph_band_metrics(result["graph_fourier"]))

    return row


def analyze_layer_from_coordinates(
    coord: CoordinateBundle,
    *,
    layer: int,
    fourier_modes: int = 32,
    fourier_backend: FourierBackend = "direct",
    graph_eigs: int = 32,
    k_neighbors: int = 8,
    do_fourier: bool = True,
    do_graph: bool = True,
    do_hodge: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Analyze one layer using a precomputed coordinate system."""

    field = token_trajectory_field(coord.coords, layer=layer)
    result: Dict[str, Any] = {
        "coordinates": coord,
        "field": field,
        "trajectory_signed": signed_circulation_metrics(field.points, field.vectors),
        "trajectory_turning": trajectory_turning_metrics(field.vectors),
        "local_vorticity": local_jacobian_vorticity_metrics(field.points, field.vectors, k_neighbors=k_neighbors),
    }

    if do_fourier:
        try:
            fspec = vector_spectrum(field.points, field.vectors, modes=fourier_modes, backend=fourier_backend)
            hspec = helmholtz_project_spectrum(fspec)
            bands = spectral_curl_band_metrics(fspec, hspec)
            signed = spectral_signed_curl_metrics(fspec, hspec)
            result.update({
                "fourier": fspec,
                "spectral_helmholtz": hspec,
                "spectral_curl_bands": bands,
                "spectral_signed": signed,
            })
        except Exception as e:
            warnings.warn(f"Layer {field.layer}: Fourier/Helmholtz failed: {e}")

    if do_graph:
        try:
            gspec = graph_fourier_spectrum(field.points, field.vectors, k_neighbors=k_neighbors, n_eigs=graph_eigs)
            result["graph_fourier"] = gspec
        except Exception as e:
            warnings.warn(f"Layer {field.layer}: Graph Fourier failed: {e}")

    if do_hodge:
        try:
            node_points = coord.coords[field.layer, :, :2]
            hodge = hodge_from_token_field(node_points, field)
            result["hodge"] = hodge
            result["hodge_signed"] = hodge_signed_curl_metrics(hodge)
        except Exception as e:
            warnings.warn(f"Layer {field.layer}: Discrete Hodge failed: {e}")

    return result


def run_layer_sweep_from_hidden(
    hidden: Array,
    *,
    reducer: Literal["pca", "umap"] = "pca",
    n_components: int = 2,
    null_models: Sequence[NullModelName] = ("real",),
    seed: int = 0,
    fourier_modes: int = 32,
    fourier_backend: FourierBackend = "direct",
    graph_eigs: int = 32,
    k_neighbors: int = 8,
    do_fourier: bool = True,
    do_graph: bool = True,
    do_hodge: bool = True,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]]:
    """Run all-layer analysis for real and optional null-model variants."""

    H0 = np.asarray(hidden)
    if H0.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim], got {H0.shape}")
    L, T, D = H0.shape
    rows: List[Dict[str, Any]] = []
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for vi, variant in enumerate(null_models):
        log(f"[sweep] variant={variant}", verbose=verbose)
        H = make_null_hidden(H0, variant, seed=seed + 1009 * vi)
        coord = make_semantic_coordinates(
            H,
            method=reducer,
            n_components=n_components,
            random_state=seed,
            verbose=verbose,
        )
        results[variant] = {}
        for layer in range(L):
            log(f"[sweep] {variant}: layer {layer}/{L - 1}", verbose=verbose)
            result = analyze_layer_from_coordinates(
                coord,
                layer=layer,
                fourier_modes=fourier_modes,
                fourier_backend=fourier_backend,
                graph_eigs=graph_eigs,
                k_neighbors=k_neighbors,
                do_fourier=do_fourier,
                do_graph=do_graph,
                do_hodge=do_hodge,
                verbose=False,
            )
            results[variant][layer] = result
            rows.append(layer_metric_row(variant=variant, layer=layer, hidden_shape=(L, T, D), coord=coord, result=result))

    return rows, results


def write_layer_metrics_csv(rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> Path:
    """Write layer-sweep rows to CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LAYER_SWEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float("nan")) for k in LAYER_SWEEP_FIELDS})
    return out


def _rows_for_variant(rows: Sequence[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    out = [r for r in rows if str(r.get("variant")) == variant]
    return sorted(out, key=lambda r: int(r.get("layer", 0)))


def plot_layer_energy_ratios(
    rows: Sequence[Dict[str, Any]],
    *,
    variant: str = "real",
    source: Literal["spectral", "hodge"] = "spectral",
    title: Optional[str] = None,
):
    """Plot grad/curl/harmonic ratios across layers for one variant/source."""

    import matplotlib.pyplot as plt

    rr = _rows_for_variant(rows, variant)
    if not rr:
        raise ValueError(f"No rows for variant={variant}")
    layers = np.asarray([int(r["layer"]) for r in rr])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ["grad_ratio", "curl_ratio", "harmonic_ratio"]:
        key = f"{source}_{name}"
        y = np.asarray([float(r.get(key, float("nan"))) for r in rr])
        ax.plot(layers, y, marker="o", label=name.replace("_ratio", ""))
    ax.set_title(title or f"{source.title()} Helmholtz/Hodge ratios by layer ({variant})")
    ax.set_xlabel("layer")
    ax.set_ylabel("energy ratio")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_curl_comparison(rows: Sequence[Dict[str, Any]], *, variant: str = "real"):
    """Compare spectral and discrete-Hodge curl ratios across layers."""

    import matplotlib.pyplot as plt

    rr = _rows_for_variant(rows, variant)
    if not rr:
        raise ValueError(f"No rows for variant={variant}")
    layers = np.asarray([int(r["layer"]) for r in rr])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, [float(r.get("spectral_curl_ratio", float("nan"))) for r in rr], marker="o", label="spectral curl")
    ax.plot(layers, [float(r.get("hodge_curl_ratio", float("nan"))) for r in rr], marker="o", label="hodge curl")
    ax.set_title(f"Curl ratio comparison by layer ({variant})")
    ax.set_xlabel("layer")
    ax.set_ylabel("curl energy ratio")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_spectral_curl_bands(rows: Sequence[Dict[str, Any]], *, variant: str = "real"):
    """Plot Fourier Helmholtz curl energy split by low/mid/high frequency."""

    import matplotlib.pyplot as plt

    rr = _rows_for_variant(rows, variant)
    if not rr:
        raise ValueError(f"No rows for variant={variant}")
    layers = np.asarray([int(r["layer"]) for r in rr])
    metrics = [
        ("spectral_curl_low_ratio", "low-frequency curl"),
        ("spectral_curl_mid_ratio", "mid-frequency curl"),
        ("spectral_curl_high_ratio", "high-frequency curl"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for key, label in metrics:
        y = np.asarray([float(r.get(key, float("nan"))) for r in rr], dtype=np.float64)
        if np.all(np.isnan(y)):
            continue
        ax.plot(layers, y, marker="o", label=label)
        plotted = True
    if not plotted:
        raise ValueError(f"No spectral curl band metrics for variant={variant}")
    ax.set_title(f"Spectral curl frequency bands by layer ({variant})")
    ax.set_xlabel("layer")
    ax.set_ylabel("curl-band energy / total spectral energy")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_null_model_curl(
    rows: Sequence[Dict[str, Any]],
    *,
    source: Literal["spectral", "hodge"] = "spectral",
):
    """Plot curl-ratio curves for every variant in the CSV rows."""

    import matplotlib.pyplot as plt

    variants = []
    for r in rows:
        v = str(r.get("variant"))
        if v not in variants:
            variants.append(v)
    if not variants:
        raise ValueError("No layer rows to plot")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    key = f"{source}_curl_ratio"
    for variant in variants:
        rr = _rows_for_variant(rows, variant)
        layers = np.asarray([int(r["layer"]) for r in rr])
        y = np.asarray([float(r.get(key, float("nan"))) for r in rr])
        ax.plot(layers, y, marker="o", label=variant)
    ax.set_title(f"{source.title()} curl ratio: real vs null models")
    ax.set_xlabel("layer")
    ax.set_ylabel("curl energy ratio")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_signed_circulation_comparison(rows: Sequence[Dict[str, Any]], *, variant: str = "real"):
    """Compare signed orientation metrics across layers for one variant."""

    import matplotlib.pyplot as plt

    rr = _rows_for_variant(rows, variant)
    if not rr:
        raise ValueError(f"No rows for variant={variant}")
    layers = np.asarray([int(r["layer"]) for r in rr])
    metrics = [
        ("trajectory_signed_circulation_alignment", "trajectory circulation"),
        ("turning_alignment", "trajectory turning"),
        ("spectral_signed_curl_alignment", "spectral curl circulation"),
        ("local_signed_vorticity_ratio", "local Jacobian vorticity"),
        ("hodge_signed_curl_alignment", "hodge face curl"),
        ("spectral_signed_vorticity_ratio", "spectral vorticity"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for key, label in metrics:
        y = np.asarray([float(r.get(key, float("nan"))) for r in rr], dtype=np.float64)
        if np.all(np.isnan(y)):
            continue
        ax.plot(layers, y, marker="o", label=label)
        plotted = True
    if not plotted:
        raise ValueError(f"No signed metrics for variant={variant}")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Signed circulation/curl orientation by layer ({variant})")
    ax.set_xlabel("layer")
    ax.set_ylabel("signed orientation [-1, 1]")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_null_model_signed_metric(
    rows: Sequence[Dict[str, Any]],
    *,
    metric: str = "spectral_signed_curl_alignment",
    title: Optional[str] = None,
):
    """Plot one signed metric across layers for every variant."""

    import matplotlib.pyplot as plt

    variants = []
    for r in rows:
        v = str(r.get("variant"))
        if v not in variants:
            variants.append(v)
    if not variants:
        raise ValueError("No layer rows to plot")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for variant in variants:
        rr = _rows_for_variant(rows, variant)
        layers = np.asarray([int(r["layer"]) for r in rr])
        y = np.asarray([float(r.get(metric, float("nan"))) for r in rr], dtype=np.float64)
        if np.all(np.isnan(y)):
            continue
        ax.plot(layers, y, marker="o", label=variant)
        plotted = True
    if not plotted:
        raise ValueError(f"No finite values for metric={metric}")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title or f"{metric}: real vs null models")
    ax.set_xlabel("layer")
    ax.set_ylabel("signed orientation [-1, 1]")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_layer_sweep_plots(rows: Sequence[Dict[str, Any]], output_dir: Union[str, Path]) -> List[Path]:
    """Save standard layer-sweep plots and return their paths."""

    import matplotlib.pyplot as plt

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    variants = []
    for r in rows:
        v = str(r.get("variant"))
        if v not in variants:
            variants.append(v)

    for variant in variants:
        for source in ["spectral", "hodge"]:
            # Skip sources that are entirely missing.
            key = f"{source}_curl_ratio"
            vals = [float(r.get(key, float("nan"))) for r in _rows_for_variant(rows, variant)]
            if not vals or np.all(np.isnan(vals)):
                continue
            fig, _ = plot_layer_energy_ratios(rows, variant=variant, source=source)  # type: ignore[arg-type]
            path = outdir / f"layer_energy_ratios_{source}_{variant}.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

        real_rows = _rows_for_variant(rows, variant)
        has_spectral = any(not np.isnan(float(r.get("spectral_curl_ratio", float("nan")))) for r in real_rows)
        has_hodge = any(not np.isnan(float(r.get("hodge_curl_ratio", float("nan")))) for r in real_rows)
        if has_spectral and has_hodge:
            fig, _ = plot_curl_comparison(rows, variant=variant)
            path = outdir / f"curl_comparison_{variant}.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

        has_bands = any(not np.isnan(float(r.get("spectral_curl_high_ratio", float("nan")))) for r in real_rows)
        if has_bands:
            fig, _ = plot_spectral_curl_bands(rows, variant=variant)
            path = outdir / f"spectral_curl_bands_{variant}.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

        signed_keys = [
            "trajectory_signed_circulation_alignment",
            "turning_alignment",
            "spectral_signed_curl_alignment",
            "local_signed_vorticity_ratio",
            "hodge_signed_curl_alignment",
            "spectral_signed_vorticity_ratio",
        ]
        has_signed = any(
            any(not np.isnan(float(r.get(key, float("nan")))) for r in real_rows)
            for key in signed_keys
        )
        if has_signed:
            fig, _ = plot_signed_circulation_comparison(rows, variant=variant)
            path = outdir / f"signed_circulation_comparison_{variant}.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

    for source in ["spectral", "hodge"]:
        key = f"{source}_curl_ratio"
        vals = [float(r.get(key, float("nan"))) for r in rows]
        if vals and not np.all(np.isnan(vals)):
            fig, _ = plot_null_model_curl(rows, source=source)  # type: ignore[arg-type]
            path = outdir / f"null_model_curl_{source}.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

    signed_plot_specs = [
        ("trajectory_signed_circulation_alignment", "Trajectory signed circulation: real vs null models", "null_model_signed_trajectory.png"),
        ("turning_alignment", "Trajectory turning alignment: real vs null models", "null_model_turning_alignment.png"),
        ("spectral_signed_curl_alignment", "Spectral signed curl circulation: real vs null models", "null_model_signed_spectral_curl.png"),
        ("local_signed_vorticity_ratio", "Local Jacobian signed vorticity: real vs null models", "null_model_local_signed_vorticity.png"),
        ("hodge_signed_curl_alignment", "Hodge signed face curl: real vs null models", "null_model_signed_hodge_curl.png"),
        ("spectral_signed_vorticity_ratio", "Spectral signed vorticity: real vs null models", "null_model_signed_spectral_vorticity.png"),
    ]
    for metric, title, filename in signed_plot_specs:
        vals = [float(r.get(metric, float("nan"))) for r in rows]
        if vals and not np.all(np.isnan(vals)):
            fig, _ = plot_null_model_signed_metric(rows, metric=metric, title=title)
            path = outdir / filename
            fig.savefig(path, dpi=160)
            saved.append(path)
            plt.close(fig)

    return saved


def print_layer_sweep_summary(rows: Sequence[Dict[str, Any]]) -> None:
    """Human-readable summary of peaks and rough agreement."""

    variants = []
    for r in rows:
        v = str(r.get("variant"))
        if v not in variants:
            variants.append(v)

    for variant in variants:
        rr = _rows_for_variant(rows, variant)
        print(f"\n[layer sweep] variant={variant}")
        for source in ["spectral", "hodge"]:
            key = f"{source}_curl_ratio"
            vals = np.asarray([float(r.get(key, float("nan"))) for r in rr], dtype=np.float64)
            if vals.size == 0 or np.all(np.isnan(vals)):
                continue
            idx = int(np.nanargmax(vals))
            layer = int(rr[idx]["layer"])
            print(f"  {source} curl peak: layer={layer}, ratio={vals[idx]:.4f}")
        for key, label in [
            ("spectral_curl_low_ratio", "low-frequency spectral curl"),
            ("spectral_curl_high_ratio", "high-frequency spectral curl"),
        ]:
            vals = np.asarray([float(r.get(key, float("nan"))) for r in rr], dtype=np.float64)
            if vals.size == 0 or np.all(np.isnan(vals)):
                continue
            idx = int(np.nanargmax(vals))
            layer = int(rr[idx]["layer"])
            print(f"  {label} peak: layer={layer}, ratio={vals[idx]:.4f}")
        signed_specs = [
            ("trajectory_signed_circulation_alignment", "trajectory signed circulation"),
            ("turning_alignment", "trajectory turning"),
            ("spectral_signed_curl_alignment", "spectral signed curl"),
            ("local_signed_vorticity_ratio", "local Jacobian vorticity"),
            ("hodge_signed_curl_alignment", "hodge signed curl"),
            ("spectral_signed_vorticity_ratio", "spectral signed vorticity"),
        ]
        for key, label in signed_specs:
            vals = np.asarray([float(r.get(key, float("nan"))) for r in rr], dtype=np.float64)
            if vals.size == 0 or np.all(np.isnan(vals)):
                continue
            idx = int(np.nanargmax(np.abs(vals)))
            layer = int(rr[idx]["layer"])
            print(f"  {label} strongest: layer={layer}, signed={vals[idx]:+.4f}")
        s = np.asarray([float(r.get("spectral_curl_ratio", float("nan"))) for r in rr], dtype=np.float64)
        h = np.asarray([float(r.get("hodge_curl_ratio", float("nan"))) for r in rr], dtype=np.float64)
        mask = np.isfinite(s) & np.isfinite(h)
        if int(mask.sum()) >= 3:
            corr = float(np.corrcoef(s[mask], h[mask])[0, 1])
            print(f"  spectral-vs-hodge curl corr: {corr:.4f}")

# -----------------------------------------------------------------------------
# End-to-end wrappers
# -----------------------------------------------------------------------------


def run_pipeline_from_hidden(
    hidden: Array,
    *,
    layer: int = -1,
    reducer: Literal["pca", "umap"] = "pca",
    n_components: int = 2,
    fourier_modes: int = 32,
    fourier_backend: FourierBackend = "direct",
    graph_eigs: int = 32,
    k_neighbors: int = 8,
    do_fourier: bool = True,
    do_graph: bool = True,
    do_hodge: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the prototype from an existing [layers,tokens,dim] hidden array."""

    coord = make_semantic_coordinates(hidden, method=reducer, n_components=n_components, verbose=verbose)
    field = token_trajectory_field(coord.coords, layer=layer)
    result: Dict[str, Any] = {
        "coordinates": coord,
        "field": field,
        "trajectory_signed": signed_circulation_metrics(field.points, field.vectors),
        "trajectory_turning": trajectory_turning_metrics(field.vectors),
        "local_vorticity": local_jacobian_vorticity_metrics(field.points, field.vectors, k_neighbors=k_neighbors),
    }

    if do_fourier:
        log(f"[stage] nonuniform Fourier spectrum, backend={fourier_backend}", verbose=verbose)
        try:
            fspec = vector_spectrum(field.points, field.vectors, modes=fourier_modes, backend=fourier_backend)
            hspec = helmholtz_project_spectrum(fspec)
            bands = spectral_curl_band_metrics(fspec, hspec)
            signed = spectral_signed_curl_metrics(fspec, hspec)
            result.update({
                "fourier": fspec,
                "spectral_helmholtz": hspec,
                "spectral_curl_bands": bands,
                "spectral_signed": signed,
            })
        except Exception as e:
            warnings.warn(f"Fourier/Helmholtz failed: {e}")

    if do_graph:
        log("[stage] graph Fourier spectrum", verbose=verbose)
        try:
            gspec = graph_fourier_spectrum(field.points, field.vectors, k_neighbors=k_neighbors, n_eigs=graph_eigs)
            result["graph_fourier"] = gspec
        except Exception as e:
            warnings.warn(f"Graph Fourier failed: {e}")

    if do_hodge:
        log("[stage] discrete Hodge decomposition", verbose=verbose)
        try:
            node_points = coord.coords[field.layer, :, :2]
            hodge = hodge_from_token_field(node_points, field)
            result["hodge"] = hodge
            result["hodge_signed"] = hodge_signed_curl_metrics(hodge)
        except Exception as e:
            warnings.warn(f"Discrete Hodge failed: {e}")

    return result


def run_pipeline(
    model_name: str,
    text: str,
    *,
    layer: int = -1,
    reducer: Literal["pca", "umap"] = "pca",
    max_length: Optional[int] = None,
    fourier_modes: int = 32,
    fourier_backend: FourierBackend = "direct",
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu",
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    do_fourier: bool = True,
    do_graph: bool = True,
    do_hodge: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    bundle = extract_hidden_states(
        model_name,
        text,
        device=device,
        dtype=dtype,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        verbose=verbose,
    )
    result = run_pipeline_from_hidden(
        bundle.hidden,
        layer=layer,
        reducer=reducer,
        fourier_modes=fourier_modes,
        fourier_backend=fourier_backend,
        do_fourier=do_fourier,
        do_graph=do_graph,
        do_hodge=do_hodge,
        verbose=verbose,
    )
    result["hidden_bundle"] = bundle
    return result


def _format_energy(name: str, energy: Dict[str, float]) -> str:
    keys = ["grad_ratio", "curl_ratio", "harmonic_ratio"]
    bits = [f"{k}={energy.get(k, float('nan')):.4f}" for k in keys]
    return f"{name}: " + ", ".join(bits)


def _format_signed(name: str, metrics: Dict[str, float]) -> str:
    preferred = [
        "alignment",
        "signed_circulation_alignment",
        "signed_curl_alignment",
        "signed_vorticity_ratio",
        "signed_circulation_ratio",
        "signed_curl_circulation_ratio",
    ]
    bits = []
    for key in preferred:
        if key in metrics:
            bits.append(f"{key}={metrics.get(key, float('nan')):+.4f}")
    return f"{name}: " + ", ".join(bits)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fourier, graph, Hodge, and signed-circulation probes for transformer hidden-state trajectories"
    )
    parser.add_argument("--model", default="gpt2", help="HF causal LM name or local HF model directory")
    parser.add_argument(
        "--model-path",
        "--model-dir",
        dest="model_path",
        default=None,
        help="Local HF model directory. Overrides --model and implies --local-files-only.",
    )
    parser.add_argument(
        "--text",
        nargs="?",
        const=DEFAULT_TEXT,
        default=DEFAULT_TEXT,
        help="Input text. If --text is present but blank, a demo text is used.",
    )
    parser.add_argument("--text-file", default=None, help="Read input text from file")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to analyze in single-layer mode")
    parser.add_argument("--all-layers", action="store_true", help="Analyze every layer and export layer-wise metrics")
    parser.add_argument(
        "--null-models",
        default="real",
        help="Comma-separated controls for --all-layers: real,shuffle_tokens,reverse_tokens,random_hidden,all",
    )
    parser.add_argument("--csv-output", default="layer_metrics.csv", help="CSV filename/path for --all-layers")
    parser.add_argument("--output-dir", default=".", help="Directory for CSV and plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data/null models")
    parser.add_argument("--reducer", choices=["pca", "umap"], default="pca")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--fourier-modes", "--nufft-modes", dest="fourier_modes", type=int, default=32)
    parser.add_argument("--fourier-backend", choices=["direct", "finufft", "jax"], default="direct")
    parser.add_argument("--graph-eigs", type=int, default=32)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic hidden states; skips HF model loading")
    parser.add_argument("--diagnose", action="store_true", help="Print package/device diagnostics and exit")
    parser.add_argument("--no-fourier", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-hodge", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.diagnose:
        print_diagnostics()
        return 0

    verbose = not args.quiet
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.text_file:
        try:
            text_file_path = resolve_text_file_path(args.text_file)
        except (FileNotFoundError, IsADirectoryError, ValueError) as e:
            parser.error(str(e))
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    if not str(text).strip():
        text = DEFAULT_TEXT

    # Load or synthesize hidden states once. Both single-layer and all-layer
    # modes share this bundle, avoiding duplicate model forward passes.
    if args.synthetic:
        bundle = synthetic_hidden_states(seed=args.seed)
    else:
        try:
            model_ref, model_is_local = resolve_hf_model_ref(args.model, args.model_path)
        except (FileNotFoundError, ValueError) as e:
            parser.error(str(e))
        try:
            bundle = extract_hidden_states(
                model_ref,
                text,
                device=args.device,
                dtype=args.dtype,
                max_length=args.max_length,
                trust_remote_code=args.trust_remote_code,
                local_files_only=bool(args.local_files_only or model_is_local),
                verbose=verbose,
            )
        except HFModelLoadError as e:
            raise SystemExit(str(e)) from e

    hidden = bundle.hidden
    print(f"hidden shape [layers,tokens,dim] = {hidden.shape}")

    if args.all_layers:
        null_models = parse_null_models(args.null_models)
        rows, _ = run_layer_sweep_from_hidden(
            hidden,
            reducer=args.reducer,
            null_models=null_models,
            seed=args.seed,
            fourier_modes=args.fourier_modes,
            fourier_backend=args.fourier_backend,
            graph_eigs=args.graph_eigs,
            k_neighbors=args.k_neighbors,
            do_fourier=not args.no_fourier,
            do_graph=not args.no_graph,
            do_hodge=not args.no_hodge,
            verbose=verbose,
        )

        csv_path = Path(args.csv_output)
        if not csv_path.is_absolute():
            csv_path = output_dir / csv_path
        write_layer_metrics_csv(rows, csv_path)
        print(f"saved layer metrics CSV: {csv_path}")
        print_layer_sweep_summary(rows)

        if args.save_plots:
            saved = save_layer_sweep_plots(rows, output_dir)
            if saved:
                print("saved layer plots:")
                for path in saved:
                    print(f"  {path}")
            else:
                print("no layer plots saved; metrics were missing or disabled")
        return 0

    result = run_pipeline_from_hidden(
        hidden,
        layer=args.layer,
        reducer=args.reducer,
        fourier_modes=args.fourier_modes,
        fourier_backend=args.fourier_backend,
        graph_eigs=args.graph_eigs,
        k_neighbors=args.k_neighbors,
        do_fourier=not args.no_fourier,
        do_graph=not args.no_graph,
        do_hodge=not args.no_hodge,
        verbose=verbose,
    )
    result["hidden_bundle"] = bundle

    coord: CoordinateBundle = result["coordinates"]
    if coord.explained_variance_ratio is not None:
        print("PCA explained variance ratio:", np.array2string(coord.explained_variance_ratio, precision=4))

    if "spectral_helmholtz" in result:
        backend = result["fourier"].backend
        print(_format_energy(f"spectral Helmholtz ({backend})", result["spectral_helmholtz"].energy))
    if "spectral_curl_bands" in result:
        b = result["spectral_curl_bands"]
        print(
            "spectral curl bands: "
            f"low={b.get('curl_low_ratio', float('nan')):.4f}, "
            f"mid={b.get('curl_mid_ratio', float('nan')):.4f}, "
            f"high={b.get('curl_high_ratio', float('nan')):.4f}"
        )
    if "trajectory_signed" in result:
        print(_format_signed("trajectory signed circulation", result["trajectory_signed"]))
    if "trajectory_turning" in result:
        print(_format_signed("trajectory turning", result["trajectory_turning"]))
    if "local_vorticity" in result:
        print(_format_signed("local Jacobian vorticity", result["local_vorticity"]))
    if "spectral_signed" in result:
        print(_format_signed("spectral signed curl", result["spectral_signed"]))
    if "hodge" in result:
        print(_format_energy("discrete Hodge", result["hodge"].energy))
    if "hodge_signed" in result:
        print(_format_signed("hodge signed curl", result["hodge_signed"]))
    if "graph_fourier" in result:
        g = result["graph_fourier"]
        print("graph spectrum eigenvalues:", np.array2string(g.eigenvalues[:10], precision=4))
        print("graph spectrum power:", np.array2string(g.power[:10], precision=4))

    if args.save_plots:
        import matplotlib.pyplot as plt

        field: VectorFieldBundle = result["field"]
        tokens = result["hidden_bundle"].tokens
        saved = []
        fig, _ = plot_token_field(field, tokens=tokens)
        path = output_dir / "token_field.png"
        fig.savefig(path, dpi=160)
        saved.append(path)
        if "fourier" in result:
            fig, _ = plot_fourier_power(result["fourier"])
            path = output_dir / "fourier_power.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
        if "graph_fourier" in result:
            fig, _ = plot_graph_spectrum(result["graph_fourier"])
            path = output_dir / "graph_spectrum.png"
            fig.savefig(path, dpi=160)
            saved.append(path)
        plt.close("all")
        print("saved plots:")
        for path in saved:
            print(f"  {path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
