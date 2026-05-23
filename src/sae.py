"""
sae.py  —  Owner: Person A  —  Week 1

PURPOSE: Expose encode / decode / ablate as clean primitives.
causal.py and circuits.py call these — never load SAEs directly elsewhere.
Keep focused on single-sample ops; loops belong in causal.py / circuits.py.

WHAT TO IMPLEMENT
-----------------
get_sae(layer=None, device=None)
    Load and cache vit_prisma SAE for given layer (default: cfg.sae.primary_layer).
    Checkpoint path: outputs/saes/layer_{N}/sae_weights.pt
                     outputs/saes/layer_{N}/cfg.json
    Download SAE weights and place them there before calling this.

encode(activations, layer=None)
    (batch, seq_len, d_model) → (batch, seq_len, d_sae). Most values 0 (sparse).
    Internally: subtract b_dec → W_enc @ x + b_enc → ReLU → feature_acts.

decode(features, layer=None)
    (batch, seq_len, d_sae) → (batch, seq_len, d_model). Approximate reconstruction.
    Internally: W_dec @ features + b_dec.

ablate_feature(activations, feature_idx, layer=None)
    encode → clone → set feature_idx to 0 → decode → return.
    Input/output shape: (batch, seq_len, d_model)

get_l0_sparsity(activations, layer=None)
    Mean active features per token. Return: float.
    Active = feature_act > 0. Target: < cfg.sae.l0_target.

get_reconstruction_loss(activations, layer=None)
    ||x - decode(encode(x))||^2 / ||x||^2. Return: float. Target < 0.05.

TESTS (notebooks/01_sae_setup.ipynb)
  - get_sae(11) loads; sae.W_enc.shape == (d_model, d_sae)
  - encode() output shape is (batch, seq_len, d_sae) and is sparse
  - decode(encode(x)) approx reconstructs x
  - ablate_feature zeros target feature, leaves others unchanged
  - get_l0_sparsity() < cfg.sae.l0_target
  - get_reconstruction_loss() < 0.05

DEPENDENCIES: pip install vit-prisma torch
"""

from pathlib import Path

import torch
from vit_prisma.sae.sae import SparseAutoencoder

from src.config import get_config
import src.config as _cfg_mod

# --- cache: one SAE per layer, loaded on first request ---
_sae_cache: dict[int, SparseAutoencoder] = {}


def _sae_dir(layer: int) -> Path:
    repo_root = _cfg_mod._DEFAULT_CONFIG.parent.parent
    return repo_root / "outputs" / "saes" / f"layer_{layer}"


def _sae_paths(layer: int) -> tuple[Path, Path]:
    """Return (weights_path, config_path) for a given layer."""
    sae_dir = _sae_dir(layer)
    # HF repos use weights.pt + config.json; load_from_pretrained finds config.json automatically
    return sae_dir / "weights.pt", sae_dir / "config.json"


def get_sae(layer: int = None, device: str = None) -> SparseAutoencoder:
    """
    Load and cache the pre-trained SAE for the given layer.

    Weights must be present at outputs/saes/layer_{N}/weights.pt.
    If missing, run: python utils/download_saes.py --layers {N}

    Returns a vit_prisma SparseAutoencoder in eval mode.
    """
    cfg = get_config()
    if layer is None:
        layer = cfg.sae.primary_layer
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if layer in _sae_cache:
        return _sae_cache[layer]

    weights_path, _ = _sae_paths(layer)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"SAE weights not found at {weights_path}\n"
            f"Download them first: python utils/download_saes.py --layers {layer}"
        )

    # load_from_pretrained finds config.json in the same directory automatically
    sae = SparseAutoencoder.load_from_pretrained(str(weights_path))
    sae.to(device)
    sae.device = device  # sync string attribute so encode/decode move tensors to the right place
    sae.eval()

    _sae_cache[layer] = sae
    print(f"Loaded SAE layer {layer} — d_in={sae.d_in}, d_sae={sae.d_sae}")
    return sae


def encode(activations: torch.Tensor, layer: int = None) -> torch.Tensor:
    """
    Encode residual-stream activations into sparse SAE features.

    Args:
        activations: (batch, seq_len, d_model) on any device
        layer:       SAE layer; defaults to cfg.sae.primary_layer

    Returns:
        feature_acts: (batch, seq_len, d_sae) — sparse, non-negative
    """
    sae = get_sae(layer)
    activations = activations.to(sae.device).to(sae.dtype)
    # sae.encode() returns (sae_in, feature_acts) — we only need feature_acts
    _, feature_acts = sae.encode(activations)
    return feature_acts


def decode(features: torch.Tensor, layer: int = None) -> torch.Tensor:
    """
    Decode sparse SAE features back to the residual-stream space.

    Args:
        features: (batch, seq_len, d_sae)
        layer:    SAE layer; defaults to cfg.sae.primary_layer

    Returns:
        reconstruction: (batch, seq_len, d_model)
    """
    sae = get_sae(layer)
    features = features.to(sae.device).to(sae.dtype)
    return sae.decode(features)


def ablate_feature(
    activations: torch.Tensor,
    feature_idx: int,
    layer: int = None,
) -> torch.Tensor:
    """
    Zero out one SAE feature and decode back to activation space.

    The operation is:
        features = encode(activations)
        features[..., feature_idx] = 0
        return decode(features)

    Args:
        activations: (batch, seq_len, d_model)
        feature_idx: index of the feature to ablate
        layer:       SAE layer; defaults to cfg.sae.primary_layer

    Returns:
        modified activations: (batch, seq_len, d_model)
    """
    original_device = activations.device
    features = encode(activations, layer)
    features = features.clone()          # don't mutate the encoded tensor in-place
    features[..., feature_idx] = 0.0
    return decode(features, layer).to(original_device)


def get_l0_sparsity(activations: torch.Tensor, layer: int = None) -> float:
    """
    Mean number of active (non-zero) SAE features per token.

    Args:
        activations: (batch, seq_len, d_model)
        layer:       SAE layer; defaults to cfg.sae.primary_layer

    Returns:
        l0: float — target < cfg.sae.l0_target
    """
    features = encode(activations, layer)
    # features > 0 gives a bool tensor; sum over d_sae gives active count per token
    return (features > 0).float().sum(dim=-1).mean().item()


def get_reconstruction_loss(activations: torch.Tensor, layer: int = None) -> float:
    """
    Normalised reconstruction loss: ||x - decode(encode(x))||^2 / ||x||^2.

    Averaged over all tokens and batch elements.

    Args:
        activations: (batch, seq_len, d_model)
        layer:       SAE layer; defaults to cfg.sae.primary_layer

    Returns:
        loss: float — target < 0.05
    """
    features = encode(activations, layer)
    reconstruction = decode(features, layer)
    # cast back to input dtype for comparison
    activations = activations.to(reconstruction.dtype).to(reconstruction.device)
    numerator   = (activations - reconstruction).norm(dim=-1) ** 2
    denominator = activations.norm(dim=-1) ** 2
    return (numerator / denominator.clamp(min=1e-8)).mean().item()
