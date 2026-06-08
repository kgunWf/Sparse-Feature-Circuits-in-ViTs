"""SAE quality evaluation helpers."""

from src.config import get_config
from src.sae import encode
import torch

def compute_dead_feature_fraction(layer: int, activations: torch.Tensor, batch_size: int = 50) -> float:
    """
    Fraction of SAE features that never activate above
    cfg.sae.dead_feature_threshold across the provided activations.
    Processes in batches to avoid OOM.
    """
    cfg = get_config()
    threshold = cfg.sae.dead_feature_threshold

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    max_activation = None

    for i in range(0, len(activations), batch_size):
        batch = activations[i:i + batch_size].to(device)
        with torch.no_grad():
            features = encode(batch, layer)  # (batch, seq_len, d_sae)

        # max per feature across batch and tokens
        batch_max = features.max(dim=0).values.max(dim=0).values  # (d_sae,)

        if max_activation is None:
            max_activation = batch_max.cpu()
        else:
            max_activation = torch.max(max_activation, batch_max.cpu())

        del batch, features, batch_max
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    dead = (max_activation <= threshold).sum().item()
    total = max_activation.shape[0]
    return dead / total
