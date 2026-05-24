"""
evaluate.py  [Owner: Person C — Week 3]
-----------
Evaluation metrics for the circuit and SAE features.

Public API (implement these)
-----------------------------
compute_faithfulness(circuit, model, class_a_activations,
                     class_b_activations) -> dict
    Measure how faithfully the circuit explains the classification behavior.

    Steps:
        1. Compute baseline logit_diff = logit(class_a) - logit(class_b)
           on the original model.
        2. Ablate ALL nodes in the circuit simultaneously and re-run.
        3. faithfulness = (baseline - ablated_logit_diff) / baseline

    Target: cfg.circuit.faithfulness_target (>= 0.70).

    Returns a dict with:
        baseline_logit_diff:  float
        ablated_logit_diff:   float
        faithfulness:         float  (proportion of logit diff explained)
        n_nodes:              int    (number of nodes ablated)
        n_edges:              int

    Reference: Conmy et al. (2023) Section 3 for faithfulness definition.

compute_dead_feature_fraction(layer, activations) -> float
    Fraction of SAE features that never activate above
    cfg.sae.dead_feature_threshold across the provided activations.
    Used in notebook 01 to audit SAE quality.

summarise_circuit(circuit) -> dict
    Return a human-readable summary of the circuit:
        n_nodes, n_edges, nodes_per_layer, mean_edge_weight,
        top_5_nodes_by_degree (feature_idx + label),
        top_5_edges_by_weight

Implementation notes
--------------------
- compute_faithfulness requires ablating multiple features at once.
  Extend sae.ablate_feature() or add a new ablate_features() helper
  that zeros a list of (layer, feature_idx) pairs simultaneously.
  Coordinate with Person A if sae.py needs to be extended.
- summarise_circuit uses networkx graph attributes set by circuits.py.
  Read circuits.py node attribute spec before implementing.

Depends on: src/config.py, src/model.py, src/sae.py, src/circuits.py
Used by:    notebooks/04_circuit.ipynb
"""

# TODO (Person C, Week 3 Days 16–19):
#   1. Implement compute_dead_feature_fraction() — needed for notebook 01
#      audit, so do this early and share with Person A.
#   2. Implement compute_faithfulness() once circuit is available from
#      Person A (coordinate: you need the circuit by Day 17).
#   3. Implement summarise_circuit() for the gap analysis section.
#   4. If faithfulness < 0.70, document why and discuss in the report
#      (this is an informative result, not a failure).

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
