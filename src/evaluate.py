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
