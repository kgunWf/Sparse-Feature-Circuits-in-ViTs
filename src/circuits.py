"""
circuits.py  [Owner: Person A — Week 3]
-----------
Sparse feature circuit construction.

Takes the causally important features from causal.py and measures
pairwise causal connections between them across layers to form edges.
The result is a directed graph where nodes are SAE features (concepts)
and edges are causal influences.

Reference:
    Marks et al. (2024). Sparse Feature Circuits: Discovering and Editing
    Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
    Read Section 3 carefully before implementing.

Public API (implement these)
-----------------------------
measure_pairwise_edges(layer_a, features_a, layer_b, features_b,
                       activations_a, activations_b,
                       class_a_images, class_b_images,
                       model) -> torch.Tensor
    For each pair (f_i in layer_a, f_j in layer_b), measure whether
    ablating f_i causes a significant change in the activation of f_j.

    Edge weight = how much does f_j's activation change when we zero
    f_i and re-run the model from layer_a to layer_b?

    features_a: list of feature indices in layer_a (from causal.py)
    features_b: list of feature indices in layer_b (from causal.py)

    Returns a (len(features_a), len(features_b)) matrix of effect sizes.

    Implementation notes:
    - For each f_i in features_a:
        1. Ablate f_i at layer_a using sae.ablate_feature().
        2. Re-run the model from layer_a onwards using Prisma hooks.
        3. Encode layer_b's resulting activations with the SAE.
        4. Effect size = mean |activation(f_j) before - after| over images.
    - Process in batches (cfg.causal.logit_diff_batch_size).
    - Limit to top ~20 features per layer — this keeps the matrix small.

build_circuit(edge_matrix, features_a, features_b,
              feature_labels_a, feature_labels_b,
              layer_a, layer_b, percentile=None) -> networkx.DiGraph
    Threshold edge_matrix and build a directed graph.
    Keep edges above cfg.circuit.edge_effect_percentile.

    Node attributes: feature_idx (int), layer (int), label (str)
    Edge attributes: effect_size (float)

    Coordinate node attribute names with Person B before implementing
    so visualise.py can read them without changes.

build_full_circuit(layer_pairs, ...) -> networkx.DiGraph
    Call build_circuit for each pair in cfg.circuit.layer_pairs
    and merge into one graph.

save_circuit(graph, path=None)
    Save to cfg.outputs.circuit_path as JSON (networkx node-link format).

load_circuit(path=None) -> networkx.DiGraph
    Load a previously saved circuit from JSON.

Depends on: src/config.py, src/model.py, src/sae.py, src/causal.py
Used by:    src/evaluate.py, src/visualise.py,
            notebooks/04_circuit.ipynb
"""

# TODO (Person A, Week 3 Days 15–18):
#   1. Implement measure_pairwise_edges() — test 5x5 before scaling to 20x20.
#   2. Implement build_circuit() and build_full_circuit().
#   3. Implement save_circuit() / load_circuit().
#   4. Share node attribute format with Person B by Day 17.
