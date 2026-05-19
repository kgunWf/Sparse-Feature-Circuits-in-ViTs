"""
visualise.py  [Owner: Person B — Week 2–3]
------------
All plotting functions. Every figure in the report should be
produced by a function in this file — no inline plotting in notebooks.

Each function should:
  1. Accept a save_path argument (default None).
  2. If save_path is provided, save the figure there.
  3. Return the matplotlib Figure object so notebooks can display inline.

Public API (implement these)
-----------------------------
plot_feature_gallery(top_patches_dict, feature_indices,
                     save_path=None) -> Figure
    Grid of top-activating patches for a set of SAE features.
    Rows = features, Columns = top patches (cfg.features.top_k_patches).
    Annotate each row with the CLIP label.
    Used in notebook 02 to visually inspect feature quality.

plot_monosemanticity_distribution(scores_dict, layer,
                                  save_path=None) -> Figure
    Histogram of Monosemanticity Scores across all features at a layer.
    Mark the median and 80th percentile with vertical lines.

plot_layer_evolution(category_counts_per_layer,
                     save_path=None) -> Figure
    Stacked bar chart showing how concept category composition
    (texture/color/part/scene/semantic/unclear) changes across
    layers 6, 9, 11.
    category_counts_per_layer: dict {layer: {category: count}}

plot_ablation_ranking(importance_scores, feature_labels,
                      top_n=20, layer=None,
                      save_path=None) -> Figure
    Horizontal bar chart of the top_n most causally important features
    at a given layer, labelled with their CLIP concept strings.

plot_cafe_comparison(cafe_results, feature_idx,
                     save_path=None) -> Figure
    Side-by-side visualisation for a single feature:
        Left:  image with activation location highlighted (patch box)
        Right: image with gradient attribution heatmap
    Used to illustrate the CaFE sanity check.

plot_circuit(circuit, save_path=None) -> Figure
    Directed graph visualisation of the sparse feature circuit.
    Nodes: positioned by layer (x-axis) and importance (y-axis).
    Node color: concept category.
    Node label: CLIP concept string (short).
    Edge width: proportional to effect_size.
    Use networkx + matplotlib. Keep it readable — truncate long labels.

    Read circuits.py for the node attribute spec (feature_idx, layer,
    label) before implementing.

Implementation notes
--------------------
- Use seaborn style for clean report-quality figures.
- Set figure DPI to 150 for report figures (use cfg or a local constant).
- For plot_circuit: if the graph has > 30 nodes, consider a hierarchical
  layout (nodes arranged in columns by layer) rather than spring layout,
  which becomes unreadable at that scale.
- Coordinate with Person A on node attribute names from circuits.py
  before implementing plot_circuit.

Depends on: src/config.py
Used by:    all notebooks
"""

# TODO (Person B):
#   Week 2:
#   1. Implement plot_feature_gallery() — needed for notebook 02.
#   2. Implement plot_monosemanticity_distribution().
#   3. Implement plot_layer_evolution().
#   4. Implement plot_ablation_ranking() — needed for notebook 03.
#   5. Implement plot_cafe_comparison().
#
#   Week 3:
#   6. Implement plot_circuit() once node attribute format is confirmed
#      with Person A (circuits.py). Target: Day 17.
