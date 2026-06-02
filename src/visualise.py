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

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.config import get_config
from src.features import crop_patch_images

_FIG_DPI = 150

# Colour palette for the six annotation categories
CATEGORY_COLORS: dict[str, str] = {
    "texture":  "#e07b39",
    "color":    "#f5c842",
    "part":     "#4caf50",
    "scene":    "#2196f3",
    "semantic": "#9c27b0",
    "unclear":  "#9e9e9e",
}


def _save(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    return fig


def plot_monosemanticity_distribution(
    scores_dict: dict[int, float],
    layer: int,
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of MS scores with median and 80th-pct vertical markers."""
    values = sorted(v for v in scores_dict.values() if not np.isnan(v))
    dead   = sum(1 for v in scores_dict.values() if np.isnan(v))
    median = float(np.median(values))
    p80    = float(np.percentile(values, 80))

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=_FIG_DPI)
        ax.hist(values, bins=60, color="#4a90d9", edgecolor="white", linewidth=0.4)
        ax.axvline(median, color="#e07b39", linewidth=1.8, linestyle="--",
                   label=f"Median  {median:.3f}")
        ax.axvline(p80,    color="#9c27b0", linewidth=1.8, linestyle=":",
                   label=f"80th pct  {p80:.3f}")
        ax.set_xlabel("Monosemanticity Score (Pach et al. 2025, Eq. 9)", fontsize=11)
        ax.set_ylabel("Feature count", fontsize=11)
        ax.set_title(
            f"MS distribution — Layer {layer}"
            f"  ({len(values):,} active, {dead:,} dead/NaN)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

    return _save(fig, save_path)


def make_patch_grid(
    feature_patches: dict[int, list[dict]],
    labels: dict[int, list[str]] | None = None,
    context_patches: int = 2,
    crop_size: int = 128,
    max_patches: int | None = None,
) -> Image.Image:
    """Render a visual inspection grid — one row per feature.

    Each row shows the feature index + CLIP labels in a left panel, then
    ``max_patches`` image crops with the active patch outlined in red.

    Args:
        feature_patches: ``{feature_idx: [patch dicts]}`` from
            :func:`src.features.get_top_patches_all_features`.
        labels: optional ``{feature_idx: ["label1", ...]}`` from
            :func:`src.features.label_features_clip`.
        context_patches: neighbour patches to include around the active patch.
        crop_size: pixel size to resize each crop to in the output image.
        max_patches: max patches per row (defaults to ``cfg.features.top_k_patches``).

    Returns:
        A PIL Image suitable for ``display()`` in a Jupyter cell.
    """
    cfg = get_config()
    _max = max_patches or cfg.features.top_k_patches
    labels = labels or {}
    items = list(feature_patches.items())
    if not items:
        return Image.new("RGB", (320, 80), "white")

    label_w = 260
    row_h = crop_size + 42
    grid = Image.new("RGB", (label_w + _max * crop_size, row_h * len(items)), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, (feat_idx, patches) in enumerate(items):
        y = row_idx * row_h
        title = f"feature {feat_idx}"
        if feat_idx in labels:
            title += " | " + ", ".join(labels[feat_idx])
        draw.text((8, y + 8), title, fill="black")
        acts_str = ", ".join(f"{p['activation_value']:.2f}" for p in patches[:_max])
        draw.text((8, y + 28), "acts: " + acts_str, fill="black")

        for col_idx, crop in enumerate(
            crop_patch_images(patches[:_max], context_patches=context_patches, mark_patch=True)
        ):
            grid.paste(crop.resize((crop_size, crop_size)), (label_w + col_idx * crop_size, y))

    return grid


def plot_feature_gallery(
    all_top_patches: dict[int, list[dict]],
    feature_indices: list[int],
    labels: dict[int, list[str]] | None = None,
    scores: dict[int, float] | None = None,
    context_patches: int = 2,
    crop_size: int = 128,
    max_patches: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Matplotlib figure wrapping make_patch_grid for the top-50 gallery.

    Delegates crop rendering to :func:`make_patch_grid` and embeds the
    resulting PIL image in a matplotlib figure so it can be saved at report
    quality and displayed inline in Jupyter.

    Args:
        all_top_patches: full top-patches dict from get_top_patches_all_features.
        feature_indices: ordered list of feature indices to display (e.g. top-50).
        labels: optional {feat_idx: [label1, label2, label3]}.
        scores: optional {feat_idx: ms_score} — appended to row annotations.
        context_patches: neighbour patches around the active patch in each crop.
        crop_size: pixel size of each crop in the output grid.
        max_patches: patches per feature row (defaults to cfg.features.top_k_patches).
        save_path: if given, save as PNG.
    """
    labels = labels or {}
    scores = scores or {}

    # Annotate labels with MS score suffix
    annotated_labels = {
        fi: (labels.get(fi, []) + [f"MS={scores[fi]:.3f}"])
        if fi in scores else labels.get(fi, [])
        for fi in feature_indices
    }

    subset = {fi: all_top_patches[fi] for fi in feature_indices if fi in all_top_patches}
    grid_img = make_patch_grid(
        subset,
        labels=annotated_labels,
        context_patches=context_patches,
        crop_size=crop_size,
        max_patches=max_patches,
    )

    w, h = grid_img.size
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(grid_img)
    ax.axis("off")
    fig.tight_layout(pad=0)

    return _save(fig, save_path)


def plot_layer_evolution(
    category_counts_per_layer: dict[int, dict[str, int]],
    save_path: str | None = None,
) -> plt.Figure:
    """Stacked bar chart of concept category composition across layers.

    Args:
        category_counts_per_layer: {layer: {category: count}}
            e.g. {4: {"texture": 12, "color": 5, ...}, 9: {...}}
    """
    layers = sorted(category_counts_per_layer)
    categories = list(CATEGORY_COLORS)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIG_DPI)
        bottoms = np.zeros(len(layers))
        x = np.arange(len(layers))

        for cat in categories:
            vals = np.array([
                category_counts_per_layer[l].get(cat, 0) for l in layers
            ], dtype=float)
            ax.bar(x, vals, bottom=bottoms, label=cat,
                   color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.5)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.set_ylabel("Feature count", fontsize=11)
        ax.set_title("Concept category evolution across layers", fontsize=13,
                     fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        fig.tight_layout()

    return _save(fig, save_path)


def plot_ablation_ranking(
    importance_scores: dict[int, float],
    feature_labels: dict[int, list[str]],
    top_n: int = 20,
    layer: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N causally important features.

    Args:
        importance_scores: {feature_idx: logit_diff_change} from ablation loop.
        feature_labels: {feature_idx: [label1, ...]} from label_features_clip.
        top_n: how many features to display.
        layer: layer number for the title (optional).
    """
    ranked = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_n]
    values = [importance_scores[fi] for fi in ranked]
    names  = [
        f"f{fi}  " + " / ".join(feature_labels.get(fi, ["?"])[:2])
        for fi in ranked
    ]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(8, top_n * 0.38 + 1), dpi=_FIG_DPI)
        y = np.arange(len(ranked))
        ax.barh(y, values, color="#4a90d9", edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|Δ logit diff| (flamingo − spoonbill)", fontsize=10)
        title = f"Top-{top_n} causal features"
        if layer is not None:
            title += f" — Layer {layer}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()

    return _save(fig, save_path)


# --- Week 3 stubs (implement once circuit format confirmed with Person A) ---

def plot_cafe_comparison(cafe_results, feature_idx, save_path=None):
    raise NotImplementedError("plot_cafe_comparison — implement in Week 2 (Person B)")


def plot_circuit(circuit, save_path=None):
    raise NotImplementedError("plot_circuit — implement in Week 3 after circuits.py spec confirmed")
