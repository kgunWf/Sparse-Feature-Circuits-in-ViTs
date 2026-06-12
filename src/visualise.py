"""
visualise.py  [Owner: Person B — Week 2-3]
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

Implementation notes
--------------------
- Use seaborn style for clean report-quality figures.
- Set figure DPI to 150 for report figures (use cfg or a local constant).

Depends on: src/config.py
Used by:    all notebooks
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

from src.config import get_config
from src.features import crop_patch_images

_FIG_DPI = 150
_VIT_BASE_LAYERS = 12

# Colour palette for the six annotation categories
CATEGORY_COLORS: dict[str, str] = {
    "texture":  "#e07b39",
    "color":    "#f5c842",
    "part":     "#4caf50",
    "scene":    "#2196f3",
    "semantic": "#9c27b0",
    "unclear":  "#9e9e9e",
}


def _rate_summary(value, key: str = "agreement_rate") -> tuple[float, float | None]:
    """Return mean and standard error from a scalar, sequence, or result mapping."""
    if isinstance(value, dict):
        value = value[key] if key in value else [
            result[key] for result in value.values()
            if isinstance(result, dict) and key in result
        ]
    values = np.asarray(value if np.iterable(value) else [value], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, None
    error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else None
    return float(values.mean()), error


def _depths(layers) -> np.ndarray:
    """Convert ViT-B layer numbers to relative depth percentages."""
    return np.asarray(list(layers), dtype=float) / _VIT_BASE_LAYERS * 100


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


def plot_cafe_comparison(cafe_results, feature_idx, save_path=None):
    """Show activation patch vs. CaFE ERF heatmap for one feature."""
    cfg = get_config()
    results = cafe_results.get("results", [])[:3]
    if not results:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=_FIG_DPI)
        ax.text(0.5, 0.5, f"No CaFE results for feature {feature_idx}", ha="center", va="center")
        ax.axis("off")
        return _save(fig, save_path)

    ps = cfg.model.patch_size
    img_size = cfg.model.image_size
    fig, axes = plt.subplots(len(results), 2, figsize=(7, 3.2 * len(results)), dpi=_FIG_DPI)
    axes = np.atleast_2d(axes)

    for row, result in enumerate(results):
        with Image.open(result["image_path"]) as img:
            img = img.convert("RGB").resize((img_size, img_size))

        act_row, act_col = result["activation_location"]
        erf_row, erf_col = result.get("erf_location", result.get("gradient_location"))
        erf_scores = np.asarray(result["erf_scores"], dtype=float)

        axes[row, 0].imshow(img)
        axes[row, 0].add_patch(Rectangle(
            (act_col * ps, act_row * ps), ps, ps,
            fill=False, edgecolor="#e53935", linewidth=2,
        ))
        axes[row, 0].set_title(f"Activation patch ({act_row}, {act_col})", fontsize=10)

        axes[row, 1].imshow(img)
        axes[row, 1].imshow(
            erf_scores,
            cmap="magma",
            alpha=0.55,
            extent=(0, img_size, img_size, 0),
            interpolation="nearest",
        )
        axes[row, 1].add_patch(Rectangle(
            (erf_col * ps, erf_row * ps), ps, ps,
            fill=False, edgecolor="#00e5ff", linewidth=2,
        ))
        axes[row, 1].set_title(f"CaFE ERF patch ({erf_row}, {erf_col})", fontsize=10)

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(
        f"Feature {feature_idx} CaFE agreement = {cafe_results['agreement_rate']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Stubs — Person B (D3/D5/D6)
# ---------------------------------------------------------------------------

def plot_locality_by_depth(agreement_by_layer: dict, model_label: str = "DINO",
                           save_path=None):
    """Fig 4 — CaFE locality agreement rate vs. relative depth (Run 1 index-based).

    Args:
        agreement_by_layer: {layer: agreement_rate} from Run 1 results.
        model_label: 'DINO' or 'CLIP'.
        save_path: optional path to save PNG.

    Returns: matplotlib Figure.

    Plots top-1 and top-5 agreement separately, with standard-error bars
    when per-feature result dictionaries are provided.
    """
    layers = sorted(agreement_by_layer)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=_FIG_DPI)
        for key, label, style in (
            ("agreement_rate", "Top-1 agreement", "o-"),
            ("top5_agreement_rate", "Top-5 agreement", "s--"),
        ):
            summaries = [_rate_summary(agreement_by_layer[layer], key) for layer in layers]
            values = np.array([mean for mean, _ in summaries])
            if np.isfinite(values).any():
                errors = [error or 0 for _, error in summaries]
                ax.errorbar(_depths(layers), values, yerr=errors, fmt=style,
                            capsize=3, linewidth=2, label=label)
        ax.set(xlabel="Relative depth (%)", ylabel="CaFE agreement rate",
               ylim=(0, 1.05), title=f"{model_label} feature locality by depth")
        ax.set_xticks(_depths(layers), [f"{depth:.0f}%" for depth in _depths(layers)])
        ax.legend()
        fig.tight_layout()
    return _save(fig, save_path)


def plot_locality_by_category(category_agreements: dict, save_path=None):
    """Fig 5 — DINO locality agreement rate by feature category, Run 2 MS-ranked.

    Args:
        category_agreements: {layer: {category: agreement_rate}} from Run 2 results.
        save_path: optional path to save PNG.

    Returns: matplotlib Figure.

    Categories use the shared palette; hatches distinguish layers.
    """
    layers = sorted(category_agreements)
    categories = list(CATEGORY_COLORS)
    x = np.arange(len(categories))
    width = 0.8 / len(layers)
    hatches = ("", "//", "xx")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=_FIG_DPI)
        for index, layer in enumerate(layers):
            values = [_rate_summary(category_agreements[layer].get(cat, np.nan))[0]
                      for cat in categories]
            ax.bar(x + (index - (len(layers) - 1) / 2) * width, values, width,
                   color=[CATEGORY_COLORS[cat] for cat in categories],
                   hatch=hatches[index % len(hatches)], edgecolor="#555555",
                   linewidth=0.5)
        ax.set_xticks(x, categories)
        ax.set(xlabel="Feature category", ylabel="CaFE agreement rate",
               ylim=(0, 1.05), title="DINO locality by category (MS-ranked sample)")
        ax.legend(handles=[Patch(facecolor="white", edgecolor="grey",
                                hatch=hatches[i % len(hatches)], label=f"Layer {layer}")
                           for i, layer in enumerate(layers)])
        fig.text(0.5, 0.01, "MS-ranked sample; not directly comparable to index-based CaFE figures.",
                 ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, save_path)


def plot_locality_comparison(dino_rates: dict, clip_rates: dict,
                              cafe_reference: dict = None, save_path=None):
    """Fig 6 ★ — Main comparison: DINO Run 1 vs. CLIP Run 3 locality by relative depth.

    Args:
        dino_rates: {layer: agreement_rate} from DINO Run 1.
        clip_rates: {layer: agreement_rate} from CLIP Run 3.
        cafe_reference: optional {relative_depth: non_locality_rate} digitized from
                        Han et al. Fig 5 — converted to agreement rates (1 - non_locality)
                        before overlaying.
        save_path: optional path to save PNG.

    Returns: matplotlib Figure.

    CaFE reference values are non-locality rates and are converted to
    agreement before plotting.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=_FIG_DPI)
        for rates, label, color in (
            (dino_rates, "DINO ViT-B/16", "#4a90d9"),
            (clip_rates, "CLIP ViT-B/32", "#e07b39"),
        ):
            layers = sorted(rates)
            values = [_rate_summary(rates[layer])[0] for layer in layers]
            ax.plot(_depths(layers), values, "o-", color=color, linewidth=2, label=label)
        if cafe_reference:
            reference_keys = sorted(cafe_reference)
            depths = np.asarray(reference_keys, dtype=float)
            if depths.max() <= 1:
                depths *= 100
            ax.plot(depths, [1 - cafe_reference[depth] for depth in reference_keys], "--",
                    color="#666666", linewidth=2,
                    label="CaFE CLIP-L/14 (est. Fig 5)")
            fig.text(0.5, 0.01, "Reference digitized from Han et al. Fig. 5; not raw data.",
                     ha="center", fontsize=8, color="#555555")
        ax.set(xlabel="Relative depth (%)", ylabel="CaFE agreement rate",
               ylim=(0, 1.05), title="Feature locality across vision transformers")
        model_depths = sorted(set(_depths(dino_rates)) | set(_depths(clip_rates)))
        ax.set_xticks(model_depths, [f"{depth:.0f}%" for depth in model_depths])
        ax.legend()
        fig.tight_layout(rect=(0, 0.04 if cafe_reference else 0, 1, 1))
    return _save(fig, save_path)


def plot_ms_locality_scatter(ms_scores: dict, agreement_rates: dict,
                              categories: dict = None, model_label: str = "DINO",
                              save_path=None):
    """Fig 8 — MS score vs. CaFE locality agreement rate, colored by feature category.

    Args:
        ms_scores:      {feat_idx: ms_score}
        agreement_rates:{feat_idx: agreement_rate}
        categories:     {feat_idx: category_str} — optional, colours points by category.
        model_label:    'DINO' or 'CLIP'.
        save_path: optional path to save PNG.

    Returns: matplotlib Figure.

    Computes Spearman statistics over finite overlapping features and adds
    a least-squares trend line when the MS scores are non-constant.
    """
    feature_ids = [
        feature_idx for feature_idx in sorted(ms_scores.keys() & agreement_rates.keys())
        if np.isfinite(ms_scores[feature_idx]) and np.isfinite(agreement_rates[feature_idx])
    ]
    if not feature_ids:
        raise ValueError("ms_scores and agreement_rates have no finite overlapping features")
    x = np.array([ms_scores[feature_idx] for feature_idx in feature_ids], dtype=float)
    y = np.array([agreement_rates[feature_idx] for feature_idx in feature_ids], dtype=float)
    rho, pvalue = spearmanr(x, y)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIG_DPI)
        if categories:
            category_names = [
                categories.get(feature_idx, "unclear") for feature_idx in feature_ids
            ]
            category_names = [name if name in CATEGORY_COLORS else "unclear"
                              for name in category_names]
            for category in CATEGORY_COLORS:
                mask = np.array([name == category for name in category_names])
                if mask.any():
                    ax.scatter(x[mask], y[mask], s=28, alpha=0.75,
                               color=CATEGORY_COLORS[category], label=category)
            ax.legend(fontsize=8, ncol=2)
        else:
            ax.scatter(x, y, s=28, alpha=0.75, color="#4a90d9")
        if len(np.unique(x)) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color="#333333", linewidth=1.5)
        ax.text(0.03, 0.97, f"Spearman rho = {rho:.2f}\np = {pvalue:.2g}\nN = {len(x)}",
                transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set(xlabel="Monosemanticity score", ylabel="CaFE agreement rate",
               ylim=(0, 1.05), title=f"{model_label}: monosemanticity vs. locality")
        fig.tight_layout()
    return _save(fig, save_path)
