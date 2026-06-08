"""
causal.py  [Owner: Person A (ablation loop) + Person B (CaFE check) — Week 2]
----------
Causal feature importance analysis.

Two distinct responsibilities in this file — coordinate before coding:
  Person A owns: compute_feature_importance(), get_top_causal_features()
  Person B owns: cafe_sanity_check()

Public API
----------

--- Person A ---

compute_feature_importance(layer, class_a_activations,
                           class_b_activations, model) -> torch.Tensor
    Two-pass approach:

    Pass 1 — gradient pre-ranking (O(n_batches), independent of d_sae):
        For each image batch, set requires_grad on SAE feature activations,
        decode back to residual stream, inject via hook, forward + backward.
        Accumulates |∂logit_diff/∂feat| × |feat| per feature over all batches.
        Ranks all d_sae features without a single ablation call.

    Pass 2 — exact ablation (O(grad_top_k × n_batches)):
        Ablates only the top cfg.causal.grad_top_k candidates from Pass 1.
        Measures exact logit_diff drop for each candidate.

    Returns a 1D tensor of shape (d_sae,) — non-zero only for ablated features.

get_top_causal_features(importance_scores, layer,
                        percentile=None) -> list[int]
    Return indices above cfg.causal.importance_percentile of the ablated
    candidates. percentile argument overrides config if provided.

--- Person B ---

cafe_sanity_check(layer, feature_idx, activations,
                  image_paths, model) -> dict
    CaFE-style comparison: compare the spatial location of maximum SAE
    activation (top patch) against the location of maximum gradient
    attribution (causally responsible patch).

    Reference: Han, Kim, Kwak (2025). CaFE. arXiv:2509.00749.

    Steps:
        1. Encode activations → SAE feature map (n, seq_len).
        2. Per image: find patch token with max activation → activation location.
        3. Per image: compute gradient of the selected token's feature
           activation w.r.t. input pixels via torch.autograd. Map gradient
           magnitudes to the patch grid → ERF location.
        4. Spatial agreement rate over top-k images.

    Returns dict:
        activation_locations  list of (row, col) per image
        gradient_locations    list of (row, col) per image
        agreement_rate        float in [0, 1]
        example_images        list of image paths for visualisation

    Note: gradient computation requires model inputs with requires_grad=True
    and a full forward pass from pixels — not from cached activations.

Depends on: src/config.py, src/model.py, src/sae.py, src/cache.py
Used by:    notebooks/03_causal_features.ipynb
"""

from __future__ import annotations

import math

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

from src.config import get_config
from src.sae import encode, decode, ablate_feature, get_sae


def compute_feature_importance(
    layer: int,
    class_a_activations: torch.Tensor,
    class_b_activations: torch.Tensor,
    model,
) -> torch.Tensor:
    """
    Measure causal importance of every SAE feature at ``layer`` for the
    flamingo-vs-spoonbill logit difference.

    importance[f] = mean_over_images(logit_diff_original - logit_diff_ablated_f)

    Positive score: feature promotes flamingo over spoonbill.
    Negative score: feature promotes spoonbill over flamingo.
    Zero: feature was not in the top-grad_top_k candidates (not ablated).

    Returns: torch.Tensor shape (d_sae,)
    """
    cfg = get_config()
    batch_size    = cfg.causal.logit_diff_batch_size
    grad_top_k    = cfg.causal.grad_top_k
    flamingo_idx  = cfg.data.flamingo_logit_idx
    spoonbill_idx = cfg.data.spoonbill_logit_idx
    hook_key      = f"blocks.{layer}.hook_resid_post"
    device        = next(model.parameters()).device
    img_shape     = (3, cfg.model.image_size, cfg.model.image_size)

    all_acts = torch.cat([class_a_activations, class_b_activations], dim=0)
    n_total  = all_acts.shape[0]

    with torch.no_grad():
        d_sae = encode(all_acts[:1].to(device), layer).shape[2]

    # ── Pass 1: gradient pre-ranking ─────────────────────────────────────────
    # One forward+backward per batch ranks all d_sae features simultaneously.
    # Importance proxy: |∂logit_diff/∂feat| × |feat|, averaged over tokens/images.
    grad_scores = torch.zeros(d_sae)

    for i in tqdm(range(0, n_total, batch_size), desc="Pass 1 — gradient ranking"):
        batch = all_acts[i:i + batch_size].to(device)

        with torch.no_grad():
            feat = encode(batch, layer)             # (bs, seq_len, d_sae)
        feat_v = feat.detach().requires_grad_(True)
        recon  = decode(feat_v, layer)              # (bs, seq_len, d_model) — has grad_fn

        def _hook(*_, _r=recon, **__):                # default-arg captures current recon
            return _r                                 # **__ absorbs the hook= kwarg

        logits    = model.run_with_hooks(
            torch.zeros(batch.shape[0], *img_shape, device=device),
            fwd_hooks=[(hook_key, _hook)],
        )
        logit_diff = (logits[:, 0, flamingo_idx] - logits[:, 0, spoonbill_idx]).mean()
        logit_diff.backward()

        with torch.no_grad():
            grad_scores += (feat_v.grad.abs() * feat.abs()).mean(dim=(0, 1)).cpu()

        del feat, feat_v, recon, logits, logit_diff

    grad_scores /= math.ceil(n_total / batch_size)
    top_k_idx = torch.topk(grad_scores, k=min(grad_top_k, d_sae)).indices.tolist()

    # ── Pass 2: exact ablation on top-K candidates ────────────────────────────
    # Compute baseline logit diffs once — reused for every candidate.
    baseline_diffs = []
    for i in range(0, n_total, batch_size):
        batch = all_acts[i:i + batch_size].to(device)
        def _hook(*_, _b=batch, **__):
            return _b
        with torch.no_grad():
            logits = model.run_with_hooks(
                torch.zeros(batch.shape[0], *img_shape, device=device),
                fwd_hooks=[(hook_key, _hook)],
            )
        baseline_diffs.append(
            (logits[:, 0, flamingo_idx] - logits[:, 0, spoonbill_idx]).cpu()
        )
    baseline_diffs = torch.cat(baseline_diffs)      # (n_total,)

    importance = torch.zeros(d_sae)

    for f in tqdm(top_k_idx, desc=f"Pass 2 — ablation (top {grad_top_k})"):
        ablated_diffs = []
        for i in range(0, n_total, batch_size):
            batch   = all_acts[i:i + batch_size].to(device)
            ablated = ablate_feature(batch, f, layer)
            def _hook(*_, _a=ablated, **__):
                return _a
            with torch.no_grad():
                logits = model.run_with_hooks(
                    torch.zeros(batch.shape[0], *img_shape, device=device),
                    fwd_hooks=[(hook_key, _hook)],
                )
            ablated_diffs.append(
                (logits[:, 0, flamingo_idx] - logits[:, 0, spoonbill_idx]).cpu()
            )
        ablated_diffs  = torch.cat(ablated_diffs)
        importance[f] = (baseline_diffs - ablated_diffs).mean()

    return importance


def get_top_causal_features(
    importance: torch.Tensor,
    layer: int,
    percentile: int | None = None,
) -> list[int]:
    """Return feature indices above the importance_percentile threshold.

    Percentile is computed over non-zero entries only (features that were
    ablated in Pass 2). Features not in grad_top_k have importance == 0
    and are always excluded.
    """
    cfg = get_config()
    pct = percentile if percentile is not None else cfg.causal.importance_percentile

    nonzero_mask = importance.abs() > 0
    if not nonzero_mask.any():
        return []

    threshold = torch.quantile(importance.abs()[nonzero_mask], pct / 100.0)
    return (importance.abs() >= threshold).nonzero(as_tuple=True)[0].tolist()


def cafe_sanity_check(layer, feature_idx, activations, image_paths, model, top_k: int = 10,
                      attribution_method: str = "gradient", ig_steps: int = 16):
    """CaFE-style ERF check for the top activations of one SAE feature.

    CaFE treats each feature activation at a specific token as the target and
    attributes that scalar back to input patches.  By default we use vanilla
    input gradients, with the selected token's residual skip detached at the
    target layer so the ERF is not just the activated patch's identity path.
    """
    from src.model import preprocess_image

    cfg = get_config()
    device = next(model.parameters()).device
    hook_key = f"blocks.{layer}.hook_resid_post"
    patch_size = cfg.model.patch_size
    grid_size = cfg.model.image_size // patch_size
    patch_count = grid_size * grid_size
    batch_size = cfg.causal.logit_diff_batch_size
    image_paths = list(image_paths)

    if activations.shape[0] != len(image_paths):
        raise ValueError("activations and image_paths must have the same image count")
    method = attribution_method.lower()
    if method in {"gradient", "gradients", "input_gradient", "vanilla_gradient"}:
        method = "gradient"
    elif method in {"ig", "integrated_gradient", "integrated_gradients"}:
        method = "integrated_gradients"
        if ig_steps <= 0:
            raise ValueError("ig_steps must be positive")
    else:
        raise ValueError("attribution_method must be 'gradient' or 'integrated_gradients'")
    method_label = "integrated_gradients" if method == "integrated_gradients" else "vanilla_gradient"

    if top_k <= 0:
        return {
            "activation_locations": [],
            "erf_locations": [],
            "gradient_locations": [],
            "erf_scores": [],
            "agreement_rate": 0.0,
            "example_images": [],
            "attribution_method": method_label,
            "results": [],
        }

    peaks = []
    for start in range(0, activations.shape[0], batch_size):
        batch = activations[start:start + batch_size].to(device)
        with torch.no_grad():
            feature_map = encode(batch, layer)[:, 1:1 + patch_count, feature_idx].detach().cpu()
        values, positions = feature_map.max(dim=1)
        for offset, (value, position) in enumerate(zip(values, positions)):
            token_offset = int(position)
            row, col = divmod(token_offset, grid_size)
            peaks.append((float(value), start + offset, 1 + token_offset, row, col))

    selected = sorted(peaks, reverse=True)[:min(top_k, len(peaks))]
    sae = get_sae(layer)
    params = list(model.parameters()) + list(sae.parameters())
    old_requires_grad = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    results = []
    try:
        for activation_value, image_idx, token_idx, act_row, act_col in tqdm(selected, desc="CaFE"):
            pixels = preprocess_image(image_paths[image_idx]).to(device).detach()

            def _input_grad(inputs):
                inputs = inputs.detach().requires_grad_(True)
                resid_pre = None
                objective = None

                def _pre_hook(resid, *_, **__):
                    nonlocal resid_pre
                    resid_pre = resid
                    return resid

                def _post_hook(resid, *_, **__):
                    nonlocal objective
                    if resid_pre is None:
                        raise RuntimeError(f"Hook {hook_key.replace('post', 'pre')} did not run")
                    target_resid = resid.clone()
                    target_resid[:, token_idx] = (
                        resid[:, token_idx]
                        - resid_pre[:, token_idx]
                        + resid_pre[:, token_idx].detach()
                    )
                    feats = encode(target_resid, layer)
                    objective = feats[:, token_idx, feature_idx].sum()
                    return resid

                model.run_with_hooks(
                    inputs,
                    fwd_hooks=[
                        (f"blocks.{layer}.hook_resid_pre", _pre_hook),
                        (hook_key, _post_hook),
                    ],
                )
                if objective is None:
                    raise RuntimeError(f"Hook {hook_key} did not run")
                return torch.autograd.grad(objective, inputs)[0]

            if method == "integrated_gradients":
                baseline = torch.zeros_like(pixels)
                total_grad = torch.zeros_like(pixels)
                for alpha in torch.linspace(0, 1, ig_steps + 1, device=device, dtype=pixels.dtype)[1:]:
                    total_grad += _input_grad(baseline + alpha * (pixels - baseline))
                saliency = ((pixels - baseline) * total_grad / ig_steps).detach().abs().sum(dim=1)[0]
            else:
                saliency = _input_grad(pixels).detach().abs().sum(dim=1)[0]

            saliency = saliency.float().cpu()
            patch_scores = saliency.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            patch_scores = patch_scores.mean(dim=(-1, -2))
            grad_row, grad_col = divmod(int(patch_scores.argmax()), grid_size)
            results.append({
                "image_path": image_paths[image_idx],
                "image_idx": image_idx,
                "token_idx": token_idx,
                "activation_location": (act_row, act_col),
                "erf_location": (grad_row, grad_col),
                "gradient_location": (grad_row, grad_col),
                "activation_value": activation_value,
                "gradient_value": float(patch_scores[grad_row, grad_col]),
                "erf_scores": patch_scores.tolist(),
                "matches": (act_row, act_col) == (grad_row, grad_col),
            })
    finally:
        for p, requires_grad in zip(params, old_requires_grad):
            p.requires_grad_(requires_grad)

    matches = [r["matches"] for r in results]
    return {
        "activation_locations": [r["activation_location"] for r in results],
        "erf_locations": [r["erf_location"] for r in results],
        "gradient_locations": [r["gradient_location"] for r in results],
        "erf_scores": [r["erf_scores"] for r in results],
        "agreement_rate": sum(matches) / len(matches) if matches else 0.0,
        "example_images": [r["image_path"] for r in results],
        "attribution_method": method_label,
        "ig_steps": ig_steps if method == "integrated_gradients" else None,
        "results": results,
    }


def cafe_integrated_gradients_check(*args, **kwargs):
    kwargs["attribution_method"] = "integrated_gradients"
    return cafe_sanity_check(*args, **kwargs)
