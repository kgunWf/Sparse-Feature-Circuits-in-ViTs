"""
causal.py  [Owner: Person B — CaFE locality check]

CaFE-style ERF check: compare the spatial location of maximum SAE feature
activation against the location of maximum integrated-gradient attribution.

Reference: Han, Kim, Kwak (2025). CaFE. arXiv:2509.00749.

Depends on: src/config.py, src/model.py, src/sae.py
Used by:    notebooks/03_causal_features.ipynb
"""

from __future__ import annotations

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

from src.config import get_config
from src.sae import encode, get_sae


def cafe_sanity_check(layer, feature_idx, activations, image_paths, model, top_k: int = 10,
                      attribution_method: str = "gradient", ig_steps: int = 16,
                      feature_selection: str = "index"):
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
    batch_size = cfg.cafe.batch_size
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
            "top5_agreement_rate": 0.0,
            "example_images": [],
            "attribution_method": method_label,
            "feature_selection": feature_selection,
            "results": [],
        }

    peaks = []
    top5_tokens_per_image: dict[int, list[tuple[int, int]]] = {}
    for start in range(0, activations.shape[0], batch_size):
        batch = activations[start:start + batch_size].to(device)
        with torch.no_grad():
            feature_map = encode(batch, layer)[:, 1:1 + patch_count, feature_idx].detach().cpu()
        values, positions = feature_map.max(dim=1)
        top5_k = min(5, patch_count)
        _, top5_pos = feature_map.topk(top5_k, dim=1)
        for offset, (value, position) in enumerate(zip(values, positions)):
            img_idx = start + offset
            token_offset = int(position)
            row, col = divmod(token_offset, grid_size)
            peaks.append((float(value), img_idx, 1 + token_offset, row, col))
            top5_tokens_per_image[img_idx] = [
                divmod(int(p), grid_size) for p in top5_pos[offset].tolist()
            ]

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
            top5 = top5_tokens_per_image.get(image_idx, [(act_row, act_col)])
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
                "top5_match": (grad_row, grad_col) in top5,
            })
    finally:
        for p, requires_grad in zip(params, old_requires_grad):
            p.requires_grad_(requires_grad)

    matches      = [r["matches"]     for r in results]
    top5_matches = [r["top5_match"]  for r in results]
    return {
        "activation_locations": [r["activation_location"] for r in results],
        "erf_locations": [r["erf_location"] for r in results],
        "gradient_locations": [r["gradient_location"] for r in results],
        "erf_scores": [r["erf_scores"] for r in results],
        "agreement_rate":      sum(matches)      / len(matches)      if matches else 0.0,
        "top5_agreement_rate": sum(top5_matches) / len(top5_matches) if top5_matches else 0.0,
        "example_images": [r["image_path"] for r in results],
        "attribution_method": method_label,
        "feature_selection": feature_selection,
        "ig_steps": ig_steps if method == "integrated_gradients" else None,
        "results": results,
    }


def cafe_integrated_gradients_check(*args, **kwargs):
    kwargs["attribution_method"] = "integrated_gradients"
    return cafe_sanity_check(*args, **kwargs)


