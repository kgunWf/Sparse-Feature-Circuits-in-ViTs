"""
features.py  [Owner: Person B — Week 1–2]
------------------------------------------
SAE feature analysis: top-activating patch retrieval, CLIP-based concept
labeling, and the Monosemanticity Score.

Public API
----------
get_top_patches            — top-k patches for a single feature (batched)
get_top_patches_all_features — top-k patches for all features (streaming)
get_top_patches_all_features_from_cache — same, reading HDF5 in chunks
precompute_patch_embeddings — encode all unique patch crops once (fast path)
load_clip_labeler          — load CLIP model + processor
label_feature_clip         — label one feature (uses precomputed embs if given)
label_features_clip        — label all features with progress bar
compute_monosemanticity_score — MS per Pach et al. 2025, Eq. 9
crop_clip_images           — 224px centered crops for CLIP encoding
crop_patch_images          — contextual crops for visual inspection grids
Depends on: src/config.py, src/sae.py
Used by:    notebooks/02_feature_analysis.ipynb,
            notebooks/03_causal_features.ipynb
"""


from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from src.config import get_config


_clip_text_cache: dict[tuple[int, str, tuple[str, ...]], torch.Tensor] = {}


def _patch_layout(seq_len: int) -> tuple[int, int]:
    cfg = get_config()
    grid_size = cfg.model.image_size // cfg.model.patch_size
    patch_count = grid_size * grid_size
    n_registers = getattr(cfg.model, "num_registers", 0)
    expected_seq_len = 1 + patch_count + n_registers
    if seq_len == expected_seq_len:
        return grid_size, patch_count

    inferred = seq_len - 1 - n_registers
    inferred_grid = math.isqrt(max(inferred, 0))
    if inferred > 0 and inferred_grid * inferred_grid == inferred:
        return inferred_grid, inferred
    raise ValueError(
        f"Cannot map seq_len={seq_len} to square patch grid "
        f"(expected {expected_seq_len} from config)."
    )


def _token_indices(seq_len: int, token_type: str, device: torch.device) -> torch.Tensor:
    _, patch_count = _patch_layout(seq_len)
    if token_type == "patch":
        return torch.arange(1, 1 + patch_count, device=device)
    if token_type == "cls":
        return torch.tensor([0], device=device)
    if token_type == "all":
        return torch.arange(seq_len, device=device)
    raise ValueError("token_type must be one of: 'patch', 'cls', 'all'")


def _patch_coords(token_idx: int, grid_size: int, patch_count: int) -> tuple[int | None, int | None]:
    patch_offset = token_idx - 1
    if 0 <= patch_offset < patch_count:
        return divmod(patch_offset, grid_size)
    return None, None


def _entry(
    image_paths: list[str],
    image_idx: int,
    token_idx: int,
    value: float,
    grid_size: int,
    patch_count: int,
) -> dict:
    row, col = _patch_coords(token_idx, grid_size, patch_count)
    return {
        "image_path": image_paths[image_idx],
        "image_idx": image_idx,
        "token_idx": token_idx,
        "activation_value": value,
        "patch_row": row,
        "patch_col": col,
    }


def _entries_from_topk(
    values: torch.Tensor,
    positions: torch.Tensor,
    selected_tokens: torch.Tensor,
    image_paths: list[str],
    grid_size: int,
    patch_count: int,
) -> list[dict]:
    token_count = selected_tokens.numel()
    selected_tokens = selected_tokens.detach().cpu()
    results = []
    for value, position in zip(values.cpu(), positions.cpu()):
        image_idx = int(position // token_count)
        token_idx = int(selected_tokens[int(position % token_count)])
        results.append(_entry(image_paths, image_idx, token_idx, float(value), grid_size, patch_count))
    return results


def get_top_patches(
    layer,
    feature_idx,
    activations,
    image_paths,
    k=None,
    token_type="patch",
    batch_size: int = 64,
) -> list[dict]:
    """Return top-k patches for a single SAE feature.

    Streams through activations in batches of ``batch_size`` images so the full
    (n_images, seq_len, d_sae) tensor is never materialised — only one batch at
    a time, then only the single feature slice is kept.
    Peak GPU memory ≈ batch_size × seq_len × d_sae × 4 bytes
    (64 × 197 × 49152 × 4 ≈ 2.4 GB).
    """
    from src.sae import encode

    cfg = get_config()
    image_paths = list(image_paths)
    n_images = activations.shape[0]
    if n_images != len(image_paths):
        raise ValueError("activations and image_paths must have the same image count")

    k = cfg.features.top_k_patches if k is None else k
    if k <= 0:
        return []

    # Probe a single image to learn layout and validate feature_idx
    with torch.no_grad():
        probe = encode(activations[:1], layer).detach()
    seq_len = probe.shape[1]
    d_sae   = probe.shape[2]
    if not 0 <= feature_idx < d_sae:
        raise IndexError(f"feature_idx={feature_idx} outside d_sae={d_sae}")
    grid_size, patch_count = _patch_layout(seq_len)
    selected_tokens = _token_indices(seq_len, token_type, probe.device)
    n_tokens = selected_tokens.numel()
    del probe

    # Running top-k accumulators kept on CPU
    running_vals = torch.full((k,), float("-inf"))
    running_pos  = torch.zeros((k,), dtype=torch.long)

    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        with torch.no_grad():
            feats = encode(activations[start:end], layer).detach()  # (bs, seq_len, d_sae)

        # Extract only the one feature we care about → (bs * n_tokens,)
        batch_vals = feats[:, selected_tokens, feature_idx].reshape(-1).cpu()
        batch_pos  = (
            torch.arange(end - start, dtype=torch.long).unsqueeze(1) * n_tokens
            + torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)
        ).reshape(-1) + start * n_tokens  # absolute flat indices

        combined_vals = torch.cat([running_vals, batch_vals])
        combined_pos  = torch.cat([running_pos,  batch_pos])
        top_k_actual  = min(k, combined_vals.numel())
        running_vals, idx = torch.topk(combined_vals, top_k_actual)
        running_pos  = combined_pos[idx]

        del feats, batch_vals, combined_vals, combined_pos

    return _entries_from_topk(
        running_vals, running_pos, selected_tokens.cpu(),
        image_paths, grid_size, patch_count,
    )


def get_top_patches_all_features(
    layer,
    activations,
    image_paths,
    k=None,
    token_type="patch",
    batch_size: int = 64,
) -> dict[int, list[dict]]:
    """
    Compute top-k activating patches for every SAE feature.

    Uses a streaming batch loop so the full (n_images, seq_len, d_sae) tensor
    is never materialised in one go — peak GPU memory is batch_size rows instead
    of all n_images rows (e.g. 64 × 197 × 49152 × 4 ≈ 2.4 GB vs ~45 GB for 5000 images).

    Args:
        batch_size: number of images to encode per GPU batch. Reduce if OOM.
                    64 fits comfortably on a 24 GB GPU; try 16–32 on 8 GB.
    """
    from src.sae import encode

    cfg = get_config()
    image_paths = list(image_paths)
    n_images = activations.shape[0]
    if n_images != len(image_paths):
        raise ValueError("activations and image_paths must have the same image count")

    k = cfg.features.top_k_patches if k is None else k
    if k <= 0:
        return {}

    # Infer layout from a single probe batch
    probe = activations[:1]
    with torch.no_grad():
        probe_feats = encode(probe, layer).detach()
    seq_len = probe_feats.shape[1]
    d_sae   = probe_feats.shape[2]
    grid_size, patch_count = _patch_layout(seq_len)
    selected_tokens = _token_indices(seq_len, token_type, probe_feats.device)
    n_tokens = selected_tokens.numel()
    del probe_feats

    # Running top-k: shape (k, d_sae) on CPU to save GPU memory
    running_vals = torch.full((k, d_sae), float("-inf"))
    running_pos  = torch.zeros((k, d_sae), dtype=torch.long)

    for start in tqdm(range(0, n_images, batch_size), desc="Encoding batches"):
        end   = min(start + batch_size, n_images)
        batch = activations[start:end]                           # (bs, seq_len, d_model)
        with torch.no_grad():
            feats = encode(batch, layer).detach()                # (bs, seq_len, d_sae)

        # Flatten to (bs*n_tokens, d_sae) and build absolute position indices
        batch_vals = feats[:, selected_tokens, :].reshape(-1, d_sae).cpu()
        batch_pos  = (
            torch.arange(end - start, dtype=torch.long).unsqueeze(1) * n_tokens
            + torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)
        ).reshape(-1) + start * n_tokens                        # absolute flat indices

        # Merge running top-k with this batch and keep top-k
        combined_vals = torch.cat([running_vals, batch_vals], dim=0)  # (k+bs*n_tokens, d_sae)
        combined_pos  = torch.cat([
            running_pos,
            batch_pos.unsqueeze(1).expand(-1, d_sae),
        ], dim=0)

        top_k_actual = min(k, combined_vals.shape[0])
        running_vals, idx = torch.topk(combined_vals, top_k_actual, dim=0)
        running_pos = combined_pos.gather(0, idx)

        del feats, batch_vals, combined_vals, combined_pos

    # Convert flat positions back to (image_idx, token_idx) dicts
    result = {}
    for feature_idx in range(d_sae):
        result[feature_idx] = _entries_from_topk(
            running_vals[:, feature_idx],
            running_pos[:, feature_idx],
            selected_tokens.cpu(),
            image_paths,
            grid_size,
            patch_count,
        )
    return result


def load_clip_labeler(model_name=None):
    cfg = get_config()
    model_name = cfg.features.clip_model if model_name is None else model_name
    from transformers import CLIPModel, CLIPProcessor

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # use_safetensors=True avoids the torch.load CVE-2025-32434 check that
    # transformers ≥ 4.51 enforces for PyTorch < 2.6.
    # All major HuggingFace CLIP checkpoints ship .safetensors files.
    # If you point model_name at a local checkpoint that only has .bin weights,
    # set use_safetensors=False here (and accept the security caveat).
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    except Exception:
        # fallback for local-only .bin checkpoints — requires PyTorch >= 2.6
        model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, CLIPProcessor.from_pretrained(model_name)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clip_features(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    for output_item in output if isinstance(output, (list, tuple)) else ():
        if torch.is_tensor(output_item) and output_item.ndim == 2:
            return output_item
    raise TypeError(f"Expected CLIP features tensor, got {type(output).__name__}")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_space_image(image: Image.Image) -> Image.Image:
    """Resize-then-center-crop to match the model's preprocessing geometry."""
    cfg = get_config()
    size = cfg.model.image_size
    return ImageOps.fit(
        image.convert("RGB"), (size, size),
        method=Image.Resampling.BILINEAR, centering=(0.5, 0.5),
    )


def _patch_box(patch: dict) -> tuple[int, int, int, int]:
    cfg = get_config()
    left = patch["patch_col"] * cfg.model.patch_size
    top  = patch["patch_row"] * cfg.model.patch_size
    return left, top, left + cfg.model.patch_size, top + cfg.model.patch_size


def _centered_square_box(
    patch_box: tuple[int, int, int, int],
    crop_size: int,
    image_size: int,
) -> tuple[int, int, int, int]:
    crop_size = max(1, min(crop_size, image_size))
    left, top, right, bottom = patch_box
    x0 = round((left + right) / 2 - crop_size / 2)
    y0 = round((top + bottom) / 2 - crop_size / 2)
    x0 = min(max(0, x0), image_size - crop_size)
    y0 = min(max(0, y0), image_size - crop_size)
    return x0, y0, x0 + crop_size, y0 + crop_size


def _clip_image_size(processor) -> int:
    ip = getattr(processor, "image_processor", processor)
    for attr in ("crop_size", "size"):
        val = getattr(ip, attr, None)
        if isinstance(val, int) and val > 0:
            return val
        if isinstance(val, dict):
            for key in ("shortest_edge", "height", "width"):
                if val.get(key, 0) > 0:
                    return int(val[key])
    return 224


def _text_embeddings(vocab: list[str], clip_model, processor, device: torch.device) -> torch.Tensor:
    key = (id(clip_model), str(device), tuple(vocab))
    if key not in _clip_text_cache:
        text_inputs = processor(text=vocab, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = _clip_features(clip_model.get_text_features(**_to_device(text_inputs, device)))
        _clip_text_cache[key] = F.normalize(embeddings, dim=-1).cpu()
    return _clip_text_cache[key].to(device)


def precompute_patch_embeddings(
    all_top_patches: dict[int, list[dict]],
    clip_model,
    processor,
    batch_size: int = 256,
) -> dict[tuple, torch.Tensor]:
    """Precompute normalised CLIP image embeddings for every unique patch crop
    that appears across all features in ``all_top_patches``.

    Returns a ``{(image_path, patch_row, patch_col): tensor}`` lookup dict.
    Pass it to :func:`label_features_clip` as ``patch_embeddings=`` to skip
    per-feature CLIP encoding — the label step drops from ~1 hour to seconds.

    Why fast: instead of 49k separate 20-image CLIP calls (one per feature),
    this encodes each unique (image, row, col) crop exactly once in large batches,
    then :func:`label_features_clip` just averages the pre-looked-up vectors.

    ``batch_size`` controls GPU peak memory during encoding (256 ≈ safe for 24 GB).
    """
    unique: dict[tuple, dict] = {}
    for patches in all_top_patches.values():
        for p in patches:
            row, col = p.get("patch_row"), p.get("patch_col")
            if row is not None and col is not None:
                key = (p["image_path"], row, col)
                if key not in unique:
                    unique[key] = p

    keys = list(unique.keys())
    device = _model_device(clip_model)
    embeddings: dict[tuple, torch.Tensor] = {}

    clip_size = _clip_image_size(processor)
    for start in tqdm(range(0, len(keys), batch_size), desc="Precomputing patch embeddings"):
        batch_keys = keys[start : start + batch_size]
        crops = crop_clip_images([unique[k] for k in batch_keys], clip_image_size=clip_size)
        if not crops:
            continue
        image_inputs = processor(images=crops, return_tensors="pt")
        with torch.no_grad():
            embs = _clip_features(
                clip_model.get_image_features(**_to_device(image_inputs, device))
            )
        embs = F.normalize(embs.float(), dim=-1).cpu()
        for key, emb in zip(batch_keys, embs):
            embeddings[key] = emb.clone()  # clone breaks the shared batch storage (else pickle is batch_size× too large)

    print(f"Precomputed embeddings for {len(embeddings):,} unique patch crops")
    return embeddings


def label_feature_clip(
    top_patches,
    vocab,
    clip_model,
    processor,
    top_n: int = 3,
    patch_embeddings: dict | None = None,
) -> list[str]:
    """Find the top-n closest vocabulary concepts for a single SAE feature.

    If ``patch_embeddings`` (from :func:`precompute_patch_embeddings`) is supplied,
    embeddings are looked up from the cache instead of re-encoding — O(1) per patch.
    Otherwise, contextual crops (2-patch context radius, no red box) are encoded
    on the fly.
    """
    vocab = [str(item) for item in vocab]
    if not top_patches or not vocab or top_n <= 0:
        return []

    device = _model_device(clip_model)

    if patch_embeddings is not None:
        embs = [
            patch_embeddings[(p["image_path"], p["patch_row"], p["patch_col"])]
            for p in top_patches
            if p.get("patch_row") is not None
            and (p["image_path"], p["patch_row"], p["patch_col"]) in patch_embeddings
        ]
        if not embs:
            return []
        image_embedding = F.normalize(
            torch.stack(embs).mean(dim=0, keepdim=True), dim=-1
        ).to(device)
    else:
        images = crop_clip_images(top_patches, clip_image_size=_clip_image_size(processor))
        if not images:
            return []
        image_inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = _clip_features(
                clip_model.get_image_features(**_to_device(image_inputs, device))
            )
        image_embedding = F.normalize(
            F.normalize(image_embeddings, dim=-1).mean(dim=0, keepdim=True), dim=-1
        )

    text_embeddings = _text_embeddings(vocab, clip_model, processor, device)
    scores = image_embedding @ text_embeddings.T
    top_scores = torch.topk(scores.squeeze(0), k=min(top_n, len(vocab))).indices.cpu().tolist()
    return [vocab[idx] for idx in top_scores]


def compute_monosemanticity_score(
    all_top_patches: dict[int, list[dict]],
    patch_embeddings: dict[tuple, torch.Tensor],
    max_patches: int = 20,
) -> dict[int, float]:
    """Compute the Monosemanticity Score per feature (Pach et al. 2025, Eq. 9).

    MS^k = Σ_{n<m} (ã^k_n · ã^k_m · s_{nm}) / Σ_{n<m} (ã^k_n · ã^k_m)

      s_{nm}    CLIP cosine similarity between crop embeddings of patches n, m
      ã^k_n     activation of feature k for patch n, min-max normalised to [0,1]
      r^k_{nm}  ã^k_n × ã^k_m  (shared-activation relevance weight)

    ``patch_embeddings`` must come from :func:`precompute_patch_embeddings`.
    Call that function first — it encodes every unique patch crop once and
    returns a ``{(image_path, patch_row, patch_col): tensor}`` lookup dict.

    MS → 1: patches visually similar → monosemantic.
    MS → 0: patches visually diverse → polysemantic.
    nan: dead feature (< 2 activating patches with precomputed embeddings).
    """
    cfg = get_config()
    dead_threshold = cfg.sae.dead_feature_threshold
    scores: dict[int, float] = {}

    for feat_idx, patches in tqdm(all_top_patches.items(), desc="MS scores"):
        # Pach et al. 2025 compliance: exclude features whose dataset-wide maximum
        # activation never exceeds the dead-feature threshold.  The top-k patches
        # already contain the highest activations for this feature, so their max
        # equals the global max — no re-encoding needed.
        max_act = max((p["activation_value"] for p in patches), default=float("-inf"))
        if max_act <= dead_threshold:
            scores[feat_idx] = float("nan")
            continue

        valid = [
            p for p in patches
            if p.get("patch_row") is not None
            and (p["image_path"], p["patch_row"], p["patch_col"]) in patch_embeddings
        ][:max_patches]

        if len(valid) < 2:
            scores[feat_idx] = float("nan")
            continue

        acts = torch.tensor([p["activation_value"] for p in valid], dtype=torch.float32)
        a_min, a_max = acts.min(), acts.max()
        if (a_max - a_min).item() < 1e-8:
            scores[feat_idx] = 1.0
            continue
        acts_norm = (acts - a_min) / (a_max - a_min)

        emb = torch.stack([
            patch_embeddings[(p["image_path"], p["patch_row"], p["patch_col"])]
            for p in valid
        ])  # already normalised by precompute_patch_embeddings

        n = emb.shape[0]
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        r = acts_norm[idx_i] * acts_norm[idx_j]
        s = (emb[idx_i] * emb[idx_j]).sum(dim=-1)
        denom = r.sum().item()
        scores[feat_idx] = ((r * s).sum() / denom).item() if denom > 1e-10 else float("nan")

    return scores


def label_features_clip(
    patches_dict: dict[int, list[dict]],
    vocab: list[str],
    clip_model,
    processor,
    top_n: int = 3,
    feature_batch_size: int = 32,
    patch_embeddings: dict | None = None,
) -> dict[int, list[str]]:
    """CLIP-label every feature in ``patches_dict``.

    Fast path (``patch_embeddings`` given): dict lookup + average — seconds.
    Slow path: crops from ``feature_batch_size`` features are encoded in one
    GPU forward pass, giving ~32× better utilisation than one feature at a time.
    """
    vocab = [str(item) for item in vocab]
    result: dict[int, list[str]] = {}
    keys = list(patches_dict.keys())

    if patch_embeddings is not None:
        for i in tqdm(range(0, len(keys), feature_batch_size), desc="CLIP labeling"):
            for feat_idx in keys[i : i + feature_batch_size]:
                result[feat_idx] = label_feature_clip(
                    patches_dict[feat_idx], vocab, clip_model, processor,
                    top_n=top_n, patch_embeddings=patch_embeddings,
                )
        return result

    # Slow path: batch crops from multiple features into one CLIP call
    device = _model_device(clip_model)
    text_embeddings = _text_embeddings(vocab, clip_model, processor, device)
    clip_size = _clip_image_size(processor)
    top_k = min(top_n, len(vocab))

    for i in tqdm(range(0, len(keys), feature_batch_size), desc="CLIP labeling"):
        batch_keys = keys[i : i + feature_batch_size]
        all_crops: list[Image.Image] = []
        slices: list[tuple[int, int, int]] = []  # (feat_idx, start, end)

        for feat_idx in batch_keys:
            crops = crop_clip_images(patches_dict[feat_idx], clip_image_size=clip_size)
            if not crops:
                result[feat_idx] = []
                continue
            start = len(all_crops)
            all_crops.extend(crops)
            slices.append((feat_idx, start, len(all_crops)))

        if not all_crops:
            continue

        image_inputs = processor(images=all_crops, return_tensors="pt")
        with torch.no_grad():
            embs = F.normalize(
                _clip_features(clip_model.get_image_features(**_to_device(image_inputs, device))),
                dim=-1,
            )
        for feat_idx, start, end in slices:
            feat_emb = F.normalize(embs[start:end].mean(dim=0, keepdim=True), dim=-1)
            top_idxs = torch.topk(feat_emb @ text_embeddings.T, k=top_k).indices.squeeze(0).cpu().tolist()
            result[feat_idx] = [vocab[idx] for idx in top_idxs]

    return result


def crop_patch_images(
    patches: list[dict],
    context_patches: int = 2,
    mark_patch: bool = True,
) -> list[Image.Image]:
    """Crop the image region around each top patch, with optional red-box marking.

    Each crop contains the patch itself plus ``context_patches`` neighbours on
    every side (clamped at image edges).  When ``mark_patch=True`` a red
    rectangle is drawn around the centre patch so it is easy to identify.

    Returns one PIL Image per patch dict that has ``patch_row`` / ``patch_col``
    set (CLS-token entries without coords are skipped).
    """
    cfg = get_config()
    ps = cfg.model.patch_size
    img_size = cfg.model.image_size
    n_per_side = img_size // ps

    crops: list[Image.Image] = []
    for patch in patches:
        row, col = patch.get("patch_row"), patch.get("patch_col")
        if row is None or col is None:
            continue
        with Image.open(Path(patch["image_path"])) as img:
            img = img.convert("RGB").resize((img_size, img_size))

        r0 = max(0, row - context_patches)
        c0 = max(0, col - context_patches)
        r1 = min(n_per_side, row + context_patches + 1)
        c1 = min(n_per_side, col + context_patches + 1)

        crop = img.crop((c0 * ps, r0 * ps, c1 * ps, r1 * ps))

        if mark_patch:
            draw = ImageDraw.Draw(crop)
            rel_r = (row - r0) * ps
            rel_c = (col - c0) * ps
            draw.rectangle(
                [rel_c, rel_r, rel_c + ps - 1, rel_r + ps - 1],
                outline="red",
                width=2,
            )
        crops.append(crop)
    return crops


def crop_clip_images(
    patches: list[dict],
    clip_image_size: int = 224,
    context_size: int | None = None,
) -> list[Image.Image]:
    """Return crops centered on each patch token, sized for CLIP input.

    ``context_size`` controls the crop window in the model's 224×224 image
    space before upscaling to ``clip_image_size``.  Must be < image_size
    (224) — passing 224 would always return the full image because the crop
    window can't be centered when it equals the image size.

    Defaults to ``features.clip_context_size`` from config (96 px), giving a
    ~6×6 patch neighbourhood (each patch is 16 px) — large enough for CLIP to
    read texture/colour/shape without including irrelevant scene background.

    Why this matters: if context_size == image_size, _centered_square_box
    clamps both x0 and y0 to 0, so every crop is (0,0,224,224) — the full
    image — and CLIP labels the dominant ImageNet subject rather than the
    activated patch region.
    """
    cfg = get_config()
    if context_size is None:
        context_size = cfg.features.clip_context_size
    # Guard: context_size must be strictly less than image_size
    context_size = min(context_size, cfg.model.image_size - cfg.model.patch_size)
    crops: list[Image.Image] = []
    for patch in patches:
        if patch.get("patch_row") is None or patch.get("patch_col") is None:
            continue
        with Image.open(Path(patch["image_path"])) as img:
            img = _model_space_image(img)
        box = _centered_square_box(_patch_box(patch), context_size, cfg.model.image_size)
        crop = img.crop(box)
        if clip_image_size != context_size:
            crop = crop.resize((clip_image_size, clip_image_size), Image.Resampling.BILINEAR)
        crops.append(crop)
    return crops


def get_top_patches_all_features_from_cache(
    layer,
    image_paths: list[str],
    cachepath,
    k: int | None = None,
    batch_size: int = 64,
    image_count: int | None = None,
    token_type: str = "patch",
) -> dict[int, list[dict]]:
    """Like :func:`get_top_patches_all_features` but streams activations from
    the HDF5 cache in chunks instead of requiring the full tensor in RAM.

    ``image_count`` limits how many images to process (``None`` = all).
    ``batch_size`` controls GPU peak memory — same formula as
    :func:`get_top_patches_all_features`.
    """
    from src.sae import encode
    from src.cache import load_layer as _load_layer

    cfg = get_config()
    image_paths = list(image_paths)
    n_images = min(image_count, len(image_paths)) if image_count is not None else len(image_paths)
    k = cfg.features.top_k_patches if k is None else k
    if k <= 0:
        return {}

    # Probe a single image to learn seq_len, d_sae, and patch layout
    probe = _load_layer(layer, indices=[0], cachepath=str(cachepath))
    with torch.no_grad():
        probe_feats = encode(probe, layer).detach()
    seq_len      = probe_feats.shape[1]
    d_sae        = probe_feats.shape[2]
    grid_size, patch_count = _patch_layout(seq_len)
    selected_tokens = _token_indices(seq_len, token_type, probe_feats.device)
    n_tokens = selected_tokens.numel()
    del probe_feats, probe

    running_vals = torch.full((k, d_sae), float("-inf"))
    running_pos  = torch.zeros((k, d_sae), dtype=torch.long)

    for start in tqdm(range(0, n_images, batch_size), desc="Encoding batches (from cache)"):
        end  = min(start + batch_size, n_images)
        acts = _load_layer(layer, indices=list(range(start, end)), cachepath=str(cachepath))
        with torch.no_grad():
            feats = encode(acts, layer).detach()

        batch_vals = feats[:, selected_tokens, :].reshape(-1, d_sae).cpu()
        batch_pos  = (
            torch.arange(end - start, dtype=torch.long).unsqueeze(1) * n_tokens
            + torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)
        ).reshape(-1) + start * n_tokens

        combined_vals = torch.cat([running_vals, batch_vals], dim=0)
        combined_pos  = torch.cat([
            running_pos,
            batch_pos.unsqueeze(1).expand(-1, d_sae),
        ], dim=0)

        top_k_actual = min(k, combined_vals.shape[0])
        running_vals, idx = torch.topk(combined_vals, top_k_actual, dim=0)
        running_pos  = combined_pos.gather(0, idx)

        del feats, acts, batch_vals, combined_vals, combined_pos

    result: dict[int, list[dict]] = {}
    for feature_idx in range(d_sae):
        result[feature_idx] = _entries_from_topk(
            running_vals[:, feature_idx],
            running_pos[:, feature_idx],
            selected_tokens.cpu(),
            image_paths,
            grid_size,
            patch_count,
        )
    return result
