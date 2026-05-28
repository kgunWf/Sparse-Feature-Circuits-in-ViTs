"""
features.py  [Owner: Person B — Week 1–2]
------------
SAE feature analysis: top-activating patch retrieval, CLIP-based
automated concept labeling, and the Monosemanticity Score.

Reads activations via src/cache.py and encodes via src/sae.py.
No direct model loading or HDF5 access in this file.

Public API (implement these)
-----------------------------
get_top_patches(layer, feature_idx, activations, image_paths,
                k=None, token_type="patch") -> list[dict]
    Return the k image patches that most strongly activate a given
    SAE feature.

    activations: torch.Tensor shape (n_images, seq_len, d_model)
                 loaded from cache.load_layer()
    image_paths: corresponding list of image file paths
    token_type:  "patch"  — exclude CLS and register tokens
                 "cls"    — CLS token only
                 "all"    — all tokens

    Token layout is derived from cfg.model. Current DINO v1 ViT-B/16
    uses seq_len=197 with no registers:
        index 0          = CLS token
        index 1..196     = patch tokens  (224px image, 16px patches → 14×14 grid)

    Each returned dict should contain:
        image_path, image_idx, token_idx,
        activation_value, patch_row, patch_col

get_top_patches_all_features(layer, activations, image_paths,
                              k=None, token_type="patch")
    -> dict[int, list[dict]]
    Run get_top_patches for every SAE feature at a given layer.
    Returns {feature_idx: [top patch dicts]}.
    Encode activations ONCE outside the per-feature loop (not per call).
    Use tqdm for progress.

load_clip_labeler(model_name=None) -> (model, processor)
    Load CLIP model and processor for concept labeling.
    model_name defaults to cfg.features.clip_model.

label_feature_clip(top_patches, vocab, clip_model, processor,
                   top_n=3) -> list[str]
    Given a feature's top_patches list, crop those patches from
    their source images and find the closest vocabulary concepts
    by cosine similarity in CLIP embedding space.
    Returns the top_n concept strings.

    Patch crop formula (pixels):
        top    = patch_row * cfg.model.patch_size
        left   = patch_col * cfg.model.patch_size
        bottom = top  + cfg.model.patch_size
        right  = left + cfg.model.patch_size

    Steps:
        1. Load + crop each image using the patch_row/col.
        2. Encode cropped patches with CLIP image encoder.
        3. Encode vocabulary strings with CLIP text encoder
           (cache these — they don't change between features).
        4. Cosine similarity between mean patch embedding and text embeddings.
        5. Return top_n concept strings.

compute_monosemanticity_score(top_labels_per_feature) -> dict[int, float]
    Compute the Monosemanticity Score for a set of features.
    Follow the metric definition from Pach et al. (NeurIPS 2025) Section 3.2.
    Read the paper before implementing — the exact formula matters
    for comparison against their reported numbers.
    Returns {feature_idx: score} where score in [0, 1].

Implementation notes
--------------------
- Normalise CLIP embeddings to unit norm before cosine similarity.
- Cache CLIP text embeddings for the vocabulary after the first call.
- For get_top_patches_all_features: encode all activations once,
  then slice per feature — do not re-call encode() in a loop.

Depends on: src/config.py, src/sae.py
Used by:    notebooks/02_feature_analysis.ipynb,
            notebooks/03_causal_features.ipynb
"""

# Done (Person B, Week 1):
#   1. get_top_patches() and get_top_patches_all_features().
#   2. load_clip_labeler() and label_feature_clip().
#   3. Verified on the local layer 9 cache, SAE weights, and source images.
#
# TODO (Person B, Week 2):
#   4. Implement compute_monosemanticity_score() per Pach et al.
#   5. Run on all features at cfg.sae.primary_layer; produce score distribution.
#   6. Inspect top 50 features manually; annotate with categories
#      (texture/color/part/scene/semantic/unclear) and save to
#      report/notes/feature_catalog_layer{cfg.sae.primary_layer}.md


from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
) -> list[dict]:
    from src.sae import encode

    cfg = get_config()
    image_paths = list(image_paths)
    if activations.shape[0] != len(image_paths):
        raise ValueError("activations and image_paths must have the same image count")

    k = cfg.features.top_k_patches if k is None else k
    if k <= 0:
        return []

    with torch.no_grad():
        feature_acts = encode(activations, layer).detach()

    if not 0 <= feature_idx < feature_acts.shape[-1]:
        raise IndexError(f"feature_idx={feature_idx} outside d_sae={feature_acts.shape[-1]}")

    grid_size, patch_count = _patch_layout(feature_acts.shape[1])
    selected_tokens = _token_indices(feature_acts.shape[1], token_type, feature_acts.device)
    values = feature_acts[:, selected_tokens, feature_idx].reshape(-1)
    top_k = min(k, values.numel())
    top_values, top_positions = torch.topk(values, top_k)
    return _entries_from_topk(top_values, top_positions, selected_tokens, image_paths, grid_size, patch_count)


def get_top_patches_all_features(layer, activations, image_paths, k=None, token_type="patch") -> dict[int, list[dict]]:
    from src.sae import encode

    cfg = get_config()
    image_paths = list(image_paths)
    if activations.shape[0] != len(image_paths):
        raise ValueError("activations and image_paths must have the same image count")

    k = cfg.features.top_k_patches if k is None else k
    if k <= 0:
        return {}

    with torch.no_grad():
        feature_acts = encode(activations, layer).detach()

    grid_size, patch_count = _patch_layout(feature_acts.shape[1])
    selected_tokens = _token_indices(feature_acts.shape[1], token_type, feature_acts.device)
    values = feature_acts[:, selected_tokens, :].reshape(-1, feature_acts.shape[-1])
    top_k = min(k, values.shape[0])
    top_values, top_positions = torch.topk(values, top_k, dim=0)
    result = {}
    for feature_idx in tqdm(range(values.shape[-1]), desc="Top patches per feature"):
        result[feature_idx] = _entries_from_topk(
            top_values[:, feature_idx],
            top_positions[:, feature_idx],
            selected_tokens,
            image_paths,
            grid_size,
            patch_count,
        )
    return result


def load_clip_labeler(model_name=None):
    cfg = get_config()
    model_name = cfg.features.clip_model if model_name is None else model_name
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def crop_patch_images(top_patches, context_patches: int = 0, mark_patch: bool = False) -> list[Image.Image]:
    """Return image crops for patch-token entries, optionally with local context."""
    cfg = get_config()
    crops = []
    for patch in top_patches:
        if patch.get("patch_row") is None or patch.get("patch_col") is None:
            continue
        with Image.open(Path(patch["image_path"])) as image:
            image = image.convert("RGB").resize((cfg.model.image_size, cfg.model.image_size))
            left = patch["patch_col"] * cfg.model.patch_size
            top = patch["patch_row"] * cfg.model.patch_size
            right = left + cfg.model.patch_size
            bottom = top + cfg.model.patch_size
            margin = max(0, int(context_patches)) * cfg.model.patch_size
            crop_box = (
                max(0, left - margin),
                max(0, top - margin),
                min(cfg.model.image_size, right + margin),
                min(cfg.model.image_size, bottom + margin),
            )
            crop = image.crop(crop_box).copy()
            if mark_patch:
                draw = ImageDraw.Draw(crop)
                mark_box = (
                    left - crop_box[0],
                    top - crop_box[1],
                    right - crop_box[0],
                    bottom - crop_box[1],
                )
                draw.rectangle(mark_box, outline="red", width=2)
            crops.append(crop)
    return crops


def _text_embeddings(vocab: list[str], clip_model, processor, device: torch.device) -> torch.Tensor:
    key = (id(clip_model), str(device), tuple(vocab))
    if key not in _clip_text_cache:
        text_inputs = processor(text=vocab, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = _clip_features(clip_model.get_text_features(**_to_device(text_inputs, device)))
        _clip_text_cache[key] = F.normalize(embeddings, dim=-1).cpu()
    return _clip_text_cache[key].to(device)


def label_feature_clip(top_patches, vocab, clip_model, processor, top_n=3) -> list[str]:
    cfg = get_config()
    vocab = [str(item) for item in vocab]
    if not top_patches or not vocab or top_n <= 0:
        return []

    crops = crop_patch_images(top_patches)
    if not crops:
        raise ValueError("label_feature_clip requires patch-token entries with patch_row/patch_col")

    device = _model_device(clip_model)
    image_inputs = processor(images=crops, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = _clip_features(clip_model.get_image_features(**_to_device(image_inputs, device)))
    image_embedding = F.normalize(F.normalize(image_embeddings, dim=-1).mean(dim=0, keepdim=True), dim=-1)
    text_embeddings = _text_embeddings(vocab, clip_model, processor, device)
    scores = image_embedding @ text_embeddings.T
    top_scores = torch.topk(scores.squeeze(0), k=min(top_n, len(vocab))).indices.cpu().tolist()
    return [vocab[idx] for idx in top_scores]

def compute_monosemanticity_score(
    layer: int,
    activations,
    image_paths: list,
    clip_model,
    processor,
    batch_size: int = 16,
) -> dict:
    """
    Monosemanticity Score per Pach et al. (NeurIPS 2025) metric.py.
    MS(f) = sum_{i<j} a_i*a_j*cos(e_i,e_j) / sum_{i<j} a_i*a_j
    Dead features return float("nan").
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm.auto import tqdm
    from src.sae import encode

    image_paths = list(image_paths)
    n_images = activations.shape[0]
    device = next(clip_model.parameters()).device

    # Step 1 — SAE encode in small batches to avoid OOM, keep on CPU
    cls_acts_list = []
    for i in range(0, n_images, batch_size):
        batch = activations[i:i+batch_size].cuda()
        with torch.no_grad():
            feat = encode(batch, layer)   # (B, seq_len, d_sae)
        cls_acts_list.append(feat[:, 0, :].cpu().float())
        del batch, feat
        torch.cuda.empty_cache()
    cls_acts = torch.cat(cls_acts_list, dim=0)   # (n_images, d_sae)

    # Min-max normalise per feature
    min_vals = cls_acts.min(dim=0, keepdim=True).values
    max_vals = cls_acts.max(dim=0, keepdim=True).values
    cls_acts_norm = (cls_acts - min_vals) / (max_vals - min_vals).clamp(min=1e-8)

    # Step 2 — CLIP embeddings via vision_model (avoids BaseModelOutputWithPooling bug)
    all_embeddings = []
    for start in range(0, n_images, batch_size):
        batch_paths = image_paths[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model.vision_model(**inputs)
            embs = out.pooler_output   # always a plain tensor (B, 768)
        all_embeddings.append(F.normalize(embs, dim=-1).cpu().float())
        for img in images:
            img.close()
        del inputs, out, embs
        torch.cuda.empty_cache()
    embeddings = torch.cat(all_embeddings, dim=0)   # (n_images, embed_dim)

    # Step 3 — Pairwise similarity entirely on CPU
    d_sae = cls_acts_norm.shape[1]
    weighted_sim_sum = torch.zeros(d_sae)
    weight_sum       = torch.zeros(d_sae)

    for i in tqdm(range(n_images), desc=f"MS score layer {layer}"):
        for j_start in range(i + 1, n_images, batch_size):
            j_end = min(j_start + batch_size, n_images)
            emb_i = embeddings[i]
            emb_j = embeddings[j_start:j_end]
            act_i = cls_acts_norm[i]
            act_j = cls_acts_norm[j_start:j_end]
            cos_ij = F.cosine_similarity(
                emb_i.unsqueeze(0).expand(j_end - j_start, -1),
                emb_j, dim=1,
            )
            weights = act_i.unsqueeze(0) * act_j
            weighted_sim_sum += (weights * cos_ij.unsqueeze(1)).sum(dim=0)
            weight_sum       += weights.sum(dim=0)

    scores = torch.where(
        weight_sum > 0,
        weighted_sim_sum / weight_sum,
        torch.tensor(float("nan")),
    )
    return {int(idx): float(scores[idx]) for idx in range(d_sae)}
