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

get_top_patches_all_features_from_cache(layer, image_paths, cachepath=None,
                                        k=None, token_type="patch",
                                        batch_size=2, image_count=None)
    Streaming version for the full 5k cache. Loads cache rows in small
    batches and keeps only top-k entries per SAE feature.

load_clip_labeler(model_name=None) -> (model, processor)
    Load CLIP model and processor for concept labeling.
    model_name defaults to cfg.features.clip_model.

load_concept_vocab(path=None, size=10000, extra_terms=None) -> list[str]
    Load one concept per line from a vocab file or build a 10k English
    vocabulary via wordfreq, with project-specific visual terms prepended.

label_feature_clip(top_patches, vocab, clip_model, processor,
                   top_n=3, clip_image_size=None) -> list[str]
    Given a feature's top_patches list, make CLIP-sized crops
    centered on those patch locations and find the closest vocabulary concepts
    by cosine similarity in CLIP embedding space.
    Returns the top_n concept strings.

label_features_clip(top_patches_by_feature, vocab, clip_model, processor,
                    top_n=3, clip_image_size=None,
                    feature_batch_size=4) -> dict[int, list[str]]
    Batched version for checkpoint runs. Labels each feature with top_n
    vocabulary concepts while reusing cached CLIP text embeddings.

    Steps:
        1. Load each source image into the same 224px model-space used by DINO.
        2. Crop a CLIP-sized window around patch_row/col, not a 16x16 patch.
        3. Encode crops with CLIP image encoder.
        4. Encode vocabulary strings with CLIP text encoder
           (cache these — they don't change between features).
        5. Cosine similarity between mean image embedding and text embeddings.
        6. Return top_n concept strings.

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
- For full-cache runs: stream image batches and keep top-k per feature;
  do not materialize the full (images, tokens, d_sae) activation tensor.

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
from collections.abc import Iterable

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

_DEFAULT_VISUAL_TERMS = [
    "bird", "beak", "wing", "feather", "eye", "head", "leg",
    "grass", "water", "sky", "tree", "branch", "leaf", "rock",
    "white", "pink", "black", "brown", "yellow", "red",
    "stripe", "spot", "edge", "texture", "background",
]

_VOCAB_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "with", "you", "your",
}


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


def _batched_topk_entries(
    layer,
    activation_batches: Iterable[tuple[int, torch.Tensor]],
    image_paths,
    k=None,
    token_type="patch",
) -> dict[int, list[dict]]:
    from src.sae import encode

    cfg = get_config()
    image_paths = list(image_paths)
    k = cfg.features.top_k_patches if k is None else int(k)
    if k <= 0:
        return {}

    top_values = None
    top_positions = None
    selected_tokens = None
    grid_size = None
    patch_count = None
    token_count = None
    d_sae = None
    total_positions_seen = 0

    for batch_start, activations in tqdm(activation_batches, desc="Streaming top patches"):
        if activations.shape[0] == 0:
            continue
        if batch_start < 0 or batch_start + activations.shape[0] > len(image_paths):
            raise ValueError("activation batch range is outside image_paths")

        with torch.no_grad():
            feature_acts = encode(activations, layer).detach()

        if selected_tokens is None:
            grid_size, patch_count = _patch_layout(feature_acts.shape[1])
            selected_tokens = _token_indices(feature_acts.shape[1], token_type, feature_acts.device)
            token_count = int(selected_tokens.numel())
            d_sae = int(feature_acts.shape[-1])
        elif feature_acts.shape[-1] != d_sae:
            raise ValueError(f"SAE width changed from {d_sae} to {feature_acts.shape[-1]}")

        values = feature_acts[:, selected_tokens, :].reshape(-1, feature_acts.shape[-1])
        batch_top_k = min(k, values.shape[0])
        batch_values, batch_positions = torch.topk(values, batch_top_k, dim=0)
        batch_positions = batch_positions.to(torch.long)
        batch_image_offsets = batch_positions // token_count
        batch_token_offsets = batch_positions % token_count
        batch_global_positions = (int(batch_start) + batch_image_offsets) * token_count + batch_token_offsets

        batch_values = batch_values.cpu()
        batch_global_positions = batch_global_positions.cpu()
        if top_values is None:
            top_values = batch_values
            top_positions = batch_global_positions
        else:
            candidate_values = torch.cat([top_values, batch_values], dim=0)
            candidate_positions = torch.cat([top_positions, batch_global_positions], dim=0)
            keep_k = min(k, candidate_values.shape[0])
            kept_values, kept_rows = torch.topk(candidate_values, keep_k, dim=0)
            kept_positions = torch.gather(candidate_positions, 0, kept_rows)
            top_values = kept_values
            top_positions = kept_positions

        total_positions_seen += int(values.shape[0])
        del feature_acts, values, batch_values, batch_positions, batch_global_positions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if top_values is None or top_positions is None or selected_tokens is None:
        return {}

    selected_tokens = selected_tokens.detach().cpu()
    result = {}
    for feature_idx in tqdm(range(top_values.shape[-1]), desc="Materializing top patch entries"):
        result[feature_idx] = _entries_from_topk(
            top_values[:, feature_idx],
            top_positions[:, feature_idx],
            selected_tokens,
            image_paths,
            int(grid_size),
            int(patch_count),
        )
    expected_positions = len(image_paths) * int(token_count)
    if total_positions_seen != expected_positions:
        print(f"Warning: ranked {total_positions_seen:,} token positions, expected {expected_positions:,}.")
    return result


def get_top_patches_all_features_from_cache(
    layer,
    image_paths,
    cachepath=None,
    k=None,
    token_type="patch",
    batch_size: int = 2,
    image_count: int | None = None,
) -> dict[int, list[dict]]:
    """Stream cached activations and keep top-k patch entries for every SAE feature."""
    from src.cache import load_layer

    image_paths = list(image_paths)
    if image_count is None:
        image_count = len(image_paths)
    image_count = min(int(image_count), len(image_paths))
    batch_size = max(1, int(batch_size))

    def batches():
        for start in range(0, image_count, batch_size):
            end = min(start + batch_size, image_count)
            yield start, load_layer(layer, indices=range(start, end), cachepath=cachepath)

    return _batched_topk_entries(
        layer=layer,
        activation_batches=batches(),
        image_paths=image_paths[:image_count],
        k=k,
        token_type=token_type,
    )


def _clean_vocab_term(term: str) -> str | None:
    term = str(term).strip().lower().replace("_", " ")
    term = " ".join(term.split())
    if len(term) < 2 or term in _VOCAB_STOPWORDS:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyz -'")
    if any(ch not in allowed for ch in term):
        return None
    if not any(ch.isalpha() for ch in term):
        return None
    return term


def _dedupe_terms(terms: Iterable[str], limit: int) -> list[str]:
    vocab = []
    seen = set()
    for term in terms:
        cleaned = _clean_vocab_term(term)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        vocab.append(cleaned)
        if len(vocab) >= limit:
            break
    return vocab


def load_concept_vocab(path=None, size: int = 10_000, extra_terms=None) -> list[str]:
    """Load a concept vocabulary from a text file or wordfreq's English word list."""
    size = int(size)
    if size <= 0:
        return []

    terms = list(extra_terms or _DEFAULT_VISUAL_TERMS)
    if path is not None:
        with open(Path(path), "r", encoding="utf-8") as f:
            terms.extend(line.split("#", 1)[0] for line in f)
    else:
        try:
            from wordfreq import top_n_list
        except ImportError as exc:
            raise ImportError(
                "load_concept_vocab(size=10000) needs the optional wordfreq package. "
                "Install requirements.txt or pass a vocab text file via path=..."
            ) from exc

        n_candidates = max(size * 4, size + 5_000)
        try:
            terms.extend(top_n_list("en", n_candidates, ascii_only=True))
        except TypeError:
            terms.extend(top_n_list("en", n_candidates))

    vocab = _dedupe_terms(terms, size)
    if len(vocab) < size:
        raise ValueError(f"Only built {len(vocab)} vocabulary terms; requested {size}.")
    return vocab


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


def _model_space_image(image: Image.Image) -> Image.Image:
    """Match preprocess_image's resize-then-center-crop geometry in PIL space."""
    cfg = get_config()
    size = int(cfg.model.image_size)
    return ImageOps.fit(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )


def _patch_box(patch: dict) -> tuple[int, int, int, int]:
    cfg = get_config()
    left = int(patch["patch_col"]) * int(cfg.model.patch_size)
    top = int(patch["patch_row"]) * int(cfg.model.patch_size)
    return left, top, left + int(cfg.model.patch_size), top + int(cfg.model.patch_size)


def _centered_square_box(
    patch_box: tuple[int, int, int, int],
    crop_size: int,
    image_size: int,
) -> tuple[int, int, int, int]:
    crop_size = max(1, min(int(crop_size), int(image_size)))
    left, top, right, bottom = patch_box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(center_x - crop_size / 2)
    crop_top = round(center_y - crop_size / 2)
    crop_left = min(max(0, crop_left), image_size - crop_size)
    crop_top = min(max(0, crop_top), image_size - crop_size)
    return crop_left, crop_top, crop_left + crop_size, crop_top + crop_size


def _draw_patch_marker(
    crop: Image.Image,
    patch_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = patch_box
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (
            left - crop_box[0],
            top - crop_box[1],
            right - crop_box[0],
            bottom - crop_box[1],
        ),
        outline="red",
        width=2,
    )


def crop_patch_images(top_patches, context_patches: int = 0, mark_patch: bool = False) -> list[Image.Image]:
    """Return patch-token crops for galleries, optionally with local context."""
    cfg = get_config()
    crops = []
    for patch in top_patches:
        if patch.get("patch_row") is None or patch.get("patch_col") is None:
            continue
        with Image.open(Path(patch["image_path"])) as image:
            image = _model_space_image(image)
            patch_box = _patch_box(patch)
            left, top, right, bottom = patch_box
            margin = max(0, int(context_patches)) * int(cfg.model.patch_size)
            crop_box = (
                max(0, left - margin),
                max(0, top - margin),
                min(int(cfg.model.image_size), right + margin),
                min(int(cfg.model.image_size), bottom + margin),
            )
            crop = image.crop(crop_box).copy()
            if mark_patch:
                _draw_patch_marker(crop, patch_box, crop_box)
            crops.append(crop)
    return crops


def crop_clip_images(top_patches, clip_image_size: int = 224) -> list[Image.Image]:
    """Return CLIP-sized crops centered on patch-token entries."""
    cfg = get_config()
    clip_image_size = max(1, int(clip_image_size))
    crops = []
    for patch in top_patches:
        if patch.get("patch_row") is None or patch.get("patch_col") is None:
            continue
        with Image.open(Path(patch["image_path"])) as image:
            image = _model_space_image(image)
            crop_box = _centered_square_box(
                _patch_box(patch),
                int(clip_image_size),
                int(cfg.model.image_size),
            )
            crop = image.crop(crop_box).copy()
            if crop.size != (clip_image_size, clip_image_size):
                crop = crop.resize((clip_image_size, clip_image_size), Image.Resampling.BILINEAR)
            crops.append(crop)
    return crops


def _square_image_size(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and value:
        return int(min(value))
    if isinstance(value, dict):
        height = value.get("height") or value.get("shortest_edge")
        width = value.get("width") or value.get("shortest_edge") or height
        if height is not None and width is not None:
            return int(min(height, width))
    return None


def _clip_image_size(processor) -> int:
    image_processor = getattr(processor, "image_processor", processor)
    for attr in ("crop_size", "size"):
        size = _square_image_size(getattr(image_processor, attr, None))
        if size is not None and size > 0:
            return size
    return 224


def _text_embeddings(vocab: list[str], clip_model, processor, device: torch.device) -> torch.Tensor:
    key = (id(clip_model), str(device), tuple(vocab))
    if key not in _clip_text_cache:
        text_inputs = processor(text=vocab, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = _clip_features(clip_model.get_text_features(**_to_device(text_inputs, device)))
        _clip_text_cache[key] = F.normalize(embeddings, dim=-1).cpu()
    return _clip_text_cache[key].to(device)


def label_feature_clip(top_patches, vocab, clip_model, processor, top_n=3, clip_image_size=None) -> list[str]:
    vocab = [str(item) for item in vocab]
    if not top_patches or not vocab or top_n <= 0:
        return []

    clip_image_size = _clip_image_size(processor) if clip_image_size is None else int(clip_image_size)
    crops = crop_clip_images(top_patches, clip_image_size=clip_image_size)
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


def label_features_clip(
    top_patches_by_feature,
    vocab,
    clip_model,
    processor,
    top_n=3,
    clip_image_size=None,
    feature_batch_size: int = 4,
) -> dict[int, list[str]]:
    """Label many features with CLIP, batching crop encoding across features."""
    vocab = [str(item) for item in vocab]
    feature_items = [(int(feature_idx), patches) for feature_idx, patches in top_patches_by_feature.items()]
    if not feature_items:
        return {}
    if not vocab or top_n <= 0:
        return {feature_idx: [] for feature_idx, _ in feature_items}

    clip_image_size = _clip_image_size(processor) if clip_image_size is None else int(clip_image_size)
    feature_batch_size = max(1, int(feature_batch_size))
    device = _model_device(clip_model)
    text_embeddings = _text_embeddings(vocab, clip_model, processor, device)
    top_k = min(int(top_n), len(vocab))
    labels: dict[int, list[str]] = {}

    for start in tqdm(range(0, len(feature_items), feature_batch_size), desc="CLIP labels"):
        batch_items = feature_items[start:start + feature_batch_size]
        crops = []
        feature_slices = []
        for feature_idx, patches in batch_items:
            feature_crops = crop_clip_images(patches, clip_image_size=clip_image_size)
            if not feature_crops:
                labels[feature_idx] = []
                continue
            crop_start = len(crops)
            crops.extend(feature_crops)
            feature_slices.append((feature_idx, crop_start, len(crops)))

        if not crops:
            continue

        image_inputs = processor(images=crops, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = _clip_features(
                clip_model.get_image_features(**_to_device(image_inputs, device))
            )
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            feature_embeddings = []
            feature_indices = []
            for feature_idx, crop_start, crop_end in feature_slices:
                embedding = F.normalize(
                    image_embeddings[crop_start:crop_end].mean(dim=0, keepdim=True),
                    dim=-1,
                )
                feature_embeddings.append(embedding)
                feature_indices.append(feature_idx)

            feature_embeddings = torch.cat(feature_embeddings, dim=0)
            scores = feature_embeddings @ text_embeddings.T
            top_indices = torch.topk(scores, k=top_k, dim=-1).indices.cpu().tolist()

        for feature_idx, indices in zip(feature_indices, top_indices):
            labels[feature_idx] = [vocab[idx] for idx in indices]

        del crops, image_inputs, image_embeddings, feature_embeddings, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return labels
