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

    Token layout for DINOv2-reg4:
        index 0          = CLS token
        index 1..196     = patch tokens  (224px image, 14px patches)
        index 197..200   = register tokens

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

# TODO (Person B):
#   Week 1 (Days 3–5):
#   1. Implement get_top_patches() and get_top_patches_all_features().
#   2. Implement load_clip_labeler() and label_feature_clip().
#   3. Manual sanity check on 10 features: do the top patches look
#      visually coherent? Do CLIP labels match what you see?
#
#   Week 2:
#   4. Implement compute_monosemanticity_score() per Pach et al.
#   5. Run on all features at layer 11; produce score distribution.
#   6. Inspect top 50 features manually; annotate with categories
#      (texture/color/part/scene/semantic/unclear) and save to
#      report/notes/feature_catalog_layer11.md
