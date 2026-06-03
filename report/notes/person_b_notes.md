# Person B — Running Notes

## Week 1 — Feature analysis + CLIP labeling

### Decisions

- [x] Concept vocabulary is explicit input to `label_feature_clip` and `label_features_clip`.
      Lives in `utils/clip_vocab.py` (not `features.py`) as a 4-tier structured vocab:
      Tier 1 textures/colors/shapes (~600), Tier 2 body parts (~400), Tier 3 scenes (~300),
      Tier 4 ImageNet-1k or filtered 21k labels (~1k–3.9k). Notebook uses `tier4_subset="1k"`
      for a cleaner ~2.3k vocab with no 21k noise.
- [x] CLIP labels use resized model-space images and CLIP-sized crops centered on the
      top-activating patch (`context_size=96` px in 224px space → ~6×6 patch neighbourhood),
      not raw 16×16 patch crops. Passing `context_size=224` returns the full image every time
      because `_centered_square_box` clamps to (0,0,224,224); the guard in `crop_clip_images`
      enforces `context_size <= image_size - patch_size`.
- [x] Token layout is config-derived: CLS at index 0, patch tokens 1..196 for DINO v1
      ViT-B/16 (14×14 grid, no register tokens). `_patch_layout` infers grid size from
      `seq_len` and validates against config.
- [x] Full-cache all-feature top-patch retrieval streams over activation cache in batches
      of `batch_size` images (default 64), keeping only `(k, d_sae)` running top-k
      accumulators on CPU. The full `(n_images, seq_len, d_sae)` tensor is never materialised
      (~45 GB for 5k images at d_sae=49152; streaming peak is ~2.4 GB GPU).
- [x] `make_patch_grid` lives in `src/visualise.py`, not `features.py`. The split:
      PIL composite image (visualise) vs. matplotlib Figure (visualise.plot_feature_gallery).
      `crop_patch_images` and `crop_clip_images` remain in `features.py` as data-layer helpers.
- [x] `compute_monosemanticity_score` takes `patch_embeddings` as a required positional
      argument (no `clip_model` / `processor`). The slow CLIP-encoding path is removed.
      Caller must run `precompute_patch_embeddings` first.
- [x] Two thresholds kept strictly separate in `configs/default.yaml`:
      `sae.dead_feature_threshold: 0.01` — algorithmic, marks truly dormant features in MS score.
      `features.min_activation_threshold: 1.0` — analytical, gates reliable CLIP labels/gallery.
      `features.ms_max_patches: 5` — top-5 patches per feature for Pach et al. Eq. 9
      (wider distribution than top-20; patches 6-20 dilute the pairwise similarity signal).

### Efficiency and caching design

The pipeline produces four independently cacheable artifacts under `outputs/features/`:

| Artifact | File | Reuse condition |
|---|---|---|
| Top-k patches | `top_patches_layer{N}_full.pkl.gz` | Rebuild if `top_k_patches`, layer, or dataset changes |
| Patch embeddings | In-memory only (`patch_embeddings` dict) | Recomputed each kernel session from the pkl; ~20 min on A100 |
| CLIP labels | `clip_labels_layer{N}_full.json` | Rebuild if vocab or `top_n` changes; seconds with precomputed embeddings |
| MS scores | `ms_scores_layer{N}_top{K}.json` | Rebuild if `ms_max_patches` changes; seconds with precomputed embeddings |

**Why `patch_embeddings` is not persisted to disk:** The dict maps `(image_path, row, col)` tuples to 512-d float32 tensors. For 5k images × 196 patch tokens = ~980k entries × 2 KB each ≈ 1.9 GB. Saving and loading this is slower than re-encoding (~20 min) and the pkl.gz already holds the top-k patch metadata needed to reconstruct it. Callers that need it across sessions should pickle it themselves.

**Why top-patches pkl uses gzip:** The raw dict of 49k × 20 patch-dicts is ~200 MB uncompressed. gzip brings it to ~15–20 MB, which matters on Colab where `/content` disk is limited.

**Cross-machine path portability:** The pkl stores absolute image paths from the machine where it was built (e.g. `/home/gunaydin/...`). The notebook remaps paths by filename on load:
```python
_name_to_local = {Path(p).name: p for p in image_paths}
for _p in all_top_patches.values():
    if not Path(_p["image_path"]).exists():
        _p["image_path"] = _name_to_local.get(Path(_p["image_path"]).name, _p["image_path"])
```
This is notebook-level logic (not in `features.py`) because it depends on the local `image_paths` list built during cache loading.

**`precompute_patch_embeddings` is the critical fast path:** Without it, `label_features_clip` and `compute_monosemanticity_score` would each do ~49k × 20 = 980k CLIP image encoder calls (≈1 hour each). With it, all unique `(image, row, col)` crops are encoded exactly once in batches of 256, then both label and MS score steps run in seconds via dict lookup.

### Findings

- Implemented top-patch retrieval for single features and all features.
  The all-feature path keeps only top-k entries per feature while scanning cached
  activations in small batches; accumulators live on CPU between batches.
- Implemented CLIP label loading and patch-crop concept labeling with cached text embeddings.
  `_text_embeddings` caches by `(model_id, device, vocab_tuple)` — switching vocab or device
  invalidates the cache automatically.
- Week 1 notebook runs the all-feature top-patch scan on all cached images and caches
  the result under `outputs/features/`. CLIP labeling and MS scores also cached there.
- MS score uses Pach et al. 2025 Eq. 9 exactly: pairwise CLIP image similarity weighted
  by min-max-normalised activation products. Features with max activation ≤ 0.01 get `nan`
  (dead). Features with < 2 patches in `patch_embeddings` also get `nan`.
- Feature gallery (top 50 by MS) filtered by both `min_activation_threshold` and
  `ms_max_patches` support to avoid the MS=1.0 artefact spike from low-support features.

### Remaining checks

- Manual annotation of top-50 gallery: assign texture / color / part / scene / semantic / unclear.
  Save to `report/notes/feature_catalog_layer9.md`.
- Repeat top-patch retrieval and CLIP labeling for layers 4 and 6 (Week 2 dependency).
- Confirm CLIP labels are sensible for ≥ 60% of inspected features before treating as
  first-pass semantic annotations.

---

## Week 2 — Multi-layer ablation + CaFE check

### Decisions

- [x] Reused Person A's two-pass ablation loop
      implementation. Notebook 03 now iterates over `cfg.sae.target_layers`
      (`[4, 6, 9]`) and calls `compute_feature_importance()` once per layer.
- [x] Store per-layer artifacts under `outputs/features/`:
      `importance_layer{N}.pt` for dense importance tensors and
      `importance_ranking_layer{N}.json` for reviewable sorted rankings.
- [x] Keep `importance` and `top_features` as backwards-compatible aliases for
      `cfg.sae.primary_layer` so downstream primary-layer CaFE/circuit cells still run.
- [x] CaFE sanity check follows the core paper idea: explain a specific SAE feature
      activation at a specific patch token by attributing that scalar back to input
      patches. For Week 2, the backend is input gradients rather than full AttnLRP.
- [x] CaFE uses fresh image forward passes with `pixels.requires_grad_(True)`.
      Cached residual activations are only used to select the top activation events;
      they cannot provide pixel-level attribution.
- [x] CaFE results are saved per feature as JSON in `outputs/features/cafe/`, and
      activation-vs-ERF comparison figures are saved in `report/figures/`.

### Findings

- Notebook 03 now extracts residual-stream activations for all target SAE layers in
  one pass per image, storing `act_flamingo_by_layer[layer]` and
  `act_spoonbill_by_layer[layer]`.
- The ablation-ranking cell now produces one plot per layer:
  `ablation_ranking_layer4.png`, `ablation_ranking_layer6.png`, and
  `ablation_ranking_layer9.png`.
- `cafe_sanity_check()` now returns activation locations, ERF/gradient locations,
  per-example ERF heatmaps, agreement rate, image paths, and detailed per-example rows.
- `plot_cafe_comparison()` overlays the CaFE ERF heatmap on the image and marks:
  red = max activation patch, cyan = max ERF patch.
- Local smoke checks passed: `src/causal.py` and `src/visualise.py` compile,
  notebook code cells parse, and the CaFE plotting path saves a PNG.

### Blockers

- Full exact ablation for layers 4 and 6 is still compute-heavy on the MacBook Air M2.
  The code is ready, but the full `grad_top_k=200` run should ideally be done on an
  A100/Colab runtime or overnight with thermal throttling expected.
- If running locally just to verify the loop, temporarily set `cfg.causal.grad_top_k`
  to 20-50, then restore 200 before reporting final numbers.
- CaFE is a Week 2 sanity check, not a full reproduction of Han et al.'s AttnLRP-based
  CaFE pipeline. Report it as input-gradient CaFE-style ERF validation.

---

## Week 3 — Visualisations + steering

### Decisions

### Findings

### Blockers
