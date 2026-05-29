# Person B — Running Notes

## Week 1 — Feature analysis + CLIP labeling

### Decisions

- [x] Concept vocabulary is explicit input to `label_feature_clip`.
      Week 1 now builds a 10k English concept vocabulary with targeted
      visual terms prepended, instead of using the toy 25-word notebook list.
- [x] CLIP labels use resized model-space images and CLIP-sized crops
      centered on the top-activating patch, not raw 16x16 patch crops.
- [x] Token layout is config-derived: CLS at 0 and patch tokens from
      `1..grid_size^2`. Current DINO v1 has no register tokens.
- [x] Full-cache all-feature top-patch retrieval is streaming over cache
      batches, so it can run on all 5,000 images without materialising the
      entire `(images, tokens, d_sae)` tensor.

### Findings
- Implemented top-patch retrieval for single features and all features.
  The full all-feature path keeps only top-k entries per feature while
  scanning cached activations in small batches.
- Implemented CLIP label loading and patch-crop concept labeling with
  cached text embeddings.
- Week 1 notebook is configured to run the all-feature top-patch scan on
  all cached images and cache the result under `outputs/features/`.
- Week 1 notebook can also run full-feature CLIP labeling with the 10k
  vocabulary and cache `clip_labels_layer{layer}_full.json` under
  `outputs/features/`.
- CLIP labels are usable as first-pass hints, but should be checked
  against patch grids because auto-labeling is only a semantic proxy.

### Remaining Checks
- Run a small manual review over several features before treating the
  CLIP labels as final semantic annotations.

---

## Week 2 — Multi-layer ablation + CaFE check

### Decisions

### Findings

### Blockers

---

## Week 3 — Visualisations + steering

### Decisions

### Findings

### Blockers
