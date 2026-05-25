# Person B — Running Notes

## Week 1 — Feature analysis + CLIP labeling

### Decisions

- [x] Concept vocabulary is explicit input to `label_feature_clip`.
      Use ImageNet class names plus targeted texture, color, body-part,
      scene, and bird-part terms for the Week 1 sanity run.
- [x] CLIP labels use the resized model input image and crop one
      `cfg.model.patch_size` square at the top-activating patch.
- [x] Token layout is config-derived: CLS at 0 and patch tokens from
      `1..grid_size^2`. Current DINO v1 has no register tokens.

### Findings
- Implemented top-patch retrieval for single features and all features.
  The all-feature path encodes activations once before ranking.
- Implemented CLIP label loading and patch-crop concept labeling with
  cached text embeddings.
- Verified the Week 1 path on the local 5k-image cache:
  layer 9 activations -> SAE encode -> top patches -> CLIP labels.
- CLIP labels are usable as first-pass hints, but should be checked
  against patch grids because the current crop is only one ViT patch.

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
