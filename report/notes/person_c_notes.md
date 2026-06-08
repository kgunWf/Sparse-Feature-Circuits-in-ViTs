# Person C — Running Notes

## Week 1 — Activation cache

### Decisions
- [ ] Agree HDF5 schema with team BEFORE writing build_cache()
      (seq_len, d_model, chunk size — see cache.py docstring)
- [ ] Confirm total cache size estimate:
      5000 images x 201 tokens x 768 dims x 4 bytes x 3 layers = ~3 GB
      Check available Colab disk space before building
- [ ] Decide batch size for cache build (default 32, may need to lower on Colab)

### Findings
<!-- Record actual file size and build time -->

### Blockers
<!-- Depends on Person A finishing model.py (need get_model and run_with_cache) -->
### Cache Update

Implemented `src/cache.py` on branch `dev-person-c`.

Confirmed cache schema:
- `/metadata`: model_name, image_size, layers, n_images
- `/images`: paths, labels, class_ids
- `/activations/layer_{L}`: float32, shape `(n_images, seq_len, d_model)`

Confirmed hook key format:
- `blocks.{L}.hook_resid_post`

Current team setup:
- model: `facebook/dino-vitb16`
- image size: 224
- target layers: `[6, 9, 11]`
- observed activation shape per image: `(197, 768)`

Implemented:
- `build_cache(...)`
- `load_layer(...)`
- `load_metadata(...)`
- `load_image_index(...)`
- `get_class_indices(...)`

---

## Week 2 — Patch-CLS comparison + layer evolution

### Decisions

### Findings

### Blockers
