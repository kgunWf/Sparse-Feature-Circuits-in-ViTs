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

---

## Week 3 — Faithfulness eval + gap analysis

### Decisions

### Findings

### Blockers
## Week 1 — Monosemanticity Score

This week I implemented compute_monosemanticity_score() in src/features.py, based on the metric introduced by Pach et al. (NeurIPS 2025). 
The implementation follows their metric.py approach: SAE activations from the CLS token are first min–max normalised, CLIP whole-image embeddings are unit-normalised using the vision model’s pooler_output, 
cand monosemanticity is then computed using activation-weighted pairwise cosine similarity across all image pairs.

I ran the evaluation on layers 4, 6, and 9 using the SAE checkpoints currently available in the Prisma zoo. Layer 11, which was part of the original plan, could not be included
 because this project uses DINO v1 (facebook/dino-vitb16), and matching DINOv2-reg4 SAE weights for that layer are not available.

### Results (5000 ImageNet-val images)

| Layer | Live Features | Dead Features | Mean MS |
|-------|--------------|---------------|---------|
| 4     | 2,651        | 46,501        | 0.3312  |
| 6     | 2,614        | 46,538        | 0.3221  |
| 9     | 10,567       | 38,585        | 0.3667  |

### Observations
- Monosemanticity scores were fairly stable across all three layers, ranging from 0.32 to 0.37, which suggests a moderate level of monosemanticity overall.
- Layer 9 stood out, with roughly four times more live features than layers 4 and 6. This likely reflects richer or more specialised representations emerging in the deeper layers of the model.
- A large proportion of features remained inactive across all layers. This aligns with the relatively sparse L0 targets used in the SAE configuration (876, 962, and 1105 respectively), so the high dead-feature count is expected rather than surprising.
