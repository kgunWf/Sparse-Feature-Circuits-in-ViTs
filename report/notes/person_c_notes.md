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
