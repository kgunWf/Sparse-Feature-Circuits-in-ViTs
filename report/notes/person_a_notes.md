# Person A — Running Notes

Use this file to log decisions, findings, and blockers as you work.
Commit updates regularly so the team can follow your progress.

## Week 1 — Model + SAE setup

### Decisions

- **Switched model from `facebook/dinov2-vitb14-reg` to `facebook/dino-vitb16`.**
  DINOv2 is not registered in the installed vit_prisma (v2.0.0). No config, no weight
  converter, no SAE support. DINO v1 ViT-B/16 is the closest supported model.
  Config updated: `model.name`, `patch_size: 16`, `num_registers: 0`.
  See `report/model_loading_findings.md` for full investigation.

- **Monkey-patched `_load_dino_weights` in `src/model.py`.**
  The vit_prisma weight converter (`convert_dino_weights`) expects the old HF key format
  (`encoder.layer.{N}.attention.attention.query.weight`, `intermediate.dense.*`) but the
  current transformers release uses a different format (`layers.{N}.attention.q_proj.weight`,
  `mlp.fc1.*`). Rather than editing the installed package, `src/model.py` patches
  `vit_prisma.models.model_loader._load_dino_weights` at import time to remap keys
  before they reach the converter. See `_remap_dino_keys()` in `src/model.py`.

- **Used `load_hooked_model()` directly instead of `HookedViT.from_pretrained()`.**
  The wrapper doesn't forward `allow_failing=True` correctly. Calling the lower-level
  function directly is more reliable and explicit.

### Findings

- **Hook key format: `blocks.{N}.hook_resid_post`** — confirmed for all layers 0–11.
  Full list also includes `hook_resid_pre` and `hook_resid_mid` per block.
  → **Share with Person C before Day 4** (needed for `build_cache()` in `src/cache.py`).

- **Activation shape: `(batch, 197, 768)`** at image_size=224.
  197 = 1 CLS token + 196 patch tokens (14×14 grid). No register tokens (DINO v1).
  Note: if we later switch to DINOv2-reg4, seq_len becomes 201 (+ 4 registers).

- **Warnings that are safe to ignore:**
  - `pooler.dense.weight MISSING` — HuggingFace's `ViTModel` always includes a pooler
    projection for downstream fine-tuning, but DINO was never trained with one so the
    checkpoint doesn't contain it. HF initialises it randomly and warns. vit_prisma's
    weight converter never maps these keys into `HookedViT`, so the random values are
    dropped entirely. We use `return_type: "pre_logits"` and never call the pooler —
    these weights have zero effect on any computation in the pipeline.
  - `head.W_H missing` — same reason, no classification head.
  - `ln_pre not set` — DINO v1 has no pre-layer-norm.

### Blockers

- **DINOv2 with registers is still not loadable via vit_prisma.** The project goal requires
  `facebook/dinov2-vitb14-reg` (patch14, 4 registers, 201 tokens). Currently using DINO v1
  (patch16, no registers, 197 tokens) as a stand-in. Pre-trained SAE availability for
  DINO v1 layers 6, 9, 11 needs to be confirmed before Week 2.
  → Raise with the group: do we patch vit_prisma for DINOv2, or continue with DINO v1?

---

## Week 2 — Ablation loop

### Decisions

### Findings

### Blockers

---

## Week 3 — Circuit construction

### Decisions

### Findings

### Blockers
