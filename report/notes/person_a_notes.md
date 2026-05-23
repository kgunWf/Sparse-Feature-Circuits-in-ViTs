# Person A — Running Notes

Use this file to log decisions, findings, and blockers as you work.
Commit updates regularly so the team can follow your progress.

## Week 1 — Model + SAE setup

### Decisions

- **SAE implementation: all 6 primitives in `src/sae.py`.**
  `get_sae()`, `encode()`, `decode()`, `ablate_feature()`, `get_l0_sparsity()`,
  `get_reconstruction_loss()`. One SAE per layer, cached in `_sae_cache` on first call.
  Checkpoint path convention: `outputs/saes/layer_{N}/weights.pt` + `outputs/saes/layer_{N}/config.json`.
  Both paths derived from `_DEFAULT_CONFIG.parent.parent` (repo root anchor) — no hardcoded paths.
  These filenames match the HuggingFace repo structure exactly; `load_from_pretrained` finds
  `config.json` automatically from the same directory as the weights.

- **Pre-trained DINO SAEs found on Prisma-Multimodal HuggingFace.**
  Repos follow the pattern `Prisma-Multimodal/DINO-vanilla-x64-all_patches_{N}-resid_post-{L0}-{var}`.
  `DINO-vanilla` = trained on DINO v1 activations (correct model match). `x64` = d_sae = 768 × 64 = 49,152.
  `all_patches` + `resid_post` = hook point matches `blocks.{N}.hook_resid_post` on patch tokens.
  Target layers updated to [4, 6, 9]; primary layer = 9. Repo IDs stored in `cfg.sae.sae_repos`.

- **Auto-download implemented in `get_sae()` + `utils/download_saes.py`.**
  `get_sae()` calls `_download_sae(layer)` automatically if `weights.pt` is missing.
  For large SAE files (288 MB each), prefer the standalone script to download before running
  notebooks: `python utils/download_saes.py --layers 9` (or `--layers 4 6 9` for all).
  Uses `vit_prisma.sae.sae_utils.download_sae_from_huggingface` internally (lazy import).

- **Used `SparseAutoencoder.load_from_pretrained()` to load checkpoints.**
  `SparseAutoencoder` itself is abstract; `load_from_pretrained()` instantiates the correct
  concrete subclass (`StandardSparseAutoencoder`) from the saved config. The `config_path`
  parameter is ignored for non-legacy checkpoints — it always looks for `config.json` in the
  same directory as the weights file.

- **Device selection: explicit MPS detection required throughout the pipeline.**
  `torch.cuda.is_available()` returns False on Apple Silicon; MPS must be checked separately
  with `torch.backends.mps.is_available()`. Both `get_model()` and `get_sae()` now detect
  CUDA → MPS → CPU in that priority order.
  Two additional fixes were needed:
  1. `load_hooked_model()` with `device="mps"` leaves some parameters on CPU — fixed by
     calling `_model.to(device)` explicitly after loading in `get_model()`.
  2. `sae.device` is a plain string attribute not updated by `.to()` — fixed by setting
     `sae.device = device` after `sae.to(device)` in `get_sae()`.
  `ablate_feature()` casts its output back to `original_device = activations.device` so it is
  safe to compare against the input activations regardless of SAE device.

- **`sae.encode(x)` returns a tuple `(sae_in, feature_acts)` — only `feature_acts` is used.**
  `sae_in` is the pre-activation input (x − b_dec); `feature_acts` is the post-ReLU output.
  This is the vit_prisma SAE API. Documented in `src/sae.py:119`.

- **`ablate_feature` clones features before zeroing.**
  `features[..., feature_idx] = 0.0` is in-place; clone prevents mutating the encoded
  tensor and avoids downstream aliasing bugs.

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

- ~~**Pre-trained SAE weights not yet located.**~~ **RESOLVED.**
  Pre-trained weights found on Prisma-Multimodal HuggingFace for layers 4, 6, 9.
  Config updated: `target_layers: [4, 6, 9]`, `primary_layer: 9`.
  Repo IDs stored in `cfg.sae.sae_repos`; weights auto-downloaded via
  `python utils/download_saes.py`. Verified: L0=1098.8 (< target 1200),
  reconstruction loss=0.0097 (< 0.05). All three notebook cells (SAE load, L0,
  reconstruction) now pass with real weights.

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
