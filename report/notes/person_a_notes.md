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

- **Dead feature fraction is ~96% on a single image — this is expected and not a bug.**
  `compute_dead_feature_fraction` measures features whose max activation never exceeds
  `cfg.sae.dead_feature_threshold` (0.01). With only one image and 197 tokens, ~96.6% of
  the 49,152 SAE features never fire — they are specialists for other visual patterns
  (other textures, object parts, scenes) that simply don't appear in one bird photo.
  Only ~1,671 features activate at all for a single spoonbill image.
  The metric is only meaningful across the full dataset (200+ images); a well-trained SAE
  should have near-0% truly dead features over many images.
  → Do not assert on this cell until `build_cache()` is available and the HDF5 is populated.

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

- **Two-pass importance ranking in `compute_feature_importance()`.**
  Naive approach (ablate all 49,152 features × n_batches forward passes) is intractable.
  Instead:
  - **Pass 1 — gradient pre-ranking.** One forward+backward per batch ranks *all* d_sae
    features at once. Encode the cached activations → SAE features, set `requires_grad`
    on them, decode back to the residual stream, inject at `blocks.9.hook_resid_post`,
    forward, and backprop the flamingo−spoonbill logit diff. Accumulate the proxy score
    `|∂logit_diff/∂feat| × |feat|` per feature, averaged over tokens/images. Cost is
    O(n_batches), independent of d_sae.
  - **Pass 2 — exact ablation.** Only the top `cfg.causal.grad_top_k` (200) candidates
    from Pass 1 are individually ablated and re-run to measure the *exact* logit-diff
    drop. Cost O(grad_top_k × n_batches).
  Final `importance[f] = mean(baseline_diff − ablated_diff)`; positive ⇒ feature promotes
  flamingo, negative ⇒ promotes spoonbill, zero ⇒ not in the top-k (never ablated).

- **Activation-injection trick: forward pass on a zeros input.**
  Both passes call `run_with_hooks(torch.zeros(...), fwd_hooks=[(hook_key, _hook)])`.
  The model runs on a dummy zeros image, but the layer-9 `hook_resid_post` hook overwrites
  the residual stream with our decoded/ablated activations, so layers 10–11 compute on the
  injected value. This lets us evaluate from cached activations without needing the original
  pixels — only the residual stream at the primary layer matters for the downstream logits.

- **Baseline diffs computed once, reused across all 200 candidates.**
  Pass 2 computes `baseline_diffs` (no ablation, just re-inject the unmodified batch) a single
  time before the candidate loop, rather than per-feature. Saves ~200× redundant forward passes.

- **`get_top_causal_features()` thresholds over ablated features only.**
  Percentile (`cfg.causal.importance_percentile`, default 80) is computed over the non-zero
  (ablated) entries via `torch.quantile`, not all 49k. Features outside grad_top_k have
  importance == 0 and are always excluded. `percentile` arg overrides config when passed.

### Findings

- **Hook callback signature bug — hooks must accept a `hook=` keyword.**
  vit_prisma's `HookPoint.full_hook` invokes registered forward hooks as
  `hook(module_output, hook=self)`. The original closures used `def _hook(*_, _r=recon)`,
  which only catches positional args, so the `hook=` keyword raised
  `TypeError: unexpected keyword argument 'hook'` (notebook 03 cell 4 / `src/causal.py:132`).
  Fixed all three closures to `def _hook(*_, _r=..., **__)` — `**__` absorbs the keyword.
  → Any future hook closure in this file must absorb `**kwargs` (or take an explicit `hook`
  param, as the notebook-01 `zero_hook` does).

- **Verified dataset: 181 flamingo + 148 spoonbill = 329 images** correctly classified
  (out of 200 + 200 fetched). act shapes `(181, 197, 768)` and `(148, 197, 768)`, kept on CPU.
  Pass 1 ran 42 batches (`cfg.causal.logit_diff_batch_size = 8`).

- **Ranking results (layer 9), saved to `outputs/features/importance_layer9.pt`:**
  - 200 candidates ablated; max |importance| = 0.3780.
  - Top-5 features: 17825 (0.378), 32842 (0.325), 30072 (0.266), 40410 (0.258), 49054 (0.232).
  - 40 features pass the 80th-percentile cut → `top_features` for the ablation-ranking plot
    and downstream CaFE / circuit work.
  - Result is cached; re-running cell 4 loads from disk instead of recomputing. **Delete the
    `.pt` if the importance code changes**, or the stale cache is silently reused.

### Blockers

- **CaFE sanity check (`cafe_sanity_check`) is Person B's and still a `NotImplementedError`.**
  Notebook 03 cell 4 (section 4) stays commented out until that lands. My importance output
  (`top_features[:10]`) is the input it depends on — already saved, so Person B is unblocked.

- **Still running on DINO v1 (patch16, 197 tokens), not DINOv2-reg.** Carried over from Week 1
  — all causal results above are for the DINO v1 stand-in. If the group switches models, the
  importance ranking must be recomputed against the new SAEs.

---

## Week 3 — Circuit construction

### Decisions

### Findings

### Blockers
