# P3 — Mechanistic Interpretability of Vision Transformers

### Sparse Feature Circuits in ViTs: From SAE Concept Discovery to Causal Graph

**Explainable and Trustworthy AI — Politecnico di Torino 2025/2026**
Teachers: Gabriele Ciravegna, Eliana Pastor

---

## Research direction

We build spatially-resolved sparse feature circuits for a contrastive visual discrimination behavior in DINOv2. Nodes in the circuit are interpretable SAE concepts; edges are measured causal connections between those concepts across layers. Unlike token-aggregated prior work (Kim et al., RRM, NeurIPS 2025), our circuits preserve patch-level spatial resolution throughout — enabling spatial attribution of each circuit node to specific image regions, and CaFE-style validation that circuit features are causally grounded at the right spatial location.

Sparse feature circuits were originally demonstrated for LLMs (Marks et al., 2024). RRM extended a token-aggregated variant to ViTs for single-class recognition. Our contribution is the first spatially-resolved sparse feature circuit for a **contrastive two-class behavior** in **DINOv2 with register tokens**.

**Target model:** DINOv2 ViT-B/14 with register tokens (Darcet et al., ICLR 2024)
**Current model:** `facebook/dino-vitb16` (DINO v1 ViT-B/16) — stand-in while vit_prisma
DINOv2 support is unresolved. See `report/notes/person_a_notes.md`.
**Behavior under study:** flamingo vs. spoonbill classification
**Tooling:** Prisma / vit_prisma (Joseph et al., CVPR 2025)

**Fallback:** If circuit construction is intractable within the timeline, the SAE analysis (Stages 1–2) constitutes a standalone deliverable: monosemanticity scores, CLIP-based concept labeling, and a CaFE-style causal sanity check on the configured DINO model.

---

## Pipeline overview

```
Stage 1 — src/model.py + src/sae.py
  Load the configured DINO model via Prisma HookedViT. Load pre-trained SAEs.
  Verify reconstruction quality and L0 sparsity.

Stage 2 — src/cache.py + src/features.py
  Build HDF5 activation cache for 5,000 ImageNet val images.
  Retrieve top-activating patches per SAE feature.
  Auto-label features via CLIP cosine similarity.
  Compute Monosemanticity Score (Pach et al., 2025).

Stage 3 — src/causal.py
  Curate flamingo/spoonbill dataset (200+200 images).
  Run per-feature ablation: zero a feature, measure logit diff change.
  Rank features by causal importance per layer.
  CaFE-style sanity check: causal attribution vs. activation location.

Stage 4 — src/circuits.py + src/evaluate.py
  Measure pairwise causal edges between top features across layers.
  Build directed graph (networkx). Threshold and visualise.
  Faithfulness evaluation: ablate all circuit nodes, measure logit drop.
```

---

## Repository structure

```
configs/default.yaml        All hyperparameters — never hardcode values elsewhere
data/                       ImageNet val split — NOT tracked by git (see data/README.md)
src/
  config.py                 YAML loader — everyone imports get_config() from here
  model.py                  DINO + HookedViT loading (patched)  [Person A]
  sae.py                    SAE loading + encode/decode/ablate   [Person A]
  cache.py                  HDF5 activation cache build + read   [Person C]
  features.py               Top patches, CLIP labeling, MS score [Person B]
  causal.py                 Ablation loops, logit diff ranking   [Person A+B]
  circuits.py               Pairwise edge measurement, graph     [Person A]
  evaluate.py               Faithfulness metric, MS wrapper      [Person C]
  visualise.py              All plotting functions               [Person B]
notebooks/
  01_sae_setup.ipynb        Stage 1: model + SAE smoke test      [Person A]
  02_feature_analysis.ipynb Stage 2: feature catalog             [Person B]
  03_causal_features.ipynb  Stage 3: ablation + CaFE check       [Person B+C]
  04_circuit.ipynb          Stage 4: circuit construction        [Person A]
outputs/                    Generated files — NOT tracked by git
report/figures/             Final figures for the report         [tracked]
report/notes/               Running notes per person             [tracked]
```

---

## Setup

```bash
git clone <repo-url>
cd p3-vit-mech-interp
conda create --name vit_mech python=3.10
conda activate vit_mech
pip install -r requirements.txt
pip install -e .          # makes `from src.x import ...` work in notebooks
python -m ipykernel install --user --name vit_mech --display-name "Python (vit_mech)"
```

In VS Code / Jupyter, select the **Python (vit_mech)** kernel before running any notebook.

Download a small test set (flamingo + spoonbill, 5 images each):

```bash
python data/load_data.py
```

For the full pipeline, download ImageNet val and update `data/imagenet_val_path` in
`configs/default.yaml`.

## SAE Weights

Pre-trained DINO SAE weights are stored outside this directory at:

```
outputs/saes/
├── layer_4/
│   ├── weights.pt    (~288 MB)
│   └── config.json
├── layer_6/
│   ├── weights.pt    (~288 MB)
│   └── config.json
└── layer_9/
    ├── weights.pt    (~288 MB)
    └── config.json
```

The `outputs/` directory is git-ignored. Download weights before running any notebook.

### Download command

```bash
# All target layers (4, 6, 9) — ~864 MB total
python utils/download_saes.py

# Single layer (recommended for first run)
python utils/download_saes.py --layers 9

# Force re-download if files are present but corrupted
python utils/download_saes.py --layers 9 --force
```

Weights are fetched from the `Prisma-Multimodal` organisation on HuggingFace.
Repo IDs are stored in `cfg.sae.sae_repos` in `configs/default.yaml`.
No HuggingFace account is required (repos are public).

### Known issue — model

vit_prisma v2.0.0 does not support `facebook/dinov2-vitb14-reg`. The project currently
uses `facebook/dino-vitb16` (DINO v1) as a stand-in. `src/model.py` includes a runtime
patch (`_remap_dino_keys`) to fix a key-name mismatch between the installed vit_prisma
weight converter and the current HuggingFace transformers release. No manual action
required — the patch applies automatically on import.

---

## Key rule

**All hyperparameters live in `configs/default.yaml`.** Never hardcode a layer index,
threshold, or path in a notebook or src/ file. Import `get_config()` from `src/config.py`.

**All reusable logic lives in `src/`.** Notebooks call functions from `src/` — they do not
contain loops, model loading, or metric computation directly.

---

## Key references

- Marks et al. (2024). Sparse Feature Circuits. arXiv:2403.19647
- Kim et al. (2025). Interpreting ViTs via Residual Replacement Model. arXiv:2509.17401 — closest prior work; token-aggregated
- Joseph et al. (2025). Prisma. arXiv:2504.19475
- Pach et al. (2025). Sparse Autoencoders Learn Monosemantic Features in VLMs. NeurIPS 2025
- Żukowska et al. (2026). Seeing Through Circuits. arXiv:2604.14477
- Han et al. (2025). CaFE. arXiv:2509.00749
- Darcet et al. (2024). Vision Transformers Need Registers. ICLR 2024
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits
- Conmy et al. (2023). Towards Automated Circuit Discovery. NeurIPS 2023
