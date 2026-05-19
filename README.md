# P3 — Mechanistic Interpretability of Vision Transformers
### Sparse Feature Circuits in ViTs: From SAE Concept Discovery to Causal Graph
**Explainable and Trustworthy AI — Politecnico di Torino 2025/2026**
Teachers: Gabriele Ciravegna, Eliana Pastor

---

## Research direction

We extend the sparse feature circuits approach of Marks et al. (2024) — originally demonstrated for LLMs — to Vision Transformers. Instead of building circuits at the attention-head/MLP level, we build them at the SAE feature level: nodes are interpretable visual concepts recovered by a Sparse Autoencoder, edges are measured causal connections between those concepts across layers.

**Model:** DINOv2 ViT-B/14 with register tokens (Darcet et al., ICLR 2024)
**Behavior under study:** flamingo vs. spoonbill classification
**Tooling:** Prisma (Joseph et al., CVPR 2025)

**Fallback:** If circuit construction is intractable within the timeline, the SAE analysis (Stages 1–2) constitutes a standalone deliverable: monosemanticity scores, CLIP-based concept labeling, and a CaFE-style causal sanity check on DINOv2.

---

## Pipeline overview

```
Stage 1 — src/model.py + src/sae.py
  Load DINOv2 via Prisma HookedViT. Load pre-trained SAEs.
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
  model.py                  DINOv2 + HookedViT loading          [Person A]
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
pip install -r requirements.txt
```

Download ImageNet val and update `data/imagenet_val_path` in `configs/default.yaml`.

---

## Key rule

**All hyperparameters live in `configs/default.yaml`.** Never hardcode a layer index,
threshold, or path in a notebook or src/ file. Import `get_config()` from `src/config.py`.

**All reusable logic lives in `src/`.** Notebooks call functions from `src/` — they do not
contain loops, model loading, or metric computation directly.

---

## Key references

- Marks et al. (2024). Sparse Feature Circuits. arXiv:2403.19647
- Joseph et al. (2025). Prisma. arXiv:2504.19475
- Pach et al. (2025). Sparse Autoencoders Learn Monosemantic Features in VLMs. NeurIPS 2025
- Żukowska et al. (2026). Seeing Through Circuits. arXiv:2604.14477
- Han et al. (2025). CaFE. arXiv:2509.00749
- Darcet et al. (2024). Vision Transformers Need Registers. ICLR 2024
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits
- Conmy et al. (2023). Towards Automated Circuit Discovery. NeurIPS 2023
