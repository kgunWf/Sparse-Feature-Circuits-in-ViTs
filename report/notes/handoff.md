# P3 Handoff — Person B & C

**Compute constraints**

- Person A: NVIDIA RTX A5000 (25 GB) — runs nb02 and nb04 only; all of nb03 delegated
- Person C: NVIDIA RTX 5070 (12 GB) — runs all of nb03 (CaFE loops process one image at a time, fits in 8 GB)
- Person B: CPU only — code implementation + annotations + figure rendering

---

## Person A — remaining tasks (handoff boundary)

Person A's only remaining job is producing the feature outputs and sharing the cache.

| Task | Notebook | Output |
|------|----------|--------|
| Run DINO feature pipeline | nb02, all cells | `top_patches_layer{4,6,9}_full.pkl.gz`, `clip_labels_layer{4,6,9}_full.json`, `ms_scores_layer{4,6,9}_top5.json` |
| Run CLIP feature pipeline | nb04, all cells | `top_patches_clip_layer{4,6,9}_full.pkl.gz`, `clip_labels_clip_layer{4,6,9}_full.json`, `ms_scores_clip_layer{4,6,9}_top5.json` |
| Share HDF5 caches | — | `outputs/caches/activations.h5`, `outputs/caches/activations_clip.h5` |

**Person A is done after nb02 + nb04.** Everything in nb03 is Person C's.

---

## Person C — all of nb03 (with GPU)

The CaFE loops process one image at a time — DINO ViT-B/16 (~330 MB) + SAE (~300 MB) fits in 8 GB. Activations are loaded from the HDF5 cache into CPU RAM, not GPU.

Run nb03 top to bottom in this order:

### Step 1 — Load data (cells env, 1, 2a, 2b)

No GPU needed. Reads cache + JSON files from Person A.

### Step 2 — DINO CaFE Run 1 (cell 3a) · ~50 min

Index-based, first 100 features per layer. Validate 5 features at layer 9 first: IG maps must be spatially coherent (non-uniform). If uniform, check the residual-skip detach in `cafe_sanity_check` before proceeding.
```
Saves: outputs/features/cafe_ig/run1/cafe_ig_layer{4,6,9}_feat*.json
```

### Step 3 — DINO CaFE Run 2 (cell 5) · ~25 min

MS-ranked, top 50 per layer.
```
Saves: outputs/features/cafe_ig/run2/cafe_ig_layer{4,6,9}_feat*.json
```

### Step 4 — ERF visualisation (cell 6)

Run immediately after cell 5 in the same session — needs `erf_scores` in memory.

### Step 5 — CLIP Run 3 (cells 7a → 7b) · ~35 min

Requires nb04 outputs from Person A.
```
Saves: outputs/features/cafe_ig/run3_clip/cafe_ig_layer{4,6,9}_feat*.json
```

### Step 6 — Spearman analysis (cell 8)

CPU/scipy. Implement from scratch — the stub has the full input/output spec.

Required output — `spearman_results` with three variants:

| Variant | N | MS range | Purpose |
|---------|---|----------|---------|
| DINO Run 1 pooled | 300 | full | Headline DINO ρ, symmetric with CLIP |
| DINO Run 2 pooled | 150 | restricted | Conservative ρ — report separately, note range restriction |
| CLIP Run 3 pooled | 300 | full | Headline CLIP ρ |

Always report per-layer ρ alongside pooled (pooled may reflect a shared depth trend).

Call `plot_ms_locality_scatter` → **Fig 8**, `plot_locality_by_depth` → **Fig 4**, `plot_locality_comparison` → **Fig 6 ★**.  
For Fig 6 ★: digitize the CaFE CLIP-L/14 reference from Han et al. Fig 5 (non-locality → agreement = 1 − rate); label as `'CaFE CLIP-L/14 (est. Fig 5)'` with a dashed line.

### Step 7 — Layer evolution (cell 9)

CPU/matplotlib. Reads CLIP labels from disk. Requires Person B's annotation files to exist. Produces **Fig 2** (category composition by layer — replaces the two separate Fig 1/Fig 2 from the original plan with a single summary).

### Report writing (D3 methods draft + D7 assembly)

- **Methods:** model descriptions, SAE provenance justification (compute-infeasible; Prisma quality controls), token layouts, MS formula note (top-5 vs Pach's top-20 — state explicitly), three-run CaFE design and rationale, IG limitation (AttnLRP is CaFE's best-performing method; we use IG for implementation simplicity — acknowledge in Limitations)
- **Results:** locality comparability (§ Fig 4 + Fig 6 ★), quality-stratified analysis (§ Fig 5 + Fig 8), MS distributions and category composition in text/tables
- **Gap statement:** three gaps as specified in plan §D7
- **Limitations:** IG vs AttnLRP, ViT-B vs ViT-L, Prisma vs Matryoshka SAEs, deepest layer 75% ≠ CaFE's 92%, top-5 MS vs Pach top-20, CLIP self-labeling circularity, Run 2 range restriction
- **Direction evolution note** (required subsection): original circuit goal → RRM paper overlap → supervisor meeting → Direction A
- **Final assembly:** abstract, introduction, proofreading pass

---

## Person B — CPU only: annotations + 4 figure stubs

### 1. Implement 4 figure stubs — `src/visualise.py` ·

**Prioritise in this order**
All stubs are at the bottom of `visualise.py`, each with a detailed docstring.

| Function | Figure | Data available after | Priority |
|----------|--------|----------------------|----------|
| `plot_locality_by_depth` | Fig 4 | CaFE Run 1 (Person C) | 1st — unblocks Spearman cell |
| `plot_locality_comparison` | Fig 6 ★ | Runs 1 + 3 (Person C) | 2nd — headline result |
| `plot_ms_locality_scatter` | Fig 8 | Spearman results (Person C) | 3rd |
| `plot_locality_by_category` | Fig 5 | CaFE Run 2 + annotations (Person C + B) | 4th |

All functions: accept `save_path=None`, save if provided, return the `Figure` object.

MS distributions (Fig 1) and category composition (Fig 2) are replaced by tables/text in the report — no figure stubs needed for them. Insertion test (Fig 3/7) is dropped — acknowledge IG vs. AttnLRP in Limitations text instead.

### 2. Category annotations (D1 + D3)

**DINO** — `report/notes/feature_catalog_layer{4,6,9}.md`  
Input: `clip_labels_layer{4,6,9}_full.json` + `top_patches_layer{4,6,9}_full.pkl.gz` from Person A.  
For each of the top-50 MS-ranked features: top-20 patches + top-3 CLIP labels → assign one category.

**CLIP** — `report/notes/feature_catalog_clip_layer{4,6,9}.md`  
Same scheme, input from nb04 outputs.

```
Categories: texture / color / part / scene / semantic / unclear

Decision rules:
  repeating surface pattern          → texture
  body part / component              → part
  background / environment           → scene  (check compounds first: "blue sky", "green grass")
  whole object class                 → semantic
  conflicting or ambiguous           → unclear
```

150 DINO + 150 CLIP annotations total. AI first pass + genuine human review required (course requirement).
Document inter-annotator agreement rate for ≥ 30 features per model in the Methods section.

**Note:** CLIP features labeled by CLIP-B/32 embeddings is mildly circular — acknowledge in Methods.

### 3. Run figure-generating cells after data arrives (CPU only)

- nb03 cell 6 (ERF vis) — ask Person C to run it in the same session as cell 5 (needs `erf_scores` in memory)
- Figs 4, 5, 6 ★, 8 — once Person C's JSONs and Person B's stubs are both complete

---

## Dependency order

```
Person A: nb02, nb04 ──────────────────────────────────────────────────────┐
                                                                            │ JSON + HDF5 files
                                                                            ▼
                                              Person C: nb03 cells env→1→2a→2b→3a→5→6→7a→7b
Person B: figure stubs (plot_locality_by_depth first) ──┐                  │
Person B: annotations ──────────────────────────────────┤                  ▼
                                                         └─► Person C: nb03 cells 8, 9
                                                                     │
                                                                     ▼
                                                         Person C: report assembly (D7)
```

---

## File locations

```
outputs/features/
  top_patches_layer{4,6,9}_full.pkl.gz              ← Person A (nb02)
  clip_labels_layer{4,6,9}_full.json                ← Person A (nb02)
  ms_scores_layer{4,6,9}_top5.json                  ← Person A (nb02)
  top_patches_clip_layer{4,6,9}_full.pkl.gz         ← Person A (nb04)
  clip_labels_clip_layer{4,6,9}_full.json            ← Person A (nb04)
  ms_scores_clip_layer{4,6,9}_top5.json             ← Person A (nb04)
  cafe_ig/run1/cafe_ig_layer{N}_feat{M}.json        ← Person C (nb03 cell 3a)
  cafe_ig/run2/cafe_ig_layer{N}_feat{M}.json        ← Person C (nb03 cell 5)
  cafe_ig/run3_clip/cafe_ig_layer{N}_feat{M}.json   ← Person C (nb03 cell 7b)

outputs/caches/
  activations.h5                                    ← Person A (share with Person C)
  activations_clip.h5                               ← Person A (share with Person C)

report/notes/
  feature_catalog_layer{4,6,9}.md                   ← Person B (D1)
  feature_catalog_clip_layer{4,6,9}.md              ← Person B (D3)

report/figures/
  feature_gallery_layer9.png                        ← Person A (nb02)
  feature_gallery_clip.png                          ← Person A (nb04)
  locality_by_depth_dino.png                        ← Person B (Fig 4)
  locality_by_category_dino.png                     ← Person B (Fig 5)
  locality_comparison.png                           ← Person B (Fig 6 ★)
  ms_locality_scatter.png                           ← Person B (Fig 8)
  spearman_ms_vs_cafe_layer{N}.png                  ← Person C (nb03 cell 8)
  layer_evolution.png                               ← Person C (nb03 cell 9)

  [MS distributions and category composition are tables/text in the report — no figure files]
  [Insertion test figures dropped — IG vs AttnLRP acknowledged in Limitations text]
```
