# Gap Analysis — Sparse Feature Circuits in ViTs: Spatial Resolution vs. Token Aggregation

Owner: Person C — fill in during Week 3 after circuit is built.

## 1. Circuit compactness

| Method                         | Nodes | Edges | Spatial resolution | Interpretable? |
|--------------------------------|-------|-------|--------------------|----------------|
| Sparse feature — ours          | ?     | ?     | Patch-level ✓      | ?              |
| RRM (Kim et al., NeurIPS 2025) | ~11*  | ?     | Token-aggregated ✗ | Yes (small)    |
| Head-level (estimated)         | ~144  | ?     | N/A                | Harder         |

*RRM's Granny Smith example circuit had 11 nodes across 10 layers.

## 2. Does spatial resolution matter for this behavior?

This is the core question your project must answer empirically.

For the flamingo/spoonbill circuit, check:
- Do the causally important circuit nodes activate at spatially
  meaningful locations? (e.g., beak region for the beak-shape feature)
- Does the CaFE sanity check (causal.py) show that activation location
  ≠ gradient location for any nodes? If yes, token aggregation would
  have misattributed those features.
- Would RRM's token-aggregated version of this circuit look different?
  (You can approximate this by averaging your patch activations and
  re-running the edge measurement — compare the resulting circuit.)

(Fill in after running the CaFE check in notebook 03)

## 3. Interpretability of the circuit

Can you read the circuit as a coarse-to-fine story?
(e.g., texture features in layer 6 → part features in layer 9
→ semantic/discriminative features in layer 9)

(Fill in after inspecting the graph)

## 4. Faithfulness

Target: >= 0.70. Achieved: ?

If below target, explain why and whether this is expected.
Compare to RRM's faithfulness numbers (Table 1 in their paper:
94.1% for ViT, 85.1% for DINOv2) — if yours is lower, explain
the likely cause (contrastive vs. single-class behavior,
fewer nodes, different model variant).

## 5. What spatial resolution reveals that RRM cannot

(Fill in — this is the core contribution paragraph for the report.
Should be concrete: specific features, specific image regions,
specific spatial disagreements from the CaFE check.)

## 6. Open questions for future work

(Fill in — these are the documented gaps your project leaves open)
