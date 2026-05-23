"""
causal.py  [Owner: Person A (ablation loop) + Person B (CaFE check) — Week 2]
----------
Causal feature importance analysis.

Two distinct responsibilities in this file — coordinate before coding:
  Person A owns: compute_feature_importance(), get_top_causal_features()
  Person B owns: cafe_sanity_check()

Public API (implement these)
-----------------------------

--- Person A ---

compute_feature_importance(layer, class_a_activations,
                           class_b_activations, model) -> torch.Tensor
    For every SAE feature at the given layer, measure its causal
    importance for the flamingo-vs-spoonbill classification by
    ablation (zeroing the feature) and measuring the resulting
    change in logit difference.

    logit_diff = logit(class_a) - logit(class_b)

    Importance of feature f =
        mean_over_images( logit_diff_original - logit_diff_with_f_ablated )

    A high positive score means feature f contributes positively to
    predicting class_a over class_b.

    activations: output of cache.load_layer() for each class.
    model: the HookedViT from model.get_model().
    Returns a 1D tensor of shape (d_sae,) with importance scores.

    Implementation notes:
    - Use sae.ablate_feature() as the primitive.
    - Process images in batches (cfg.causal.logit_diff_batch_size)
      to avoid OOM on Colab.
    - For each image, re-inject the modified activations into the
      model's residual stream at the correct layer using Prisma's
      hooks, then read off the logits.
    - Use tqdm for progress.

get_top_causal_features(importance_scores, layer,
                        percentile=None) -> list[int]
    Return the indices of features whose importance score is above
    cfg.causal.importance_percentile (default 80th percentile).
    percentile argument overrides config if provided.

--- Person B ---

cafe_sanity_check(layer, feature_idx, activations,
                  image_paths, model) -> dict
    CaFE-style comparison: for a given SAE feature, compare the
    spatial location of maximum activation (the "top patch") against
    the location of maximum gradient attribution (the causally
    responsible patch).

    Reference: Han, Kim, Kwak (2025). CaFE: Causal Interpretation of
    Sparse Autoencoder Features in Vision. arXiv:2509.00749.

    Steps:
        1. Encode activations to get SAE feature map (n, seq_len).
        2. For each image, find the patch token with max activation
           for this feature -> "activation location".
        3. For each image, compute gradient of the feature's activation
           (summed over tokens) w.r.t. the input pixel values using
           torch.autograd. Map gradient magnitudes back to patch grid
           -> "gradient location".
        4. Measure spatial agreement: are the two locations the same
           patch? Compute agreement rate over the top-k images.

    Returns a dict with:
        activation_locations:  list of (row, col) per image
        gradient_locations:    list of (row, col) per image
        agreement_rate:        float in [0, 1]
        example_images:        list of image paths for visualisation

Implementation notes (shared)
------------------------------
- The key question for re-injection (Person A): how does Prisma's
  run_with_cache handle modified activations? Read Prisma docs on
  hook-based interventions before implementing.
- For cafe_sanity_check (Person B): gradient computation requires
  model inputs with requires_grad=True and a full forward pass
  through the model (not from cached activations).

Depends on: src/config.py, src/model.py, src/sae.py, src/cache.py
Used by:    notebooks/03_causal_features.ipynb
"""

# TODO (Person A, Week 2 Days 8–10):
#   0. Verify model.run_with_hooks() in notebook 01 BEFORE writing any code here.
#      Hook signature: fwd_hooks=[(hook_name, fn)] where fn(value, hook) -> tensor.
#   1. Implement compute_feature_importance() with a two-pass approach:
#      Pass 1 — gradient ranking: one backward pass of logit_diff w.r.t. SAE
#               feature activations → importance ∝ |grad| × |feature_act|.
#               This ranks all d_sae features in O(1) backward passes.
#      Pass 2 — ablation confirmation: run ablate_feature() only on top-K candidates
#               to get exact importance scores for the final ranking.
#      Do NOT loop ablate_feature() over all 3072 features — that is 3072 forward
#      passes per layer and will take ~30 min on CPU.
#   2. Implement get_top_causal_features().
#   3. Run on layers 6, 9, 11 and save ranked lists to outputs/.
#
# TODO (Person B, Week 2 Days 10–12):
#   1. Read CaFE paper (arXiv:2509.00749) before implementing.
#   2. Implement cafe_sanity_check().
#   3. Run on top 10 causally important features from Person A's output.
#   4. Report agreement rate; flag features where activation != gradient
#      location for discussion in the report.
