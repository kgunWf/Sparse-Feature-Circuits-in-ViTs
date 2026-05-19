"""
cache.py  [Owner: Person C — Week 1]
---------
Build and read the HDF5 activation cache.

IMPORTANT: Agree on and freeze the HDF5 schema (see below) before
Person A or Person B write any code that reads activations. The
schema is the contract between cache.py and every other module.

HDF5 Schema
-----------
The cache file lives at cfg.outputs.cache_path and has this layout:

    /metadata
        model_name      str     e.g. "facebook/dinov2-vitb14-reg"
        image_size      int     e.g. 224
        layers          int[]   e.g. [6, 9, 11]
        n_images        int

    /images
        paths           str[]   absolute paths to source images
        labels          str[]   ImageNet class name per image
        class_ids       int[]   ImageNet class index per image

    /activations
        layer_{L}       float32  shape (n_images, seq_len, d_model)
                        seq_len = (image_size/patch_size)^2
                                  + n_registers + 1  (for CLS token)

Public API (implement these)
-----------------------------
build_cache(image_paths, labels, class_ids,
            output_path=None, layers=None, batch_size=32) -> str
    Run DINOv2 on all images and save residual stream activations
    for the target layers to an HDF5 file.

    Steps:
        1. Create the HDF5 file with pre-allocated datasets
           (use h5py chunked storage for efficient slice access).
        2. Process images in batches — call model.run_with_cache()
           and write each batch to disk immediately to avoid OOM.
        3. Store metadata and image index.
    Returns the path to the created file.

load_layer(layer, indices=None, cache_path=None) -> torch.Tensor
    Load activations for a given layer.
    indices: optional list of row indices (load a subset).
    Returns shape (n, seq_len, d_model), float32 torch.Tensor.
    NOTE: h5py requires sorted indices for fancy indexing.

load_metadata(cache_path=None) -> dict
    Return the metadata group as a plain dict.

load_image_index(cache_path=None) -> dict
    Return {"paths": str[], "labels": str[], "class_ids": int[]}

get_class_indices(class_name, cache_path=None) -> list[int]
    Return cache row indices for all images of a given class.
    Used by causal.py to load only flamingo or spoonbill activations.

Implementation notes
--------------------
- Use h5py chunked datasets — chunk size = (batch_size, seq_len, d_model).
  This makes row-slice reads fast (which is the common read pattern).
- Use tqdm to show progress during build_cache — it takes a while.
- Do a first forward pass on a single image to infer seq_len and
  d_model before pre-allocating the datasets.
- model.run_with_cache() returns an activation dict; the key for
  layer L's residual stream is "blocks.{L}.hook_resid_post".
  Confirm this key format with Person A once model.py is working.
- Inform Person B of the confirmed key format so features.py
  can read activations correctly.

Depends on: src/config.py, src/model.py
Used by:    src/features.py, src/causal.py, src/circuits.py,
            notebooks/02_feature_analysis.ipynb,
            notebooks/03_causal_features.ipynb
"""

# TODO (Person C, Week 1 Days 2–4):
#   1. Implement build_cache() — test with 100 images before full 5,000.
#   2. Implement load_layer(), load_metadata(), load_image_index().
#   3. Implement get_class_indices().
#   4. Verify: load a slice and confirm shape (n, seq_len, d_model).
#   5. Post confirmed HDF5 schema + hook key format to report/notes/
#      before end of Day 4 so A and B can proceed.
