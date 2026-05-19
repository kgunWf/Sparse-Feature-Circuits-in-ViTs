"""
model.py  —  Owner: Person A  —  Week 1

PURPOSE: Single entry point for loading DINOv2 ViT-B/14-reg via Prisma HookedViT.
Never load the model anywhere else. Cache after first call.

WHAT TO IMPLEMENT
-----------------
get_model(model_name=None, device=None)
    Load HookedViT via Prisma. Auto-detect CUDA/CPU.
    Cache so repeated calls return the same object.
    Prisma call: HookedViT.from_pretrained(name, is_timm=False, is_clip=False)
    Return: Prisma HookedViT with hooks registered.

get_device()
    Return "cuda" or "cpu".

preprocess_image(image_path, image_size=None)
    Load image → (1, 3, H, W) float tensor.
    Resize to cfg.model.image_size, normalise with ImageNet mean/std.

TESTS (notebooks/01_sae_setup.ipynb)
  - get_model() runs without error on a single image
  - logits, cache = model.run_with_cache(pixels) succeeds
  - cache contains key "blocks.11.hook_resid_post"
  - preprocess_image() returns shape (1, 3, 224, 224)
  - get_model() twice returns same object (cache check)

DEPENDENCIES: pip install prisma-interp torch torchvision pillow
"""

from src.config import get_config


def get_model(model_name=None, device=None):
    raise NotImplementedError("Implement get_model() — see docstring above.")

def get_device():
    raise NotImplementedError("Implement get_device() — see docstring above.")

def preprocess_image(image_path, image_size=None):
    raise NotImplementedError("Implement preprocess_image() — see docstring above.")
