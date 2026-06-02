"""
model.py  —  Owner: Person A  —  Week 1

PURPOSE: Single entry point for loading the configured DINO ViT via Prisma HookedViT.
Never load the model anywhere else. Cache after first call.

WHAT TO IMPLEMENT
-----------------
get_model(model_name=None, device=None)
    Load HookedViT via Prisma. Auto-detect CUDA/CPU.
    Cache so repeated calls return the same object.
    Prisma call: HookedViT.from_pretrained(name, is_timm=False, is_clip=False)
    Return: Prisma HookedViT with hooks registered.

preprocess_image(image_path, image_size=None)
    Load image → (1, 3, H, W) float tensor.
    Resize to cfg.model.image_size, normalise with ImageNet mean/std.

TESTS (notebooks/01_sae_setup.ipynb)
  - get_model() runs without error on a single image
  - logits, cache = model.run_with_cache(pixels) succeeds
  - cache contains key f"blocks.{cfg.sae.primary_layer}.hook_resid_post"
  - preprocess_image() returns shape (1, 3, 224, 224)
  - get_model() twice returns same object (cache check)

DEPENDENCIES: pip install vit-prisma torch torchvision pillow
"""

import re

import torch
import torchvision.transforms as T
from PIL import Image
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.models.model_loader import load_hooked_model
import vit_prisma.models.model_loader as _model_loader

from src.config import get_config


def _remap_dino_keys(state_dict):
    """Remap newer HF transformers DINO key names to the format convert_dino_weights expects."""
    _ATTN = {
        "attention.q_proj": "attention.attention.query",
        "attention.k_proj": "attention.attention.key",
        "attention.v_proj": "attention.attention.value",
        "attention.o_proj": "attention.output.dense",
        "mlp.fc1":          "intermediate.dense",
        "mlp.fc2":          "output.dense",
    }
    remapped = {}
    for k, v in state_dict.items():
        new_k = re.sub(r"^layers\.(\d+)\.", lambda m: f"encoder.layer.{m.group(1)}.", k)
        for old, new in _ATTN.items():
            new_k = new_k.replace(old, new)
        remapped[new_k] = v
    return remapped


def _patched_load_dino_weights(model_name, dtype, **kwargs):
    """Load DINO ViT weights directly, bypassing transformers' from_pretrained.

    transformers >= 5.x refuses torch.load on torch < 2.6 (CVE-2025-32434) even
    with weights_only=True, which breaks facebook/dino-vitb16 because its `main`
    revision only ships pytorch_model.bin. We sidestep that gate by fetching the
    checkpoint ourselves: prefer safetensors, else torch.load(weights_only=True),
    which is safe on torch 2.5.x. The raw checkpoint keys are already in the
    `encoder.layer.N...` layout convert_dino_weights expects; _remap_dino_keys is
    a no-op here but kept in case a future revision uses the renamed layout.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    revision = kwargs.get("revision")
    try:
        from safetensors.torch import load_file
        path = hf_hub_download(model_name, "model.safetensors", revision=revision)
        state_dict = load_file(path)
    except EntryNotFoundError:
        path = hf_hub_download(model_name, "pytorch_model.bin", revision=revision)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    return _remap_dino_keys(state_dict)


_model_loader._load_dino_weights = _patched_load_dino_weights

_model = None

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_model(model_name=None, device=None):
    global _model
    if _model is not None:
        return _model

    cfg = get_config()
    if model_name is None:
        model_name = cfg.model.name
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    _model = load_hooked_model(
        model_name,
        model_class=HookedViT,
        pretrained=True,
        device=device,
        allow_failing=True
    )
    _model = _model.to(device)   # load_hooked_model leaves some params on CPU
    n_params = sum(p.numel() for p in _model.parameters())
    print(f"Loaded {model_name} on {device} — {n_params:,} params")
    return _model


def preprocess_image(image_path, image_size=None):
    cfg = get_config()
    if image_size is None:
        image_size = cfg.model.image_size

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)
