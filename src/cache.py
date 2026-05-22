from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import get_config
from src.model import get_model


HOOK_KEY_TEMPLATE = "blocks.{layer}.hook_resid_post"


def _get_cfg():
    return get_config()


def _resolve_cache_path(cache_path: str | Path | None = None) -> Path:
    cfg = _get_cfg()
    path = Path(cache_path) if cache_path is not None else Path(cfg.outputs.cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_layers(layers: Iterable[int] | None = None) -> list[int]:
    cfg = _get_cfg()
    if layers is None:
        return list(cfg.sae.target_layers)
    return [int(layer) for layer in layers]


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _decode_if_bytes(x: Any):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _load_batch_inputs(model, image_paths: list[str]) -> torch.Tensor:
    batch = []
    for image_path in image_paths:
        x = model.preprocess_image(image_path)
        if isinstance(x, dict):
            if "pixel_values" in x:
                x = x["pixel_values"]
            else:
                raise KeyError("model.preprocess_image returned a dict without 'pixel_values'.")
        if not isinstance(x, torch.Tensor):
            raise TypeError("model.preprocess_image must return a torch.Tensor or dict with 'pixel_values'.")
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected preprocessed image shape (C,H,W), got {tuple(x.shape)}")
        batch.append(x)

    pixel_values = torch.stack(batch, dim=0)
    device = next(model.parameters()).device
    return pixel_values.to(device)


def _get_probe_shape(model, sample_image_path: str, sample_layer: int) -> tuple[int, int]:
    pixel_values = _load_batch_inputs(model, [sample_image_path])
    with torch.no_grad():
        _, cache = model.run_with_cache(pixel_values)

    key = HOOK_KEY_TEMPLATE.format(layer=sample_layer)
    if key not in cache:
        resid_keys = [k for k in cache.keys() if "resid" in k]
        raise KeyError(
            f"Hook key '{key}' not found in activation cache. "
            f"Available residual keys include: {resid_keys[:20]}"
        )

    acts = cache[key]
    if acts.ndim != 3:
        raise ValueError(f"Expected activation shape (batch, seq_len, d_model), got {tuple(acts.shape)}")

    _, seq_len, d_model = acts.shape
    return int(seq_len), int(d_model)


def build_cache(
    image_paths,
    labels,
    class_ids,
    output_path=None,
    layers=None,
    batch_size: int = 32,
) -> str:
    """
    Run DINO on all images and save residual stream activations
    for the target layers to an HDF5 file.

    Returns the path to the created cache file.
    """
    if not (len(image_paths) == len(labels) == len(class_ids)):
        raise ValueError("image_paths, labels, and class_ids must have the same length.")
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty.")

    cfg = _get_cfg()
    output_path = _resolve_cache_path(output_path)
    layers = _resolve_layers(layers)
    n_images = len(image_paths)

    model = get_model()
    model.eval()

    seq_len, d_model = _get_probe_shape(model, image_paths[0], layers[0])

    with h5py.File(output_path, "w") as f:
        str_dtype = _string_dtype()

        metadata = f.create_group("metadata")
        metadata.create_dataset("model_name", data=str(cfg.model.name), dtype=str_dtype)
        metadata.create_dataset("image_size", data=int(cfg.model.image_size))
        metadata.create_dataset("layers", data=np.asarray(layers, dtype=np.int32))
        metadata.create_dataset("n_images", data=int(n_images))

        images_group = f.create_group("images")
        images_group.create_dataset("paths", data=np.asarray(list(image_paths), dtype=object), dtype=str_dtype)
        images_group.create_dataset("labels", data=np.asarray(list(labels), dtype=object), dtype=str_dtype)
        images_group.create_dataset("class_ids", data=np.asarray(list(class_ids), dtype=np.int32))

        acts_group = f.create_group("activations")
        for layer in layers:
            acts_group.create_dataset(
                f"layer_{layer}",
                shape=(n_images, seq_len, d_model),
                dtype=np.float32,
                chunks=(min(batch_size, n_images), seq_len, d_model),
                compression="gzip",
            )

        for start in tqdm(range(0, n_images, batch_size), desc="Building activation cache"):
            end = min(start + batch_size, n_images)
            batch_paths = list(image_paths[start:end])

            pixel_values = _load_batch_inputs(model, batch_paths)

            with torch.no_grad():
                _, cache = model.run_with_cache(pixel_values)

            for layer in layers:
                key = HOOK_KEY_TEMPLATE.format(layer=layer)
                if key not in cache:
                    resid_keys = [k for k in cache.keys() if "resid" in k]
                    raise KeyError(
                        f"Hook key '{key}' not found in activation cache. "
                        f"Available residual keys include: {resid_keys[:20]}"
                    )

                acts = cache[key].detach().to(torch.float32).cpu().numpy()
                acts_group[f"layer_{layer}"][start:end] = acts

            del pixel_values
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return str(output_path)


def load_layer(layer: int, indices=None, cache_path=None) -> torch.Tensor:
    """
    Load activations for a given layer.

    Returns shape (n, seq_len, d_model), float32 torch.Tensor.
    """
    cache_path = _resolve_cache_path(cache_path)

    with h5py.File(cache_path, "r") as f:
        dataset = f["activations"][f"layer_{int(layer)}"]

        if indices is None:
            array = dataset[:]
        else:
            indices = list(indices)
            if len(indices) == 0:
                shape = (0,) + dataset.shape[1:]
                return torch.empty(shape, dtype=torch.float32)

            sorted_positions = np.argsort(indices)
            sorted_indices = np.asarray(indices, dtype=np.int64)[sorted_positions]
            sorted_array = dataset[sorted_indices]

            inverse = np.argsort(sorted_positions)
            array = sorted_array[inverse]

    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def load_metadata(cache_path=None) -> dict:
    """
    Return the metadata group as a plain dict.
    """
    cache_path = _resolve_cache_path(cache_path)

    with h5py.File(cache_path, "r") as f:
        meta = f["metadata"]
        return {
            "model_name": _decode_if_bytes(meta["model_name"][()]),
            "image_size": int(meta["image_size"][()]),
            "layers": [int(x) for x in meta["layers"][:]],
            "n_images": int(meta["n_images"][()]),
        }


def load_image_index(cache_path=None) -> dict:
    """
    Return {"paths": str[], "labels": str[], "class_ids": int[]}
    """
    cache_path = _resolve_cache_path(cache_path)

    with h5py.File(cache_path, "r") as f:
        images = f["images"]
        return {
            "paths": [_decode_if_bytes(x) for x in images["paths"][:]],
            "labels": [_decode_if_bytes(x) for x in images["labels"][:]],
            "class_ids": [int(x) for x in images["class_ids"][:]],
        }


def get_class_indices(class_name: str, cache_path=None) -> list[int]:
    """
    Return cache row indices for all images of a given class name.
    """
    index = load_image_index(cache_path)
    target = class_name.strip().lower()
    return [i for i, label in enumerate(index["labels"]) if str(label).strip().lower() == target]
