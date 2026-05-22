from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import get_config
from src.model import get_model, preprocess_image


HOOK_KEY_TEMPLATE = "blocks.{layer}.hook_resid_post"


def _get_cfg():
    return get_config()


def _resolve_cache_path(cachepath: str | Path | None = None) -> Path:
    cfg = _get_cfg()
    path = Path(cachepath) if cachepath is not None else Path(cfg.outputs.cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_layers(layers: Iterable[int] | None = None) -> list[int]:
    cfg = _get_cfg()
    if layers is None:
        return [int(x) for x in cfg.sae.target_layers]
    return [int(layer) for layer in layers]


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _decode_if_bytes(x: Any):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _load_batch_inputs(model, imagepaths: list[str]) -> torch.Tensor:
    batch = []

    for imagepath in imagepaths:
        x = preprocess_image(imagepath)

        if isinstance(x, dict):
            if "pixel_values" in x:
                x = x["pixel_values"]
            else:
                raise KeyError("preprocess_image returned a dict without 'pixel_values'.")

        if not isinstance(x, torch.Tensor):
            raise TypeError("preprocess_image must return a torch.Tensor or dict with 'pixel_values'.")

        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim != 3:
            raise ValueError(f"Expected preprocessed image shape (C, H, W), got {tuple(x.shape)}")

        batch.append(x)

    pixelvalues = torch.stack(batch, dim=0)
    device = next(model.parameters()).device
    return pixelvalues.to(device)


def _get_probe_shape(model, sampleimagepath: str, samplelayer: int) -> tuple[int, int]:
    pixelvalues = _load_batch_inputs(model, [sampleimagepath])

    with torch.no_grad():
        _, cache = model.run_with_cache(pixelvalues)

    key = HOOK_KEY_TEMPLATE.format(layer=samplelayer)
    if key not in cache:
        residkeys = [k for k in cache.keys() if "resid" in k]
        raise KeyError(
            f"Hook key '{key}' not found in activation cache. "
            f"Available residual keys include: {residkeys[:20]}"
        )

    acts = cache[key]
    if acts.ndim != 3:
        raise ValueError(f"Expected activation shape (batch, seq_len, d_model), got {tuple(acts.shape)}")

    _, seqlen, dmodel = acts.shape
    return int(seqlen), int(dmodel)


def build_cache(
    imagepaths,
    labels,
    classids,
    outputpath=None,
    layers=None,
    batchsize: int = 32,
) -> str:
    """
    Run DINOv2 on all images and save residual stream activations
    for the target layers to an HDF5 file.

    Returns the path to the created cache file.
    """
    if not (len(imagepaths) == len(labels) == len(classids)):
        raise ValueError("imagepaths, labels, and classids must have the same length.")
    if len(imagepaths) == 0:
        raise ValueError("imagepaths is empty.")

    cfg = _get_cfg()
    outputpath = _resolve_cache_path(outputpath)
    layers = _resolve_layers(layers)
    nimages = len(imagepaths)

    model = get_model()
    model.eval()

    seqlen, dmodel = _get_probe_shape(model, imagepaths[0], layers[0])

    with h5py.File(outputpath, "w") as f:
        strdtype = _string_dtype()

        metadata = f.create_group("metadata")
        metadata.create_dataset("modelname", data=str(cfg.model.name), dtype=strdtype)
        metadata.create_dataset("imagesize", data=int(cfg.model.image_size))
        metadata.create_dataset("layers", data=np.asarray(layers, dtype=np.int32))
        metadata.create_dataset("nimages", data=int(nimages))

        imagesgroup = f.create_group("images")
        imagesgroup.create_dataset("paths", data=np.asarray(list(imagepaths), dtype=object), dtype=strdtype)
        imagesgroup.create_dataset("labels", data=np.asarray(list(labels), dtype=object), dtype=strdtype)
        imagesgroup.create_dataset("classids", data=np.asarray(list(classids), dtype=np.int32))

        activations = f.create_group("activations")
        for layer in layers:
            activations.create_dataset(
                f"layer_{layer}",
                shape=(nimages, seqlen, dmodel),
                dtype=np.float32,
                chunks=(min(batchsize, nimages), seqlen, dmodel),
                compression="gzip",
            )

        for start in tqdm(range(0, nimages, batchsize), desc="Building activation cache"):
            end = min(start + batchsize, nimages)
            batchpaths = list(imagepaths[start:end])

            pixelvalues = _load_batch_inputs(model, batchpaths)

            with torch.no_grad():
                _, cache = model.run_with_cache(pixelvalues)

            for layer in layers:
                key = HOOK_KEY_TEMPLATE.format(layer=layer)
                if key not in cache:
                    residkeys = [k for k in cache.keys() if "resid" in k]
                    raise KeyError(
                        f"Hook key '{key}' not found in activation cache. "
                        f"Available residual keys include: {residkeys[:20]}"
                    )

                acts = cache[key].detach().to(torch.float32).cpu().numpy()
                activations[f"layer_{layer}"][start:end] = acts

            del pixelvalues
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return str(outputpath)


def load_layer(layer: int, indices=None, cachepath=None) -> torch.Tensor:
    """
    Load activations for a given layer.

    Returns shape (n, seq_len, d_model), float32 torch.Tensor.
    """
    cachepath = _resolve_cache_path(cachepath)

    with h5py.File(cachepath, "r") as f:
        dataset = f["activations"][f"layer_{int(layer)}"]

        if indices is None:
            array = dataset[:]
        else:
            indices = list(indices)
            if len(indices) == 0:
                shape = (0,) + dataset.shape[1:]
                return torch.empty(shape, dtype=torch.float32)

            sortedpositions = np.argsort(indices)
            sortedindices = np.asarray(indices, dtype=np.int64)[sortedpositions]
            sortedarray = dataset[sortedindices]

            inverse = np.argsort(sortedpositions)
            array = sortedarray[inverse]

    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def load_metadata(cachepath=None) -> dict:
    """
    Return the metadata group as a plain dict.
    """
    cachepath = _resolve_cache_path(cachepath)

    with h5py.File(cachepath, "r") as f:
        meta = f["metadata"]
        return {
            "modelname": _decode_if_bytes(meta["modelname"][()]),
            "imagesize": int(meta["imagesize"][()]),
            "layers": [int(x) for x in meta["layers"][:]],
            "nimages": int(meta["nimages"][()]),
        }


def load_image_index(cachepath=None) -> dict:
    """
    Return {"paths": str[], "labels": str[], "classids": int[]}
    """
    cachepath = _resolve_cache_path(cachepath)

    with h5py.File(cachepath, "r") as f:
        images = f["images"]
        return {
            "paths": [_decode_if_bytes(x) for x in images["paths"][:]],
            "labels": [_decode_if_bytes(x) for x in images["labels"][:]],
            "classids": [int(x) for x in images["classids"][:]],
        }


def get_class_indices(classname: str, cachepath=None) -> list[int]:
    """
    Return cache row indices for all images of a given class name.
    """
    index = load_image_index(cachepath)
    target = classname.strip().lower()
    return [i for i, label in enumerate(index["labels"]) if str(label).strip().lower() == target]
