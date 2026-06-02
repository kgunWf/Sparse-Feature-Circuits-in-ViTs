import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

# ImageNet-1k label indices for our two behavior classes
CLASSES = {
    "flamingo":  130,   # n02007558
    "spoonbill": 129,   # n02006656
}

N_TARGET = 200  # target images per class


def _pixel_hash(img: Image.Image) -> str:
    """MD5 of raw RGB pixel values — invariant to JPEG encoder settings."""
    return hashlib.md5(np.array(img.convert("RGB")).tobytes()).hexdigest()


def _build_hash_set(cls_dir: Path) -> set:
    hashes = set()
    for p in cls_dir.glob("*.JPEG"):
        try:
            with Image.open(p) as img:
                hashes.add(_pixel_hash(img))
        except Exception:
            pass
    return hashes


def download_images(out_dir="data/imagenet_val", n=N_TARGET):
    """Download up to n unique images per class from ImageNet validation then train splits.

    Counts images already present (e.g. from an extracted zip) and only fetches
    what is missing. Deduplicates by pixel-content hash so images already present
    on disk are never re-downloaded under a different filename, even if the zip and
    the stream share some content.
    Streaming without shuffle is deterministic — the gap images are always the same.
    """
    out_dir = Path(out_dir)

    counts = {}
    seen: dict[str, set] = {}
    for cls in CLASSES:
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        counts[cls] = sum(1 for _ in cls_dir.glob("*.JPEG"))
        seen[cls]   = _build_hash_set(cls_dir)
        print(f"{cls}: {counts[cls]} images present, {len(seen[cls])} unique hashes indexed")

    for split in ("validation", "train"):
        if all(counts[cls] >= n for cls in CLASSES):
            break

        print(f"\nStreaming {split} split ...")
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            streaming=True,
        )

        for example in dataset:
            if all(counts[cls] >= n for cls in CLASSES):
                break
            for cls, label_idx in CLASSES.items():
                if example["label"] == label_idx and counts[cls] < n:
                    h = _pixel_hash(example["image"])
                    if h in seen[cls]:
                        continue  # already on disk — skip
                    path = out_dir / cls / f"{cls}_{counts[cls]:03d}.JPEG"
                    example["image"].save(str(path), format="JPEG")
                    seen[cls].add(h)
                    counts[cls] += 1
                    print(f"  saved {path.name}")

    print("\nDone.", {cls: f"{c}/{n}" for cls, c in counts.items()})


if __name__ == "__main__":
    download_images()
    exit(0)
