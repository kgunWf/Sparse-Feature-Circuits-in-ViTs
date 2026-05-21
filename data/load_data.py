from pathlib import Path
from datasets import load_dataset

# ImageNet-1k label indices for our two behavior classes
CLASSES = {
    "flamingo":  130,   # n02007558
    "spoonbill": 129,   # n02006656
}

N_IMAGES = 5   # images per class — enough for smoke tests

def download_test_images(out_dir="data/imagenet_val", n=N_IMAGES):
    out_dir = Path(out_dir)

    # streaming=True avoids pulling the full ~150GB before filtering
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    counts = {cls: 0 for cls in CLASSES}
    for example in dataset:
        for cls, label_idx in CLASSES.items():
            if example["label"] == label_idx and counts[cls] < n:
                save_dir = out_dir / cls
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / f"{cls}_{counts[cls]:03d}.JPEG"
                example["image"].save(str(path), format="JPEG")
                counts[cls] += 1
                print(f"Saved {path}")

        if all(v >= n for v in counts.values()):
            break

    print("Done.", {cls: f"{c}/{n}" for cls, c in counts.items()})


if __name__ == "__main__":
    download_test_images()
    exit(0)
