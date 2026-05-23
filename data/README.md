# Data

This directory is NOT tracked by git. Do not commit datasets.

## Setup

Download the ImageNet 2012 validation split and place it here:

```
data/
└── imagenet_val/
    ├── n02007558/     # flamingo (200 images minimum)
    ├── n02006656/     # spoonbill (200 images minimum)
    └── ...            # other classes for the 5,000-image cache
```

Update `data.imagenet_val_path` in `configs/default.yaml` if you
place the folder elsewhere.

## ImageNet class IDs

| Class      | Synset ID  |
|------------|------------|
| flamingo   | n02007558  |
| spoonbill  | n02006656  |

## Download

ImageNet requires registration at https://image-net.org.
On Colab, mount your Google Drive and point the config path there.
Do not re-download per session — store once in a shared Drive folder.

---

## SAE Weights

Pre-trained DINO SAE weights are stored outside this directory at:

```
outputs/saes/
├── layer_4/
│   ├── weights.pt    (~288 MB)
│   └── config.json
├── layer_6/
│   ├── weights.pt    (~288 MB)
│   └── config.json
└── layer_9/
    ├── weights.pt    (~288 MB)
    └── config.json
```

The `outputs/` directory is git-ignored. Download weights before running any notebook.

### Download command

```bash
# All target layers (4, 6, 9) — ~864 MB total
python utils/download_saes.py

# Single layer (recommended for first run)
python utils/download_saes.py --layers 9

# Force re-download if files are present but corrupted
python utils/download_saes.py --layers 9 --force
```

Weights are fetched from the `Prisma-Multimodal` organisation on HuggingFace.
Repo IDs are stored in `cfg.sae.sae_repos` in `configs/default.yaml`.
No HuggingFace account is required (repos are public).
