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

| Class     | Synset ID |
| --------- | --------- |
| flamingo  | n02007558 |
| spoonbill | n02006656 |

## Download

ImageNet requires registration at https://image-net.org.
On Colab, mount your Google Drive and point the config path there.
Do not re-download per session — store once in a shared Drive folder.

---
