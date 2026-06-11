# Data

This directory is NOT tracked by git. Do not commit datasets.

## Setup

Download the ImageNet 2012 validation split and place it here:

```
data/
└── imagenet_val/
    ├── n01440764/     # any class — 5,000 images sampled across classes
    ├── n01443537/
    └── ...
```

Update `data.imagenet_val_path` in `configs/default.yaml` if you place the folder elsewhere.

## Usage

The 5,000-image cache (`outputs/caches/activations.h5`) is built from a random
sample across ImageNet val classes. No specific class is required. The same
dataset is used for both DINO and CLIP pipelines.

## Download

ImageNet requires registration at https://image-net.org.
On Colab, mount your Google Drive and point the config path there.
Do not re-download per session — store once in a shared Drive folder.
