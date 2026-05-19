# Data

Not tracked by git. Each team member sets this up locally.

## ImageNet validation set

We use 5,000 images from ImageNet-1k val for the activation cache,
and 400 images (200 flamingo + 200 spoonbill) for circuit analysis.

### Download via HuggingFace (recommended for Colab)

```python
from datasets import load_dataset
ds = load_dataset("imagenet-1k", split="validation", streaming=True)
```

### Manual download

Register at https://image-net.org/download.php, download
`ILSVRC2012_img_val.tar` (~6.3 GB) and extract to `data/imagenet_val/`:

```
data/imagenet_val/
├── n02007558/   # flamingo  (class index 130)
├── n02006656/   # spoonbill (class index 129)
└── ...
```
