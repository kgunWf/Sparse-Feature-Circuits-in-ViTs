"""
Download pre-trained SAE weights from HuggingFace.

Usage:
    python utils/download_saes.py              # download all target layers
    python utils/download_saes.py --layers 9   # specific layer(s)
    python utils/download_saes.py --force       # re-download even if present

Saves to: outputs/saes/layer_{N}/weights.pt + config.json
Repo IDs are read from cfg.sae.sae_repos in configs/default.yaml.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without pip install -e
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
import src.config as _cfg_mod
from vit_prisma.sae.sae_utils import download_sae_from_huggingface


def download_layer(layer: int, force: bool = False) -> None:
    cfg = get_config()
    sae_repos = getattr(cfg.sae, "sae_repos", {})

    if layer not in sae_repos:
        print(f"[skip] layer {layer}: no repo ID in cfg.sae.sae_repos")
        return

    repo_id = sae_repos[layer]
    repo_root = _cfg_mod._DEFAULT_CONFIG.parent.parent
    sae_dir = repo_root / "outputs" / "saes" / f"layer_{layer}"
    weights_path = sae_dir / "weights.pt"
    config_path = sae_dir / "config.json"

    if weights_path.exists() and config_path.exists() and not force:
        print(f"[skip] layer {layer}: already present at {sae_dir}")
        return

    print(f"[download] layer {layer} <- {repo_id}")
    sae_dir.mkdir(parents=True, exist_ok=True)
    download_sae_from_huggingface(repo_id, "weights.pt", str(sae_dir))
    download_sae_from_huggingface(repo_id, "config.json", str(sae_dir))
    print(f"[done] layer {layer} -> {sae_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download SAE weights from HuggingFace")
    parser.add_argument(
        "--layers", nargs="+", type=int,
        help="Layer indices to download (default: all cfg.sae.target_layers)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist"
    )
    args = parser.parse_args()

    cfg = get_config()
    layers = args.layers if args.layers else cfg.sae.target_layers

    print(f"Downloading SAEs for layers: {layers}")
    for layer in layers:
        download_layer(layer, force=args.force)

    print("All done.")


if __name__ == "__main__":
    main()
