"""
config.py  —  shared utility, do not modify without consulting the group.

PURPOSE: Single config loader. Every src/ module and notebook imports
get_config() from here. Never read the YAML directly anywhere else.

WHAT TO IMPLEMENT
-----------------
get_config(path=None)
    Read configs/default.yaml (or custom path).
    Return dot-accessible object: cfg.sae.primary_layer not cfg["sae"]["primary_layer"].
    Cache after first load so repeated calls don't re-read disk.

USAGE
-----
    from src.config import get_config
    cfg = get_config()
    cfg.model.name          # "facebook/dino-vitb16"
    cfg.sae.primary_layer   # 9

DEPENDENCIES: pip install pyyaml
"""

from pathlib import Path
import yaml
from types import SimpleNamespace

_cache = None
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"

def get_config(path=None):
    global _cache
    if _cache is not None:
        return _cache
    if path is None:
        path = _DEFAULT_CONFIG
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _cache = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()})
    return _cache
