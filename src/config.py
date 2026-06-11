"""
config.py  —  shared utility, do not modify without consulting the group.

PURPOSE: Single config loader. Every src/ module and notebook imports
get_config() from here. Never read the YAML directly anywhere else.

USAGE
-----
    from src.config import get_config
    cfg = get_config()                          # default.yaml (DINO)
    cfg = get_config("configs/clip_b32.yaml")   # CLIP-B/32
    cfg = get_config("configs/clip_b32.yaml", reload=True)  # force reload

DEPENDENCIES: pip install pyyaml
"""

from pathlib import Path
import yaml
from types import SimpleNamespace

_cache = None
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"


def _dict_to_ns(obj):
    """Recursively convert dicts to SimpleNamespace; leave other types unchanged."""
    if isinstance(obj, dict):
        # Only convert if all keys are valid Python identifiers (string keys).
        # Dicts with integer keys (e.g. sae_repos: {4: ..., 9: ...}) stay as dicts.
        if all(isinstance(k, str) and k.isidentifier() for k in obj):
            return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
        return {k: _dict_to_ns(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dict_to_ns(i) for i in obj]
    return obj


def get_config(path=None, reload=False):
    """Load and return the project config as a dot-accessible namespace.

    Args:
        path:   Path to a YAML config file. Defaults to configs/default.yaml.
        reload: If True, discard the cached config and reload from disk.
                Required when switching between DINO and CLIP configs in the
                same process (e.g. notebook cells that import both models).
    """
    global _cache
    if _cache is not None and not reload:
        return _cache
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(resolved, "r") as f:
        raw = yaml.safe_load(f)
    _cache = _dict_to_ns(raw)
    return _cache
