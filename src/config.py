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
    cfg.model.name          # "facebook/dinov2-vitb14-reg"
    cfg.sae.primary_layer   # 11
    cfg.circuit.layer_pairs # [[6, 9], [9, 11]]

DEPENDENCIES: pip install pyyaml
"""


def get_config(path=None):
    raise NotImplementedError("Implement get_config() — see docstring above.")
