"""
sae.py  —  Owner: Person A  —  Week 1

PURPOSE: Expose encode / decode / ablate as clean primitives.
causal.py and circuits.py call these — never load SAEs directly elsewhere.
Keep focused on single-sample ops; loops belong in causal.py / circuits.py.

WHAT TO IMPLEMENT
-----------------
get_sae(layer=None, device=None)
    Load and cache Prisma SAE for given layer (default: cfg.sae.primary_layer).
    Prisma call: from prisma.saes import load_sae

encode(activations, layer=None)
    (batch, seq_len, d_model) → (..., d_sae). Most values 0 (sparse).

decode(features, layer=None)
    (..., d_sae) → (..., d_model). Approximate reconstruction.

ablate_feature(activations, feature_idx, layer=None)
    encode → clone → set feature_idx to 0 → decode → return.
    Input/output shape: (batch, seq_len, d_model)

get_l0_sparsity(activations, layer=None)
    Mean active features per token. Return: float.

get_reconstruction_loss(activations, layer=None)
    ||x - decode(encode(x))||^2 / ||x||^2. Return: float. Target < 0.05.

TESTS (notebooks/01_sae_setup.ipynb)
  - get_sae(11) loads; sae.W_enc exists
  - encode() output is sparse
  - decode(encode(x)) approx reconstructs x
  - ablate_feature zeros target feature, leaves others unchanged
  - get_l0_sparsity() < cfg.sae.l0_target
  - get_reconstruction_loss() < 0.05

DEPENDENCIES: pip install prisma-interp torch
"""

from src.config import get_config


def get_sae(layer=None, device=None):
    raise NotImplementedError("Implement get_sae() — see docstring above.")

def encode(activations, layer=None):
    raise NotImplementedError("Implement encode() — see docstring above.")

def decode(features, layer=None):
    raise NotImplementedError("Implement decode() — see docstring above.")

def ablate_feature(activations, feature_idx, layer=None):
    raise NotImplementedError("Implement ablate_feature() — see docstring above.")

def get_l0_sparsity(activations, layer=None):
    raise NotImplementedError("Implement get_l0_sparsity() — see docstring above.")

def get_reconstruction_loss(activations, layer=None):
    raise NotImplementedError("Implement get_reconstruction_loss() — see docstring above.")
