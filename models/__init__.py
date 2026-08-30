"""Model definitions and weights for DRSMT.

The reconstruction backbone (the paper's VAE) is pluggable: ``VAE`` (default)
or :class:`TransformerAE` -- see :mod:`models.reconstruction` for the shared
training / penalty / persistence interface used by the pipeline.
"""

from models.vae.vae_model import VAE, WindowScaler
from models.transformer.transformer_model import TransformerAE
from models.reconstruction import (
    compute_penalty_array,
    load_reconstructor,
    reconstruction_errors,
    save_reconstructor,
    train_reconstructor,
)
from models.dqn.q_network import QNetwork
from models.dqn.replay_buffer import ReplayBuffer, Transition

__all__ = [
    "VAE",
    "TransformerAE",
    "WindowScaler",
    "compute_penalty_array",
    "load_reconstructor",
    "reconstruction_errors",
    "save_reconstructor",
    "train_reconstructor",
    "QNetwork",
    "ReplayBuffer",
    "Transition",
]
