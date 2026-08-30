"""Transformer autoencoder for multivariate reconstruction.

Drop-in replacement for the VAE of the paper (Sec. III-B / IV-A): it is
trained exclusively on normal windows, learns a compact latent representation
and its per-window reconstruction error becomes the unsupervised anomaly score
that feeds the RL agent as the intrinsic reward R2 (scaled by lambda(t)).

Architecture (token = one time step of the multivariate window):

* input: a flattened window of size ``n_steps * d`` is reshaped to the
  sequence ``(n_steps, d)`` of synchronised sensor readings; a linear layer
  embeds every step into ``d_model`` and sinusoidal positional encodings
  restore the temporal order;
* encoder: ``num_encoder_layers`` x TransformerEncoderLayer (self-attention
  over the ``n_steps`` tokens, GELU, pre-processing identical to standard
  post-norm blocks) followed by mean pooling over time;
* latent bottleneck: Linear(d_model -> latent_dim) -- the paper's "compact
  latent representation".  With ``variational=True`` a second head produces
  log sigma^2 (clamped to [-10, 10]) and z is sampled with the
  reparameterisation trick, giving a Transformer-VAE with the same ELBO
  objective as the original VAE; with the default ``variational=False`` the
  model is a deterministic Transformer autoencoder trained with pure
  reconstruction loss;
* decoder: the latent vector is projected back to ``d_model``, replicated to
  ``n_steps`` tokens with positional encodings, passed through
  ``num_decoder_layers`` Transformer blocks, and mapped per step through a
  sigmoid Linear back to the ``d`` sensor channels.

Losses and I/O contract match ``models/vae/vae_model.py`` exactly:

* ``compute_loss(x_flat) -> (loss, recon_mean, kl_mean)`` where the
  reconstruction term is the per-sample *sum* of squared errors (i.e.
  ``mse * original_dim``, the original implementation's scale) plus the KL
  term when variational;
* ``reconstruct(x_flat) -> recon_flat`` for the shared helpers in
  ``models/reconstruction.py`` (penalty arrays, error statistics);
* flattened windows in, flattened windows out -- the WindowScaler, the
  chunked COMPUTEPENALTY logic and the whole DRSMT pipeline are unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding")
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


def _transformer_block(
    d_model: int, nhead: int, dim_feedforward: int, dropout: float
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )


class TransformerAE(nn.Module):
    """Transformer autoencoder over multivariate sliding windows."""

    def __init__(
        self,
        input_dim: int,
        n_steps: int,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        latent_dim: int = 10,
        dropout: float = 0.1,
        variational: bool = False,
    ) -> None:
        super().__init__()
        if input_dim != int(n_steps) * int(n_features):
            raise ValueError(
                f"input_dim {input_dim} != n_steps * n_features "
                f"({n_steps} * {n_features})"
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        self.input_dim = int(input_dim)
        self.n_steps = int(n_steps)
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.variational = bool(variational)

        self.input_embed = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(512, n_steps),
                                          dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            _transformer_block(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_encoder_layers,
        )

        self.to_latent = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim) if variational else None

        self.from_latent = nn.Linear(latent_dim, d_model)
        self.pos_dec = PositionalEncoding(d_model, max_len=max(512, n_steps),
                                          dropout=dropout)
        self.decoder = nn.TransformerEncoder(
            _transformer_block(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers,
        )
        self.output = nn.Linear(d_model, n_features)

    # ------------------------------------------------------------------ flow
    def encode(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Flattened windows -> (mu, logvar|None, z)."""
        batch = x_flat.shape[0]
        seq = x_flat.view(batch, self.n_steps, self.n_features)
        h = self.pos_enc(self.input_embed(seq))
        h = self.encoder(h)                     # (B, T, d_model)
        pooled = h.mean(dim=1)                  # (B, d_model)
        mu = self.to_latent(pooled)
        if self.variational:
            logvar = torch.clamp(self.to_logvar(pooled), -10.0, 10.0)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            logvar = None
            z = mu
        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        h = self.from_latent(z)                             # (B, d_model)
        seq = h.unsqueeze(1).expand(batch, self.n_steps, self.d_model)
        seq = self.pos_dec(seq)                             # positional variety
        h = self.decoder(seq)                               # (B, T, d_model)
        out = torch.sigmoid(self.output(h))                 # (B, T, n_features)
        return out.reshape(batch, self.input_dim)

    def forward(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mu, logvar, z = self.encode(x_flat)
        recon = self.decode(z)
        return recon, mu, logvar

    def reconstruct(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Reconstruction only (used by the shared penalty helpers)."""
        recon, _, _ = self.forward(x_flat)
        return recon

    # ----------------------------------------------------------------- loss
    def compute_loss(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same loss contract as VAE.elbo_loss.

        Reconstruction = per-sample sum of squared errors over the flattened
        window (== mse * original_dim); the KL term is added when
        ``variational=True``.  All terms are means over the batch.
        """
        recon, mu, logvar = self.forward(x_flat)
        recon_elem = F.mse_loss(recon, x_flat, reduction="none")
        recon_loss = recon_elem.sum(dim=1)
        loss = recon_loss.mean()
        if self.variational and logvar is not None:
            kl_loss = -0.5 * torch.sum(
                1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
            )
            kl = kl_loss.mean()
            loss = loss + kl
        else:
            kl = torch.zeros((), device=x_flat.device)
        return loss, recon_loss.mean(), kl


# --------------------------------------------------------------------------- #
# Training helper (mirrors train_vae; kept here for the transformer backbone)
# --------------------------------------------------------------------------- #


def train_transformer(
    model: TransformerAE,
    windows: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> list:
    """Minimise the reconstruction (+ KL) loss on flattened normal windows."""
    from models.reconstruction import train_reconstructor

    return train_reconstructor(
        model, windows,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        grad_clip=grad_clip, device=device, verbose=verbose,
    )
