"""Architecture-agnostic reconstruction-model interface for DRSMT.

The reconstruction model of the paper (the VAE, Sec. III-B / IV-A) is
pluggable: ``--model vae`` keeps the original Variational Autoencoder and
``--model transformer`` swaps in the Transformer autoencoder of
``models/transformer/transformer_model.py``.  Both backbones share the same
I/O contract (flattened windows in/out, :class:`~models.vae.vae_model.WindowScaler`
for scaling, per-window element-mean MSE as the anomaly score R2), so the
WARMUP / TRAINRL / VALIDATE stages of the pipeline are completely unchanged.

This module provides the shared pieces:

* :func:`train_reconstructor` -- one training loop for any model exposing
  ``compute_loss(x_flat) -> (loss, recon_mean, aux_mean)``;
* :func:`reconstruction_errors` -- per-window element-mean MSE via
  ``model.reconstruct(x_flat)``;
* :func:`compute_penalty_array` -- the chunked COMPUTEPENALTY of Algorithm 1
  (per-time-step penalty, zero-padded for the first ``n_steps - 1`` steps);
* :func:`save_reconstructor` / :func:`load_reconstructor` -- persistence and
  the factory that dispatches on the ``architecture`` field of ``meta.json``
  (``"vae"`` or ``"transformer_ae"``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch

from models.transformer.transformer_model import TransformerAE
from models.vae.vae_model import VAE, WindowScaler

# --------------------------------------------------------------------------- #
# Training (any backbone with the compute_loss contract)
# --------------------------------------------------------------------------- #


def train_reconstructor(
    model,
    windows: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> list:
    """Minimise the model's reconstruction (+ KL) loss on flattened windows.

    Mirrors the original training loop: Adam optimiser, per-epoch shuffling,
    gradient clipping (the original VAE used ``clipnorm=1.0``).  Returns the
    per-epoch losses.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_all = torch.as_tensor(np.asarray(windows, dtype=np.float32))
    n = x_all.shape[0]
    losses = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss, epoch_recon, batches = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = x_all[idx].to(device)
            optimizer.zero_grad()
            loss, recon, _ = model.compute_loss(batch)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_recon += float(recon.item())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
        if verbose:
            print(
                f"[recon] epoch {epoch + 1:03d}/{epochs} "
                f"loss={losses[-1]:.6f} recon={epoch_recon / max(batches, 1):.6f}"
            )
    model.eval()
    return losses


# --------------------------------------------------------------------------- #
# Inference (any backbone with the reconstruct contract)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def reconstruction_errors(
    model,
    windows_flat: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-window MSE  ||x - x_hat||^2 / (n_steps * d)  (paper Eq. of R2)."""
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    x = np.asarray(windows_flat, dtype=np.float32)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        chunk = x[start : start + batch_size]
        tensor = torch.as_tensor(chunk, device=device)
        recon = model.reconstruct(tensor)
        err = torch.mean((recon - tensor) ** 2, dim=1)
        out[start : start + batch_size] = err.cpu().numpy()
    return out


def compute_penalty_array(
    model,
    scaler: WindowScaler,
    values: np.ndarray,
    n_steps: int,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    window_chunk: int = 4096,
) -> np.ndarray:
    """Per-time-step reconstruction penalty p[t] for a whole series.

    Mirrors Algorithm 1 (COMPUTEPENALTY): slide length-``n_steps`` windows
    over the series, score each with the reconstruction model, and zero-pad
    the first ``n_steps - 1`` steps so the returned array aligns with the
    time axis (p[t] is the error of the window ending at t).  Windows are
    processed in chunks so WADI-scale series never materialise all windows
    at once.
    """
    from utils.data_loader import flatten_windows, make_windows

    values = np.asarray(values, dtype=np.float32)
    T = values.shape[0]
    n_windows = T - n_steps + 1
    errors = np.empty(n_windows, dtype=np.float64)
    for start in range(0, n_windows, window_chunk):
        stop = min(start + window_chunk, n_windows)
        # the sub-array rows [start, stop + n_steps - 1) yield exactly the
        # windows (start .. stop-1) of the full series
        chunk = make_windows(values[start : stop + n_steps - 1], n_steps)
        scaled = scaler.transform(flatten_windows(chunk))
        errors[start:stop] = reconstruction_errors(
            model, scaled, device=device, batch_size=batch_size
        )
    penalty = np.zeros(T, dtype=np.float32)
    penalty[n_steps - 1 :] = errors.astype(np.float32)
    return penalty


# --------------------------------------------------------------------------- #
# Persistence + factory (models/<kind>/<run>/: weights + scaler.pkl + meta.json)
# --------------------------------------------------------------------------- #


def save_reconstructor(
    out_dir: str,
    model,
    scaler: WindowScaler,
    meta: Dict[str, Any],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    architecture = str(meta.get("architecture", "vae"))
    torch.save(model.state_dict(), os.path.join(out_dir, f"{architecture}.pt"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    full_meta = dict(meta)
    full_meta.setdefault("architecture", architecture)
    full_meta.setdefault("format_version", 1)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2)
    return out_dir


def load_reconstructor(
    model_dir: str,
    map_location: str = "cpu",
) -> Tuple[object, WindowScaler, Dict[str, Any]]:
    """Factory: rebuild the reconstruction model stored in ``model_dir``.

    Dispatches on ``meta.json``'s ``architecture`` field, so every pipeline
    stage accepts either a VAE directory or a Transformer directory.
    """
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    architecture = str(meta.get("architecture", "vae"))

    if architecture == "vae":
        model = VAE(
            input_dim=meta["input_dim"],
            latent_dim=meta.get("latent_dim", 10),
            intermediate_dim=meta.get("intermediate_dim", 64),
            encoder_layers=meta.get("encoder_layers", 3),
        )
        weights_file = "vae.pt"
    elif architecture == "transformer_ae":
        model = TransformerAE(
            input_dim=meta["input_dim"],
            n_steps=meta["n_steps"],
            n_features=meta["n_features"],
            d_model=meta.get("d_model", 128),
            nhead=meta.get("nhead", 4),
            num_encoder_layers=meta.get("num_encoder_layers", 2),
            num_decoder_layers=meta.get("num_decoder_layers", 2),
            dim_feedforward=meta.get("dim_feedforward", 256),
            latent_dim=meta.get("latent_dim", 10),
            dropout=meta.get("dropout", 0.1),
            variational=meta.get("variational", False),
        )
        weights_file = "transformer_ae.pt"
    else:
        raise ValueError(
            f"unknown reconstruction-model architecture '{architecture}' "
            f"in {model_dir} (expected 'vae' or 'transformer_ae')"
        )

    state = torch.load(
        os.path.join(model_dir, weights_file),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return model, scaler, meta
