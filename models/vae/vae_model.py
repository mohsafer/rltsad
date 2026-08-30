"""Variational Autoencoder for multivariate reconstruction (paper Sec. IV-A).

Architecture (mirrors ``build_vae`` of the original code, paper Sec. IV-A):

* input: a sliding window of shape (n_steps, d) flattened to a vector of
  size n_steps * d;
* encoder: ``vae_encoder_layers`` Dense layers (ReLU, He init) of
  ``intermediate_dim`` units, producing mu(x) and log sigma^2(x)
  (log-variance clamped to [-10, 10]);
* reparameterisation z = mu + exp(0.5 logvar) * eps;
* decoder: Dense (ReLU) layer(s) then a Dense sigmoid layer reconstructing
  the flattened window;
* ELBO loss: reconstruction (sum of squared errors over the window, i.e.
  ``mse * original_dim`` as in the original implementation) + KL divergence.

The per-window reconstruction error  ||x - x_hat||^2 / (n_steps * d)  is the
unsupervised anomaly score that becomes the intrinsic reward R2 of the RL
agent after scaling by the dynamic coefficient lambda(t).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Window scaler (VAE side)
# --------------------------------------------------------------------------- #


class WindowScaler:
    """Feature-wise scaler fitted on *flattened* windows (N, n_steps * d).

    ``"standard"`` reproduces the SMD variant (StandardScaler), ``"robust"``
    the WADI variant (RobustScaler, median/IQR) which avoids overflow when
    later transforming outlier windows.
    """

    def __init__(self, kind: str = "robust", clip: float = 10.0) -> None:
        if kind not in {"standard", "robust"}:
            raise ValueError(f"unknown scaler kind '{kind}'")
        self.kind = kind
        self.clip = float(clip)
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "WindowScaler":
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "standard":
            self.center_ = x.mean(axis=0)
            std = x.std(axis=0)
            self.scale_ = np.where(std < 1e-12, 1.0, std)
        else:  # robust: median and inter-quartile range (sklearn convention)
            self.center_ = np.median(x, axis=0)
            q75, q25 = np.percentile(x, [75.0, 25.0], axis=0)
            iqr = q75 - q25
            self.scale_ = np.where(iqr < 1e-12, 1.0, iqr)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("WindowScaler.transform called before fit")
        scaled = (np.asarray(x, dtype=np.float64) - self.center_) / self.scale_
        if self.clip is not None and self.clip > 0:
            scaled = np.clip(scaled, -self.clip, self.clip)
        return scaled.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# --------------------------------------------------------------------------- #
# VAE module
# --------------------------------------------------------------------------- #


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        intermediate_dim: int = 64,
        encoder_layers: int = 3,
    ) -> None:
        super().__init__()
        if encoder_layers < 1:
            raise ValueError("encoder_layers must be >= 1")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        enc: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(encoder_layers):
            linear = nn.Linear(in_dim, intermediate_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            enc += [linear, nn.ReLU()]
            in_dim = intermediate_dim
        self.encoder_body = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        dec: list[nn.Module] = [nn.Linear(latent_dim, intermediate_dim), nn.ReLU()]
        self.decoder_body = nn.Sequential(*dec)
        self.decoder_out = nn.Linear(intermediate_dim, input_dim)

    # ------------------------------------------------------------------ flow
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder_out(self.decoder_body(z)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # ----------------------------------------------------------------- loss
    def elbo_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO loss exactly as in the original implementation.

        ``recon = sum of squared errors over the flattened window`` (equal to
        ``mse * original_dim``) and the closed-form KL term, both averaged
        over the batch.
        """
        recon, mu, logvar = self.forward(x)
        recon_elem = torch.nn.functional.mse_loss(recon, x, reduction="none")
        recon_loss = recon_elem.sum(dim=1)  # per-sample sum of squared errors
        kl_loss = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        loss = torch.mean(recon_loss + kl_loss)
        return loss, recon_loss.mean(), kl_loss.mean()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction only (shared contract, see models/reconstruction.py)."""
        recon, _, _ = self.forward(x)
        return recon

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias of :meth:`elbo_loss` (shared contract across backbones)."""
        return self.elbo_loss(x)


# --------------------------------------------------------------------------- #
# Training and inference helpers
# --------------------------------------------------------------------------- #


def train_vae(
    model: VAE,
    windows: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> list:
    """Minimise the ELBO on flattened normal windows; returns the epoch losses."""
    device = device or torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # original: Adam(0.001, clipnorm=1.0)
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
            loss, recon, _ = model.elbo_loss(batch)
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
                f"[VAE] epoch {epoch + 1:03d}/{epochs} "
                f"loss={losses[-1]:.6f} recon={epoch_recon / max(batches, 1):.6f}"
            )
    model.eval()
    return losses


@torch.no_grad()
def reconstruction_errors(
    model: VAE,
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
        recon, _, _ = model.forward(tensor)
        err = torch.mean((recon - tensor) ** 2, dim=1)
        out[start : start + batch_size] = err.cpu().numpy()
    return out


def compute_penalty_array(
    model: VAE,
    scaler: WindowScaler,
    values: np.ndarray,
    n_steps: int,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    window_chunk: int = 4096,
) -> np.ndarray:
    """Per-time-step reconstruction penalty p[t] for a whole series.

    Mirrors Algorithm 1 (COMPUTEPENALTY): slide length-``n_steps`` windows
    over the series, score each with the VAE, and zero-pad the first
    ``n_steps - 1`` steps so the returned array aligns with the time axis
    (p[t] is the error of the window ending at t).  Windows are processed in
    chunks so WADI-scale series never materialise all windows at once.
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
# Persistence (models/vae/<name>/: vae.pt + scaler.pkl + meta.json)
# --------------------------------------------------------------------------- #


def save_vae(
    out_dir: str,
    model: VAE,
    scaler: WindowScaler,
    meta: Dict[str, Any],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    full_meta = dict(meta)
    full_meta.update(
        {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "intermediate_dim": meta.get("intermediate_dim", 64),
            "encoder_layers": meta.get("encoder_layers", 3),
            "architecture": "vae",
            "format_version": 1,
        }
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2)
    return out_dir


def load_vae(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[VAE, WindowScaler, Dict[str, Any]]:
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = VAE(
        input_dim=meta["input_dim"],
        latent_dim=meta["latent_dim"],
        intermediate_dim=meta.get("intermediate_dim", 64),
        encoder_layers=meta.get("encoder_layers", 3),
    )
    state = torch.load(
        os.path.join(model_dir, "vae.pt"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return model, scaler, meta
