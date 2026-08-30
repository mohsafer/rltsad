"""LSTM-based Deep Q-Network (paper Sec. III-A / IV-B).

The Q-network processes the action-augmented state s_a (n_steps, d + 1)
through an LSTM with 64 hidden units (paper Sec. IV-B) and a dense output
layer producing Q(s,0) and Q(s,1).

Action-value convention (identical to ``myasp-wadi.py``'s selection rule
``a = 1 if q1[1] > q0[0] else 0``): feeding the window augmented with the
indicator of action ``a`` yields the action-value of *that* action through
the corresponding output head,

    Q(s, a) = QNet(s_a)[a],

so the greedy policy compares ``QNet(s_0)[0]`` against ``QNet(s_1)[1]`` and
the Bellman backup of the taken action reads

    target(a) = r[a] + gamma * max(Q(s'_0)[0], Q(s'_1)[1])

(no bootstrap at terminal steps).  ``q_values_pair`` implements this readout
batched over many windows.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.env import state_pair


class QNetwork(nn.Module):
    def __init__(self, n_steps: int, n_features: int, n_hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.n_features = int(n_features)
        self.n_hidden_dim = int(n_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=n_features + 1,  # + 1 action-indicator channel
            hidden_size=n_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(n_hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_steps, n_features + 1) -> Q-values (B, 2)."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@torch.no_grad()
def q_values_pair(
    model: QNetwork,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonal action-values Q(s,0) and Q(s,1) for a batch of windows.

    ``windows``: (N, n_steps, d).  Returns two (N,) arrays.  Both candidate
    states of a window are evaluated in one forward pass.
    """
    model.eval()
    windows = np.asarray(windows, dtype=np.float32)
    n = windows.shape[0]
    q0 = np.empty(n, dtype=np.float32)
    q1 = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        chunk = windows[start : start + batch_size]
        pairs = np.stack([state_pair(w) for w in chunk], axis=0)  # (B, 2, T, d+1)
        b = pairs.shape[0]
        x = torch.as_tensor(pairs.reshape(b * 2, *pairs.shape[2:]), device=device)
        q = model(x)  # (2B, 2)
        q = q.view(b, 2, 2)
        q0[start : start + batch_size] = q[:, 0, 0].cpu().numpy()
        q1[start : start + batch_size] = q[:, 1, 1].cpu().numpy()
    return q0, q1


@torch.no_grad()
def q_values_max_next(
    target: QNetwork,
    next_windows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """V(s') = max_a Q(s', a) under the target network for a batch of windows.

    This is the bootstrap term of the Bellman backup (both candidate states
    are evaluated; terminal windows must be filtered out by the caller).
    """
    q0, q1 = q_values_pair(target, next_windows, device)
    return np.maximum(q0, q1)


def sync_target(policy: QNetwork, target: QNetwork) -> None:
    """Q' <- Q hard sync (Algorithm 1 line 32)."""
    target.load_state_dict(policy.state_dict())


# --------------------------------------------------------------------------- #
# Persistence (models/dqn/<name>/: q_network.pt + meta.json)
# --------------------------------------------------------------------------- #


def save_q_network(out_dir: str, model: QNetwork, meta: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "q_network.pt"))
    full_meta = dict(meta)
    full_meta.update(
        {
            "n_steps": model.n_steps,
            "n_features": model.n_features,
            "n_hidden_dim": model.n_hidden_dim,
            "architecture": "lstm-dqn",
            "format_version": 1,
        }
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2)
    return out_dir


def load_q_network(
    model_dir: str,
    map_location: str = "cpu",
) -> Tuple[QNetwork, Dict[str, Any]]:
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = QNetwork(
        n_steps=meta["n_steps"],
        n_features=meta["n_features"],
        n_hidden_dim=meta.get("n_hidden_dim", 64),
    )
    state = torch.load(
        os.path.join(model_dir, "q_network.pt"),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, meta
