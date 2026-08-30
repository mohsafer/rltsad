"""Experience replay for the DQN agent (paper Sec. III-A, Algorithm 1 line 29).

A transition stores the *raw* sensor window (n_steps, d); the action-indicator
channel is appended on the fly when batches are built, so the memory stays
compact for high-dimensional datasets (the action indicator of the *next*
state is determined by the action under consideration, hence only one next
window is stored and both candidate branches s'_0, s'_1 are reconstructed in
``sample``).
"""

from __future__ import annotations

import pickle
from collections import deque
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Transition(NamedTuple):
    window: np.ndarray        # (n_steps, d) float32, window ending at t
    action: int               # action taken at t (0 normal, 1 anomaly)
    reward: np.ndarray        # (2,) reward vector [r_0, r_1] incl. lambda * R2
    next_window: Optional[np.ndarray]  # (n_steps, d) window ending at t+1; None at terminal
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.memory: deque = deque(maxlen=self.capacity)

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def extend(self, transitions: Sequence[Transition]) -> None:
        for tr in transitions:
            self.memory.append(tr)

    def is_ready(self, init_size: int) -> bool:
        return len(self.memory) >= init_size

    # ---------------------------------------------------------------- sample
    def sample(
        self, batch_size: int, n_steps: int, n_features: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform minibatch.

        Returns
        -------
        windows      : (B, n_steps, d)        states s (indicator added later)
        actions      : (B,) int64             taken actions a
        rewards      : (B, 2) float32         full reward vectors r
        next_windows : (B, n_steps, d)        s' contents (zeros where terminal)
        dones        : (B,) bool              terminal flags
        """
        if batch_size >= len(self.memory):
            idx = np.arange(len(self.memory))
        else:
            # uniform sample *without* replacement, like random.sample in the
            # original implementation
            idx = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        windows = np.stack([b.window for b in batch]).astype(np.float32)
        actions = np.asarray([int(b.action) for b in batch], dtype=np.int64)
        rewards = np.stack([np.asarray(b.reward, dtype=np.float32) for b in batch])
        next_windows = np.zeros(
            (len(batch), n_steps, n_features), dtype=np.float32
        )
        for i, b in enumerate(batch):
            if b.next_window is not None:
                next_windows[i] = b.next_window
        dones = np.asarray([bool(b.done) for b in batch], dtype=bool)
        return windows, actions, rewards, next_windows, dones

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> str:
        with open(path, "wb") as fh:
            pickle.dump(
                {"capacity": self.capacity, "memory": list(self.memory)}, fh
            )
        return path

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        buffer = cls(payload.get("capacity", len(payload["memory"]) or 1))
        buffer.memory = deque(payload["memory"], maxlen=buffer.capacity)
        return buffer
