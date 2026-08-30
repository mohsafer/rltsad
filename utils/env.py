"""RL environment for DRSMT (paper Sec. IV-B, Algorithm 1).

Semantics (identical to the original ``myasp-wadi.py`` / ``env_smd.py`` pair,
with the WADI window alignment):

* The episode classifies time steps ``t = n_steps .. T-1``.  At every step the
  agent observes a sliding window ``w(t) = values[t-n_steps+1 : t+1]`` of
  shape ``(n_steps, d)`` holding synchronised readings of all ``d`` sensors.
* To let the agent distinguish its prediction action, the window is augmented
  with an action indicator channel, so ``s_a(t)`` has shape
  ``(n_steps, d + 1)`` (:func:`augment_state`) and ``s_a in R^{NSTEPS x (d+1)}``.
* Actions are binary: 0 = predict normal, 1 = predict anomaly.  After each
  action the environment shifts the sliding window forward.
* Rewards follow the *semi-supervised* label discipline of the original code:
  the environment keeps a working label column that starts fully unlabelled
  (``-1``) and is filled by the warm-up and by the per-episode active-learning
  / LabelSpreading loop.  The extrinsic reward R1 is computed from that
  working column (Algorithm 1 line 27 "Compute extrinsic rcls from y");
  unlabelled steps yield a zero classification reward.  The intrinsic reward
  R2 is the VAE reconstruction error of the current window scaled by the
  dynamic coefficient lambda(t):

      r = [ R1(0) + lam*p[t],  R1(1) + lam*p[t] ]

  where index 0 is the reward for predicting *normal* and index 1 for
  predicting *anomaly* (TP=10, TN=1, FP=-1, FN=-10), and ``p`` is the
  pre-computed per-step reconstruction-error array (Algorithm 1,
  COMPUTEPENALTY, zero-padded for the first ``n_steps - 1`` steps).
  ``reward_unlabeled="anomaly"`` reproduces the WADI variant of the original
  code, which treats unlabelled steps as anomalies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from utils.data_loader import SeriesData, make_windows

NOT_ANOMALY = 0
ANOMALY = 1
UNLABELED = -1
ACTION_SPACE = (NOT_ANOMALY, ANOMALY)
ACTION_SPACE_N = len(ACTION_SPACE)


def augment_state(window: np.ndarray, action: int) -> np.ndarray:
    """Append the constant action-indicator channel -> (n_steps, d + 1)."""
    window = np.asarray(window, dtype=np.float32)
    indicator = np.full((window.shape[0], 1), float(action), dtype=np.float32)
    return np.concatenate([window, indicator], axis=1).astype(np.float32)


def state_pair(window: np.ndarray) -> np.ndarray:
    """Both action-augmented candidate states stacked -> (2, n_steps, d + 1)."""
    return np.stack([augment_state(window, 0), augment_state(window, 1)], axis=0)


def classification_reward(
    label: float,
    tp: float = 10.0,
    tn: float = 1.0,
    fp: float = -1.0,
    fn: float = -10.0,
    unlabeled: str = "zero",
) -> Tuple[float, float]:
    """Asymmetric extrinsic reward R1 as a vector [predict-normal, predict-anomaly].

    ``label`` comes from the working label column: 0 -> (TN, FP),
    1 -> (FN, TP); unlabelled steps yield (0, 0) unless ``unlabeled ==
    "anomaly"`` (WADI variant of the original code).
    """
    if label == 0:
        return tn, fp
    if label == 1:
        return fn, tp
    if unlabeled == "anomaly":
        return fn, tp
    return 0.0, 0.0


class TimeSeriesEnv:
    """Single-series environment with the DRSMT reward structure."""

    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        penalty: Optional[np.ndarray] = None,
        n_steps: int = 25,
        tp: float = 10.0,
        tn: float = 1.0,
        fp: float = -1.0,
        fn: float = -10.0,
        dynamic_coef: float = 1.0,
        reward_unlabeled: str = "zero",
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        self.values = values
        self.ground_truth = np.asarray(labels, dtype=np.float32)
        # Working (semi-supervised) label column revealed by active learning.
        self.work_labels = np.full(len(self.ground_truth), UNLABELED, dtype=np.float32)
        T = len(self.values)
        if penalty is None:
            penalty = np.zeros(T, dtype=np.float32)
        penalty = np.asarray(penalty, dtype=np.float32).reshape(-1)
        if len(penalty) != T:
            raise ValueError(
                f"penalty array length {len(penalty)} does not match series length {T}"
            )
        self.penalty = penalty
        self.n_steps = int(n_steps)
        if len(self.values) <= self.n_steps:
            raise ValueError(
                f"series length {len(self.values)} must exceed n_steps={self.n_steps}"
            )
        self.tp, self.tn, self.fp, self.fn = tp, tn, fp, fn
        self.reward_unlabeled = reward_unlabeled
        self.dynamic_coef = float(dynamic_coef)
        self.action_space_n = ACTION_SPACE_N
        self.t = self.n_steps  # cursor: index of the time step being classified

    # ------------------------------------------------------------------ info
    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def length(self) -> int:
        return len(self.values)

    def set_dynamic_coef(self, coef: float) -> None:
        """Update lambda(t) between episodes (Algorithm 1, line 34)."""
        self.dynamic_coef = float(coef)

    def reveal(self, index: int, value: float) -> None:
        """Write a (ground-truth or propagated) label into the working column."""
        self.work_labels[int(index)] = float(value)

    def labeled_indices(self) -> np.ndarray:
        return np.where(self.work_labels != UNLABELED)[0]

    def window_at(self, t: int) -> np.ndarray:
        """Window of the ``n_steps`` rows ending (inclusive) at index ``t``."""
        return self.values[t - self.n_steps + 1 : t + 1]

    def get_states_list(self) -> np.ndarray:
        """All classified windows (t = n_steps .. T-1) -> (N, n_steps, d).

        Used by active learning and the warm-up; window ``i`` corresponds to
        time index ``i + n_steps``.
        """
        return make_windows(self.values, self.n_steps)[1:]

    # ------------------------------------------------------------- gym-like
    def reset(self) -> np.ndarray:
        """Restart the episode; returns the first window (n_steps, d)."""
        self.t = self.n_steps
        return self.window_at(self.t)

    def reward_vector(self, t: Optional[int] = None) -> np.ndarray:
        """Reward vector [r_0, r_1] for the step at cursor ``t``.

        r_0 is earned when predicting normal, r_1 when predicting anomaly;
        the VAE reconstruction penalty R2 scaled by lambda is added to both
        entries (Algorithm 1 line 28).
        """
        if t is None:
            t = self.t
        r0, r1 = classification_reward(
            self.work_labels[t],
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            unlabeled=self.reward_unlabeled,
        )
        vae_penalty = self.dynamic_coef * float(self.penalty[t])
        return np.asarray([r0 + vae_penalty, r1 + vae_penalty], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
        """Apply ``action``; returns ``(next_window, reward_vec, done, info)``."""
        reward = self.reward_vector(self.t)
        info = {
            "index": self.t,
            "label": float(self.work_labels[self.t]),
            "ground_truth": float(self.ground_truth[self.t]),
        }
        self.t += 1
        done = int(self.t >= self.length)
        if done:
            next_window = np.zeros((0,), dtype=np.float32)  # terminal marker
        else:
            next_window = self.window_at(self.t)
        return next_window, reward, done, info


def env_from_series(
    series: SeriesData,
    penalty: np.ndarray,
    n_steps: int,
    dynamic_coef: float,
    tp: float,
    tn: float,
    fp: float,
    fn: float,
    reward_unlabeled: str = "zero",
) -> TimeSeriesEnv:
    return TimeSeriesEnv(
        series.values,
        series.labels,
        penalty=penalty,
        n_steps=n_steps,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        dynamic_coef=dynamic_coef,
        reward_unlabeled=reward_unlabeled,
    )
