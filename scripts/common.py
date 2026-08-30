"""Shared pipeline helpers for the DRSMT scripts.

Contains the pieces that ``train_rl.py``, ``warmup_replay.py`` and
``evaluate.py`` all need: penalty-backed environment construction, the
train/validation series split of the original code, the epsilon and lambda
schedules, the WARMUP routine (Algorithm 1 lines 12-15), the epsilon-greedy
rollout and replay-memory persistence (including the semi-supervised label
state revealed during warm-up).

Efficiency note: the sensor content of a window does not depend on the
agent's actions (only the appended action-indicator does), and the Q-network
is frozen during a rollout (replay updates happen afterwards), so the two
Q-values of every window of an episode can be pre-computed in one batched
forward pass (:func:`precompute_q_table`) instead of two forwards per step.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from models.dqn.q_network import QNetwork, q_values_pair, q_values_max_next  # noqa: E402
from models.dqn.replay_buffer import ReplayBuffer, Transition  # noqa: E402
from models.reconstruction import compute_penalty_array  # noqa: E402
from utils.config import DRSMTConfig  # noqa: E402
from utils.data_loader import SeriesData, flatten_windows, normalize_data, prepare_data  # noqa: E402
from utils.env import UNLABELED, TimeSeriesEnv, env_from_series  # noqa: E402

# --------------------------------------------------------------------------- #
# Data / environments
# --------------------------------------------------------------------------- #


def load_and_prepare(cfg: DRSMTConfig):
    """Load the dataset, drop zero-variance sensors, Min-Max normalise."""
    data = prepare_data(cfg.dataset, cfg.data_dir, cfg.drop_zero_variance)
    data = normalize_data(data)
    return data


def split_train_valid(
    series_list: Sequence[SeriesData], validation_separate_ratio: float
) -> Tuple[List[SeriesData], List[SeriesData]]:
    """Split the labelled series into RL-train / validation parts.

    Reproduces the original behaviour: with ``validation_separate_ratio=0.8``
    the first 80% of the series train the agent (SMD: 22 of 28 machines) and
    the tail is held out for validation.  Single-series datasets (WADI, one
    synthetic machine, ...) keep the whole series on both sides; validation
    then slices it into K equal parts (Algorithm 1, VALIDATE).
    """
    series_list = list(series_list)
    if len(series_list) <= 1:
        return series_list, series_list
    n_train = max(1, int(len(series_list) * validation_separate_ratio))
    return series_list[:n_train], series_list[n_train:]


def build_envs(
    series_list: Sequence[SeriesData],
    recon_model,
    scaler,
    cfg: DRSMTConfig,
    device,
    dynamic_coef: Optional[float] = None,
) -> Dict[str, TimeSeriesEnv]:
    """Create one penalty-backed environment per series (COMPUTEPENALTY).

    ``recon_model`` is any reconstruction backbone loaded via
    :func:`models.reconstruction.load_reconstructor` (VAE or Transformer).
    """
    coef = cfg.lambda_init if dynamic_coef is None else dynamic_coef
    envs: Dict[str, TimeSeriesEnv] = {}
    for series in series_list:
        print(f"[env] computing reconstruction penalty array for '{series.name}' "
              f"({len(series)} steps, d={series.n_features})")
        penalty = compute_penalty_array(
            recon_model, scaler, series.values, cfg.n_steps,
            device=device, batch_size=cfg.penalty_batch_size,
        )
        env = env_from_series(
            series,
            penalty,
            n_steps=cfg.n_steps,
            dynamic_coef=coef,
            tp=cfg.tp_value,
            tn=cfg.tn_value,
            fp=cfg.fp_value,
            fn=cfg.fn_value,
            reward_unlabeled=cfg.reward_unlabeled,
        )
        envs[series.name] = env
    return envs


def precompute_q_table(
    policy: QNetwork, env: TimeSeriesEnv, device, batch_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """Q(s,0) and Q(s,1) for every window of the series (two batched passes)."""
    return q_values_pair(policy, env.get_states_list(), device, batch_size)


# --------------------------------------------------------------------------- #
# Schedules (Algorithm 1 lines 25, 34)
# --------------------------------------------------------------------------- #


def epsilon_at(cfg: DRSMTConfig, episode: int, total_updates: int) -> float:
    """epsilon-greedy exploration rate.

    ``"episode"`` (WADI variant): linear decay over the episodes.
    ``"linear"`` (SMD variant): linear decay over ``epsilon_decay_steps``
    gradient updates.
    """
    span = cfg.epsilon_start - cfg.epsilon_end
    if cfg.epsilon_schedule == "linear":
        frac = min(total_updates / max(cfg.epsilon_decay_steps, 1), 1.0)
    else:  # "episode"
        frac = min(max(episode - 1, 0) / max(cfg.episodes - 1, 1), 1.0)
    return float(cfg.epsilon_start - span * frac)


def update_lambda(cfg: DRSMTConfig, current_coef: float, episode_reward: float) -> float:
    """Proportional controller of the dynamic coefficient (Algorithm 1, line 34).

    ``lambda <- clip(lambda + alpha * (R_target - R_episode), lambda_min,
    lambda_max)``: when the episode reward is below target, lambda grows (the
    VAE reconstruction penalty R2 weighs more, i.e. exploration); when the
    agent performs well, lambda decays toward exploitation.
    """
    new_coef = current_coef + cfg.lambda_alpha * (cfg.lambda_target - episode_reward)
    return float(np.clip(new_coef, cfg.lambda_min, cfg.lambda_max))


# --------------------------------------------------------------------------- #
# Rollout (Algorithm 1 lines 23-33)
# --------------------------------------------------------------------------- #


def rollout(
    env: TimeSeriesEnv,
    policy: Optional[QNetwork],
    device,
    epsilon: float,
    rng: np.random.Generator,
    replay: Optional[ReplayBuffer] = None,
    record: bool = False,
    q_table: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, object]:
    """Run one full episode; optionally store transitions and predictions.

    * ``q_table`` = pre-computed (Q0, Q1) arrays from :func:`precompute_q_table`
      (fast path used during training).
    * ``policy`` given without ``q_table``: two forwards per step (used at
      evaluation time when memory matters more than speed).
    * ``policy=None`` or ``epsilon=1.0``: the random warm-up policy
      (Algorithm 1 line 14: "Play random actions to fill ReplayMem").
    """
    window = env.reset()
    total_reward = 0.0
    predictions: List[int] = []
    truths: List[int] = []
    scores: List[float] = []
    values: List[float] = []
    while True:
        i = env.t - env.n_steps  # window index in get_states_list()
        if q_table is not None:
            q0, q1 = q_table
        elif policy is not None:
            q0, q1 = q_values_pair(policy, window[None, ...], device)
        else:
            q0 = q1 = None
        if q0 is None or epsilon >= 1.0:
            action, score = int(rng.integers(0, 2)), float("nan")
        else:
            action = 1 if q1[i] > q0[i] else 0
            score = float(q1[i] - q0[i])
        if rng.random() < epsilon:  # epsilon-greedy exploration
            action, score = int(rng.integers(0, 2)), float("nan")

        next_window, reward, done, info = env.step(action)
        if replay is not None:
            replay.push(
                Transition(
                    window=np.asarray(window, dtype=np.float32),
                    action=action,
                    reward=reward,
                    next_window=None if done else np.asarray(next_window, dtype=np.float32),
                    done=bool(done),
                )
            )
        total_reward += float(reward[action])
        if record:
            predictions.append(action)
            truths.append(int(info["ground_truth"]))
            scores.append(score)
            values.append(float(window[-1, 0]))
        if done:
            break
        window = next_window
    return {
        "reward": total_reward,
        "steps": len(predictions) if record else env.length - env.n_steps,
        "predictions": np.asarray(predictions, dtype=np.int64),
        "ground_truth": np.asarray(truths, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "values": np.asarray(values, dtype=np.float64),
    }


# --------------------------------------------------------------------------- #
# Warm-up (Algorithm 1 lines 12-15)
# --------------------------------------------------------------------------- #


def warmup_replay_memory(
    envs: Dict[str, TimeSeriesEnv],
    cfg: DRSMTConfig,
    rng: np.random.Generator,
) -> Tuple[ReplayBuffer, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """WARMUP: outlier-guided labels + LabelSpreading + random rollouts.

    1. fit an IsolationForest on the last row of the collected windows;
    2. reveal the ground truth of the ``warmup_label_each`` most anomalous
       and the same number of most normal windows per series;
    3. pseudo-label the most confident unlabelled windows with LabelSpreading;
    4. play random actions until the replay memory holds
       ``replay_memory_init_size`` transitions.

    Returns the replay buffer and the revealed label state per series.
    """
    from scripts.active_learning import fit_label_spreading

    windows_per_env = {name: env.get_states_list() for name, env in envs.items()}
    stacked = np.concatenate(list(windows_per_env.values()), axis=0)
    if len(stacked) > cfg.warmup_max_samples:
        sel = rng.choice(len(stacked), size=cfg.warmup_max_samples, replace=False)
        stacked = stacked[sel]
    print(f"[warmup] IsolationForest on {len(stacked)} windows "
          f"(contamination={cfg.warmup_outliers_fraction})")
    iso = IsolationForest(
        contamination=cfg.warmup_outliers_fraction, random_state=cfg.seed
    )
    iso.fit(stacked[:, -1, :])  # last row: current readings of all d sensors

    labels_out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, env in envs.items():
        windows = windows_per_env[name]
        n_windows = len(windows)
        time_index = np.arange(n_windows) + env.n_steps
        scores = -iso.decision_function(windows[:, -1, :])  # higher = more anomalous
        order = np.argsort(scores, kind="stable")
        k = min(cfg.warmup_label_each, max(n_windows // 2, 1))
        chosen = np.concatenate([order[:k], order[len(order) - k:]])
        for i in chosen:
            t = int(time_index[i])
            env.reveal(t, env.ground_truth[t])

        revealed = env.work_labels[time_index]
        if n_windows >= 2:
            try:
                lp, fit_idx = fit_label_spreading(
                    flatten_windows(windows),
                    revealed,
                    n_neighbors=cfg.lp_neighbors,
                    max_samples=cfg.lp_max_samples,
                    seed=cfg.seed,
                )
                uncertainty = 1.0 - lp.label_distributions_.max(axis=1)
                candidates = np.where(revealed[fit_idx] == UNLABELED)[0]
                if len(candidates):
                    # the warm-up pseudo-labels the *most confident*
                    # unlabelled windows (ascending uncertainty), as in the
                    # original SMD warm-up
                    k_lp = int(min(cfg.lp_budget, len(candidates)))
                    picked = fit_idx[candidates[np.argsort(uncertainty[candidates])[:k_lp]]]
                    for i in picked:
                        propagated = int(lp.transduction_[np.where(fit_idx == i)[0][0]])
                        env.reveal(int(time_index[i]), propagated)
            except ValueError as exc:
                print(f"[warmup] LabelSpreading skipped for '{name}': {exc}")

        idx = np.where(env.work_labels != UNLABELED)[0]
        labels_out[name] = (idx, env.work_labels[idx])
        n_anom = int((env.work_labels[idx] == 1).sum())
        print(f"[warmup] '{name}': revealed {len(idx)} labels "
              f"({n_anom} anomalies, {len(idx) - n_anom} normal)")

    replay = ReplayBuffer(cfg.replay_memory_size)
    fill_target = min(cfg.replay_memory_init_size, cfg.replay_memory_size)
    while len(replay) < fill_target:
        for name, env in envs.items():
            if len(replay) >= fill_target:
                break
            before = len(replay)
            rollout(env, policy=None, device=None, epsilon=1.0, rng=rng, replay=replay)
            print(f"[warmup] random rollout on '{name}': +{len(replay) - before} "
                  f"transitions (memory {len(replay)}/{fill_target})")
    return replay, labels_out


# --------------------------------------------------------------------------- #
# Replay persistence (including the revealed label state)
# --------------------------------------------------------------------------- #


def save_replay_payload(
    path: str,
    replay: ReplayBuffer,
    cfg: DRSMTConfig,
    labels: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> str:
    """Save replay transitions + revealed labels + the lambda used.

    ``labels`` maps series name -> (time indices, label values) of the working
    label column, so that a warm-up executed as a separate process transfers
    its semi-supervised state into ``train_rl.py``.
    """
    payload = {
        "format": "warmup_v1",
        "n_steps": cfg.n_steps,
        "lambda_init": cfg.lambda_init,
        "labels": {
            name: (np.asarray(idx), np.asarray(val))
            for name, (idx, val) in (labels or {}).items()
        },
        "memory": list(replay.memory),
        "capacity": replay.capacity,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_replay_payload(
    path: str,
) -> Tuple[ReplayBuffer, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load a replay file written by :func:`save_replay_payload` or by
    :meth:`ReplayBuffer.save`.  Returns ``(buffer, labels)``."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, ReplayBuffer):  # direct buffer pickle
        return payload, {}
    memory = payload.get("memory", [])
    capacity = payload.get("capacity", max(len(memory), 1))
    buffer = ReplayBuffer(capacity)
    buffer.extend(memory)
    labels_raw = payload.get("labels", {}) or {}
    labels = {
        name: (np.asarray(idx), np.asarray(val)) for name, (idx, val) in labels_raw.items()
    }
    return buffer, labels
