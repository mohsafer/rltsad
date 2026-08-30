"""Active learning for DRSMT (paper Sec. IV-C, Algorithm 1 lines 19-21).

Two complementary mechanisms, exactly as in the paper:

1. **Margin sampling** (Sec. IV-C).  For every candidate state the margin
   ``Margin(s) = |Q(s, a_1) - Q(s, a_2)|`` is computed from the two Q-values
   of the LSTM-DQN; the ``K_AL`` windows with the *smallest* margin (the most
   uncertain predictions) are selected and labelled by the human oracle
   (here: the ground-truth column, i.e. ``Selected = argmin_s Margin(s)``).

2. **Label propagation** (Sec. IV-C / Algorithm 1 line 20).  A
   semi-supervised ``LabelSpreading`` model transmits the revealed labels to
   nearby unlabelled windows based on feature similarity, ``P(y_i | x_i) ~
   sum_{j in L} w_ij P(y_j | x_j)``; ``K_LP`` unlabelled windows receive
   propagated pseudo-labels.  *Which* windows are selected is configurable
   (``DRSMTConfig.lp_selection``): ``"certain"`` takes the most confident LP
   outputs (ascending uncertainty -- the RLAD / myasp-smd lineage, matching
   the warm-up path; the original ranked by entropy, here by
   ``1 - max prob``), ``"uncertain"`` takes the least confident ones (the
   myasp-wadi variant).

This file is both an importable module (used by ``train_rl.py`` and
``warmup_replay.py``) and a standalone CLI that demonstrates one AL+LP round
on a trained agent.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.semi_supervised import LabelSpreading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn.q_network import QNetwork, load_q_network, q_values_pair  # noqa: E402
from utils.config import DRSMTConfig, set_seed  # noqa: E402
from utils.data_loader import flatten_windows  # noqa: E402
from utils.env import TimeSeriesEnv, UNLABELED  # noqa: E402


class MarginActiveLearner:
    """Margin-sampling query strategy backed by the LSTM-DQN."""

    def __init__(self, model: QNetwork, device, batch_size: int = 1024) -> None:
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)

    def q_values(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return q_values_pair(self.model, windows, self.device, self.batch_size)

    def margins(self, windows: np.ndarray) -> np.ndarray:
        """|Q(s,0) - Q(s,1)| for every window."""
        q0, q1 = self.q_values(windows)
        return np.abs(q0 - q1)

    def select(
        self,
        windows: np.ndarray,
        exclude_mask: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Indices of the ``k`` least confident windows (smallest margins).

        ``exclude_mask`` marks windows that must not be queried (already
        labelled); they receive an infinite margin.
        """
        margins = self.margins(windows)
        margins = np.asarray(margins, dtype=np.float64).copy()
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        margins[exclude_mask] = np.inf
        k = int(min(max(k, 0), int(np.isfinite(margins).sum())))
        if k == 0:
            return np.empty(0, dtype=np.int64)
        return np.argsort(margins, kind="stable")[:k]


def fit_label_spreading(
    x_flat: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
    max_samples: int = 5_000,
    seed: int = 42,
) -> Tuple[LabelSpreading, np.ndarray]:
    """Fit LabelSpreading on (a subsample of) the flattened windows.

    ``labels`` uses -1 for unlabelled entries.  All labelled points are always
    kept in the fit subset; unlabelled points are randomly subsampled when the
    series is longer than ``max_samples`` (the kNN graph would otherwise be
    prohibitive on long series such as WADI).

    Returns the fitted model and the array mapping model positions back to
    window indices.
    """
    labels = np.asarray(labels)
    labeled_idx = np.where(labels != UNLABELED)[0]
    unlabeled_idx = np.where(labels == UNLABELED)[0]
    budget = max(max_samples - len(labeled_idx), 0)
    if len(unlabeled_idx) > budget:
        rng = np.random.default_rng(seed)
        unlabeled_idx = rng.choice(unlabeled_idx, size=budget, replace=False)
    fit_idx = np.concatenate([labeled_idx, unlabeled_idx]).astype(np.int64)
    if len(fit_idx) < 2:
        raise ValueError("not enough samples to fit LabelSpreading")
    # the kNN graph needs n_neighbors < n_samples
    n_neighbors = max(1, min(int(n_neighbors), len(fit_idx) - 1))
    lp = LabelSpreading(kernel="knn", n_neighbors=n_neighbors)
    lp.fit(x_flat[fit_idx], labels[fit_idx].astype(int))
    return lp, fit_idx


def active_learning_step(
    env: TimeSeriesEnv,
    learner: MarginActiveLearner,
    cfg: DRSMTConfig,
) -> Dict[str, np.ndarray]:
    """One AL + label-propagation round on the current series (in place).

    1. rank unlabelled windows by the Q-value margin, reveal the ``K_AL``
       most confusing ones with ground-truth labels;
    2. fit LabelSpreading with all revealed labels, then pseudo-label
       ``K_LP`` still-unlabelled windows with the propagated labels
       (``lp.transduction_``) -- the most confident ones by default
       (``cfg.lp_selection == "certain"``), or the most uncertain ones with
       ``"uncertain"``.

    Returns a bookkeeping dictionary with the revealed time indices.
    """
    windows = env.get_states_list()
    n_windows = len(windows)
    time_index = np.arange(n_windows) + env.n_steps  # window i ends at t=i+n_steps

    # -- 1) margin sampling with ground-truth revelation --------------------
    already = env.work_labels[time_index] != UNLABELED
    al_local = learner.select(windows, exclude_mask=already, k=cfg.al_budget)
    for i in al_local:
        env.reveal(time_index[i], env.ground_truth[time_index[i]])

    # -- 2) LabelSpreading propagation --------------------------------------
    lp_al = lp_lp = 0
    if n_windows >= 2:
        window_labels_col = env.work_labels[time_index]
        x_flat = flatten_windows(windows)
        try:
            lp, fit_idx = fit_label_spreading(
                x_flat,
                window_labels_col,
                n_neighbors=cfg.lp_neighbors,
                max_samples=cfg.lp_max_samples,
                seed=cfg.seed,
            )
            distributions = lp.label_distributions_
            uncertainty = 1.0 - distributions.max(axis=1)
            # candidates: still-unlabelled points inside the LP fit subset
            candidate_pos = np.where(window_labels_col[fit_idx] == UNLABELED)[0]
            lp_local = np.empty(0, dtype=np.int64)
            if len(candidate_pos) > 0:
                k = int(min(cfg.lp_budget, len(candidate_pos)))
                if getattr(cfg, "lp_selection", "certain") == "uncertain":
                    # myasp-wadi variant: propagate to the *least* confident
                    order = np.argsort(-uncertainty[candidate_pos], kind="stable")[:k]
                else:
                    # RLAD / myasp-smd lineage (and the warm-up path):
                    # propagate the *most* confident LP outputs
                    order = np.argsort(uncertainty[candidate_pos], kind="stable")[:k]
                lp_local = fit_idx[candidate_pos[order]]
            for i in lp_local:
                propagated = int(lp.transduction_[np.where(fit_idx == i)[0][0]])
                env.reveal(time_index[i], propagated)
            lp_al, lp_lp = len(al_local), len(lp_local)
        except ValueError as exc:
            # e.g. a single global class during the very first episodes
            print(f"[active-learning] LabelSpreading skipped: {exc}")

    return {
        "al_time_index": time_index[al_local],
        "lp_time_index": time_index[lp_local],
        "al_count": np.int64(len(al_local)),
        "lp_count": np.int64(lp_lp),
    }


def reset_working_labels(env: TimeSeriesEnv) -> None:
    """Clear the working label column (labels start fully unlabelled)."""
    env.work_labels[:] = UNLABELED


# --------------------------------------------------------------------------- #
# CLI demonstration
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one active-learning + label-propagation round "
        "with a trained DRSMT agent (paper Sec. IV-C)."
    )
    parser.add_argument("--dataset", default="synthetic",
                        choices=["smd", "synthetic", "wadi", "yahoo"])
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--dqn_model_dir", default="models/dqn/synthetic")
    parser.add_argument("--vae_model_dir", default="models/vae/synthetic")
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--al_budget", type=int, default=200)
    parser.add_argument("--lp_budget", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)

    from models.reconstruction import compute_penalty_array, load_reconstructor
    from utils.data_loader import prepare_data, normalize_data

    cfg = DRSMTConfig(
        n_steps=args.n_steps,
        dataset=args.dataset,
        data_dir=args.data_dir,
        al_budget=args.al_budget,
        lp_budget=args.lp_budget,
        seed=args.seed,
    )

    data = prepare_data(args.dataset, args.data_dir, cfg.drop_zero_variance)
    data = normalize_data(data)
    recon_model, scaler, _ = load_reconstructor(args.vae_model_dir)
    dqn, _ = load_q_network(args.dqn_model_dir)
    device = cfg.resolved_device()

    series = data.test[0]
    penalty = compute_penalty_array(
        recon_model, scaler, series.values, cfg.n_steps, device=device,
        batch_size=cfg.penalty_batch_size,
    )
    env = TimeSeriesEnv(
        series.values, series.labels, penalty=penalty, n_steps=cfg.n_steps,
        tp=cfg.tp_value, tn=cfg.tn_value, fp=cfg.fp_value, fn=cfg.fn_value,
        reward_unlabeled=cfg.reward_unlabeled,
    )
    learner = MarginActiveLearner(dqn.to(device), device)
    info = active_learning_step(env, learner, cfg)

    al_idx = info["al_time_index"]
    print(f"[active-learning] margin-sampled {info['al_count']} windows "
          f"(K_AL={cfg.al_budget}); time indices: {al_idx[:20]}...")
    print(f"[active-learning] propagated pseudo-labels on {info['lp_count']} "
          f"windows (K_LP={cfg.lp_budget})")
    if len(al_idx):
        truths = series.labels[al_idx]
        print(f"[active-learning] ground truth of queried windows: "
              f"{int((truths == 1).sum())} anomalies / {len(al_idx)} queried")


if __name__ == "__main__":
    main()
