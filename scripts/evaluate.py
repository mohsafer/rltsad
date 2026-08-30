"""Validate a trained DRSMT agent (Algorithm 1, VALIDATE).

For every validation episode the greedy policy (epsilon = 0) classifies each
time step of a held-out series; precision, recall, F1 and AU-PR are computed
per episode (paper Sec. V) and aggregated as mean +- std (Table II notation).
The default ``--protocol pointwise`` follows the paper (sklearn metrics,
zero_division=0); ``--protocol rlad`` additionally offers the ancestor RLAD
baseline protocol (reward-tolerance correction + add-one smoothing, no AU-PR)
so results can be compared apples-to-apples with that baseline.

Episode definition, following the original code:

* multi-series datasets (SMD, synthetic): one episode per held-out machine
  (the tail ``1 - validation_separate_ratio`` of the labelled series);
* single-series datasets (WADI, Yahoo): the series is split into ``K``
  equal slices and each slice is one episode (Algorithm 1 lines 36-40).

Outputs:

* ``--output`` JSON with per-episode and aggregated metrics
  (default ``results/<dataset>_metrics.json``);
* one ``.npz`` per episode (values, predictions, ground truth, Q-scores)
  next to it, consumed by ``scripts/visualize_results.py`` for the Fig. 3
  style plots.

Example:
    python scripts/evaluate.py --dataset synthetic --data_dir data/synthetic \
        --vae_model_dir models/vae/synthetic --dqn_model_dir models/dqn/synthetic \
        --output results/synthetic_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.dqn.q_network import load_q_network  # noqa: E402
from models.reconstruction import compute_penalty_array, load_reconstructor  # noqa: E402
from scripts.common import load_and_prepare, rollout, split_train_valid  # noqa: E402
from utils.config import DRSMTConfig, set_seed  # noqa: E402
from utils.data_loader import SeriesData  # noqa: E402
from utils.env import TimeSeriesEnv, env_from_series  # noqa: E402
from utils.metrics import (  # noqa: E402
    aggregate_metrics,
    binary_metrics,
    format_metrics_table,
    rlad_protocol_metrics,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DRSMT agent (VALIDATE).")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["smd", "synthetic", "wadi", "yahoo"])
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--vae_model_dir", default="models/vae/synthetic")
    parser.add_argument("--dqn_model_dir", default="models/dqn/synthetic")
    parser.add_argument("--output", default=None,
                        help="metrics JSON path (default results/<dataset>_metrics.json)")
    parser.add_argument("--arrays_dir", default=None,
                        help="directory for per-episode .npz (default alongside --output)")
    parser.add_argument("--k_slices", type=int, default=5,
                        help="K equal slices for single-series datasets (Algorithm 1)")
    parser.add_argument("--validation_separate_ratio", type=float, default=0.8)
    parser.add_argument("--protocol", default="pointwise",
                        choices=["pointwise", "rlad"],
                        help="metrics protocol: 'pointwise' is the paper's "
                             "(sklearn P/R/F1 + AU-PR); 'rlad' applies the "
                             "ancestor RLAD baseline protocol (reward "
                             "tolerance +-5 + add-one smoothing) so numbers "
                             "are comparable with that baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def _slice_series(series: SeriesData, k: int, n_steps: int) -> list:
    """Split a series into K equal slices (Algorithm 1 line 36)."""
    if k < 2:
        return [series]
    seg = len(series) // k
    if seg <= n_steps + 1:
        raise ValueError(
            f"cannot split series '{series.name}' of length {len(series)} "
            f"into {k} slices of size {seg} (need > n_steps={n_steps})"
        )
    slices = []
    for i in range(k):
        start, end = i * seg, (i + 1) * seg
        if i == k - 1:
            end = len(series)  # keep the tail
        slices.append(
            SeriesData(f"{series.name}_slice{i}", series.values[start:end],
                       series.labels[start:end])
        )
    return slices


def main(argv=None) -> None:
    args = _parse_args(argv)
    output = args.output or os.path.join("results", f"{args.dataset}_metrics.json")
    arrays_dir = args.arrays_dir or os.path.splitext(output)[0] + "_episodes"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)
    set_seed(args.seed)
    device = DRSMTConfig(device=args.device).resolved_device()

    # ------------------------------------------------------------- models
    dqn, dqn_meta = load_q_network(args.dqn_model_dir)
    dqn = dqn.to(device)
    recon_model, scaler, recon_meta = load_reconstructor(args.vae_model_dir)
    n_steps = int(dqn_meta["n_steps"])
    if int(recon_meta["n_steps"]) != n_steps:
        raise ValueError("the reconstruction model and DQN were trained with "
                         "different n_steps")
    if int(recon_meta["n_features"]) != int(dqn_meta["n_features"]):
        raise ValueError("the reconstruction model and DQN were trained on "
                         "different feature sets")

    cfg = DRSMTConfig(
        n_steps=n_steps,
        dataset=args.dataset,
        data_dir=args.data_dir,
        validation_separate_ratio=args.validation_separate_ratio,
        k_validation_slices=args.k_slices,
        eval_protocol=args.protocol,
        seed=args.seed,
        device=args.device,
    )
    data = load_and_prepare(cfg)
    if data.n_features != int(dqn_meta["n_features"]):
        raise ValueError(
            f"dataset provides {data.n_features} features but the models "
            f"expect {dqn_meta['n_features']} (check --data_dir / zero-variance setting)"
        )

    # ------------------------------------------------------ validation set
    _, valid_series = split_train_valid(data.test, cfg.validation_separate_ratio)
    single_series = len(data.test) == 1
    episodes: list = []
    names: list = []
    if single_series:
        episodes = _slice_series(valid_series[0], cfg.k_validation_slices, cfg.n_steps)
    else:
        episodes = list(valid_series)
    if not episodes:
        raise ValueError(
            "no validation series available: --validation_separate_ratio "
            f"{cfg.validation_separate_ratio} leaves nothing held out "
            f"(dataset has {len(data.test)} labelled series)"
        )
    print(f"[evaluate] {len(episodes)} validation episode(s) on dataset "
          f"'{data.dataset}' (device: {device})")

    # ------------------------------------------------------------- rollouts
    per_episode = []
    rng = np.random.default_rng(cfg.seed)
    for series in episodes:
        penalty = compute_penalty_array(
            recon_model, scaler, series.values, cfg.n_steps,
            device=device, batch_size=cfg.penalty_batch_size,
        )
        env = env_from_series(
            series, penalty, n_steps=cfg.n_steps,
            dynamic_coef=0.0,  # rewards are irrelevant for the greedy pass
            tp=cfg.tp_value, tn=cfg.tn_value, fp=cfg.fp_value, fn=cfg.fn_value,
        )
        info = rollout(env, dqn, device, epsilon=0.0, rng=rng, record=True)
        if cfg.eval_protocol == "rlad":
            # ancestor RLAD baseline protocol (no score-based AU-PR)
            metrics = rlad_protocol_metrics(
                info["ground_truth"], info["predictions"]
            )
        else:
            metrics = binary_metrics(
                info["ground_truth"], info["predictions"], y_score=info["scores"]
            )
        per_episode.append(metrics)
        names.append(series.name)

        npz_path = os.path.join(arrays_dir, f"{series.name}.npz")
        np.savez_compressed(
            npz_path,
            name=np.asarray(series.name),
            values=info["values"],
            predictions=info["predictions"],
            ground_truth=info["ground_truth"],
            scores=info["scores"],
            aupr=np.asarray(metrics["aupr"]),
            aupr_score=np.asarray(metrics.get("aupr_score", 0.0)),
        )
        print(
            f"[evaluate] {series.name:<24} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"AU-PR={metrics['aupr']:.4f} (score-based {metrics.get('aupr_score', 0.0):.4f})"
        )

    aggregate = aggregate_metrics(per_episode)
    report = {
        "dataset": data.dataset,
        "data_dir": args.data_dir,
        "vae_model_dir": args.vae_model_dir,
        "dqn_model_dir": args.dqn_model_dir,
        "n_steps": cfg.n_steps,
        "n_features": data.n_features,
        "protocol": cfg.eval_protocol,
        "n_episodes": len(per_episode),
        "episode_names": names,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "aggregate": aggregate,
        "per_episode": [
            {"name": name, **metrics} for name, metrics in zip(names, per_episode)
        ],
    }
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print()
    print(format_metrics_table(per_episode, names))
    print(f"\n[evaluate] metrics written to {output}")
    print(f"[evaluate] per-episode arrays written to {arrays_dir}")


if __name__ == "__main__":
    main()
