"""Fill the replay memory before RL training (Algorithm 1, WARMUP).

Reproducing the original warm-up (``myasp-smd.py`` / Algorithm 1 lines 12-15):

1. collect sliding windows of the RL-training series;
2. fit a one-class outlier detector (IsolationForest) and reveal the
   ground-truth labels of the ``warmup_label_each`` most anomalous and the
   same number of most normal windows;
3. propagate those labels with LabelSpreading and pseudo-label the most
   confident still-unlabelled windows;
4. play random actions (epsilon = 1) until the replay memory holds
   ``replay_memory_init_size`` (M) transitions, with rewards computed under
   the initial dynamic coefficient lambda_0.

The output pickle contains the transitions *and* the revealed label state so
that ``train_rl.py`` continues from exactly this semi-supervised state.

Example:
    python scripts/warmup_replay.py --dataset synthetic --data_dir data/synthetic \
        --vae_model_dir models/vae/synthetic --output_replay replay_memory.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.reconstruction import load_reconstructor  # noqa: E402
from scripts.common import (  # noqa: E402
    build_envs,
    load_and_prepare,
    save_replay_payload,
    split_train_valid,
    warmup_replay_memory,
)
from utils.config import DRSMTConfig, set_seed  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up the DRSMT replay memory (WARMUP).")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["smd", "synthetic", "wadi", "yahoo"])
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--vae_model_dir", default="models/vae/synthetic")
    parser.add_argument("--output_replay", default="replay_memory.pkl")
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--init_size", type=int, default=1_500,
                        help="M, transitions collected before training (Algorithm 1)")
    parser.add_argument("--lambda_init", type=float, default=10.0)
    parser.add_argument("--outliers_fraction", type=float, default=0.01)
    parser.add_argument("--label_each", type=int, default=5,
                        help="ground-truth labels for the N most anomalous and N most normal windows")
    parser.add_argument("--lp_budget", type=int, default=200,
                        help="pseudo-labels propagated by LabelSpreading")
    parser.add_argument("--lp_neighbors", type=int, default=10)
    parser.add_argument("--lp_max_samples", type=int, default=5_000)
    parser.add_argument("--max_isoforest_samples", type=int, default=10_000)
    parser.add_argument("--validation_separate_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = DRSMTConfig(
        n_steps=args.n_steps,
        dataset=args.dataset,
        data_dir=args.data_dir,
        replay_memory_init_size=args.init_size,
        lambda_init=args.lambda_init,
        warmup_outliers_fraction=args.outliers_fraction,
        warmup_label_each=args.label_each,
        warmup_max_samples=args.max_isoforest_samples,
        lp_budget=args.lp_budget,
        lp_neighbors=args.lp_neighbors,
        lp_max_samples=args.lp_max_samples,
        validation_separate_ratio=args.validation_separate_ratio,
        seed=args.seed,
        device=args.device,
    )
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    rng = np.random.default_rng(cfg.seed)

    data = load_and_prepare(cfg)
    train_series, _ = split_train_valid(data.test, cfg.validation_separate_ratio)
    print(f"[warmup] {len(train_series)} RL-training series: "
          f"{[s.name for s in train_series]}")

    recon_model, scaler, _ = load_reconstructor(args.vae_model_dir)
    envs = build_envs(train_series, recon_model, scaler, cfg, device)

    replay, labels_out = warmup_replay_memory(envs, cfg, rng)
    save_replay_payload(args.output_replay, replay, cfg, labels_out)
    print(f"[warmup] saved {len(replay)} transitions + label state to "
          f"{args.output_replay} (lambda_init={cfg.lambda_init})")


if __name__ == "__main__":
    main()
