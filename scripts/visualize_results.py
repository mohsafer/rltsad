"""Visualise DRSMT results (paper Fig. 2 and Fig. 3).

* Fig. 2a -- evolution of the dynamic coefficient lambda(t) over episodes;
* Fig. 2b -- the training reward curve;
* Fig. 3  -- per validation episode: normalised series, the agent's binary
  predictions, the ground-truth anomalies and the episode AU-PR (the four
  stacked panels of the paper), rendered from the ``.npz`` arrays written by
  ``scripts/evaluate.py``.

Examples:
    python scripts/visualize_results.py --run_dir models/dqn/synthetic \
        --eval_dir results/synthetic_episodes --output_dir results/plots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_lambda_and_reward(history: dict, out_dir: str) -> None:
    """Fig. 2: relationship between the dynamic coefficient and the reward."""
    episodes = history["episode"]
    lambdas = history["lambda"]
    rewards = history["reward"]

    plt.figure(figsize=(7, 4))
    plt.plot(episodes, lambdas, color="tab:orange", marker="s", markersize=3)
    plt.xlabel("Episode")
    plt.ylabel(r"Dynamic coefficient $\lambda(t)$")
    plt.title("(a) Dynamic coefficient evolution over episodes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2a_lambda_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] wrote {path}")

    plt.figure(figsize=(7, 4))
    plt.plot(episodes, rewards, color="tab:blue", marker="o", markersize=3)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("(b) Training reward curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2b_training_reward.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] wrote {path}")

    # Combined view: lambda against the reward (twin axes), as in Fig. 2.
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(episodes, rewards, color="tab:blue", label="Episode reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode reward", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(episodes, lambdas, color="tab:orange", label=r"$\lambda(t)$")
    ax2.set_ylabel(r"Dynamic coefficient $\lambda(t)$", color="tab:orange")
    ax1.set_title("Dynamic reward scaling: coefficient vs. training reward")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_lambda_vs_reward.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {path}")


def plot_validation_episode(npz_path: str, out_dir: str) -> None:
    """Fig. 3 style panels for one validation episode."""
    data = np.load(npz_path, allow_pickle=True)
    name = str(data["name"])
    values = data["values"]
    predictions = data["predictions"]
    ground_truth = data["ground_truth"]
    aupr = float(data["aupr"])
    n = len(values)
    x = np.arange(n)

    fig, axarr = plt.subplots(4, sharex=True, figsize=(9, 7))
    axarr[0].plot(x, values, color="tab:blue", lw=0.8)
    axarr[0].set_ylabel("Series")
    axarr[0].set_title(f"Validation episode: {name}")

    axarr[1].plot(x, predictions, color="tab:green", lw=0.8)
    axarr[1].set_ylabel("Prediction")
    axarr[1].set_yticks([0, 1])

    axarr[2].plot(x, ground_truth, color="tab:red", lw=0.8)
    axarr[2].set_ylabel("Ground truth")
    axarr[2].set_yticks([0, 1])

    axarr[3].plot(x, np.full(n, aupr), color="tab:purple")
    axarr[3].set_ylabel("AU-PR")
    axarr[3].set_xlabel("Time step")
    axarr[3].set_ylim(0, 1)
    axarr[3].text(
        0.99, 0.1, f"AU-PR = {aupr:.4f}", ha="right", va="bottom",
        transform=axarr[3].transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="tab:purple", alpha=0.8),
    )

    for ax in axarr:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, f"validation_{os.path.splitext(os.path.basename(npz_path))[0]}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {path}")


def plot_metrics_summary(metrics_json: str, out_dir: str) -> None:
    """Bar chart of the aggregated metrics (Table II row for this run)."""
    with open(metrics_json, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    aggregate = report["aggregate"]
    keys = ["precision", "recall", "f1", "aupr"]
    means = [aggregate[k]["mean"] for k in keys]
    stds = [aggregate[k]["std"] for k in keys]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(keys, means, yerr=stds, capsize=4,
                   color=["tab:blue", "tab:green", "tab:orange", "tab:purple"])
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"DRSMT on {report['dataset']} "
              f"({report['n_episodes']} validation episodes)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"metrics_{report['dataset']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] wrote {path}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot DRSMT results (Fig. 2 / Fig. 3).")
    parser.add_argument("--run_dir", default=None,
                        help="models/dqn/<run> containing history.json (Fig. 2)")
    parser.add_argument("--eval_dir", default=None,
                        help="directory with per-episode .npz files from evaluate.py (Fig. 3)")
    parser.add_argument("--metrics_json", default=None,
                        help="metrics JSON from evaluate.py (summary bar chart)")
    parser.add_argument("--output_dir", default="results/plots")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_dir:
        history_path = os.path.join(args.run_dir, "history.json")
        if not os.path.exists(history_path):
            raise FileNotFoundError(
                f"{history_path} not found; run scripts/train_rl.py first"
            )
        with open(history_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)
        plot_lambda_and_reward(history, args.output_dir)
    else:
        print("[visualize] --run_dir not given; skipping Fig. 2 plots")

    if args.eval_dir:
        npz_files = sorted(glob.glob(os.path.join(args.eval_dir, "*.npz")))
        if not npz_files:
            print(f"[visualize] no .npz files in {args.eval_dir}; "
                  f"run scripts/evaluate.py first")
        for npz_path in npz_files:
            plot_validation_episode(npz_path, args.output_dir)
    else:
        print("[visualize] --eval_dir not given; skipping Fig. 3 panels")

    if args.metrics_json:
        if not os.path.exists(args.metrics_json):
            raise FileNotFoundError(args.metrics_json)
        plot_metrics_summary(args.metrics_json, args.output_dir)

    print(f"[visualize] all plots under {args.output_dir}")


if __name__ == "__main__":
    main()
