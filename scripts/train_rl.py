"""Train the LSTM-DQN agent with dynamic reward scaling (Algorithm 1, TRAINRL).

Per episode (paper Sec. IV-B / IV-C, Algorithm 1 lines 18-34):

1. **Active learning + label propagation** -- reveal the ``K_AL`` most
   uncertain windows (smallest Q-value margin) with ground truth and
   pseudo-label ``K_LP`` more windows through LabelSpreading (the most
   confident ones by default; ``--lp_selection uncertain`` reproduces the
   WADI variant);
2. **Rollout** -- run one full episode with epsilon-greedy actions; every
   transition stores the reward vector ``r = [R1(0) + lam*p[t], R1(1) +
   lam*p[t]]`` with the VAE reconstruction penalty scaled by lambda(t);
3. **Replay updates** -- ``num_updates_per_episode`` minibatch Bellman
   updates (target network ``Q'`` hard-synced every ``update_target_every``
   updates, Algorithm 1 lines 30-32);
4. **Dynamic coefficient update** -- ``lambda <- clip(lambda + alpha *
   (R_target - R_episode), lambda_min, lambda_max)`` (line 34).

Fidelity notes (paper vs. original code):

* Algorithm 1 line 32 draws the minibatch update *inside* the episode loop
  (a per-step DQN schedule).  The original research code instead batches a
  fixed number of updates per episode (``num_epoches`` = 10 on SMD); this
  implementation follows the code, which is the only tractable choice for
  28k-step episodes.
* The paper does not state alpha / R_target for the lambda controller.  The
  original code used alpha=0.001, R_target=0 on the reward accumulated over
  its active-learning subset; this pipeline accumulates the reward over the
  FULL episode, so pass a smaller ``--lambda_alpha`` (1e-4..1e-5) for the
  gradual Fig. 2a-style decay over many episodes.

Outputs (under ``models/dqn/<run>/``): ``q_network.pt``,
``meta.json``, ``history.json`` and the resolved ``config.json``.

Example (synthetic data):
    python scripts/train_rl.py --dataset synthetic --data_dir data/synthetic \
        --vae_model_dir models/vae/synthetic --replay_path replay_memory.pkl \
        --output_dir models/dqn --run_name synthetic --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.dqn.q_network import (  # noqa: E402
    QNetwork,
    q_values_max_next,
    save_q_network,
    sync_target,
)
from models.reconstruction import load_reconstructor  # noqa: E402
from scripts.active_learning import MarginActiveLearner, active_learning_step  # noqa: E402
from scripts.common import (  # noqa: E402
    build_envs,
    epsilon_at,
    load_and_prepare,
    load_replay_payload,
    precompute_q_table,
    rollout,
    save_replay_payload,
    split_train_valid,
    update_lambda,
    warmup_replay_memory,
)
from utils.config import DRSMTConfig, set_seed  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DRSMT LSTM-DQN with dynamic reward scaling (TRAINRL)."
    )
    parser.add_argument("--dataset", default="synthetic",
                        choices=["smd", "synthetic", "wadi", "yahoo"])
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--vae_model_dir", default="models/vae/synthetic")
    parser.add_argument("--replay_path", default="replay_memory.pkl")
    parser.add_argument("--output_dir", default="models/dqn")
    parser.add_argument("--run_name", default=None,
                        help="sub-directory of --output_dir (default: --dataset)")

    rl = parser.add_argument_group("RL hyper-parameters (defaults = paper/repo)")
    rl.add_argument("--n_steps", type=int, default=25)
    rl.add_argument("--episodes", type=int, default=100)
    rl.add_argument("--n_hidden_dim", type=int, default=64)
    rl.add_argument("--batch_size", type=int, default=128)
    rl.add_argument("--learning_rate", type=float, default=3e-4)
    rl.add_argument("--discount_factor", type=float, default=0.96)
    rl.add_argument("--replay_memory_size", type=int, default=50_000)
    rl.add_argument("--init_size", type=int, default=1_500)
    rl.add_argument("--num_updates_per_episode", type=int, default=10)
    rl.add_argument("--update_target_every", type=int, default=10)
    rl.add_argument("--epsilon_schedule", default="episode",
                    choices=["episode", "linear"])
    rl.add_argument("--epsilon_start", type=float, default=1.0)
    rl.add_argument("--epsilon_end", type=float, default=0.1)
    rl.add_argument("--epsilon_decay_steps", type=int, default=500_000)

    dyn = parser.add_argument_group("dynamic reward scaling")
    dyn.add_argument("--lambda_init", type=float, default=10.0)
    dyn.add_argument("--lambda_alpha", type=float, default=0.001)
    dyn.add_argument("--lambda_target", type=float, default=0.0)
    dyn.add_argument("--lambda_min", type=float, default=0.1)
    dyn.add_argument("--lambda_max", type=float, default=10.0)

    al = parser.add_argument_group("active learning")
    al.add_argument("--al_budget", type=int, default=200)
    al.add_argument("--al_fraction", type=float, default=0.0,
                    help="if > 0, query this fraction of each series' windows "
                         "per episode (paper Sec. V-B: 0.05); overrides --al_budget")
    al.add_argument("--lp_budget", type=int, default=200)
    al.add_argument("--lp_selection", default="certain",
                    choices=["certain", "uncertain"],
                    help="which unlabelled windows receive LP pseudo-labels: "
                         "'certain' = most confident (RLAD/myasp-smd lineage, "
                         "matches warm-up); 'uncertain' = least confident "
                         "(myasp-wadi variant)")
    al.add_argument("--lp_neighbors", type=int, default=10)
    al.add_argument("--lp_max_samples", type=int, default=5_000)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--validation_separate_ratio", type=float, default=0.8)
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    misc.add_argument("--save_every", type=int, default=10,
                      help="checkpoint the Q-network every N episodes (0 = only at the end)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = DRSMTConfig(
        n_steps=args.n_steps,
        dataset=args.dataset,
        data_dir=args.data_dir,
        n_hidden_dim=args.n_hidden_dim,
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        replay_memory_size=args.replay_memory_size,
        replay_memory_init_size=args.init_size,
        num_updates_per_episode=args.num_updates_per_episode,
        update_target_every=args.update_target_every,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        lambda_init=args.lambda_init,
        lambda_alpha=args.lambda_alpha,
        lambda_target=args.lambda_target,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        al_budget=args.al_budget,
        al_fraction=args.al_fraction,
        lp_budget=args.lp_budget,
        lp_selection=args.lp_selection,
        lp_neighbors=args.lp_neighbors,
        lp_max_samples=args.lp_max_samples,
        validation_separate_ratio=args.validation_separate_ratio,
        seed=args.seed,
        device=args.device,
    )
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    rng = np.random.default_rng(cfg.seed)
    print(f"[train-rl] device: {device}")

    # ------------------------------------------------------------------ data
    data = load_and_prepare(cfg)
    recon_model, scaler, recon_meta = load_reconstructor(args.vae_model_dir)
    if int(recon_meta["n_steps"]) != cfg.n_steps:
        print(f"[train-rl] adopting n_steps={recon_meta['n_steps']} from the "
              f"reconstruction-model meta")
        cfg.n_steps = int(recon_meta["n_steps"])
    if int(recon_meta["n_features"]) != data.n_features:
        raise ValueError(
            f"the reconstruction model was trained on {recon_meta['n_features']} "
            f"features but the dataset provides {data.n_features} (check "
            f"--data_dir and the zero-variance setting used in train_vae.py)"
        )

    train_series, valid_series = split_train_valid(
        data.test, cfg.validation_separate_ratio
    )
    print(f"[train-rl] RL-train series: {[s.name for s in train_series]}")
    print(f"[train-rl] validation series: {[s.name for s in valid_series]}")

    envs = build_envs(train_series, recon_model, scaler, cfg, device,
                      dynamic_coef=cfg.lambda_init)

    # ---------------------------------------------------------------- replay
    if args.replay_path and os.path.exists(args.replay_path):
        replay, labels = load_replay_payload(args.replay_path)
        n_revealed = 0
        for name, (idx, vals) in labels.items():
            if name not in envs:
                continue
            for i, v in zip(idx, vals):
                envs[name].reveal(int(i), float(v))
            n_revealed += len(idx)
        print(f"[train-rl] loaded {len(replay)} warm-up transitions and "
              f"{n_revealed} revealed labels from {args.replay_path}")
    else:
        print("[train-rl] no replay file found; running inline WARMUP "
              "(recommend running scripts/warmup_replay.py for reproducibility)")
        replay, labels = warmup_replay_memory(envs, cfg, rng)
        if args.replay_path:
            save_replay_payload(args.replay_path, replay, cfg, labels)

    # ------------------------------------------------------------- networks
    policy = QNetwork(cfg.n_steps, data.n_features, cfg.n_hidden_dim).to(device)
    target = QNetwork(cfg.n_steps, data.n_features, cfg.n_hidden_dim).to(device)
    sync_target(policy, target)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[train-rl] LSTM-DQN parameters: {n_params:,} "
          f"(hidden={cfg.n_hidden_dim}, n_steps={cfg.n_steps}, d={data.n_features})")

    current_lambda = cfg.lambda_init
    total_updates = 0
    history = {
        "episode": [],
        "series": [],
        "reward": [],
        "lambda": [],
        "epsilon": [],
        "loss": [],
        "al_count": [],
        "lp_count": [],
        "labeled_total": [],
    }

    # --------------------------------------------------------------- episodes
    for ep in range(1, cfg.episodes + 1):
        series = train_series[(ep - 1) % len(train_series)]
        env = envs[series.name]
        env.set_dynamic_coef(current_lambda)

        # 1) active learning + label propagation (Algorithm 1 lines 19-21)
        learner = MarginActiveLearner(policy, device)
        al_cfg = cfg
        if cfg.al_fraction > 0:  # paper Sec. V-B: "5% of the windows"
            n_windows = env.length - cfg.n_steps
            al_cfg = cfg.with_overrides(
                {"al_budget": max(1, int(round(cfg.al_fraction * n_windows)))}
            )
        al_info = active_learning_step(env, learner, al_cfg)

        # 2) epsilon-greedy rollout (lines 23-29); the policy is frozen
        #    during the episode, so its Q-values are pre-computed batched
        epsilon = epsilon_at(cfg, ep, total_updates)
        q_table = precompute_q_table(policy, env, device)
        ep_info = rollout(env, policy, device, epsilon, rng,
                          replay=replay, record=False, q_table=q_table)
        episode_reward = ep_info["reward"]

        # 3) replay updates (lines 30-32)
        losses = []
        for _ in range(cfg.num_updates_per_episode):
            if len(replay) < 2:
                break
            bs = min(cfg.batch_size, len(replay))
            windows, actions, rewards, next_windows, dones = replay.sample(
                bs, cfg.n_steps, data.n_features
            )
            # augment each state with the indicator of its taken action
            x = np.concatenate(
                [windows, np.zeros((bs, cfg.n_steps, 1), dtype=np.float32)], axis=2
            )
            x[np.arange(bs), :, -1] = actions
            pred = policy(torch.as_tensor(x, device=device))

            target_vec = pred.detach().clone()
            non_terminal = np.where(~dones)[0]
            if len(non_terminal):
                v_next = q_values_max_next(target, next_windows[non_terminal], device)
                target_vec[
                    non_terminal, actions[non_terminal]
                ] = torch.as_tensor(
                    rewards[non_terminal, actions[non_terminal]]
                    + cfg.discount_factor * v_next,
                    dtype=torch.float32,
                    device=device,
                )
            loss = F.mse_loss(pred, target_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_updates += 1
            losses.append(float(loss.item()))
            if total_updates % cfg.update_target_every == 0:
                sync_target(policy, target)

        # 4) dynamic coefficient update (line 34)
        current_lambda = update_lambda(cfg, current_lambda, episode_reward)

        history["episode"].append(ep)
        history["series"].append(series.name)
        history["reward"].append(float(episode_reward))
        history["lambda"].append(float(current_lambda))
        history["epsilon"].append(float(epsilon))
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["al_count"].append(int(al_info["al_count"]))
        history["lp_count"].append(int(al_info["lp_count"]))
        history["labeled_total"].append(int((env.work_labels != -1).sum()))

        print(
            f"[train-rl] ep {ep:03d}/{cfg.episodes} series={series.name:<14} "
            f"eps={epsilon:.3f} lambda={current_lambda:.3f} "
            f"reward={episode_reward:12.2f} loss={history['loss'][-1]:.5f} "
            f"AL={al_info['al_count']} LP={al_info['lp_count']} "
            f"labeled={history['labeled_total'][-1]}"
        )

        if args.save_every and ep % args.save_every == 0:
            out_dir = os.path.join(args.output_dir, args.run_name or cfg.dataset)
            save_q_network(out_dir, policy, _dqn_meta(cfg, data))
            print(f"[train-rl] checkpoint saved to {out_dir}")

    # ------------------------------------------------------------------ save
    out_dir = os.path.join(args.output_dir, args.run_name or cfg.dataset)
    save_q_network(out_dir, policy, _dqn_meta(cfg, data))
    cfg.save(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[train-rl] saved Q-network, config and history to {out_dir}")
    print(f"[train-rl] final lambda = {current_lambda:.4f} "
          f"(init {cfg.lambda_init}); validate with scripts/evaluate.py")


def _dqn_meta(cfg: DRSMTConfig, data) -> dict:
    return {
        "dataset": cfg.dataset,
        "data_dir": cfg.data_dir,
        "n_steps": cfg.n_steps,
        "n_features": data.n_features,
        "episodes": cfg.episodes,
        "n_hidden_dim": cfg.n_hidden_dim,
        "discount_factor": cfg.discount_factor,
        "lambda_init": cfg.lambda_init,
        "reward_unlabeled": cfg.reward_unlabeled,
    }


if __name__ == "__main__":
    main()
