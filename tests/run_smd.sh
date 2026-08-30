#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full DRSMT pipeline on the real SMD benchmark (paper Sec. V, Table II).
#
# The ServerMachineDataset ships with this repository under
#   SMD/ServerMachineDataset/{train,test,test_label}   (28 machines, 38 dims)
#
# Hyper-parameters follow the paper (Sec. IV/V, Algorithm 1) and the
# original `myasp-smd.py`: n_steps=25, LSTM hidden=64, episodes=100,
# gamma=0.96, batch=128, lr=3e-4, TP/TN/FP/FN = 10/1/-1/-10,
# lambda_0=10, lambda in [0.1, 10], K_AL = 5% of windows/episode (paper
# Sec. V-B; repo used K_AL=200), K_LP=200, alpha=1e-4 (see LAMBDA_ALPHA
# below; the repo's 1e-3 was calibrated on its active-learning-subset
# reward).  VAE window scaler defaults to "standard" (Algorithm 1 line 6).
#
# Usage:  bash tests/run_smd.sh [EPISODES] [MODEL]
# Example: bash tests/run_smd.sh 100              # VAE backbone (paper)
#          bash tests/run_smd.sh 100 transformer  # Transformer backbone
#
# NOTE: on CPU this is a long run (~28k steps x 22 machines x 100 episodes).
# Set CUDA_VISIBLE_DEVICES to use a GPU and/or reduce --episodes for a dry run.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${1:-100}"
MODEL="${2:-vae}"                      # "vae" (paper) or "transformer"
RECON_DIR="models/${MODEL}/smd"
PY="${PYTHON:-python}"
SMD_DIR="SMD/ServerMachineDataset"

# Paper-fidelity settings overridable via environment:
#   AL_FRACTION=0.05    -> "5% of the most confusing windows per episode"
#   LAMBDA_ALPHA=1e-4   -> gradual Fig. 2a-style decay of lambda for the
#                          full-episode reward (repo-literal: 1e-3 on its
#                          active-learning-subset reward)
AL_FRACTION="${AL_FRACTION:-0.05}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-1e-4}"

echo "== 1/5 BUILDVAE (${MODEL} backbone) on the 28 normal training machines =="
"$PY" scripts/train_vae.py \
  --model "$MODEL" \
  --dataset smd --data_dir "$SMD_DIR" \
  --output_dir "models/$MODEL" --run_name smd \
  --n_steps 25 --latent_dim 10 --epochs 50

echo "== 2/5 WARMUP =="
"$PY" scripts/warmup_replay.py \
  --dataset smd --data_dir "$SMD_DIR" \
  --vae_model_dir "$RECON_DIR" \
  --output_replay "replay_memory_smd_${MODEL}.pkl" \
  --n_steps 25 --init_size 1500 --lambda_init 10.0

echo "== 3/5 TRAINRL =="
"$PY" scripts/train_rl.py \
  --dataset smd --data_dir "$SMD_DIR" \
  --vae_model_dir "$RECON_DIR" \
  --replay_path "replay_memory_smd_${MODEL}.pkl" \
  --output_dir models/dqn --run_name smd \
  --n_steps 25 --n_hidden_dim 64 --episodes "$EPISODES" \
  --batch_size 128 --learning_rate 3e-4 --discount_factor 0.96 \
  --al_fraction "$AL_FRACTION" --lp_budget 200 \
  --lambda_init 10.0 --lambda_alpha "$LAMBDA_ALPHA" \
  --lambda_target 0.0 --lambda_min 0.1 --lambda_max 10.0

echo "== 4/5 VALIDATE (held-out 20% of the machines) =="
"$PY" scripts/evaluate.py \
  --dataset smd --data_dir "$SMD_DIR" \
  --vae_model_dir "$RECON_DIR" \
  --dqn_model_dir models/dqn/smd \
  --output "results/smd_${MODEL}_metrics.json"

echo "== 5/5 Fig. 2 + Fig. 3 plots =="
"$PY" scripts/visualize_results.py \
  --run_dir models/dqn/smd \
  --eval_dir "results/smd_${MODEL}_episodes" \
  --metrics_json "results/smd_${MODEL}_metrics.json" \
  --output_dir "results/plots_smd_${MODEL}"
