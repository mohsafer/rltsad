#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DRSMT end-to-end smoke test on the SYNTHETIC benchmark.
#
# Exercises every recreated component of the paper (Algorithm 1) without
# needing the access-restricted SMD/WADI downloads:
#   1. generate the synthetic SMD-style benchmark (scripts/generate_synthetic.py)
#   2. BUILDVAE        (scripts/train_vae.py)
#   3. WARMUP          (scripts/warmup_replay.py)
#   4. TRAINRL         (scripts/train_rl.py)
#   5. VALIDATE        (scripts/evaluate.py)
#   6. Fig.2 / Fig.3   (scripts/visualize_results.py)
#
# Usage:   bash tests/smoke_test_synthetic.sh [EPISODES] [MODEL]
# Example: bash tests/smoke_test_synthetic.sh 30               # VAE backbone
#          bash tests/smoke_test_synthetic.sh 30 transformer   # Transformer backbone
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${1:-30}"
MODEL="${2:-vae}"                      # "vae" (paper) or "transformer"
RECON_DIR="models/${MODEL}/synthetic"  # weights dir of the reconstruction model
PY="${PYTHON:-python}"

# Paper-fidelity settings overridable via environment:
#   AL_FRACTION=0.05    -> "5% of the most confusing windows per episode"
#   LAMBDA_ALPHA=1e-4   -> gradual Fig. 2a-style decay of lambda
AL_FRACTION="${AL_FRACTION:-0.05}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-1e-4}"

echo "== 0/6 sanity: script import checks =="
"$PY" scripts/active_learning.py --help > /dev/null
"$PY" scripts/train_vae.py --help > /dev/null
"$PY" scripts/warmup_replay.py --help > /dev/null
"$PY" scripts/train_rl.py --help > /dev/null
"$PY" scripts/evaluate.py --help > /dev/null
"$PY" scripts/visualize_results.py --help > /dev/null
echo "all scripts import cleanly"

echo "== 1/6 synthetic benchmark =="
"$PY" scripts/generate_synthetic.py \
  --output_dir data/synthetic \
  --n_train 4 --n_test 4 --T 6000 --d 8 \
  --anomaly_fraction 0.05 --seed 7

echo "== 2/6 BUILDVAE with the ${MODEL} backbone (Algorithm 1 lines 1-6) =="
"$PY" scripts/train_vae.py \
  --model "$MODEL" \
  --dataset synthetic --data_dir data/synthetic \
  --output_dir "models/$MODEL" --run_name synthetic \
  --n_steps 25 --latent_dim 10 --epochs 50

echo "== 3/6 WARMUP (Algorithm 1 lines 12-15) =="
"$PY" scripts/warmup_replay.py \
  --dataset synthetic --data_dir data/synthetic \
  --vae_model_dir "$RECON_DIR" \
  --output_replay "replay_memory_${MODEL}.pkl" \
  --n_steps 25 --init_size 1500 --lambda_init 10.0

echo "== 4/6 TRAINRL (Algorithm 1 lines 16-34) =="
"$PY" scripts/train_rl.py \
  --dataset synthetic --data_dir data/synthetic \
  --vae_model_dir "$RECON_DIR" \
  --replay_path "replay_memory_${MODEL}.pkl" \
  --output_dir models/dqn --run_name synthetic \
  --n_steps 25 --n_hidden_dim 64 --episodes "$EPISODES" \
  --batch_size 128 --learning_rate 3e-4 --discount_factor 0.96 \
  --al_fraction "$AL_FRACTION" --lp_budget 200 \
  --lambda_init 10.0 --lambda_alpha "$LAMBDA_ALPHA" \
  --lambda_target 0.0 --lambda_min 0.1 --lambda_max 10.0

echo "== 5/6 VALIDATE (Algorithm 1 lines 36-43) =="
"$PY" scripts/evaluate.py \
  --dataset synthetic --data_dir data/synthetic \
  --vae_model_dir "$RECON_DIR" \
  --dqn_model_dir models/dqn/synthetic \
  --output "results/synthetic_${MODEL}_metrics.json"

echo "== 6/6 Fig. 2 + Fig. 3 plots =="
"$PY" scripts/visualize_results.py \
  --run_dir models/dqn/synthetic \
  --eval_dir "results/synthetic_${MODEL}_episodes" \
  --metrics_json "results/synthetic_${MODEL}_metrics.json" \
  --output_dir "results/plots_${MODEL}"

echo ""
echo "Smoke test finished (${MODEL} backbone). Inspect:"
echo "  results/synthetic_${MODEL}_metrics.json   (precision/recall/F1/AU-PR)"
echo "  results/plots_${MODEL}/fig2a_lambda_evolution.png"
echo "  results/plots_${MODEL}/fig2b_training_reward.png"
echo "  results/plots_${MODEL}/validation_*.png   (Fig. 3 style panels)"
