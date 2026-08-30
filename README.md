

A semi‐supervised RL framework for time‐series anomaly detection.  


## Features

- **VAE Module**  
  Learns “normal” patterns and provides reconstruction‐error as intrinsic reward.

- **DQN Agent (LSTM)**  
  Classifies sliding windows as normal/anomalous.

- **Dynamic Reward (λ)**  
  Balances intrinsic (VAE) and extrinsic (classification) rewards, gradually shifting focus during training.

- **Active Learning**  
  Selects the most uncertain windows for labeling and propagates labels via LabelSpreading.

---

## Datasets

Supported benchmarks:
- **Yahoo A1**
- **Yahoo A2**
- **SMD (Server Machine Dataset)**
- **WaDi (Water Distribution Dataset)**

Place raw CSV files under:
```bash
data/
├── yahoo_a1/
├── yahoo_a2/
├── smd/
└── wadi/
```
Scripts automatically handle sliding window extraction (e.g., `window_size = 50`) and normalization.

---

## Installation

```bash
git clone https://github.com/baharehgl/Dynamic-Reward-RL-VAE.git
cd Dynamic-Reward-RL-VAE
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```bash
All commands assume your virtual environment is active and you are in the repository root.
Replace DATASET with one of: yahoo_a1, yahoo_a2, smd, or wadi.
```

### 1-Train VAE
```bash
python scripts/train_vae.py \
  --data_dir data/DATASET/ \
  --output_dir models/vae/
Trains on “normal” windows and saves encoder/decoder weights to models/vae/.
```
### 2-Warm‐up Replay Memory
```bash
python scripts/warmup_replay.py \
  --data_dir data/DATASET/ \
  --vae_model_dir models/vae/ \
  --output_replay replay_memory.pkl
```
### 3-Train RL Agent
```bash
python scripts/train_rl.py \
  --data_dir data/DATASET/ \
  --vae_model_dir models/vae/ \
  --replay_path replay_memory.pkl \
  --output_dir models/dqn/
```
### 4-Evaluate & Visualize
```bash
python scripts/evaluate.py \
  --data_dir data/DATASET/ \
  --vae_model_dir models/vae/ \
  --dqn_model_dir models/dqn/ \
  --output_metrics results/DATASET_metrics.json
python scripts/visualize_results.py \
  --log_dir models/dqn/ \
  --output_dir results/plots/
```
## Repository Structure
```bash
Dynamic-Reward-RL-VAE/
├── data/
│   ├── yahoo_a1/       
│   ├── yahoo_a2/       
│   ├── smd/            
│   └── wadi/           
│
├── models/
│   ├── vae/            # VAE definitions & weights
│   └── dqn/            # DQN (LSTM) definitions & weights
│
├── scripts/
│   ├── train_vae.py
│   ├── warmup_replay.py
│   ├── train_rl.py
│   ├── evaluate.py
│   ├── visualize_results.py
│   └── active_learning.py
│
├── tests/              # runnable pipelines: smoke_test_synthetic.sh, run_smd.sh
│
├── utils/
│   ├── data_loader.py  # Sliding-window & normalization
│   ├── metrics.py      # Precision/Recall/F1 computations
│   └── config.py       # Default hyperparameters
│
├── Dynamic Reward.pdf  
├── requirements.txt
├── LICENSE
├── legacy/             # original monolithic TF1 scripts + old weights (reference only)
└── README.md
```

## Recreated implementation (DRSMT, PyTorch)

The original repository only shipped monolithic TF1 scripts; the modular
implementation below re-creates the missing `models/`, `scripts/` and `utils/`
packages so they reproduce the paper (Algorithm 1) with PyTorch instead of
TF1.  Mapping to the paper:

| Paper component | File |
| --- | --- |
| Default hyper-parameters (Sec. IV/V, Fig. 2) | `utils/config.py` |
| Sliding windows, zero-variance feature selection, Min-Max / VAE scalers (Sec. IV-A) | `utils/data_loader.py` |
| Precision / Recall / F1 / AU-PR (Sec. V, Table II) | `utils/metrics.py` |
| MDP environment: state s_a ∈ R^{Nsteps×(d+1)}, binary actions, asymmetric reward R1 + λ·R2 (Sec. IV-B) | `utils/env.py` |
| VAE: ELBO loss, latent dim 10, reconstruction-error penalty R2 (Sec. III-B / IV-A) | `models/vae/vae_model.py` |
| **Transformer autoencoder** (optional R2 backbone, same I/O contract; `--model transformer`) | `models/transformer/transformer_model.py` |
| Backbone-agnostic BUILDVAE training, COMPUTEPENALTY, save/load factory | `models/reconstruction.py` |
| LSTM-DQN (64 hidden units) + target network (Sec. III-A / IV-B) | `models/dqn/q_network.py` |
| Experience replay ⟨s, a, r, s′⟩ | `models/dqn/replay_buffer.py` |
| BUILDVAE | `scripts/train_vae.py` |
| COMPUTEPENALTY | `models/vae/vae_model.py::compute_penalty_array` (called by the scripts) |
| WARMUP (IsolationForest + LabelSpreading + random rollouts) | `scripts/warmup_replay.py`, `scripts/common.py` |
| TRAINRL (ε-greedy, replay, λ proportional controller) | `scripts/train_rl.py` |
| Active learning: margin sampling + label propagation (Sec. IV-C) | `scripts/active_learning.py` |
| VALIDATE (K equal slices, mean F1 / mean AU-PR) | `scripts/evaluate.py` |
| Fig. 2 (λ and reward curves) and Fig. 3 (episode panels) | `scripts/visualize_results.py` |
| Synthetic benchmark generator for end-to-end testing | `scripts/generate_synthetic.py` |

### Quickstart (synthetic data, no download required)

```bash
pip install -r requirements.txt
bash tests/smoke_test_synthetic.sh          # generate → VAE → warmup → RL → evaluate → plots
```

The reconstruction backbone that produces the intrinsic reward R2 is
pluggable — `tests/smoke_test_synthetic.sh [EPISODES] [MODEL]` and every
script accept `--model vae` (the paper's VAE, default) or `--model
transformer` (Transformer autoencoder: per-time-step self-attention encoder,
10-d latent bottleneck, sigmoid decoder; `--variational` keeps the paper's
ELBO objective, otherwise pure reconstruction loss).  WARMUP / TRAINRL /
VALIDATE are untouched — they simply point `--vae_model_dir` at
`models/transformer/<run>` instead of `models/vae/<run>`:

```bash
bash tests/smoke_test_synthetic.sh 30 transformer   # transformer backbone
python scripts/train_vae.py --model transformer \
  --dataset synthetic --data_dir data/synthetic \
  --output_dir models/transformer --run_name synthetic
```

The end-to-end dataflow (stages, artifacts and the per-episode RL loop) is
diagrammed in [FLOW.md](FLOW.md).

### SMD (data ships in this repo) / WADI

```bash
bash tests/run_smd.sh 100                   # full paper setup on the 28 machines
# WADI: place WADI_14days_new.csv + WADI_attackdataLABLE.csv under data/wadi/
# then swap --dataset wadi --data_dir data/wadi in the commands above.
```

### SMD pipeline, step by step (run sequentially in one terminal)

```bash
cd Dynamic-Reward-RL-VAE
source .venv/bin/activate        # or: pip install -r requirements.txt

# 1) BUILDVAE -- train the VAE on the 28 anomaly-free training machines
#    (Algorithm 1, lines 1-6)
python scripts/train_vae.py \
  --dataset smd --data_dir SMD/ServerMachineDataset \
  --output_dir models/vae --run_name smd \
  --n_steps 25 --latent_dim 10 --intermediate_dim 64 --epochs 50

# 2) WARMUP -- IsolationForest labels + LabelSpreading + random rollouts
#    fill the replay memory (Algorithm 1, lines 12-15)
python scripts/warmup_replay.py \
  --dataset smd --data_dir SMD/ServerMachineDataset \
  --vae_model_dir models/vae/smd \
  --output_replay replay_memory_smd.pkl \
  --n_steps 25 --init_size 1500 --lambda_init 10.0

# 3) TRAINRL -- LSTM-DQN + dynamic reward scaling + active learning
#    (Algorithm 1, lines 16-34); 100 episodes = the paper's setting
python scripts/train_rl.py \
  --dataset smd --data_dir SMD/ServerMachineDataset \
  --vae_model_dir models/vae/smd \
  --replay_path replay_memory_smd.pkl \
  --output_dir models/dqn --run_name smd \
  --n_steps 25 --n_hidden_dim 64 --episodes 100 \
  --batch_size 128 --learning_rate 3e-4 --discount_factor 0.96 \
  --al_budget 200 --lp_budget 200 \
  --lambda_init 10.0 --lambda_alpha 0.001 \
  --lambda_target 0.0 --lambda_min 0.1 --lambda_max 10.0

# 4) VALIDATE -- greedy policy on the held-out 20% of the machines;
#    per-machine Precision / Recall / F1 / AU-PR (Algorithm 1, lines 36-43)
python scripts/evaluate.py \
  --dataset smd --data_dir SMD/ServerMachineDataset \
  --vae_model_dir models/vae/smd \
  --dqn_model_dir models/dqn/smd \
  --output results/smd_metrics.json

# 5) Fig. 2 (lambda + reward curves) and Fig. 3 (episode panels)
python scripts/visualize_results.py \
  --run_dir models/dqn/smd \
  --eval_dir results/smd_episodes \
  --metrics_json results/smd_metrics.json \
  --output_dir results/plots_smd
```

Tips: start with `--episodes 20` for a dry run (CPU); add `--device cuda` to
steps 1/3/4 on a GPU.  The paper labels "only 5% of the most confusing
windows per episode" — pass `--al_fraction 0.05` to step 3 for that setting
(default `--al_budget 200` follows the original code).  For the Transformer
reconstruction backbone add `--model transformer` to step 1 and point
`--vae_model_dir` at `models/transformer/smd` in steps 2-4 (or just run
`bash tests/run_smd.sh 100 transformer`).

Where the two original research scripts disagreed (e.g. LSTM hidden size 128
on SMD vs 64 on WADI, ε decay over updates vs episodes), the paper's value is
the default and the alternative is a `DRSMTConfig` / CLI option; each choice
is documented in the corresponding module docstring.

## Citation
```
@inproceedings{golchin2025dynamic,
  title     = {Dynamic Reward Scaling for Reinforcement Learning in Time Series Anomaly Detection},
  author    = {Golchin, Bahareh and Rekabdar, Banafsheh and Liu, Kunpeng},
  booktitle = {Proceedings of the IEEE AIxSET 2025},
  year      = {2025},
  publisher = {IEEE}  
}
```
<!--
## Paper
```
Dynamic Reward Scaling for Reinforcement Learning in Time Series Anomaly Detection
ICML 2025.
Download the full PDF here.
```

## Citation
```
@inproceedings{golchin2025dynamic,
  title        = {Dynamic Reward Scaling for Reinforcement Learning in Time Series Anomaly Detection},
  author       = {Golchin, Bahareh and Rekabdar, Banafsheh and Liu, Kunpeng},
  booktitle    = {ICML},
  year         = {2025},
  note         = {Code: \url{https://github.com/baharehgl/Dynamic-Reward-RL-VAE}}
}
```
-->
