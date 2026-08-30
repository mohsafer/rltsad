"""Default hyper-parameters for DRSMT.

Dynamic Reward Scaling for Multivariate Time Series Anomaly Detection:
A VAE-Enhanced Reinforcement Learning Approach (Golchin & Rekabdar).

Every value below is taken either from the paper (Sec. IV/V, Algorithm 1,
Fig. 2) or from the original research code shipped in this repository
(`myasp-smd.py`, `myasp-wadi.py`, `RLVAL.py`).  Where the two variants of the
original code disagree (e.g. LSTM hidden size 128 on SMD vs 64 on WADI), the
paper value is used as the default and the alternative is documented.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class DRSMTConfig:
    # ------------------------------------------------------------------ data
    # Sliding-window length n_steps (paper Sec. IV-A "NSTEPS"; repo uses 25).
    n_steps: int = 25
    # Dataset selector understood by utils.data_loader.load_dataset.
    dataset: str = "synthetic"  # {"smd", "synthetic", "wadi", "yahoo"}
    data_dir: str = "data/synthetic"
    # Fraction of the *series* (multi-series datasets, e.g. SMD's 28 machines)
    # used for RL training; the remaining tail is held out for validation
    # (repo: validation_separate_ratio = 0.8).
    validation_separate_ratio: float = 0.8
    # Number of equal slices K used by VALIDATE (Algorithm 1, line 36-43) when
    # the dataset consists of a single long series (e.g. WADI).
    k_validation_slices: int = 5
    # Evaluation protocol: "pointwise" = paper protocol (point-wise sklearn
    # precision/recall/F1 + AU-PR, zero_division=0); "rlad" = the ancestor
    # RLAD baseline protocol (reward-tolerance correction +/-5 steps +
    # add-one smoothing, see utils.metrics.rlad_protocol_metrics) for
    # apples-to-apples comparison with that baseline.  Default is the paper's.
    eval_protocol: str = "pointwise"  # {"pointwise", "rlad"}
    # Remove sensors with zero variance across the training samples
    # (paper Sec. IV-A "feature selection").
    drop_zero_variance: bool = True
    # Cap on the number of VAE training windows (the original WADI script
    # sampled 200; a larger default gives a better behaved VAE).
    max_vae_samples: int = 100_000
    seed: int = 42

    # ------------------------------------------------------------------- VAE
    # Dense layers per paper Sec. IV-A; the SMD variant of the original code
    # stacks 3 hidden layers of 64 units, the WADI variant 1 layer of 128.
    vae_intermediate_dim: int = 64
    vae_encoder_layers: int = 3
    vae_latent_dim: int = 10  # mu(x), sigma^2(x) dimensionality
    vae_epochs: int = 50      # SMD script; WADI script used 20
    vae_batch_size: int = 32
    vae_learning_rate: float = 1e-3
    vae_grad_clip: float = 1.0  # clipnorm=1.0 in the original optimizer
    # Window scaler for the VAE input.  Algorithm 1 (line 6) says
    # "Standardize" and the SMD variant of the original code used a
    # StandardScaler; "robust" reproduces the WADI variant (RobustScaler to
    # avoid overflow on outlier windows).
    vae_scaler: str = "standard"
    # Scaled windows are clipped to +-vae_scale_clip before the VAE (WADI
    # variant clips to [-10, 10] so outlier test windows stay bounded).
    vae_scale_clip: float = 10.0

    # ------------------------------------------- reconstruction backbone
    # "vae" keeps the paper's Variational Autoencoder; "transformer" swaps in
    # the Transformer autoencoder of models/transformer/ as the source of the
    # reconstruction-error penalty R2 (same I/O contract, same penalty scale).
    recon_model: str = "vae"  # {"vae", "transformer"}
    # Transformer autoencoder hyper-parameters (token = one time step).
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_encoder_layers: int = 2
    transformer_num_decoder_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    # If True, the transformer keeps the VAE's variational objective
    # (Transformer-VAE with the same ELBO loss); otherwise it is a
    # deterministic autoencoder trained with pure reconstruction loss.
    transformer_variational: bool = False

    # ------------------------------------------------------------------- DQN
    # LSTM hidden units (paper Sec. IV-B: "LSTM layers (64 hidden units)").
    n_hidden_dim: int = 64
    episodes: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    discount_factor: float = 0.96  # gamma in the Bellman update (Sec. III-A)
    replay_memory_size: int = 50_000
    # M in Algorithm 1 (line 12-15, WARMUP): transitions collected before
    # training starts (repo: replay_memory_init_size = 1500).
    replay_memory_init_size: int = 1_500
    # Gradient updates per episode (repo: num_epoches = 10 on SMD, 5 on WADI).
    num_updates_per_episode: int = 10
    # Hard-sync the target network Q' <- Q every this many gradient updates
    # (Algorithm 1 line 31-32 "if step mod C = 0"; repo: 10).
    update_target_every: int = 10
    # epsilon-greedy exploration (paper Sec. IV-B).  Two schedules:
    #   "episode": linear decay from epsilon_start to epsilon_end over the
    #              whole training run (WADI variant).
    #   "linear":  linear decay over epsilon_decay_steps gradient updates
    #              (SMD variant).
    epsilon_schedule: str = "episode"
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 500_000

    # ------------------------------------------------------- reward (R1, R2)
    # Asymmetric classification reward R1 (paper Sec. IV-B):
    #   TP=10, TN=1, FP=-1, FN=-10.
    tp_value: float = 10.0
    tn_value: float = 1.0
    fp_value: float = -1.0
    fn_value: float = -10.0
    # Reward contributed by steps still unlabelled (-1) in the working label
    # column: "zero" reproduces the SMD variant / Algorithm 1, "anomaly" the
    # WADI variant of the original code.
    reward_unlabeled: str = "zero"

    # Dynamic scaling coefficient lambda(t) (paper Sec. IV-B, Eq. 4;
    # Algorithm 1 line 34):
    #   lambda <- clip(lambda + alpha * (R_target - R_episode), min, max)
    # If the episode reward is below target, lambda grows (more weight on the
    # VAE reconstruction penalty R2 = exploration); otherwise it decays.
    # The paper does not state alpha / R_target; the original code used
    # alpha=0.001, R_target=0 on the reward accumulated over its active-
    # learning subset.  This pipeline accumulates the reward over the FULL
    # episode (thousands of steps), so with alpha=0.001 lambda reaches
    # lambda_min within a few episodes.  For the gradual Fig. 2a-style decay
    # pass a smaller alpha (e.g. --lambda_alpha 1e-4 .. 1e-5).
    lambda_init: float = 10.0
    lambda_alpha: float = 0.001
    lambda_target: float = 0.0
    lambda_min: float = 0.1
    lambda_max: float = 10.0

    # ------------------------------------------------------ active learning
    # K_AL: ground-truth labels queried per episode via margin sampling
    # (paper Sec. IV-C: smallest |Q(s,a1) - Q(s,a2)|; the paper reports
    # labelling "only 5% of the most confusing windows per episode").
    al_budget: int = 200
    # If > 0, overrides ``al_budget`` with this *fraction* of each series'
    # windows per episode -- the paper's "5%" setting is al_fraction=0.05.
    al_fraction: float = 0.0
    # K_LP: pseudo-labels per episode propagated by LabelSpreading
    # (Algorithm 1 line 20-21).
    lp_budget: int = 200
    # LabelSpreading graph parameters.
    lp_neighbors: int = 10
    # Subsample cap for the LabelSpreading fit (the kNN graph is expensive on
    # long series such as WADI's ~172k steps).
    lp_max_samples: int = 5_000
    # Warm-up: IsolationForest contamination, how many most-anomalous /
    # most-normal windows are ground-truth labelled at warm-up time, and the
    # cap on windows used to fit the detector (repo: 10_000).
    warmup_outliers_fraction: float = 0.01
    warmup_label_each: int = 5
    warmup_max_samples: int = 10_000

    # ------------------------------------------------------------- plumbing
    device: str = "auto"  # {"auto", "cpu", "cuda"}
    penalty_batch_size: int = 256

    # ---------------------------------------------------------------- utils
    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DRSMTConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_overrides(self, overrides: dict) -> "DRSMTConfig":
        """Return a copy with the given hyper-parameters replaced.

        Unknown keys raise a KeyError so that typos in CLI flags fail fast.
        """
        known = set(self.__dataclass_fields__)
        bad = set(overrides) - known
        if bad:
            raise KeyError(
                f"Unknown configuration keys: {sorted(bad)}. "
                f"Valid keys are: {sorted(known)}"
            )
        cfg = DRSMTConfig(**{**self.to_dict(), **overrides})
        return cfg


def set_seed(seed: int) -> None:
    """Seed every RNG used by the framework for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
