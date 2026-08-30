# Legacy code (reference only)

Everything in this folder is the original **TF1 / Keras research code** shipped
with the repository before the modular PyTorch re-implementation
(`utils/`, `models/`, `scripts/`, `tests/`).  None of it is imported or
executed by the new pipeline.  It is kept because the monolithic scripts are
the only executable ground truth of the implementation details that the
DRSMT paper omits.

## The DRSMT paper implementation (most useful references)

| File | Role |
| --- | --- |
| `myasp-smd.py` | DRSMT for the Server Machine Dataset: VAE, LSTM-DQN, dynamic λ, active learning, validation (used to reconstruct `scripts/`) |
| `env_smd.py` | Its RL environment (per-machine series, reward vector, semi-supervised label column) |
| `myasp-wadi.py` | DRSMT for WADI: multivariate windows with the action-indicator channel, pre-computed penalty arrays, K-slice validation |
| `env_wadi.py` | Its RL environment |

## Earlier / side experiments (not part of the DRSMT paper)

| File | Role |
| --- | --- |
| `RLVAL.py` | Univariate predecessor (Golchin & Rekabdar, AIxSET 2024, ref. [14] of the paper); hardcoded Windows paths |
| `env.py`, `env1.py`, `environment/` | Early univariate environments |
| `myasp-A1.py`, `myasp-A2.py` | Yahoo A1 / A2 benchmark variants |
| `myasp-KPI.py`, `env_KPI.py`, `asp_DRO_KPI.py` | Tsinghua KPI-dataset experiments |
| `aps.py`, `asp-DRO.py` | Robust-optimisation (DRO) side experiments on Yahoo A1 |
| `convert_hdf_to_csv.py` | Utility: Yahoo/KPI HDF → CSV |
| `test1.py` | Trivial TF GPU check |

## Old artifacts

| Path | Role |
| --- | --- |
| `weights/vae_model.h5`, `weights/vae_model_kpi.h5`, `weights/vae_wadi.h5` | Keras/TF1 VAE weights from the authors' runs (incompatible with the PyTorch pipeline) |
| `phase2_ground_truth.hdf` | KPI-dataset ground-truth stub |
| `exp/` | Original training outputs (the paper's Fig. 2 curves) |

**Note:** the legacy scripts resolve their data paths relative to their own
location (`current_dir = os.path.dirname(__file__)`), so after the move they
would look for `legacy/SMD/...` and `legacy/normal-data/...`.  The actual data
still lives at the repository root (`SMD/`, `normal-data/`,
`ydata-labeled-time-series-anomalies-v1_0/`, `KPI_data/`, `WaDi/`); if you ever
need to run a legacy script, run it with the data folders symlinked/copied
next to it — the supported pipeline in `scripts/` is unaffected.
