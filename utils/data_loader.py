"""Data loading, sliding windows and normalisation for DRSMT.

The paper (Sec. IV-A) specifies:

* multivariate sliding windows of shape (n_steps, d) built from d
  synchronised sensor channels;
* feature selection that removes sensors with zero variance across the
  training samples;
* per-feature Min-Max normalisation of the RL observations, and a separate
  scaler (Standard/Robust) feeding the flattened windows to the VAE.

Supported layouts (``dataset`` kind):

* ``"smd"`` / ``"synthetic"`` -- a directory tree::

      <data_dir>/train/*.txt|csv          # normal training series (no labels)
      <data_dir>/test/*.txt|csv           # test series
      <data_dir>/test_label/*.txt|csv     # one 0/1 label per test time step

  This is the official Server Machine Dataset layout; ``scripts/generate_synthetic.py``
  writes the same layout so the identical pipeline runs on synthetic data.
* ``"wadi"`` -- ``WADI_14days_new.csv`` + ``WADI_attackdataLABLE.csv`` in
  ``<data_dir>`` (the cleaned single-series variant used by the original
  ``myasp-wadi.py``).
* ``"yahoo"`` -- ``real_*.csv`` files with header ``timestamp,value,is_anomaly``
  (the Yahoo Webscope A1 release shipped in ``normal-data/``).
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Core containers
# --------------------------------------------------------------------------- #


@dataclass
class SeriesData:
    """One multivariate time series plus its point-wise anomaly labels.

    ``values`` has shape (T, d) float32 and ``labels`` shape (T,) float32
    with 0 = normal, 1 = anomaly.
    """

    name: str
    values: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.ascontiguousarray(self.values, dtype=np.float32)
        self.labels = np.ascontiguousarray(self.labels, dtype=np.float32)
        if self.values.ndim != 2:
            raise ValueError(
                f"series '{self.name}': values must be 2-D (T, d), got {self.values.shape}"
            )
        if len(self.labels) != self.values.shape[0]:
            raise ValueError(
                f"series '{self.name}': {len(self.labels)} labels for "
                f"{self.values.shape[0]} time steps"
            )

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    def __len__(self) -> int:
        return self.values.shape[0]


@dataclass
class LoadedData:
    train: List[SeriesData]
    test: List[SeriesData]
    dataset: str
    feature_names: Optional[List[str]] = None

    @property
    def n_features(self) -> int:
        if self.train:
            return self.train[0].n_features
        return self.test[0].n_features


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #


def _read_matrix_file(path: str) -> np.ndarray:
    """Read a header-less comma separated numeric file (SMD style)."""
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if not np.isfinite(arr).all():
        # Fill any residual NaN (e.g. trailing newline artefacts) by
        # forward/backward fill per column.
        df = pd.read_csv(path, header=None, dtype=np.float64).apply(
            pd.to_numeric, errors="coerce"
        )
        df = df.ffill().bfill().fillna(0.0)
        arr = df.values
    return arr


def _read_label_file(path: str, length: int) -> np.ndarray:
    labels = _read_matrix_file(path).reshape(-1)
    labels = np.nan_to_num(labels, nan=0.0)
    if len(labels) < length:  # zero-pad like the original environment does
        labels = np.concatenate([labels, np.zeros(length - len(labels))])
    return np.clip(labels[:length], 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #


def _list_files(directory: str) -> List[str]:
    files = sorted(
        p
        for p in glob.glob(os.path.join(directory, "*"))
        if os.path.isfile(p) and p.lower().endswith((".txt", ".csv"))
    )
    if not files:
        raise FileNotFoundError(f"no .txt/.csv series found in {directory}")
    return files


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _stem_unique_prefer_txt(paths: List[str]) -> List[str]:
    """Collapse duplicate series given in several formats (e.g. the SMD test
    folder ships every machine as both ``.txt`` and ``.csv``): keep one file
    per stem, preferring the original ``.txt``."""
    by_stem: dict = {}
    for p in paths:
        stem = _stem(p)
        current = by_stem.get(stem)
        if current is None or (p.lower().endswith(".txt")
                               and not current.lower().endswith(".txt")):
            by_stem[stem] = p
    return [by_stem[k] for k in sorted(by_stem)]


def load_smd_style(data_dir: str, dataset: str = "smd") -> LoadedData:
    """Loader for the SMD layout (also used for the synthetic datasets)."""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    label_dir = os.path.join(data_dir, "test_label")

    train: List[SeriesData] = []
    for path in _stem_unique_prefer_txt(_list_files(train_dir)):
        values = _read_matrix_file(path)
        name = _stem(path)
        # Training series are anomaly free (paper Sec. V-A: the first five
        # days of SMD contain only normal operation).
        labels = np.zeros(len(values), dtype=np.float32)
        train.append(SeriesData(name, values, labels))

    # Labels may be stored with a different extension than the series
    # (SMD: series as .txt/.csv, labels as .txt) -- match by stem.
    label_paths = {_stem(p): p for p in _list_files(label_dir)}

    test: List[SeriesData] = []
    for path in _stem_unique_prefer_txt(_list_files(test_dir)):
        values = _read_matrix_file(path)
        name = _stem(path)
        label_path = label_paths.get(name)
        if label_path is None:
            raise FileNotFoundError(
                f"no label file for test series '{name}' "
                f"(looked for '{name}.*' in {label_dir})"
            )
        labels = _read_label_file(label_path, len(values))
        test.append(SeriesData(name, values, labels))

    return LoadedData(train=train, test=test, dataset=dataset)


def load_wadi(data_dir: str) -> LoadedData:
    """Single-series WADI loader (cleaning mirrors ``myasp-wadi.py``)."""
    sensor_csv = os.path.join(data_dir, "WADI_14days_new.csv")
    label_csv = os.path.join(data_dir, "WADI_attackdataLABLE.csv")
    for path in (sensor_csv, label_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"WADI expects {path}")

    df = pd.read_csv(sensor_csv, decimal=".")
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any").reset_index(drop=True)
    if "Row" in df.columns:
        df = df.drop(columns=["Row"])

    lbl_df = pd.read_csv(label_csv, header=1, low_memory=False)
    raw = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    anomalies = np.where(raw == 1, 0, 1).astype(np.float32)  # 1 -> no attack

    min_len = min(len(df), len(anomalies))
    values = df.iloc[:min_len].to_numpy(dtype=np.float64)
    labels = anomalies[:min_len]
    feature_names = [str(c) for c in df.columns]

    series = SeriesData("WADI", values, labels)
    return LoadedData(train=[series], test=[series], dataset="wadi",
                      feature_names=feature_names)


def load_yahoo(data_dir: str) -> LoadedData:
    """Yahoo Webscope A1 loader (``real_*.csv`` with is_anomaly column)."""
    files = _list_files(data_dir)
    train: List[SeriesData] = []
    test: List[SeriesData] = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        value_col = "value" if "value" in df.columns else df.columns[1]
        label_col = "is_anomaly" if "is_anomaly" in df.columns else df.columns[-1]
        values = df[value_col].to_numpy(dtype=np.float64).reshape(-1, 1)
        labels = np.clip(
            pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy(), 0, 1
        ).astype(np.float32)
        name = os.path.splitext(os.path.basename(path))[0]
        series = SeriesData(name, values, labels)
        # Yahoo A1 files mix normal and anomalous stretches; the whole file is
        # used for the RL environment, windows for the VAE are filtered to
        # fully-normal stretches in collect_normal_windows().
        train.append(series)
        test.append(series)
    return LoadedData(train=train, test=test, dataset="yahoo")


def load_dataset(dataset: str, data_dir: str) -> LoadedData:
    dataset = dataset.lower()
    if dataset in {"smd", "synthetic"}:
        return load_smd_style(data_dir, dataset=dataset)
    if dataset == "wadi":
        return load_wadi(data_dir)
    if dataset == "yahoo":
        return load_yahoo(data_dir)
    raise ValueError(
        f"unknown dataset '{dataset}'; expected one of smd, synthetic, wadi, yahoo"
    )


# --------------------------------------------------------------------------- #
# Feature selection and normalisation (paper Sec. IV-A)
# --------------------------------------------------------------------------- #


def zero_variance_mask(train: Sequence[SeriesData]) -> np.ndarray:
    """Boolean mask of features with non-zero variance over *all* training rows."""
    if not train:
        raise ValueError("no training series available")
    d = train[0].n_features
    total_sq = np.zeros(d, dtype=np.float64)
    total_sum = np.zeros(d, dtype=np.float64)
    n_rows = 0
    for series in train:
        v = series.values.astype(np.float64)
        total_sum += v.sum(axis=0)
        total_sq += np.square(v).sum(axis=0)
        n_rows += len(v)
    if n_rows == 0:
        raise ValueError("training series are empty")
    mean = total_sum / n_rows
    var = total_sq / n_rows - np.square(mean)
    return var > 0


def drop_zero_variance_features(data: LoadedData) -> LoadedData:
    """Remove zero-variance sensors (fit on the training split only)."""
    mask = zero_variance_mask(data.train)
    kept = np.where(mask)[0]
    dropped = len(mask) - len(kept)
    if dropped == 0:
        return data

    def _apply(series_list: List[SeriesData]) -> List[SeriesData]:
        return [
            SeriesData(s.name, s.values[:, kept], s.labels) for s in series_list
        ]

    names = None
    if data.feature_names is not None:
        names = [data.feature_names[i] for i in kept]
    return LoadedData(train=_apply(data.train), test=_apply(data.test),
                      dataset=data.dataset, feature_names=names)


def fit_minmax(train: Sequence[SeriesData]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature (min, max) fitted on the training series."""
    stacked = np.concatenate([s.values for s in train], axis=0).astype(np.float64)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    maxs = np.where(maxs - mins < 1e-12, mins + 1.0, maxs)  # guard flat sensors
    return mins, maxs


def apply_minmax(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    scaled = (values - mins) / (maxs - mins)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def normalize_data(data: LoadedData) -> LoadedData:
    """Min-Max scale every series with statistics from the training split."""
    mins, maxs = fit_minmax(data.train)

    def _apply(series_list: List[SeriesData]) -> List[SeriesData]:
        return [
            SeriesData(s.name, apply_minmax(s.values, mins, maxs), s.labels)
            for s in series_list
        ]

    return LoadedData(train=_apply(data.train), test=_apply(data.test),
                      dataset=data.dataset, feature_names=data.feature_names)


def prepare_data(dataset: str, data_dir: str, drop_zero_variance: bool = True) -> LoadedData:
    """``load_dataset`` + zero-variance feature selection (no normalisation)."""
    data = load_dataset(dataset, data_dir)
    if drop_zero_variance:
        data = drop_zero_variance_features(data)
    return data


# --------------------------------------------------------------------------- #
# Sliding windows (paper Sec. IV-A / Algorithm 1 lines 3, 8)
# --------------------------------------------------------------------------- #


def make_windows(values: np.ndarray, n_steps: int) -> np.ndarray:
    """All windows of length ``n_steps``.

    Window ``i`` covers rows ``[i, i + n_steps)`` and therefore *ends* at
    time index ``i + n_steps - 1``; the returned array has shape
    ``(T - n_steps + 1, n_steps, d)``.
    """
    values = np.asarray(values, dtype=np.float32)
    T = values.shape[0]
    if T < n_steps:
        raise ValueError(
            f"series length {T} is shorter than the window length {n_steps}"
        )
    idx = np.arange(n_steps)[None, :] + np.arange(T - n_steps + 1)[:, None]
    return values[idx]


def window_labels(labels: np.ndarray, n_steps: int) -> np.ndarray:
    """Label of the last time step of each window (the step the agent classifies)."""
    return np.asarray(labels, dtype=np.float32)[n_steps - 1:]


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten (N, n_steps, d) windows to (N, n_steps * d) for the VAE."""
    return np.ascontiguousarray(windows.reshape(windows.shape[0], -1), dtype=np.float32)


def collect_normal_windows(
    series: Sequence[SeriesData],
    n_steps: int,
    max_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Flattened windows that lie completely inside normal stretches.

    Used to train the VAE "exclusively on normal patterns" (paper Sec. IV-A).
    ``max_samples`` caps the number of windows; the cap is applied per series
    *before* materialisation so multi-machine datasets (SMD: ~780k windows of
    25 x 38) stay within memory.
    """
    rng = rng or np.random.default_rng(0)

    def normal_ends(s: SeriesData) -> np.ndarray:
        """End indices t whose window [t-n_steps+1, t] is fully normal."""
        normal = (s.labels == 0).astype(np.int64)
        if len(normal) < n_steps:
            return np.empty(0, dtype=np.int64)
        cs = np.concatenate([[0], np.cumsum(normal)])
        run = cs[n_steps:] - cs[:-n_steps]  # normal count per window start
        return np.where(run == n_steps)[0] + n_steps - 1

    ends_per_series = [normal_ends(s) for s in series]
    total = int(sum(len(e) for e in ends_per_series))
    if total == 0:
        raise ValueError("no fully-normal windows available for VAE training")

    keep_frac = min(1.0, max_samples / total)
    chunks: List[np.ndarray] = []
    for s, ends in zip(series, ends_per_series):
        if keep_frac < 1.0 and len(ends) > 1:
            k = max(1, int(round(len(ends) * keep_frac)))
            ends = np.sort(rng.choice(ends, size=k, replace=False))
        for end in ends:
            start = end - n_steps + 1
            chunks.append(s.values[start : end + 1])
    windows = np.stack(chunks, axis=0)
    if len(windows) > max_samples:  # per-series rounding may overshoot slightly
        sel = rng.choice(len(windows), size=max_samples, replace=False)
        windows = windows[sel]
    return flatten_windows(windows)
