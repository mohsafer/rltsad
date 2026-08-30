"""Generate synthetic multivariate time series with injected anomalies.

Writes a Server-Machine-Dataset-style directory tree so that the *entire*
DRSMT pipeline runs on data that needs no download or access request::

    <output_dir>/train/syn-<i>.txt        # normal operation only (like SMD's
                                          # first five days, paper Sec. V-A)
    <output_dir>/test/syn-<i>.txt         # test series with injected faults
    <output_dir>/test_label/syn-<i>.txt   # point-wise 0/1 anomaly labels
    <output_dir>/manifest.json            # generation metadata

Normal dynamics model a small industrial process with the properties the
paper emphasises (Sec. I / IV): inter-sensor correlations, setpoint
transitions and temporal dependencies, values scaled to [0, 1] like SMD:

* ch0: slow periodic process + AR(1) noise;
* ch1: correlated copy of ch0 (spatial dependency between sensors);
* ch2: square-wave setpoint with random dwell times;
* ch3: faster periodic process with a phase offset;
* ch4: driven by the setpoint channel;
* ch5..d-1: independent AR(1) channels with mixed parameters.

Anomaly types injected into the test series (~``anomaly_fraction`` of the
steps, default 5%, close to SMD's 4.16% / WADI's 5.77% in Table I):

* ``spike``             -- short 1-5 step excursions on 1-2 sensors;
* ``level_shift``       -- sustained offset on 2-4 sensors;
* ``correlation_break`` -- the ch0/ch1 relationship flips sign (an anomaly
  that "manifests through unexpected combinations of sensor values rather
  than individual sensor deviations", paper Sec. I);
* ``drift``             -- ramping deviation on 2 sensors;
* ``stuck_at``          -- frozen sensor values.

Example:
    python scripts/generate_synthetic.py --output_dir data/synthetic \
        --n_train 4 --n_test 4 --T 6000 --d 8 --anomaly_fraction 0.05 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def ar1(rho: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


def square_wave(n: int, rng: np.random.Generator, low=0.2, high=0.7,
                dwell_lo=200, dwell_hi=400) -> np.ndarray:
    out = np.empty(n)
    value = low if rng.random() < 0.5 else high
    i = 0
    while i < n:
        dwell = int(rng.integers(dwell_lo, dwell_hi + 1))
        out[i : i + dwell] = value
        value = high if value == low else low
        i += dwell
    return out


def synth_normal(T: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if d < 5:
        raise ValueError("d must be >= 5 for the designed channels")
    t = np.arange(T)
    period1 = float(rng.uniform(300, 900))
    period2 = float(rng.uniform(80, 250))
    phase = float(rng.uniform(0, 2 * np.pi))

    ch = np.zeros((T, d), dtype=np.float64)
    ch[:, 0] = 0.5 + 0.25 * np.sin(2 * np.pi * t / period1) + ar1(0.90, 0.03, T, rng)
    ch[:, 1] = 0.8 * ch[:, 0] + ar1(0.80, 0.02, T, rng)          # correlated pair
    ch[:, 2] = square_wave(T, rng) + ar1(0.5, 0.015, T, rng)     # setpoints
    ch[:, 3] = 0.55 + 0.20 * np.sin(2 * np.pi * t / period2 + phase) + ar1(0.85, 0.03, T, rng)
    ch[:, 4] = 0.25 + 0.50 * ch[:, 2] + ar1(0.6, 0.02, T, rng)   # setpoint-driven
    for j in range(5, d):
        rho = float(rng.uniform(0.8, 0.98))
        sigma = float(rng.uniform(0.02, 0.06))
        mean = float(rng.uniform(0.3, 0.7))
        ch[:, j] = mean + ar1(rho, sigma, T, rng)
    return np.clip(ch, 0.0, 1.0)


def _pick_channels(d: int, count: int, rng: np.random.Generator,
                   exclude: tuple = ()) -> list:
    pool = [c for c in range(d) if c not in exclude]
    return sorted(rng.choice(pool, size=min(count, len(pool)), replace=False).tolist())


def inject_anomalies(
    values: np.ndarray,
    labels: np.ndarray,
    anomaly_fraction: float,
    rng: np.random.Generator,
    min_len: int = 25,
) -> list:
    """Inject non-overlapping anomalous intervals; returns the manifest."""
    T, d = values.shape
    target = int(anomaly_fraction * T)
    labeled = 0
    events: list = []
    guard = 0
    while labeled < target and guard < 60:
        guard += 1
        kind = str(rng.choice(["spike", "level_shift", "correlation_break",
                               "drift", "stuck_at"]))
        if kind == "spike":
            length = int(rng.integers(1, 6))
        else:
            length = int(rng.integers(20, 151))
        start = int(rng.integers(min_len, T - length - min_len))
        end = start + length
        if any(start < ev["end"] + 10 and ev["start"] < end + 10 for ev in events):
            continue

        if kind == "spike":
            chans = _pick_channels(d, int(rng.integers(1, 3)), rng)
            amps = rng.uniform(0.5, 0.9, size=len(chans))
            for c, amp in zip(chans, amps):
                values[start:end, c] = np.clip(values[start:end, c] + amp, 0, 1)
        elif kind == "level_shift":
            chans = _pick_channels(d, int(rng.integers(2, 5)), rng)
            delta = float(rng.choice([-1, 1]) * rng.uniform(0.3, 0.5))
            values[start:end, chans] = np.clip(values[start:end, chans] + delta, 0, 1)
        elif kind == "correlation_break":
            # normal: ch1 = 0.8 * ch0 + noise; during the fault the
            # relationship flips to anticorrelated while staying in [0, 1]
            t = np.arange(start, end)
            rng_residual = rng.normal(0, 0.02, len(t))
            values[start:end, 1] = np.clip(
                0.8 * (1.0 - values[start:end, 0]) + rng_residual, 0, 1
            )
            chans = [1]
        elif kind == "drift":
            chans = _pick_channels(d, 2, rng)
            delta = float(rng.choice([-1, 1]) * rng.uniform(0.3, 0.5))
            ramp = np.linspace(0.0, delta, length)
            values[start:end, chans] = np.clip(values[start:end, chans] + ramp[:, None], 0, 1)
        else:  # stuck_at
            chans = _pick_channels(d, int(rng.integers(1, 4)), rng)
            values[start:end, chans] = float(rng.uniform(0.1, 0.9))

        labels[start:end] = 1
        events.append({"type": kind, "start": start, "end": int(end),
                       "channels": [int(c) for c in chans]})
        labeled += length
    return events


def generate_file(T: int, d: int, seed: int, anomaly_fraction: float,
                  test: bool) -> tuple:
    rng = np.random.default_rng(seed)
    values = synth_normal(T, d, rng)
    labels = np.zeros(T, dtype=np.int64)
    events = []
    if test:
        events = inject_anomalies(values, labels, anomaly_fraction, rng)
    return values, labels, events, {
        "seed": seed,
        "anomaly_fraction_actual": float(labels.mean()),
        "events": events,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SMD-style MTSAD benchmark.")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--n_train", type=int, default=4)
    parser.add_argument("--n_test", type=int, default=4)
    parser.add_argument("--T", type=int, default=6_000)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--anomaly_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    label_dir = os.path.join(args.output_dir, "test_label")
    for path in (train_dir, test_dir, label_dir):
        os.makedirs(path, exist_ok=True)

    manifest = {
        "generator": "scripts/generate_synthetic.py",
        "T": args.T,
        "d": args.d,
        "anomaly_fraction_target": args.anomaly_fraction,
        "seed": args.seed,
        "train": {},
        "test": {},
    }
    for i in range(1, args.n_train + 1):
        name = f"syn-{i}"
        values, labels, events, meta = generate_file(
            args.T, args.d, args.seed * 1000 + i, 0.0, test=False
        )
        np.savetxt(os.path.join(train_dir, f"{name}.txt"), values,
                   fmt="%.6f", delimiter=",")
        manifest["train"][name] = meta
        print(f"[synthetic] train/{name}.txt: T={args.T} d={args.d} "
              f"(normal operation)")

    for i in range(1, args.n_test + 1):
        name = f"syn-{i}"
        values, labels, events, meta = generate_file(
            args.T, args.d, args.seed * 1000 + 500 + i, args.anomaly_fraction,
            test=True,
        )
        np.savetxt(os.path.join(test_dir, f"{name}.txt"), values,
                   fmt="%.6f", delimiter=",")
        np.savetxt(os.path.join(label_dir, f"{name}.txt"), labels.reshape(-1, 1),
                   fmt="%d", delimiter=",")
        manifest["test"][name] = meta
        n_events = len(events)
        kinds = sorted({e["type"] for e in events})
        print(f"[synthetic] test/{name}.txt: {n_events} anomalies "
              f"({', '.join(kinds) or '-'}) -> {meta['anomaly_fraction_actual'] * 100:.2f}% "
              f"of steps labelled anomalous")

    with open(os.path.join(args.output_dir, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[synthetic] benchmark written to {args.output_dir} "
          f"(manifest.json includes the injected events)")


if __name__ == "__main__":
    main()
