"""Precision / Recall / F1 / AU-PR computations (paper Sec. V).

The original research code evaluates every validation episode with
``sklearn.metrics.precision_recall_fscore_support(..., average='binary',
zero_division=0)`` and reports the area under the precision-recall curve with
``sklearn.metrics.average_precision_score`` applied to the agent's *binary*
predictions (this is what Fig. 3's bottom panel plots and what Table II
reports).  ``binary_metrics`` reproduces exactly that; ``y_score`` can
additionally carry a continuous anomaly score (e.g. Q(s,1) - Q(s,0)) for a
threshold-free AU-PR.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

METRIC_KEYS = ("precision", "recall", "f1", "aupr")


def binary_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_score: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Point-wise precision, recall, F1 and AU-PR for one episode.

    AU-PR is ``average_precision_score`` on the binary predictions to match
    the original evaluation; when ``y_score`` is provided the same statistic
    is also computed on the continuous score and returned as ``aupr_score``.
    """
    y_true = np.asarray(y_true).round().astype(int)
    y_pred = np.asarray(y_pred).round().astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        aupr = float(average_precision_score(y_true, y_pred))
    except Exception:
        # Degenerate episode (single ground-truth class): the original code
        # catches the sklearn error and records 0.0.
        aupr = 0.0

    out: Dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "aupr": aupr,
    }

    if y_score is not None:
        try:
            out["aupr_score"] = float(
                average_precision_score(y_true, np.asarray(y_score, dtype=np.float64))
            )
        except Exception:
            out["aupr_score"] = 0.0
    return out


def point_adjust_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Dict[str, float]:
    """Optional point-adjusted variant (an entire segment is credited as soon
    as one of its points is detected).  Not used by the paper's Table II but
    provided for comparability with other MTSAD literature."""
    y_true = np.asarray(y_true).round().astype(int)
    y_pred = np.asarray(y_pred).round().astype(int)
    adjusted = y_pred.copy()
    hit = False
    for i, flag in enumerate(y_true):
        if flag == 1 and y_pred[i] == 1:
            hit = True
        if flag == 0:
            hit = False
        if hit and flag == 1:
            adjusted[i] = 1
    return binary_metrics(y_true, adjusted)


def rlad_protocol_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    tolerance: int = 5,
    tp_value: float = 10.0,
    tn_value: float = 1.0,
    fp_value: float = -1.0,
    fn_value: float = -10.0,
) -> Dict[str, float]:
    """The *ancestor* evaluation protocol of RLAD (twmoveon/RLAD,
    ``q_learning_validator``), provided for apples-to-apples comparison with
    that baseline.  Two steps, both translated literally from the original:

    1. **Reward-tolerance correction**: work on the per-step reward sequence
       (TP/TN/FP/FN values); any *negative* reward within ``tolerance`` steps
       of a true positive is flipped to its positive counterpart
       ("if FN or FP happened within a small range of TP, correct it").
    2. **Add-one smoothed** precision/recall:
       ``precision = (tp+1)/(tp+fp+1)``, ``recall = (tp+1)/(tp+fn+1)``.

    AU-PR is not defined in the RLAD protocol and is reported as 0.0 to keep
    the metric-dict shape consistent.  This is NOT the DRSMT paper protocol —
    use ``binary_metrics`` (the default) for that.
    """
    y_true = np.asarray(y_true).round().astype(int)
    y_pred = np.asarray(y_pred).round().astype(int)

    rewards = np.where(
        y_true == 1,
        np.where(y_pred == 1, tp_value, fn_value),
        np.where(y_pred == 1, fp_value, tn_value),
    ).astype(np.float64)

    if tolerance and tolerance > 0:  # literal translation of the original loop
        for i in range(len(rewards)):
            if rewards[i] < 0:
                lo = max(0, i - tolerance)
                hi = min(i + tolerance + 1, len(rewards))
                if np.count_nonzero(rewards[lo:hi] == tp_value) > 0:
                    rewards[i] = -rewards[i]

    tp = int(np.count_nonzero(rewards == tp_value))
    fp = int(np.count_nonzero(rewards == fp_value))
    fn = int(np.count_nonzero(rewards == fn_value))
    precision = (tp + 1) / float(tp + fp + 1)
    recall = (tp + 1) / float(tp + fn + 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "aupr": 0.0,  # not defined by the RLAD protocol
    }


def aggregate_metrics(
    per_episode: Sequence[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Mean and std of every metric across validation episodes/slices.

    This reproduces Algorithm 1 line 43 ("Aggregate mean F1, mean AUPR") and
    the ``x +- y`` notation of Table II.
    """
    if not per_episode:
        raise ValueError("no per-episode metrics to aggregate")
    keys: List[str] = []
    for entry in per_episode:
        for k in entry:
            if k not in keys:
                keys.append(k)
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [float(entry.get(key, np.nan)) for entry in per_episode], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        out[key] = {
            "mean": float(np.mean(finite)) if finite.size else 0.0,
            "std": float(np.std(finite)) if finite.size else 0.0,
            "n": int(finite.size),
        }
    return out


def format_metrics_table(
    per_episode: Sequence[Dict[str, float]],
    names: Optional[Sequence[str]] = None,
) -> str:
    """Human-readable summary table printed by ``scripts/evaluate.py``."""
    agg = aggregate_metrics(per_episode)
    names = names or [f"episode_{i}" for i in range(len(per_episode))]
    header = f"{'episode':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'aupr':>10}"
    lines = [header, "-" * len(header)]
    for name, entry in zip(names, per_episode):
        lines.append(
            f"{name:<16}"
            f"{entry.get('precision', float('nan')):>10.4f}"
            f"{entry.get('recall', float('nan')):>10.4f}"
            f"{entry.get('f1', float('nan')):>10.4f}"
            f"{entry.get('aupr', float('nan')):>10.4f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'mean +- std':<16}"
        f"{agg['precision']['mean']:>5.4f}+-{agg['precision']['std']:.4f}"
        f"{agg['recall']['mean']:>5.4f}+-{agg['recall']['std']:.4f}"
        f"{agg['f1']['mean']:>5.4f}+-{agg['f1']['std']:.4f}"
        f"{agg['aupr']['mean']:>5.4f}+-{agg['aupr']['std']:.4f}"
    )
    return "\n".join(lines)


def precision_recall_curve_points(
    y_true: Sequence[float], y_score: Sequence[float]
) -> Dict[str, np.ndarray]:
    """Raw precision/recall curve points (for optional PR-curve plots)."""
    precision, recall, _ = precision_recall_curve(
        np.asarray(y_true).round().astype(int),
        np.asarray(y_score, dtype=np.float64),
    )
    return {"precision": precision, "recall": recall}
