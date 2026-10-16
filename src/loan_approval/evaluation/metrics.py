"""Metric calculations. Every number shown in the app is produced here from saved predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "flagged_rate": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p_clip, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "prevalence": float(y.mean()),
        "n": len(y),
        "positives": int(y.sum()),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Research-only threshold: maximise F1 on the selection split."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "brier": [], "recall": [], "precision": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb, pb = y[i], p[i]
        if yb.min() == yb.max():
            continue
        pred = pb >= threshold
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["pr_auc"].append(average_precision_score(yb, pb))
        stats["brier"].append(brier_score_loss(yb, pb))
        stats["recall"].append(pred[yb == 1].mean())
        stats["precision"].append(yb[pred].mean() if pred.any() else 0.0)
    return {
        k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
        for k, v in stats.items()
        if v
    } | {"n_boot": n_boot, "method": "percentile bootstrap, 95%"}


def _downsample(xs: np.ndarray, ys: np.ndarray, points: int = 200) -> list[list[float]]:
    if len(xs) > points:
        keep = np.unique(np.linspace(0, len(xs) - 1, points).astype(int))
        xs, ys = xs[keep], ys[keep]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(xs, ys, strict=True)]


def curves(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    order = np.argsort(recall)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    calibration = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            calibration.append(
                {
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    hist_counts, hist_edges = np.histogram(p, bins=20, range=(0, 1))
    hist_pos, _ = np.histogram(p[y == 1], bins=20, range=(0, 1))
    return {
        "roc": _downsample(fpr, tpr),
        "pr": _downsample(recall[order], precision[order]),
        "calibration": calibration,
        "histogram": [
            {
                "bin_start": round(float(hist_edges[i]), 3),
                "bin_end": round(float(hist_edges[i + 1]), 3),
                "count": int(hist_counts[i]),
                "denied": int(hist_pos[i]),
            }
            for i in range(len(hist_counts))
        ],
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))
