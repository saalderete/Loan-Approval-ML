import numpy as np
import pytest

from loan_approval.evaluation import metrics


def test_confusion_matrix_invariants():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    m = metrics.threshold_metrics(y, p, 0.5)
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == 500
    assert cm["tp"] + cm["fn"] == y.sum()
    assert m["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]))


def test_perfectly_calibrated_scores_have_small_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert metrics.expected_calibration_error(y, p) < 0.01
    assert metrics.expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    pm = metrics.probability_metrics(y, p)
    assert pm["roc_auc"] == pytest.approx(0.75)
    assert pm["prevalence"] == 0.5


def test_wilson_interval_contains_estimate():
    lo, hi = metrics.wilson_interval(20, 100)
    assert lo < 0.2 < hi
    assert lo == pytest.approx(0.1333, abs=1e-3)
    assert hi == pytest.approx(0.2888, abs=1e-3)


def test_choose_threshold_in_range():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.uniform(size=300) * 0.4, 0, 1)
    t = metrics.choose_threshold(y, p)
    assert 0 < t < 1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(size=2000) * 0.7, 0, 1)
    ci = metrics.bootstrap_ci(y, p, 0.5, n_boot=50)
    auc = metrics.probability_metrics(y, p)["roc_auc"]
    assert ci["roc_auc"]["low"] <= auc <= ci["roc_auc"]["high"]
