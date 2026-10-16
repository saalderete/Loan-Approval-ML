"""Probability calibration fitted on a dedicated split."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibratedPipeline:
    """A fitted pipeline plus an isotonic map from its raw scores to calibrated probabilities."""

    def __init__(self, pipeline, method: str = "isotonic"):
        if method != "isotonic":
            raise ValueError("Only isotonic calibration is supported")
        self.pipeline = pipeline
        self.method = method
        self.calibrator: IsotonicRegression | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        raw = self.pipeline.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw, np.asarray(y))
        return self

    def predict_raw(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted")
        p = np.clip(self.calibrator.predict(self.predict_raw(X)), 0.0, 1.0)
        return np.column_stack([1 - p, p])
