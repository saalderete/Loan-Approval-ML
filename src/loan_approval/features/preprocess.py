"""Feature engineering and preprocessing pipelines.

``HmdaFeatureBuilder`` is stateless: it only applies fixed, documented formulas, so it can run on a
single scenario at inference time exactly as it ran during training. Anything that learns from data
(scaling, encoding) lives inside the scikit-learn ``Pipeline`` and is fitted on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from loan_approval.data import hmda

ENGINEERED_NUMERIC = [
    "log_loan_amount",
    "log_property_value",
    "log_income",
    "dti_value",
    "cltv",
    "ltv_this_loan",
    "loan_to_income",
    "loan_term",
    "total_units_n",
]
ENGINEERED_CATEGORICAL = [
    "dti_band",
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "preapproval",
    "conforming_loan_limit",
]

DTI_BANDS = [
    (0, 20, "<20%"),
    (20, 36, "20-35%"),
    (36, 43, "36-42%"),
    (43, 50, "43-49%"),
    (50, 61, "50-60%"),
    (61, 999, ">60%"),
]

CATEGORY_LEVELS = {
    "dti_band": [b[2] for b in DTI_BANDS],
    "loan_type": list(hmda.LOAN_TYPE),
    "loan_purpose": list(hmda.LOAN_PURPOSE),
    "lien_status": list(hmda.LIEN_STATUS),
    "occupancy_type": list(hmda.OCCUPANCY_TYPE),
    "preapproval": list(hmda.PREAPPROVAL),
    "conforming_loan_limit": list(hmda.CONFORMING_LOAN_LIMIT),
}


def _dti_band(value: float) -> str:
    for lo, hi, label in DTI_BANDS:
        if lo <= value < hi:
            return label
    return DTI_BANDS[-1][2]


class HmdaFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn the 13 raw HMDA inputs into model features with fixed formulas (no fitted state)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=hmda.INPUT_FEATURES)
        loan = X["loan_amount"].astype(float).clip(lower=1.0)
        value = X["property_value"].astype(float).clip(lower=1.0)
        income = X["income"].astype(float).clip(lower=1.0)
        dti = X["debt_to_income_ratio"].astype(str).map(hmda.DTI_VALUES).astype(float)
        out = pd.DataFrame(index=X.index)
        out["log_loan_amount"] = np.log(loan)
        out["log_property_value"] = np.log(value)
        out["log_income"] = np.log(income * 1000.0)
        out["dti_value"] = dti
        out["cltv"] = X["combined_loan_to_value_ratio"].astype(float).clip(0, 250)
        out["ltv_this_loan"] = (loan / value * 100.0).clip(0, 300)
        out["loan_to_income"] = (loan / (income * 1000.0)).clip(0, 50)
        out["loan_term"] = X["loan_term"].astype(float)
        out["total_units_n"] = pd.to_numeric(X["total_units"], errors="coerce").astype(float)
        out["dti_band"] = dti.map(_dti_band)
        for col in ENGINEERED_CATEGORICAL[1:]:
            out[col] = X[col].astype(str)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL, dtype=object)


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(
        categories=[CATEGORY_LEVELS[c] for c in ENGINEERED_CATEGORICAL],
        handle_unknown="ignore",
        sparse_output=False,
    )


def _ordinal() -> OrdinalEncoder:
    return OrdinalEncoder(
        categories=[CATEGORY_LEVELS[c] for c in ENGINEERED_CATEGORICAL],
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )


ALGORITHMS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
}


def build_pipeline(algorithm: str, seed: int = 0, **params) -> Pipeline:
    """Return an unfitted pipeline: raw inputs -> features -> encoding -> estimator."""
    if algorithm == "logistic_regression":
        encode = ColumnTransformer(
            [
                ("num", StandardScaler(), ENGINEERED_NUMERIC),
                ("cat", _onehot(), ENGINEERED_CATEGORICAL),
            ],
            verbose_feature_names_out=False,
        )
        model = LogisticRegression(C=params.get("C", 1.0), max_iter=3000, solver="lbfgs", random_state=seed)
    elif algorithm == "random_forest":
        encode = ColumnTransformer(
            [
                ("num", "passthrough", ENGINEERED_NUMERIC),
                ("cat", _onehot(), ENGINEERED_CATEGORICAL),
            ],
            verbose_feature_names_out=False,
        )
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=params.get("min_samples_leaf", 100),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
    elif algorithm == "hist_gradient_boosting":
        n_num = len(ENGINEERED_NUMERIC)
        encode = ColumnTransformer(
            [
                ("num", "passthrough", ENGINEERED_NUMERIC),
                ("cat", _ordinal(), ENGINEERED_CATEGORICAL),
            ],
            verbose_feature_names_out=False,
        )
        categorical_mask = [False] * n_num + [True] * len(ENGINEERED_CATEGORICAL)
        model = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.08),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            l2_regularization=params.get("l2_regularization", 1.0),
            max_iter=params.get("max_iter", 400),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            categorical_features=categorical_mask,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}")
    return Pipeline([("features", HmdaFeatureBuilder()), ("encode", encode), ("model", model)])
