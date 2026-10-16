"""Deterministic, stratified train / validation / test split.

The split is created once, before any transformer is fitted, and its hash is stored with every
model version so a reviewer can confirm all candidates were compared on identical rows.

Validation is further split in half:
  * ``calibration`` - fits the probability calibrator
  * ``selection``   - chooses the model and the research threshold
The test split is scored once, after selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "calibration", "selection", "test")


@dataclass(frozen=True)
class SplitSpec:
    seed: int = 20251
    test_size: float = 0.20
    validation_size: float = 0.20


def assign_splits(frame: pd.DataFrame, spec: SplitSpec, target: str = "denied") -> pd.Series:
    idx = np.arange(len(frame))
    y = frame[target].to_numpy()
    rest, test = train_test_split(idx, test_size=spec.test_size, stratify=y, random_state=spec.seed)
    val_share = spec.validation_size / (1 - spec.test_size)
    train, val = train_test_split(
        rest, test_size=val_share, stratify=y[rest], random_state=spec.seed + 1
    )
    calib, select = train_test_split(
        val, test_size=0.5, stratify=y[val], random_state=spec.seed + 2
    )
    labels = np.empty(len(frame), dtype=object)
    labels[train] = "train"
    labels[calib] = "calibration"
    labels[select] = "selection"
    labels[test] = "test"
    return pd.Series(labels, index=frame.index, name="split")


def split_hash(row_ids: pd.Series, splits: pd.Series) -> str:
    digest = hashlib.sha256()
    for name in SPLIT_NAMES:
        ids = np.sort(row_ids[splits == name].to_numpy(dtype=np.int64))
        digest.update(name.encode())
        digest.update(ids.tobytes())
    return digest.hexdigest()


def split_summary(frame: pd.DataFrame, splits: pd.Series, target: str = "denied") -> list[dict]:
    rows = []
    for name in SPLIT_NAMES:
        part = frame[splits == name]
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "positives": int(part[target].sum()),
                "prevalence": round(float(part[target].mean()), 4),
            }
        )
    return rows
