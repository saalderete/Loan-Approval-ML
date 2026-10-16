"""End-to-end training run: split -> tune -> fit -> calibrate -> select -> evaluate -> audit -> save.

Protocol (fixed in configs/training.yaml before training):
  * all candidates share one split, created before any fitting;
  * hyper-parameters are tuned with stratified CV on the training split only;
  * calibration is fitted on the calibration split;
  * the model and the research threshold are chosen on the selection split by the primary metric;
  * the test split is scored once, after selection, for every candidate.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from loan_approval.config import CONFIGS_DIR, ROOT, get_settings
from loan_approval.data import hmda
from loan_approval.data.split import SplitSpec, assign_splits, split_hash, split_summary
from loan_approval.data.validate import model_inputs
from loan_approval.evaluation import drift, fairness, metrics, profile
from loan_approval.evaluation.report import write_model_card
from loan_approval.features.preprocess import ALGORITHMS, build_pipeline
from loan_approval.features.schema import SCHEMA_VERSION, ScenarioInput, load_demo_scenarios, schema_summary
from loan_approval.inference import explain
from loan_approval.registry import artifacts
from loan_approval.training.calibration import CalibratedPipeline

PACKAGES = ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "pydantic"]


@dataclass
class DatasetInfo:
    name: str
    synthetic: bool
    source: str
    raw_sha256: str
    quality: dict
    outcome_definition: str


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIGS_DIR / "training.yaml").read_text())


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def univariate_leakage_screen(train: pd.DataFrame, holdout: pd.DataFrame) -> list[dict]:
    """ROC-AUC of each input on its own. A single field near 1.0 usually means leakage."""
    rows = []
    y_hold = holdout["denied"].to_numpy()
    for col in hmda.INPUT_FEATURES:
        if col in hmda.NUMERIC_INPUTS:
            score = holdout[col].astype(float).to_numpy()
        else:
            rates = train.groupby(train[col].astype(str))["denied"].mean()
            score = holdout[col].astype(str).map(rates).fillna(train["denied"].mean()).to_numpy()
        auc = roc_auc_score(y_hold, score)
        rows.append({"feature": col, "auc": float(max(auc, 1 - auc))})
    return sorted(rows, key=lambda r: r["auc"], reverse=True)


def _tune(algorithm: str, spec: dict, x: pd.DataFrame, y: np.ndarray, cfg: dict) -> tuple[dict, dict]:
    seed = cfg["seed"]
    max_rows = cfg["tuning"]["max_rows"]
    if len(x) > max_rows:
        idx, _ = train_test_split(np.arange(len(x)), train_size=max_rows, stratify=y, random_state=seed + 3)
        x, y = x.iloc[idx], y[idx]
    fixed = spec.get("fixed", {})
    grid = {f"model__{k}": v for k, v in spec.get("grid", {}).items()}
    search = GridSearchCV(
        build_pipeline(algorithm, seed=seed, **fixed),
        grid,
        scoring="average_precision",
        cv=StratifiedKFold(cfg["tuning"]["cv_folds"], shuffle=True, random_state=seed),
        n_jobs=1 if algorithm == "random_forest" else -1,
        refit=False,
    )
    search.fit(x, y)
    best_i = int(search.best_index_)
    folds = [
        float(search.cv_results_[f"split{k}_test_score"][best_i]) for k in range(cfg["tuning"]["cv_folds"])
    ]
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    cv = {
        "rows": len(x),
        "folds": cfg["tuning"]["cv_folds"],
        "metric": "pr_auc",
        "best_params": best,
        "fold_scores": folds,
        "mean": float(np.mean(folds)),
        "std": float(np.std(folds)),
        "grid": [
            {
                "params": {k.removeprefix("model__"): v for k, v in p.items()},
                "mean": float(m),
                "std": float(s),
            }
            for p, m, s in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                strict=True,
            )
        ],
    }
    return {**fixed, **best}, cv


def train(
    frame: pd.DataFrame,
    dataset: DatasetInfo,
    cfg: dict | None = None,
    out_base: Path | None = None,
    algorithms: list[str] | None = None,
    log=print,
) -> str:
    cfg = cfg or load_config()
    started = time.time()
    seed = cfg["seed"]
    base = out_base or get_settings().models_dir
    algorithms = algorithms or list(cfg["candidates"])

    splits = assign_splits(
        frame,
        SplitSpec(
            seed=seed, test_size=cfg["split"]["test_size"], validation_size=cfg["split"]["validation_size"]
        ),
    )
    shash = split_hash(frame["_row_id"], splits)
    parts = {name: frame[splits == name].reset_index(drop=True) for name in splits.unique()}
    x = {k: model_inputs(v) for k, v in parts.items()}
    y = {k: v["denied"].to_numpy() for k, v in parts.items()}
    log(f"Splits: { {k: len(v) for k, v in parts.items()} }  hash={shash[:12]}")

    leakage = univariate_leakage_screen(parts["train"], parts["selection"])
    log(f"Strongest single input: {leakage[0]['feature']} (AUC {leakage[0]['auc']:.3f})")

    candidates = {}
    for algorithm in algorithms:
        t0 = time.time()
        params, cv = _tune(algorithm, cfg["candidates"][algorithm], x["train"], y["train"], cfg)
        log(f"[{algorithm}] CV PR-AUC {cv['mean']:.4f} ± {cv['std']:.4f} best={cv['best_params']}")
        pipe = build_pipeline(algorithm, seed=seed, **params).fit(x["train"], y["train"])
        fit_seconds = time.time() - t0
        calibrated = CalibratedPipeline(pipe, cfg["calibration"]).fit(x["calibration"], y["calibration"])
        raw_sel = calibrated.predict_raw(x["selection"])
        p_sel = calibrated.predict_proba(x["selection"])[:, 1]
        threshold = metrics.choose_threshold(y["selection"], p_sel)
        candidates[algorithm] = {
            "name": ALGORITHMS[algorithm],
            "params": params,
            "cv": cv,
            "fit_seconds": round(fit_seconds, 1),
            "pipeline": calibrated,
            "uncalibrated": pipe,
            "threshold": threshold,
            "selection": {
                "pr_auc_uncalibrated": float(average_precision_score(y["selection"], raw_sel)),
                **metrics.probability_metrics(y["selection"], p_sel),
            },
        }
        log(
            f"[{algorithm}] selection PR-AUC {candidates[algorithm]['selection']['pr_auc_uncalibrated']:.4f} "
            f"threshold={threshold:.3f} ({fit_seconds:.0f}s)"
        )

    primary = cfg["primary_metric"]
    selected = max(candidates, key=lambda a: candidates[a]["selection"][f"{primary}_uncalibrated"])
    log(f"Selected {selected} by selection-split {primary}")

    # --- single scoring of the untouched test split -------------------------------------------
    test_results, test_probs = {}, {}
    for algorithm, c in candidates.items():
        p = c["pipeline"].predict_proba(x["test"])[:, 1]
        raw = c["pipeline"].predict_raw(x["test"])
        test_probs[algorithm] = p
        test_results[algorithm] = {
            "name": c["name"],
            "probability": metrics.probability_metrics(y["test"], p),
            "uncalibrated": {
                "brier": float(np.mean((raw - y["test"]) ** 2)),
                "ece": metrics.expected_calibration_error(y["test"], raw),
            },
            "at_threshold": metrics.threshold_metrics(y["test"], p, c["threshold"]),
            "ci": metrics.bootstrap_ci(
                y["test"], p, c["threshold"], n_boot=cfg["bootstrap_resamples"], seed=seed
            ),
            "curves": metrics.curves(y["test"], p),
        }
    prevalence = float(y["train"].mean())
    dummy = metrics.probability_metrics(y["test"], np.full(len(y["test"]), prevalence))

    # --- explanations ----------------------------------------------------------------------
    n_imp = min(cfg["importance_rows"], len(x["test"]))
    imp_idx = np.random.default_rng(seed).choice(len(x["test"]), n_imp, replace=False)
    importance = {
        a: explain.global_importance(c["pipeline"], x["test"].iloc[imp_idx], y["test"][imp_idx], seed=seed)
        for a, c in candidates.items()
    }
    coefficients = (
        explain.logistic_coefficients(candidates["logistic_regression"]["uncalibrated"])
        if "logistic_regression" in candidates
        else []
    )
    reference = explain.reference_values(x["train"])

    # --- fairness audit, drift baseline, dataset profile -------------------------------------
    fairness_report = fairness.audit(
        frame, parts["test"], test_probs[selected], candidates[selected]["threshold"]
    )
    baseline = drift.build_baseline(x["train"])
    data_profile = profile.build_profile(frame)
    data_profile["splits"] = split_summary(frame, splits)
    data_profile["quality"] = dataset.quality
    quantiles = {a: np.quantile(p, np.linspace(0, 1, 101)).round(6).tolist() for a, p in test_probs.items()}

    # --- write artifacts -----------------------------------------------------------------------
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    prefix = "synthetic" if dataset.synthetic else "hmda-tx2025"
    version = f"{prefix}-{stamp}"
    vdir = base / version
    (vdir / "pipelines").mkdir(parents=True, exist_ok=True)
    artifact_entries = {}
    for algorithm, c in candidates.items():
        path = vdir / "pipelines" / f"{algorithm}.joblib"
        joblib.dump(c["pipeline"], path, compress=3)
        size_mb = path.stat().st_size / 1e6
        if size_mb > cfg["max_artifact_mb"] and algorithm != selected:
            path.unlink()
            log(f"[{algorithm}] artifact {size_mb:.0f} MB exceeds budget; metrics kept, file not saved")
            test_results[algorithm]["artifact_saved"] = False
            continue
        test_results[algorithm]["artifact_saved"] = True
        artifact_entries[algorithm] = {
            "kind": "pipeline",
            "path": f"pipelines/{algorithm}.joblib",
            "sha256": artifacts.sha256_path(path),
            "size_mb": round(size_mb, 2),
        }

    reports = {
        "metrics": {
            "primary_metric": primary,
            "selected_algorithm": selected,
            "selection_split": {a: c["selection"] for a, c in candidates.items()},
            "cv": {a: c["cv"] for a, c in candidates.items()},
            "test": {a: {k: v for k, v in r.items() if k != "curves"} for a, r in test_results.items()},
            "dummy_baseline": dummy,
            "thresholds": {a: c["threshold"] for a, c in candidates.items()},
            "fit_seconds": {a: c["fit_seconds"] for a, c in candidates.items()},
            "params": {a: c["params"] for a, c in candidates.items()},
            "splits": split_summary(frame, splits),
        },
        "curves": {a: r["curves"] for a, r in test_results.items()},
        "importance": {"permutation": importance, "logistic_coefficients": coefficients[:40]},
        "leakage_screen": {"univariate_auc": leakage},
        "fairness": fairness_report,
        "drift_baseline": baseline,
        "profile": data_profile,
        "test_quantiles": quantiles,
    }
    for name, payload in reports.items():
        _write_json(vdir / "reports" / f"{name}.json", payload)

    sel = test_results[selected]
    gates = _promotion_gates(
        selected,
        sel,
        dummy,
        leakage,
        candidates[selected]["cv"],
        fairness_report,
        artifact_entries,
        reference,
        candidates[selected]["pipeline"],
        cfg,
    )
    manifest = {
        "version": version,
        "created_at": artifacts.now_iso(),
        "status_history": ["draft_experiment", "training", "evaluated"],
        "selected_algorithm": selected,
        "algorithms": {a: ALGORITHMS[a] for a in candidates},
        "outcome_definition": dataset.outcome_definition,
        "positive_class": "denied",
        "schema_version": SCHEMA_VERSION,
        "input_schema": schema_summary(),
        "input_features": hmda.INPUT_FEATURES,
        "excluded_columns": {
            "post_decision": hmda.POST_DECISION_COLUMNS,
            "process_artifacts": hmda.PROCESS_ARTIFACT_COLUMNS,
            "protected_audit_only": list(hmda.PROTECTED_AUDIT_COLUMNS),
            "proxy_risk": hmda.PROXY_RISK_COLUMNS,
        },
        "dataset": {
            "name": dataset.name,
            "synthetic": dataset.synthetic,
            "source": dataset.source,
            "raw_sha256": dataset.raw_sha256,
            "rows": len(frame),
            "prevalence": float(frame["denied"].mean()),
            "split_hash": shash,
            "split_seed": seed,
        },
        "training_config": cfg,
        "calibration": {"method": cfg["calibration"], "fitted_on": "calibration split"},
        "thresholds": {
            a: {"value": c["threshold"], "chosen_on": "selection split (max F1)", "purpose": "research only"}
            for a, c in candidates.items()
        },
        "reference_values": reference,
        "explanation": {
            "global": "permutation importance (PR-AUC drop) on held-out test rows",
            "local": explain.LOCAL_METHOD,
        },
        "headline_metrics": {
            "roc_auc": sel["probability"]["roc_auc"],
            "pr_auc": sel["probability"]["pr_auc"],
            "brier": sel["probability"]["brier"],
            "ece": sel["probability"]["ece"],
            "recall_at_threshold": sel["at_threshold"]["recall"],
            "precision_at_threshold": sel["at_threshold"]["precision"],
            "test_rows": sel["probability"]["n"],
        },
        "artifacts": artifact_entries,
        "promotion_gates": gates,
        "provenance": {
            "git_commit": _git_commit(),
            "packages": _package_versions(),
            "platform": platform.platform(),
            "training_seconds": round(time.time() - started, 1),
        },
    }
    _write_json(vdir / "manifest.json", manifest)
    write_model_card(vdir, manifest, reports)
    status = "review_required" if all(g["passed"] for g in gates) else "evaluated"
    artifacts.register_version(version, json.loads((vdir / "manifest.json").read_text()), status, base)
    log(f"Saved {version} ({status}); gates: {sum(g['passed'] for g in gates)}/{len(gates)} passed")
    return version


def _promotion_gates(
    selected, sel, dummy, leakage, cv, fairness_report, artifact_entries, reference, pipeline, cfg
) -> list[dict]:
    banned = (
        set(hmda.POST_DECISION_COLUMNS)
        | set(hmda.PROCESS_ARTIFACT_COLUMNS)
        | set(hmda.PROTECTED_AUDIT_COLUMNS)
        | set(hmda.PROXY_RISK_COLUMNS)
    )
    used = set(hmda.INPUT_COLUMN_MAP) | set(hmda.INPUT_FEATURES)
    smoke_ok, smoke_detail = True, "all demo scenarios scored"
    try:
        frames = pd.concat([ScenarioInput(**s["inputs"]).to_frame() for s in load_demo_scenarios()])
        p = pipeline.predict_proba(frames)[:, 1]
        smoke_ok = bool(np.all((p >= 0) & (p <= 1)))
    except Exception as exc:
        smoke_ok, smoke_detail = False, f"error: {exc}"
    required = ["roc_auc", "pr_auc", "brier", "ece", "log_loss"]
    top_auc = leakage[0]["auc"] if leakage else 0
    size = artifact_entries.get(selected, {}).get("size_mb")
    return [
        {
            "name": "no_banned_inputs",
            "passed": not (used & banned),
            "detail": "No post-decision, process-artifact, protected or proxy columns are model inputs",
        },
        {
            "name": "univariate_leakage_screen",
            "passed": top_auc < 0.95,
            "detail": f"Strongest single input AUC = {top_auc:.3f} (must be < 0.95)",
        },
        {
            "name": "beats_baseline",
            "passed": sel["probability"]["pr_auc"] > 1.5 * dummy["pr_auc"]
            and sel["probability"]["roc_auc"] > 0.6,
            "detail": f"Test PR-AUC {sel['probability']['pr_auc']:.3f} vs prevalence baseline {dummy['pr_auc']:.3f}",
        },
        {
            "name": "calibration",
            "passed": sel["probability"]["ece"] < 0.03,
            "detail": f"Test expected calibration error = {sel['probability']['ece']:.4f} (must be < 0.03)",
        },
        {
            "name": "cv_stability",
            "passed": cv["std"] < 0.03,
            "detail": f"CV PR-AUC std = {cv['std']:.4f} (must be < 0.03)",
        },
        {
            "name": "metrics_complete",
            "passed": all(k in sel["probability"] for k in required),
            "detail": "ROC-AUC, PR-AUC, Brier, ECE and log loss recorded on the test split",
        },
        {
            "name": "fairness_report",
            "passed": bool(fairness_report.get("model_audit")),
            "detail": "Group-level audit produced for every protected attribute",
        },
        {"name": "smoke_predictions", "passed": smoke_ok, "detail": smoke_detail},
        {
            "name": "artifact_size",
            "passed": size is not None and size <= cfg["max_artifact_mb"],
            "detail": f"Selected artifact {size} MB (budget {cfg['max_artifact_mb']} MB)",
        },
        {
            "name": "reference_values",
            "passed": set(reference) == set(hmda.INPUT_FEATURES),
            "detail": "Reference values recorded for every input",
        },
    ]
