"""Command-line entry point: ``loan-approval <command>`` or ``python -m loan_approval <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from loan_approval.config import get_settings


def _cmd_fetch(args) -> int:
    from loan_approval.data import ingest

    ingest.fetch(args.state, args.year, force=args.force)
    return 0


def _cmd_prepare(args) -> int:
    from loan_approval.data import ingest

    out, report = ingest.prepare(args.state, args.year)
    for stage in report.funnel:
        print(f"  {stage['stage']:<20} {stage['rows']:>9,}  denial rate {stage['denial_rate']}")
    for warning in report.warnings:
        print(f"  WARNING: {warning}")
    print(f"Wrote {out}")
    return 0


def _cmd_train(args) -> int:
    import pandas as pd

    from loan_approval.config import OUTCOME_DEFINITION
    from loan_approval.data import ingest
    from loan_approval.data.synthetic import generate_raw
    from loan_approval.data.validate import build_model_frame
    from loan_approval.registry.artifacts import sha256_bytes
    from loan_approval.training.train import DatasetInfo, load_config, train

    cfg = load_config()
    if args.synthetic:
        raw = generate_raw(args.rows, seed=cfg["seed"])
        frame, report = build_model_frame(raw)
        info = DatasetInfo(
            name=f"Synthetic HMDA-format applications ({args.rows:,} generated)",
            synthetic=True,
            source="loan_approval.data.synthetic.generate_raw",
            raw_sha256=sha256_bytes(raw.to_csv(index=False).encode()),
            quality=report.to_dict(),
            outcome_definition="Synthetic outcome from a documented formula (not real lending data).",
        )
        if args.fast:
            cfg["tuning"]["max_rows"] = min(cfg["tuning"]["max_rows"], 3000)
            cfg["bootstrap_resamples"] = 30
            cfg["importance_rows"] = 1500
            cfg["candidates"]["random_forest"]["fixed"]["n_estimators"] = 30
    else:
        settings = get_settings()
        path = settings.processed_dir / ingest.processed_filename(args.state, args.year)
        if not path.exists():
            print(f"{path} not found. Run `loan-approval fetch` and `loan-approval prepare` first.")
            return 1
        frame = pd.read_parquet(path)
        meta = json.loads(path.with_suffix(".quality.json").read_text())
        info = DatasetInfo(
            name=f"HMDA {args.year} {args.state}: approved or denied, complete, site-built 1-4 unit",
            synthetic=False,
            source=f"FFIEC HMDA Data Browser ({meta['raw_file']})",
            raw_sha256=meta["raw_sha256"],
            quality=meta["quality"],
            outcome_definition=OUTCOME_DEFINITION,
        )
    algorithms = args.algorithms.split(",") if args.algorithms else None
    version = train(frame, info, cfg, algorithms=algorithms)
    print(version)
    return 0


def _cmd_promote(args) -> int:
    from loan_approval.registry import artifacts

    result = artifacts.promote(args.version, actor=args.actor, reason=args.reason)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_list(args) -> int:
    from loan_approval.registry import artifacts

    registry = artifacts.read_registry()
    active = registry.get("active_version")
    for version, entry in sorted(registry["versions"].items()):
        flag = "*" if version == active else " "
        m = entry["headline_metrics"]
        print(
            f"{flag} {version:<36} {entry['status']:<16} {entry['selected_algorithm']:<24} "
            f"PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}"
        )
    return 0


def _cmd_predict(args) -> int:
    from loan_approval.features.schema import ScenarioInput, load_demo_scenarios
    from loan_approval.inference import predict

    if args.scenario:
        presets = {s["id"]: s for s in load_demo_scenarios()}
        if args.scenario not in presets:
            print(f"Unknown scenario. Choose from: {', '.join(presets)}")
            return 1
        inputs = presets[args.scenario]["inputs"]
    else:
        inputs = json.loads(args.json)
    try:
        result = predict.predict(ScenarioInput(**inputs))
    except predict.ModelUnavailable as exc:
        print(f"No model available: {exc}")
        return 1
    result.pop("all_contributions")
    print(json.dumps(result, indent=2))
    return 0


def _cmd_synthetic(args) -> int:
    from loan_approval.data.synthetic import generate_raw

    settings = get_settings()
    out = settings.data_dir / "synthetic" / f"synthetic_hmda_{args.rows}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_raw(args.rows, seed=args.seed).to_csv(out, index=False)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loan-approval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_scope(p):
        p.add_argument("--state", default="TX")
        p.add_argument("--year", type=int, default=2025)

    p = sub.add_parser("fetch", help="Download the HMDA file and record its checksum")
    add_scope(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("prepare", help="Validate and filter the raw file into the model table")
    add_scope(p)
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("train", help="Train, calibrate, evaluate and register a model version")
    add_scope(p)
    p.add_argument("--synthetic", action="store_true", help="Train on generated data (CI / offline)")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--fast", action="store_true", help="Smaller budgets for synthetic smoke runs")
    p.add_argument("--algorithms", help="Comma-separated subset of candidates")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("promote", help="Promote a version that passed every gate to the demo")
    p.add_argument("version")
    p.add_argument("--actor", required=True, help="Who is approving this promotion")
    p.add_argument("--reason", default="")
    p.set_defaults(func=_cmd_promote)

    p = sub.add_parser("list", help="List registered model versions")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("predict", help="Score a demo scenario or a JSON scenario")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario")
    group.add_argument("--json")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("synthetic", help="Write a synthetic HMDA-format CSV")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
