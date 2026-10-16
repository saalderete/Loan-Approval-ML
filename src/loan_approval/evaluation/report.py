"""Generate the model card from the measured results of one model version."""

from __future__ import annotations

from pathlib import Path


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _f(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def model_card_markdown(manifest: dict, reports: dict) -> str:
    m = reports["metrics"]
    sel = manifest["selected_algorithm"]
    ds = manifest["dataset"]
    lines = [
        f"# Model card: {manifest['version']}",
        "",
        "> Generated automatically from this version's saved evaluation files. "
        "Educational research only; not for real credit decisions.",
        "",
        "## Summary",
        "",
        f"- **Selected model:** {manifest['algorithms'][sel]} (`{sel}`), chosen on the selection split "
        f"by {m['primary_metric'].upper().replace('_', '-')} before the test split was scored",
        f"- **Outcome:** {manifest['outcome_definition']}",
        f"- **Positive class:** denied. Prevalence in the modelled data: {_pct(ds['prevalence'])}",
        f"- **Dataset:** {ds['name']} ({ds['rows']:,} applications{', synthetic' if ds['synthetic'] else ''})",
        f"- **Split hash:** `{ds['split_hash'][:16]}…` (seed {ds['split_seed']})",
        f"- **Calibration:** {manifest['calibration']['method']}, fitted on the {manifest['calibration']['fitted_on']}",
        f"- **Created:** {manifest['created_at']} · commit `{manifest['provenance']['git_commit']}`",
        "",
        "## Test-split results (all candidates, scored once)",
        "",
        "| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | ECE | Recall @ thr | Precision @ thr | Threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for algo, r in m["test"].items():
        p, t, ci = r["probability"], r["at_threshold"], r["ci"]
        marker = " **(selected)**" if algo == sel else ""
        lines.append(
            f"| {r['name']}{marker} | {_f(p['roc_auc'])} ({_f(ci['roc_auc']['low'])}–{_f(ci['roc_auc']['high'])}) "
            f"| {_f(p['pr_auc'])} ({_f(ci['pr_auc']['low'])}–{_f(ci['pr_auc']['high'])}) | {_f(p['brier'], 4)} "
            f"| {_f(p['ece'], 4)} | {_pct(t['recall'])} | {_pct(t['precision'])} | {_f(t['threshold'])} |"
        )
    d = m["dummy_baseline"]
    lines += [
        f"| Prevalence-only baseline | {_f(d['roc_auc'])} | {_f(d['pr_auc'])} | {_f(d['brier'], 4)} | – | – | – | – |",
        "",
        f"Test rows: {m['test'][sel]['probability']['n']:,}. Confidence intervals: "
        f"{m['test'][sel]['ci']['method']} ({m['test'][sel]['ci']['n_boot']} resamples). "
        "Thresholds maximise F1 on the selection split and exist only to illustrate trade-offs.",
        "",
        "## Inputs",
        "",
        "The model uses 13 fields known when an application is submitted:",
        "",
    ]
    for f in manifest["input_schema"]["fields"]:
        lines.append(f"- **{f['label']}** (`{f['name']}`): {f['description']}")
    ex = manifest["excluded_columns"]
    lines += [
        "",
        "Excluded on purpose:",
        "",
        f"- *Post-decision fields* (would leak the outcome): {', '.join(f'`{c}`' for c in ex['post_decision'])}",
        f'- *Process artifacts* ("not applicable" codes reveal how far an application got): '
        f"{', '.join(f'`{c}`' for c in ex['process_artifacts'])}",
        f"- *Protected characteristics* (used only in the fairness audit): "
        f"{', '.join(f'`{c}`' for c in ex['protected_audit_only'])}",
        f"- *Geography and lender identity* (proxy risk): {', '.join(f'`{c}`' for c in ex['proxy_risk'])}",
        "",
        "## Most important inputs (permutation importance, selected model)",
        "",
        "| Input | PR-AUC drop when shuffled |",
        "|---|---|",
    ]
    for row in reports["importance"]["permutation"][sel]["features"][:8]:
        lines.append(f"| {row['label']} | {row['importance']:.4f} ± {row['std']:.4f} |")
    lines += ["", "## Fairness audit (selected model, test split)", ""]
    for table in reports["fairness"]["model_audit"]:
        lines += [
            f"**{table['label']}** (reference group: {table['reference_group']})",
            "",
            "| Group | n | Observed denial rate | Mean estimate | Unexplained gap | FPR | Recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for g in table["groups"]:
            if g["status"] != "ok":
                lines.append(f"| {g['group']} | {g['n']:,} | insufficient data | | | | |")
                continue
            lines.append(
                f"| {g['group']} | {g['n']:,} | {_pct(g['observed_denial_rate'])} | "
                f"{_pct(g.get('mean_estimated_probability'))} | {_pct(g.get('unexplained_gap'))} | "
                f"{_pct(g.get('false_positive_rate'))} | {_pct(g.get('recall'))} |"
            )
        lines.append("")
    lines += [f"- {n}" for n in reports["fairness"]["notes"]]
    lines += [
        "",
        "## Promotion gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for g in manifest["promotion_gates"]:
        lines.append(f"| {g['name']} | {'pass' if g['passed'] else 'FAIL'} | {g['detail']} |")
    lines += [
        "",
        "## Limitations and prohibited uses",
        "",
        "- HMDA has no credit scores, credit history, assets, employment or cash reserves, which are "
        "central to real underwriting. The model reproduces patterns in *historical lender decisions* "
        "from limited fields, including any bias in those decisions.",
        "- The outcome is denial versus approval, not default or repayment risk.",
        "- Only complete applications for site-built 1-4 unit homes in Texas (2025) are modelled. "
        "Applications with missing fields had very different denial rates and are out of scope.",
        "- Loan amounts and property values are rounded by HMDA to $10,000 bands; DTI is banded.",
        "- Explanations describe how the model responds to inputs. They are not causes, not "
        "adverse-action reasons, and not advice on how to get approved.",
        "- **Prohibited:** approving, denying or pricing real applications; marketing or targeting "
        "consumers; any use with real personal data.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt && pip install -e .",
        "loan-approval fetch      # downloads the HMDA file and checks its SHA-256",
        "loan-approval prepare",
        "loan-approval train",
        "```",
        "",
        f"Raw file SHA-256: `{ds['raw_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def write_model_card(version_dir: Path, manifest: dict, reports: dict) -> Path:
    path = version_dir / "model_card.md"
    path.write_text(model_card_markdown(manifest, reports))
    return path
