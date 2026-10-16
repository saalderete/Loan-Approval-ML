import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import cached_model_info, cached_report, get_backend
from components.page import require_model

from loan_approval.data import hmda

status = require_model()
version = status["active_version"]
backend = get_backend()
info = cached_model_info(version)
fairness = cached_report(version, "fairness")

ui.page_title(
    "Model · fairness and governance",
    "Fairness & Responsible AI",
    "HMDA exists so the public can check mortgage lending for discrimination. This page measures disparities "
    "in the historical decisions and in the model's estimates. Protected attributes are never model inputs.",
)
ui.notice(
    "Fairness metrics are not legal compliance, and no single number makes a model fair. Gaps here are "
    "signals for review; HMDA lacks credit scores, assets and employment, which can explain part of them.",
    strong="How to read this.",
)
st.write("")

tab_audit, tab_hist, tab_card, tab_data, tab_limits = st.tabs(
    ["Model audit", "Historical disparities", "Model card", "Dataset card", "Limitations & prohibited uses"]
)

attr = None
with tab_audit:
    attr = (
        st.segmented_control(
            "Attribute",
            list(hmda.PROTECTED_AUDIT_COLUMNS),
            format_func=hmda.PROTECTED_AUDIT_COLUMNS.get,
            default="derived_race",
            key="fair_attr",
        )
        or "derived_race"
    )
    table = next(t for t in fairness["model_audit"] if t["attribute"] == attr)
    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True, key="card_fchart"):
        st.markdown(f"##### Observed denial rate vs model estimate · {table['label']}")
        chart = charts.fairness_chart(table["groups"])
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.caption(
            "Dot = observed denial rate in the test split, line = 95% Wilson interval, teal tick = the "
            "model's mean estimate for the group. Dot right of the tick = more denials than the 13 "
            "financial inputs account for."
        )
    with right, st.container(border=True, key="card_fread"):
        ok = [g for g in table["groups"] if g["status"] == "ok" and not g["not_reported"]]
        if ok:
            hi = max(ok, key=lambda g: g["unexplained_gap"])
            lo = min(ok, key=lambda g: g["unexplained_gap"])
            st.markdown("##### Largest gaps")
            ui.kv(
                [
                    ("Reference group", table["reference_group"]),
                    ("Largest positive gap", f"{hi['group']}: {hi['unexplained_gap'] * 100:+.1f} pts"),
                    ("Largest negative gap", f"{lo['group']}: {lo['unexplained_gap'] * 100:+.1f} pts"),
                    (
                        "Groups with too little data",
                        ", ".join(g["group"] for g in table["groups"] if g["status"] != "ok") or "none",
                    ),
                ]
            )
        st.write("")
        st.markdown(
            "**Unexplained gap** = observed denial rate − model's mean estimate. **FPR** = approved applications "
            "the model scored above the research threshold. **Recall** = denied applications scored above it."
        )
    rows = []
    for g in table["groups"]:
        if g["status"] != "ok":
            continue
        rows.append(
            {
                "Group": g["group"],
                "n": g["n"],
                "Observed denial rate": g["observed_denial_rate"] * 100,
                "95% CI": f"{g['observed_ci_low'] * 100:.1f}–{g['observed_ci_high'] * 100:.1f}%",
                "Model mean estimate": g["mean_estimated_probability"] * 100,
                "Unexplained gap (pts)": g["unexplained_gap"] * 100,
                "FPR": (g.get("false_positive_rate") or 0) * 100,
                "Recall": (g.get("recall") or 0) * 100,
                "Precision": (g.get("precision") or 0) * 100,
                "FPR diff vs ref (pts)": (g.get("false_positive_rate_diff_vs_reference") or 0) * 100,
            }
        )
    p = st.column_config.NumberColumn(format="%.1f%%")
    pts = st.column_config.NumberColumn(format="%+.1f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "n": st.column_config.NumberColumn("Test applications", format="localized"),
            "Observed denial rate": p,
            "Model mean estimate": p,
            "FPR": p,
            "Recall": p,
            "Precision": p,
            "Unexplained gap (pts)": pts,
            "FPR diff vs ref (pts)": pts,
        },
    )
    small = [f"{g['group']} (n={g['n']})" for g in table["groups"] if g["status"] != "ok"]
    if small:
        st.caption(
            f"Not reported (fewer than {table['min_group_size']} test applications): " + ", ".join(small)
        )
    for note in fairness["notes"]:
        st.caption("• " + note)

with tab_hist:
    hist = next(t for t in fairness["historical_disparities"] if t["attribute"] == attr)
    rows = [
        {
            "Group": g["group"],
            "Applications": g["n"],
            "Denial rate": g["observed_denial_rate"],
            "95% CI": f"{g['observed_ci_low'] * 100:.1f}–{g['observed_ci_high'] * 100:.1f}%",
            "Ratio vs reference": g.get("observed_rate_ratio_vs_reference"),
        }
        for g in hist["groups"]
        if g["status"] == "ok"
    ]
    with st.container(border=True, key="card_hist_fair"):
        st.markdown(
            f"##### All modelled applications · {hist['label']} (reference: {hist['reference_group']})"
        )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Applications": st.column_config.NumberColumn(format="localized"),
                "Denial rate": st.column_config.ProgressColumn(min_value=0, max_value=0.5, format="percent"),
                "Ratio vs reference": st.column_config.NumberColumn(format="%.2f×"),
            },
        )
        st.caption(
            "Raw denial rates in the historical decisions, before any modelling. Use the attribute selector "
            "on the Model audit tab to switch groups."
        )

with tab_card:
    card = backend.model_card()
    st.download_button(
        "Download model card (.md)", card, file_name=f"model_card_{version}.md", icon=":material/download:"
    )
    with st.container(border=True, key="card_mc"):
        st.markdown(card)

with tab_data:
    card = backend.dataset_card()
    if card:
        st.download_button(
            "Download dataset card (.md)", card, file_name="dataset_card.md", icon=":material/download:"
        )
        with st.container(border=True, key="card_dc"):
            st.markdown(card)
    else:
        st.info("Dataset card not found (reports/dataset_card.md).")

with tab_limits:
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True, key="card_lim"):
        st.markdown("##### Known limitations")
        st.markdown(
            "- **No underwriting data.** HMDA has no credit scores, credit history, assets, reserves or employment.\n"
            "- **Historical decisions, not risk.** The label is denial vs approval, not default or repayment.\n"
            "- **Bias can be learned.** Patterns in past decisions, including unfair ones, can be reproduced.\n"
            "- **Narrow scope.** Texas, 2025, complete applications, site-built 1-4 unit homes.\n"
            "- **Rounded inputs.** Amounts are rounded to $10k bands and DTI is banded by HMDA.\n"
            "- **Selection effect.** Applications with missing fields (denied far more often) are excluded.\n"
            "- **Proxies.** Geography and lender are excluded, but remaining inputs can still correlate with protected groups."
        )
    with b, st.container(border=True, key="card_prohib"):
        st.markdown("##### Prohibited uses")
        st.markdown(
            "- Approving, denying, pricing or pre-screening real applications.\n"
            "- Marketing, targeting or steering consumers.\n"
            "- Use with real personal or credit-report data.\n"
            "- Presenting estimates as a person's chance of approval.\n"
            "- Treating attributions as adverse-action reasons.\n\n"
            "A real lending model would need fair-lending testing, model-risk management (e.g. SR 11-7), "
            "ECOA/Reg B adverse-action processes, privacy review and legal sign-off."
        )
