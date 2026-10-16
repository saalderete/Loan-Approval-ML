"""Altair charts. Every chart is drawn from saved evaluation data and labels its source."""

from __future__ import annotations

import altair as alt
import pandas as pd

from components.ui import tokens

FONT = 'Inter, "Source Sans 3", "Source Sans Pro", system-ui, sans-serif'

ALGO_ORDER = ["hist_gradient_boosting", "random_forest", "logistic_regression"]


def _style(chart: alt.Chart, height: int = 300) -> alt.Chart:
    t = tokens()
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=t["muted"],
            titleColor=t["muted"],
            gridColor=t["border"],
            domainColor=t["border"],
            tickColor=t["border"],
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
        )
        .configure_legend(
            labelColor=t["text"],
            titleColor=t["muted"],
            labelFont=FONT,
            titleFont=FONT,
            orient="bottom",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_title(color=t["ink"], font=FONT, fontSize=13, anchor="start")
    )


def _model_colors(selected: str, algorithms: list[str]) -> list[str]:
    t = tokens()
    palette = {
        "hist_gradient_boosting": t["navy"],
        "random_forest": t["slate"],
        "logistic_regression": "#B39DDB",
    }
    colors = [t["accent"] if a == selected else palette.get(a, t["slate"]) for a in algorithms]
    return colors


def curve_chart(curves: dict, names: dict, selected: str, kind: str, prevalence: float | None = None):
    rows = []
    for algo, c in curves.items():
        for x, y in c[kind]:
            rows.append({"model": names[algo], "x": x, "y": y, "selected": algo == selected})
    df = pd.DataFrame(rows)
    algos = [a for a in ALGO_ORDER if a in curves]
    domain = [names[a] for a in algos]
    color = alt.Color(
        "model:N", scale=alt.Scale(domain=domain, range=_model_colors(selected, algos)), title=None
    )
    xt, yt = (
        ("False positive rate", "True positive rate (recall)") if kind == "roc" else ("Recall", "Precision")
    )
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("x:Q", title=xt, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", title=yt, scale=alt.Scale(domain=[0, 1])),
            color=color,
            strokeWidth=alt.condition("datum.selected", alt.value(3), alt.value(1.6)),
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("x:Q", title=xt, format=".3f"),
                alt.Tooltip("y:Q", title=yt, format=".3f"),
            ],
        )
    )
    t = tokens()
    if kind == "roc":
        ref = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[4, 4], color=t["slate"], strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )
    else:
        ref = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [prevalence or 0, prevalence or 0]}))
            .mark_line(strokeDash=[4, 4], color=t["slate"], strokeWidth=1)
            .encode(x="x:Q", y="y:Q")
        )
    return _style(alt.layer(ref, lines), 320)


def calibration_chart(curves: dict, names: dict, selected: str):
    rows = []
    for algo, c in curves.items():
        for b in c["calibration"]:
            rows.append({"model": names[algo], **b})
    df = pd.DataFrame(rows)
    algos = [a for a in ALGO_ORDER if a in curves]
    domain = [names[a] for a in algos]
    t = tokens()
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color=t["slate"], strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )
    pts = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=1.8)
        .encode(
            x=alt.X(
                "mean_predicted:Q",
                title="Mean estimated probability (decile)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("observed_rate:Q", title="Observed denial rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "model:N", scale=alt.Scale(domain=domain, range=_model_colors(selected, algos)), title=None
            ),
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("mean_predicted:Q", format=".3f", title="Mean estimate"),
                alt.Tooltip("observed_rate:Q", format=".3f", title="Observed rate"),
                alt.Tooltip("n:Q", format=",", title="Applications"),
            ],
        )
    )
    return _style(alt.layer(diag, pts), 320)


def histogram_chart(hist: list[dict], threshold: float):
    df = pd.DataFrame(hist)
    df["approved"] = df["count"] - df["denied"]
    long = df.melt(
        id_vars=["bin_start", "bin_end"],
        value_vars=["approved", "denied"],
        var_name="outcome",
        value_name="n",
    )
    long["outcome"] = long["outcome"].map({"approved": "Approved", "denied": "Denied"})
    t = tokens()
    bars = (
        alt.Chart(long)
        .mark_bar(binSpacing=1)
        .encode(
            x=alt.X("bin_start:Q", bin="binned", title="Estimated probability of denial"),
            x2="bin_end:Q",
            y=alt.Y("n:Q", title="Test applications", stack=True),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["Approved", "Denied"], range=[t["slate"], t["accent"]]),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("outcome:N"),
                alt.Tooltip("n:Q", format=","),
                alt.Tooltip("bin_start:Q", format=".2f", title="From"),
                alt.Tooltip("bin_end:Q", format=".2f", title="To"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold]}))
        .mark_rule(color=t["amber"], strokeDash=[5, 3])
        .encode(x="x:Q")
    )
    return _style(alt.layer(bars, rule), 260)


def importance_chart(
    features: list[dict], value_key: str = "importance", title: str = "PR-AUC drop when shuffled"
):
    df = pd.DataFrame(features)
    t = tokens()
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)))
    bars = base.mark_bar(color=t["accent"], cornerRadiusEnd=4, height=16).encode(
        x=alt.X(f"{value_key}:Q", title=title),
        tooltip=[
            alt.Tooltip("label:N", title="Input"),
            alt.Tooltip(f"{value_key}:Q", format=".4f"),
            alt.Tooltip("std:Q", format=".4f", title="Std across repeats"),
        ],
    )
    return _style(bars, max(220, 26 * len(df)))


def contribution_chart(contributions: list[dict]):
    df = pd.DataFrame(contributions)
    df["points"] = df["contribution"] * 100
    t = tokens()
    df["direction_label"] = df["points"].apply(lambda v: "Raises estimate" if v > 0 else "Lowers estimate")
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            y=alt.Y(
                "label:N",
                sort=alt.EncodingSortField("points", op="sum", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X("points:Q", title="Change vs typical application (percentage points)"),
            color=alt.Color(
                "direction_label:N",
                scale=alt.Scale(
                    domain=["Raises estimate", "Lowers estimate"], range=[t["amber"], t["accent"]]
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Inputs"),
                alt.Tooltip("value_display:N", title="This scenario"),
                alt.Tooltip("reference_display:N", title="Typical application"),
                alt.Tooltip("points:Q", format="+.1f", title="Points"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=t["muted"], strokeWidth=1).encode(x="x:Q")
    return _style(alt.layer(chart, zero), max(220, 34 * len(df)))


def sensitivity_chart(points: list[dict], label: str, numeric: bool):
    df = pd.DataFrame(points)
    df["pct"] = df["probability"] * 100
    t = tokens()
    if numeric:
        line = (
            alt.Chart(df)
            .mark_line(color=t["accent"], strokeWidth=2.4, interpolate="step-after")
            .encode(
                x=alt.X("value:Q", title=label),
                y=alt.Y("pct:Q", title="Estimated probability of denial (%)"),
                tooltip=[
                    alt.Tooltip("value_display:N", title=label),
                    alt.Tooltip("pct:Q", format=".1f", title="Estimate %"),
                ],
            )
        )
        cur = (
            alt.Chart(df[df["is_current"]])
            .mark_point(size=110, filled=True, color=t["amber"])
            .encode(x="value:Q", y="pct:Q", tooltip=[alt.Tooltip("value_display:N", title="Current")])
        )
        return _style(alt.layer(line, cur), 280)
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value_display:N", sort=None, title=label, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("pct:Q", title="Estimated probability of denial (%)"),
            color=alt.condition("datum.is_current", alt.value(t["amber"]), alt.value(t["accent"])),
            tooltip=[
                alt.Tooltip("value_display:N", title=label),
                alt.Tooltip("pct:Q", format=".1f", title="Estimate %"),
            ],
        )
    )
    return _style(bars, 280)


def fairness_chart(groups: list[dict]):
    rows = [g for g in groups if g["status"] == "ok"]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    t = tokens()
    df["observed"] = df["observed_denial_rate"] * 100
    df["lo"] = df["observed_ci_low"] * 100
    df["hi"] = df["observed_ci_high"] * 100
    df["estimate"] = df["mean_estimated_probability"] * 100
    base = alt.Chart(df).encode(y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
    ci = base.mark_rule(color=t["slate"], strokeWidth=2).encode(
        x=alt.X("lo:Q", title="Denial rate (%)"), x2="hi:Q"
    )
    obs = base.mark_point(filled=True, size=90, color=t["navy"]).encode(
        x="observed:Q",
        tooltip=[
            alt.Tooltip("group:N"),
            alt.Tooltip("n:Q", format=","),
            alt.Tooltip("observed:Q", format=".1f", title="Observed %"),
            alt.Tooltip("estimate:Q", format=".1f", title="Model mean %"),
        ],
    )
    est = base.mark_tick(color=t["accent"], thickness=3, size=18).encode(x="estimate:Q")
    return _style(alt.layer(ci, obs, est), max(200, 36 * len(df)))


def category_chart(rows: list[dict], label: str):
    df = pd.DataFrame(rows)
    df["rate"] = df["denial_rate"] * 100
    t = tokens()
    chart = (
        alt.Chart(df)
        .mark_bar(color=t["accent"], cornerRadiusEnd=4)
        .encode(
            x=alt.X("label:N", sort=None, title=label, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("rate:Q", title="Denial rate (%)"),
            tooltip=[
                alt.Tooltip("label:N", title=label),
                alt.Tooltip("n:Q", format=",", title="Applications"),
                alt.Tooltip("rate:Q", format=".1f", title="Denial rate %"),
                alt.Tooltip("share:Q", format=".1%", title="Share of data"),
            ],
        )
    )
    return _style(chart, 260)


def numeric_hist_chart(bins: list[dict], label: str):
    df = pd.DataFrame(bins)
    t = tokens()
    chart = (
        alt.Chart(df)
        .mark_bar(color=t["navy"], binSpacing=1)
        .encode(
            x=alt.X("start:Q", bin="binned", title=label),
            x2="end:Q",
            y=alt.Y("count:Q", title="Applications"),
            tooltip=[
                alt.Tooltip("start:Q", format=",.0f", title="From"),
                alt.Tooltip("end:Q", format=",.0f", title="To"),
                alt.Tooltip("count:Q", format=","),
            ],
        )
    )
    return _style(chart, 240)


def funnel_chart(funnel: list[dict]):
    df = pd.DataFrame(funnel)
    t = tokens()
    df["order"] = range(len(df))
    chart = (
        alt.Chart(df)
        .mark_bar(color=t["navy"], cornerRadiusEnd=4, height=20)
        .encode(
            y=alt.Y(
                "description:N",
                sort=alt.EncodingSortField("order"),
                title=None,
                axis=alt.Axis(labelLimit=320),
            ),
            x=alt.X("rows:Q", title="Applications"),
            tooltip=[
                alt.Tooltip("description:N", title="Stage"),
                alt.Tooltip("rows:Q", format=","),
                alt.Tooltip("denial_rate:Q", format=".1%", title="Denial rate"),
            ],
        )
    )
    return _style(chart, 40 * len(df))


def hourly_chart(hourly: list[dict]):
    df = pd.DataFrame(hourly)
    t = tokens()
    chart = (
        alt.Chart(df)
        .mark_bar(color=t["accent"], cornerRadiusEnd=3)
        .encode(
            x=alt.X("hour:T", title=None),
            y=alt.Y("requests:Q", title="Requests"),
            tooltip=[alt.Tooltip("hour:T", format="%b %d %H:00"), alt.Tooltip("requests:Q")],
        )
    )
    return _style(chart, 200)
