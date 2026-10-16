"""Design system for the dashboard: tokens, CSS and small presentational helpers."""

from __future__ import annotations

import html

import streamlit as st

LIGHT = {
    "bg": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface2": "#F0F4F8",
    "border": "#DCE3EC",
    "ink": "#0B1F33",
    "text": "#14263A",
    "muted": "#5B6B7D",
    "accent": "#0F8B8D",
    "accent_soft": "rgba(15,139,141,0.10)",
    "accent_ink": "#0B6E70",
    "navy": "#2F4A6D",
    "slate": "#8A9BB0",
    "amber": "#9A6512",
    "amber_soft": "#FDF3E1",
    "amber_border": "#F0D9A8",
    "error": "#A8423A",
    "shadow": "0 1px 2px rgba(11,31,51,0.04), 0 4px 16px rgba(11,31,51,0.05)",
}
DARK = {
    "bg": "#0B1220",
    "surface": "#111A2B",
    "surface2": "#152036",
    "border": "#24324A",
    "ink": "#EEF3F9",
    "text": "#DCE4EE",
    "muted": "#93A3B8",
    "accent": "#2BB3B1",
    "accent_soft": "rgba(43,179,177,0.14)",
    "accent_ink": "#5FD0CE",
    "navy": "#8FB3E8",
    "slate": "#6E7F96",
    "amber": "#E8B764",
    "amber_soft": "rgba(224,168,79,0.10)",
    "amber_border": "rgba(224,168,79,0.35)",
    "error": "#E58A80",
    "shadow": "0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.25)",
}


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def tokens() -> dict:
    return DARK if theme_type() == "dark" else LIGHT


def esc(text) -> str:
    return html.escape(str(text))


def inject_css() -> None:
    t = tokens()
    st.markdown(
        f"""
<style>
:root {{
  --la-surface: {t["surface"]}; --la-surface2: {t["surface2"]}; --la-border: {t["border"]};
  --la-ink: {t["ink"]}; --la-text: {t["text"]}; --la-muted: {t["muted"]};
  --la-accent: {t["accent"]}; --la-accent-soft: {t["accent_soft"]}; --la-accent-ink: {t["accent_ink"]};
  --la-amber: {t["amber"]}; --la-amber-soft: {t["amber_soft"]}; --la-amber-border: {t["amber_border"]};
  --la-shadow: {t["shadow"]};
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; color: var(--la-ink); }}
[data-testid="stMetricValue"], .la-num {{ font-variant-numeric: tabular-nums; }}
code, .la-mono {{ font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }}

/* cards: st.container(border=True, key="card_...") */
div[class*="st-key-card"] {{
  background: var(--la-surface); border: 1px solid var(--la-border) !important;
  border-radius: 14px; box-shadow: var(--la-shadow); padding: 1.1rem 1.2rem 1.0rem 1.2rem;
}}
div[class*="st-key-soft"] {{
  background: var(--la-surface2); border-radius: 14px; padding: 1rem 1.1rem;
}}

.la-eyebrow {{ text-transform: uppercase; letter-spacing: .09em; font-size: .72rem; font-weight: 600;
  color: var(--la-accent-ink); margin-bottom: .35rem; }}
.la-hero h1 {{ font-size: 2.15rem; line-height: 1.15; margin: 0 0 .5rem 0; padding: 0; }}
.la-hero p {{ color: var(--la-muted); font-size: 1.04rem; max-width: 760px; margin: 0; }}
.la-page-title {{ margin-bottom: 1.1rem; }}
.la-page-title h1 {{ font-size: 1.7rem; margin: 0; padding: 0; }}
.la-page-title p {{ color: var(--la-muted); margin: .3rem 0 0 0; max-width: 820px; }}

.la-stat {{ display: flex; flex-direction: column; gap: .15rem; }}
.la-stat .label {{ font-size: .78rem; color: var(--la-muted); font-weight: 500; }}
.la-stat .value {{ font-size: 1.55rem; font-weight: 700; color: var(--la-ink); font-variant-numeric: tabular-nums; }}
.la-stat .sub {{ font-size: .78rem; color: var(--la-muted); }}

.la-badge {{ display: inline-flex; align-items: center; gap: .3rem; padding: .14rem .55rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; border: 1px solid var(--la-border); color: var(--la-muted);
  background: var(--la-surface2); white-space: nowrap; }}
.la-badge.accent {{ color: var(--la-accent-ink); background: var(--la-accent-soft); border-color: transparent; }}
.la-badge.amber {{ color: var(--la-amber); background: var(--la-amber-soft); border-color: var(--la-amber-border); }}
.la-badge.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-weight: 500; }}

.la-notice {{ border: 1px solid var(--la-amber-border); background: var(--la-amber-soft); color: var(--la-text);
  border-radius: 12px; padding: .7rem .9rem; font-size: .86rem; line-height: 1.45; }}
.la-notice b {{ color: var(--la-amber); }}

.la-score {{ font-size: 3.1rem; font-weight: 700; line-height: 1; color: var(--la-ink); font-variant-numeric: tabular-nums; }}
.la-score-sub {{ color: var(--la-muted); font-size: .86rem; margin-top: .35rem; }}
.la-meter {{ position: relative; height: 10px; border-radius: 999px; background: var(--la-surface2);
  border: 1px solid var(--la-border); margin: 1rem 0 .4rem 0; }}
.la-meter .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: var(--la-accent); }}
.la-meter .tick {{ position: absolute; top: -5px; width: 2px; height: 18px; background: var(--la-ink); opacity: .55; }}
.la-meter-legend {{ display: flex; justify-content: space-between; font-size: .72rem; color: var(--la-muted); }}

.la-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem .9rem; font-size: .86rem; }}
.la-kv .k {{ color: var(--la-muted); }}
.la-kv .v {{ color: var(--la-text); font-variant-numeric: tabular-nums; }}

.la-empty {{ text-align: center; padding: 2.4rem 1rem; color: var(--la-muted); }}
.la-empty h3 {{ color: var(--la-ink); margin-bottom: .4rem; }}

.la-step {{ display: flex; gap: .8rem; align-items: flex-start; }}
.la-step .n {{ flex: none; width: 28px; height: 28px; border-radius: 8px; background: var(--la-accent-soft);
  color: var(--la-accent-ink); display: grid; place-items: center; font-weight: 700; font-size: .85rem; }}
.la-step .t {{ font-weight: 600; color: var(--la-ink); font-size: .92rem; }}
.la-step .d {{ color: var(--la-muted); font-size: .84rem; }}

.la-foot {{ color: var(--la-muted); font-size: .76rem; margin-top: 2.5rem; border-top: 1px solid var(--la-border);
  padding-top: .8rem; }}
[data-testid="stSidebar"] .la-side-note {{ font-size: .76rem; opacity: .8; line-height: 1.45; }}
[data-testid="stSidebar"] .la-badge {{ background: rgba(255,255,255,0.07); color: #DCE6F2;
  border-color: rgba(255,255,255,0.16); }}
[data-testid="stSidebar"] .la-badge.accent {{ color: #6FE3E0; background: rgba(60,196,194,0.16); border-color: transparent; }}
[data-testid="stSidebar"] .la-badge.amber {{ color: #F2C36B; background: rgba(224,168,79,0.14); border-color: transparent; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
@media (max-width: 640px) {{ .la-hero h1 {{ font-size: 1.6rem; }} .la-score {{ font-size: 2.4rem; }} }}
</style>
""",
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="la-hero"><div class="la-eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
        f"<p>{esc(body)}</p></div>",
        unsafe_allow_html=True,
    )


def page_title(eyebrow: str, title: str, body: str = "") -> None:
    st.markdown(
        f'<div class="la-page-title"><div class="la-eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
        + (f"<p>{esc(body)}</p>" if body else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def stat(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="la-stat"><span class="label">{esc(label)}</span><span class="value">{esc(value)}</span>'
        + (f'<span class="sub">{esc(sub)}</span>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "") -> str:
    return f'<span class="la-badge {esc(tone)}">{esc(text)}</span>'


def badges(*items: tuple[str, str]) -> None:
    st.markdown(" ".join(badge(t, tone) for t, tone in items), unsafe_allow_html=True)


def notice(text: str, strong: str = "Research use only.") -> None:
    st.markdown(f'<div class="la-notice"><b>{esc(strong)}</b> {esc(text)}</div>', unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> None:
    body = "".join(f'<span class="k">{esc(k)}</span><span class="v">{esc(v)}</span>' for k, v in rows)
    st.markdown(f'<div class="la-kv">{body}</div>', unsafe_allow_html=True)


def step(n: int, title: str, body: str) -> None:
    st.markdown(
        f'<div class="la-step"><div class="n">{n}</div><div><div class="t">{esc(title)}</div>'
        f'<div class="d">{esc(body)}</div></div></div>',
        unsafe_allow_html=True,
    )


def meter(probability: float, base_rate: float, threshold: float) -> None:
    st.markdown(
        f'<div class="la-meter" role="img" aria-label="Estimate {probability:.0%}; average {base_rate:.0%}">'
        f'<div class="fill" style="width:{probability * 100:.1f}%"></div>'
        f'<div class="tick" style="left:calc({base_rate * 100:.1f}% - 1px)" title="Test average"></div></div>'
        '<div class="la-meter-legend"><span>0%</span><span>100%</span></div>'
        f'<div class="la-meter-legend"><span>Tick = test average ({base_rate:.1%}) · research threshold '
        f"{threshold:.1%}</span></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, command: str | None = None) -> None:
    with st.container(border=True, key="card_empty"):
        st.markdown(
            f'<div class="la-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>', unsafe_allow_html=True
        )
        if command:
            st.code(command, language="bash")


def footer(text: str) -> None:
    st.markdown(f'<div class="la-foot">{esc(text)}</div>', unsafe_allow_html=True)


def pct(v, digits: int = 1) -> str:
    return "–" if v is None else f"{v * 100:.{digits}f}%"


def num(v, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"
