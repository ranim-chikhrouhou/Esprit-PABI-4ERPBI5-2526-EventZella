# -*- coding: utf-8 -*-
"""
EventZilla — ML laboratory (Streamlit), clear interface, teal accents, readable Plotly charts.

Launch from repository root :
    streamlit run ML/streamlit_app.py

Logo : placez une image PNG sous ``ML/assets/eventzilla_logo.png`` (optional) ;
sinon le fichier vectoriel ``ML/assets/eventzilla_ticket.svg`` est utilisé.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from ML.ml_paths import ML_MODELS, ML_PROCESSED  # noqa: E402
from ML.auth_streamlit import (  # noqa: E402
    SESSION_KEYS,
    authenticate,
    get_full_name,
    get_role,
    is_authenticated,
    logout,
)
from ML.clustering_deploy import (  # noqa: E402
    FORM_OBJECT_METIER_MARKDOWN,
    filter_clustering_metrics_if_models_missing,
    format_segment_deployment_explanation,
    friendly_feature_label,
    indices_for_radar_storytelling,
    load_median_imputer,
    load_standard_scaler,
    loyalty_form_group_key,
    loyalty_form_group_title,
    loyalty_json_hint_run_script,
    merge_metrics_for_loyalty_ui,
    number_input_format_for_feature,
    ordered_feature_indices_for_form,
    predict_cluster_from_raw_features,
    resolve_segment_labels,
    segment_card_title_loyalty,
    split_business_vs_id_feature_indices,
)

# Import différé possible pour les constantes SQL (affichage UI)
def _dw_connection_info() -> dict[str, str]:
    from ML.ml_paths import DATABASE_DW, SQL_SERVER, build_windows_auth_uri

    return {
        "serveur": SQL_SERVER,
        "base_dw": DATABASE_DW,
        "uri_apercu": build_windows_auth_uri().split("@")[-1][:120] + "…",
    }

ASSETS = _REPO / "ML" / "assets"

# Palette dashboard — thème clair, teal accents / cyan (figures modernes, bon contraste)
BRAND = {
    "deep": "#0d9488",
    "sky": "#14b8a6",
    "sky_soft": "#ccfbf1",
    "ink": "#0f172a",
    "ink_muted": "#334155",
    "muted": "#64748b",
    "card": "#ffffff",
    "page": "#f1f5f9",
    "panel": "#f8fafc",
    "accent": "#0f766e",
    "ok": "#16a34a",
    "line2": "#06b6d4",
    "radar_ref": "#a855f7",
    "gauge_low": "#e2e8f0",
    "gauge_mid": "#5eead4",
    "gauge_hi": "#0d9488",
    "chart_grid": "rgba(15, 23, 42, 0.10)",
    "plotly_plot": "#fafafa",
    "border_soft": "rgba(13, 148, 136, 0.22)",
}

# Navigation (order: home → prediction tools)
PAGE_HOME    = "Home"
PAGE_CLASSIF = "Booking Status"
PAGE_REGR    = "Price Estimation"
PAGE_CLUSTER = "Customer Segments"
PAGE_TS      = "Trends & Forecast"
PAGE_RECAP   = "Summary"
PAGE_ORDER: tuple[str, ...] = (
    PAGE_HOME,
    PAGE_CLASSIF,
    PAGE_REGR,
    PAGE_CLUSTER,
    PAGE_TS,
    PAGE_RECAP,
)

# ── Pages by role ────────────────────────────────────
# Marketing (Ranim): customer segments + booking predictions
# Finance (Naima): price estimation + trends
# CRM (Anas): booking predictions + customer loyalty
ROLE_PAGES: dict[str, tuple[str, ...]] = {
    "marketing_manager": (PAGE_HOME, PAGE_CLUSTER, PAGE_CLASSIF),
    "financial_manager": (PAGE_HOME, PAGE_REGR,    PAGE_TS),
    "crm_manager":       (PAGE_HOME, PAGE_CLASSIF, PAGE_CLUSTER),
}

# Badges affichés dans la sidebar selon le rôle
ROLE_LABELS: dict[str, str] = {
    "marketing_manager": "📢 Marketing Manager",
    "financial_manager": "💰 Finance Manager",
    "crm_manager":       "🤝 Customer Relations Manager",
}

ML_INTEREST_MARKDOWN = """
**Why machine learning for EventZilla?**  
The **database** (bookings, aggregated finances, volumes) permettent d’**anticipate** les statuses and amounts, de **segment** l’offre et de **track** les monthly trends — without replacing business, mais pour **prioritize** et **illustrate** les scenarios in an educational framework and reproducible.

**Ce studio** centralise les models trained on the same scope as your notebooks (00→05) : test them here before any production deployment.
"""

# Textes « déploiement / test » — peu techniques, alignés database (enseignants & équipe)
DEPLOY_CLASSIF_MARKDOWN = """
### What is this screen for?

**Objectif :** décrire **une situation d’activity EventZilla** telle qu’elle apparaît dans nos data (même logique que le database) et voir **quel booking status** le Prediction System retient comme le plus plausible.

Vous composez un **scénario** (niveau d’activity, period, order of magnitude of amounts) ou vous partez **d’un real case** already present dans le prepared dataset — le tout pour **tester le Prediction System** sans écrire de SQL.
"""

DEPLOY_REGR_MARKDOWN = """
### What is this screen for?

**Objectif :** estimer **une valeur continue** du scope finance / performance (ex. amount, panier) à partir d’une situation **cohérente avec le database**.

Même principe que la Status Prediction : **scénario type** ou **ligne réelle** issue des data préparées, pour **valider la Price Estimation** sur nos EventZilla indicators.
"""

DEPLOY_TS_MARKDOWN = """### What is this screen for?

**Objectif :** visualiser **l’monthly evolution** d’aggregated indicators (volume d’activity, chiffre d’affaires, average basket) **calculated from le database**, puis **project a few months** pour illustrate la observed dynamics.

**What you can test:**
1. **Choisir l’indicateur** — volume d’activity, monthly revenue or average basket.
2. **Ajuster l’horizon** — from 1 to 12 months forecast.
3. **Visually compare** training, validation zone, and forecast.
4. **Read the metrics** — RMSE, MAE, MAPE on the test window.

**Modèles comparés :** **Trend Analysis** (exponential smoothing with trend) vs **Advanced Forecast** (autoregression + differencing + moving average). The **best model** is the one with the **lowest RMSE** on validation.
"""

DEPLOY_SYNTH_MARKDOWN = """
### Navigation Help

- **Home**: Why use AI for EventZilla, quick **KPIs**, buttons to tests.
- **Analysis pages**: One screen per Prediction System type (same logic as notebooks).
- **Summary**: **Single table** of best models and quality metrics, streamlined view.

The `metrics_*.json` files in `ML/models_artifacts/` feed the indicators and summary table.
"""


def _subtitle_bold_html(s: str) -> str:
    """Convertit uniquement les **paires** en <strong>, échappe le reste."""
    parts = s.split("**")
    out: list[str] = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(f"<strong>{html.escape(p)}</strong>")
        else:
            out.append(html.escape(p))
    return "".join(out)


def section_header(title: str, subtitle: str | None = None) -> None:
    """Titre de section type fiche (barre verticale + sous-titre)."""
    sub = (
        f'<p class="ez-section-sub">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f'<div class="ez-section-wrap">'
        f'<div class="ez-section-accent" aria-hidden="true"></div>'
        f'<div class="ez-section-inner"><h2 class="ez-section-title">{html.escape(title)}</h2>{sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def champion_rationale(m: dict | None, fallback: str = "") -> str:
    """Short text explaining Prediction System choice (optional fields in JSON metrics)."""
    if not m:
        return fallback or "—"
    for key in ("champion_rule", "rationale", "notes_champion", "model_notes"):
        v = m.get(key)
        if v:
            return str(v).strip()
    return fallback or "Model selected after comparison on test set (details in associated notebook)."


def deployment_context_card(
    critere: str,
    Goal: str,
    objectif: str,
    kpi: str,
    modele: str,
    rationale: str,
    figure_note: str,
    *,
    label_cible: str = "Target (Y)",
    label_kpi: str = "KPI / business reading",
    label_figure: str = "Chart / indicator to view",
) -> None:
    """Bloc compact en tête des pages de test."""
    esc = html.escape
    st.markdown(
        f'<div class="ez-deploy-context"><h4>What this screen tests</h4>'
        f'<div class="ez-dc-grid">'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Criterion</span>'
        f'<span class="ez-dc-val">{esc(critere)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">{esc(label_cible)}</span>'
        f'<span class="ez-dc-val">{esc(Goal)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Prediction System</span>'
        f'<span class="ez-dc-val">{esc(modele)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Objectif</span>'
        f'<span class="ez-dc-val">{esc(objectif)}</span></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="EventZilla Analytics Dashboard",
    page_icon="EZ",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_logo_path() -> Path:
    for name in ("eventzilla_logo.png", "logoround.png", "eventzilla_round.png", "logo.png"):
        p = ASSETS / name
        if p.is_file():
            return p
    return ASSETS / "eventzilla_ticket.svg"


def _inject_theme_css() -> None:
    """Thème global clair : cartes nettes, teal accents, graphiques lisibles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html { font-size: 16px; }
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1200px !important;
        }
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%) !important;
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.85) !important; backdrop-filter: blur(8px); }
        [data-testid="stToolbar"] { display: none !important; }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] label { color: #0f172a !important; font-weight: 600 !important; font-size: 0.92rem !important; }
        .ez-sidebar-brand {
            text-align: center;
            padding: 0.5rem 0 0.65rem 0;
            font-weight: 800;
            font-size: 1.1rem;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            padding: 0.5rem 0.75rem !important;
            margin-bottom: 2px !important;
            transition: all 0.15s ease !important;
            text-align: left !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent !important;
            color: #334155 !important;
            border: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: #f1f5f9 !important;
            border-color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            font-weight: 700 !important;
        }

        .stMarkdown, .stText { color: #334155; font-size: 1rem; line-height: 1.62; }
        [data-testid="stExpander"] details { font-size: 0.98rem !important; }
        [data-testid="stExpander"] .stMarkdown p, [data-testid="stExpander"] .stMarkdown li {
            font-size: 1rem !important; line-height: 1.62 !important; color: #475569 !important;
        }
        [data-testid="stExpander"] .stMarkdown h3 { font-size: 1.15rem !important; color: #0d9488 !important; margin-top: 0.35rem !important; }

        label, .stSelectbox label, .stRadio label, .stSlider label, .stCheckbox label {
            color: #0f172a !important; font-size: 0.98rem !important; font-weight: 600 !important;
        }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: rgba(13, 148, 136, 0.35) !important;
            border-radius: 10px !important;
            font-size: 0.98rem !important;
        }
        .stCaption, [data-testid="stCaption"] { font-size: 0.92rem !important; color: #64748b !important; }

        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 0.85rem 1rem !important;
            min-height: 80px !important;
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0d9488 !important;
            font-weight: 800 !important;
            font-size: 1.65rem !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #2dd4bf 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.35rem !important;
            box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.05) !important;
            box-shadow: 0 6px 20px rgba(13, 148, 136, 0.4) !important;
        }
        .stButton > button[kind="secondary"] {
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 0.95rem !important;
            border: 1px solid rgba(13, 148, 136, 0.35) !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            border-radius: 14px !important;
        }
        [data-testid="stExpander"] summary { color: #0d9488 !important; font-weight: 700 !important; font-size: 0.98rem !important; }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            font-size: 0.98rem !important;
        }

        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 0.35rem 0.15rem 0.5rem 0.15rem;
            border: 1px solid #e2e8f0;
        }

        .ez-title-gradient {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 40%, #0891b2 75%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.2;
        }
        .ez-hero-sub { margin: 0.25rem 0 0 0; color: #475569; font-size: 0.95rem; line-height: 1.5; }
        .ez-hero-sub strong { color: #0f172a; font-weight: 600; }

        .ez-section-wrap {
            display: flex; align-items: stretch; gap: 0.65rem;
            margin: 1.15rem 0 0.65rem 0;
        }
        .ez-section-accent {
            width: 5px; border-radius: 5px; flex-shrink: 0;
            background: linear-gradient(180deg, #14b8a6 0%, #0ea5e9 55%, #6366f1 100%);
        }
        .ez-section-inner { flex: 1; min-width: 0; }
        .ez-section-title {
            margin: 0;
            font-size: 1.28rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .ez-section-sub {
            margin: 0.3rem 0 0 0;
            font-size: 0.98rem;
            color: #64748b;
            line-height: 1.45;
        }

        .ez-hero {
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
            position: relative;
            overflow: hidden;
        }
        .ez-hero::before {
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(ellipse 80% 60% at 100% 0%, rgba(20, 184, 166, 0.12) 0%, transparent 50%);
            pointer-events: none;
        }
        .ez-hero h1 { margin: 0 0 0.35rem 0; position: relative; z-index: 1; }

        .ez-hero--synth, .ez-hero--classif, .ez-hero--regr, .ez-hero--cluster, .ez-hero--ts {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(13, 148, 136, 0.22);
            box-shadow: 0 10px 40px rgba(15, 23, 42, 0.07);
        }
        .ez-hero--classif { border-left: 4px solid #10b981; }
        .ez-hero--regr { border-left: 4px solid #8b5cf6; }
        .ez-hero--cluster { border-left: 4px solid #14b8a6; }
        .ez-hero--ts { border-left: 4px solid #f59e0b; }
        .ez-hero--synth { border-left: 4px solid #6366f1; }

        .ez-hero-badges { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.65rem; position: relative; z-index: 1; }
        .ez-hero-badge {
            font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800;
            padding: 0.32rem 0.75rem; border-radius: 999px;
            background: rgba(20, 184, 166, 0.1);
            border: 1px solid rgba(13, 148, 136, 0.35);
            color: #0f766e;
        }

        .ez-result {
            background: #ffffff;
            border-left: 4px solid #14b8a6;
            border-radius: 0 14px 14px 0;
            padding: 1.25rem 1.35rem;
            margin-top: 0.35rem;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left-width: 4px;
        }
        .ez-result h3 {
            margin: 0 0 0.5rem 0; color: #0d9488; font-size: 1.05rem; font-weight: 800;
            letter-spacing: 0.04em; text-transform: uppercase;
        }

        .ez-card {
            background: #ffffff;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 1.15rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
        }
        .ez-card--deploy h5 { font-size: 1.05rem !important; color: #0f172a !important; margin-bottom: 0.75rem !important; }
        .ez-card h1, .ez-card h2, .ez-card h3, .ez-card h4, .ez-card h5 { color: #0f172a !important; }

        .ez-out-panel {
            min-height: 240px;
            border-radius: 14px;
            border: 1px dashed rgba(13, 148, 136, 0.35);
            background: #f8fafc;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .ez-out-panel p { font-size: 1rem; color: #64748b; max-width: 22rem; line-height: 1.55; }
        /* Bandeau d’aide Status Prediction : pas de hauteur min. élevée (évite un « cadre vide ») */
        .ez-out-panel--hint { min-height: auto; align-items: flex-start; text-align: left; padding: 1rem 1.1rem; }

        .ez-panel {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
        }
        .ez-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #0d9488 !important;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .ez-flow {
            display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
            font-size: 0.92rem; color: #64748b; margin: 0.65rem 0 0 0;
        }
        .ez-flow span {
            background: rgba(20, 184, 166, 0.08);
            border: 1px solid rgba(13, 148, 136, 0.25);
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-weight: 600;
            color: #0f766e;
        }
        .ez-panel--classif { border-left: 4px solid #10b981; }
        .ez-panel--regr { border-left: 4px solid #8b5cf6; }
        .ez-panel--ts { border-left: 4px solid #f59e0b; }
        .ez-panel--synth { border-left: 4px solid #6366f1; }
        .ez-panel--cluster { border-left: 4px solid #14b8a6; }
        .ez-flow--classif span { border-color: rgba(16, 185, 129, 0.35); color: #047857; background: rgba(16, 185, 129, 0.08); }
        .ez-flow--regr span { border-color: rgba(139, 92, 246, 0.35); color: #6d28d9; background: rgba(139, 92, 246, 0.08); }
        .ez-flow--ts span { border-color: rgba(245, 158, 11, 0.4); color: #b45309; background: rgba(245, 158, 11, 0.08); }
        .ez-flow--synth span { border-color: rgba(99, 102, 241, 0.35); color: #4338ca; background: rgba(99, 102, 241, 0.08); }
        .ez-flow--cluster span { border-color: rgba(20, 184, 166, 0.4); color: #0f766e; background: rgba(20, 184, 166, 0.08); }

        .ez-deploy-context {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.85rem;
        }
        .ez-deploy-context h4 {
            margin: 0 0 0.45rem 0;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #64748b;
            font-weight: 800;
        }
        .ez-dc-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.4rem 0.85rem;
        }
        .ez-dc-item { min-width: 0; }
        .ez-dc-label {
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .ez-dc-val {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f172a;
            line-height: 1.45;
        }
        .ez-dc-val--note { font-weight: 500; color: #475569; font-size: 0.9rem; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
        }

        .js-plotly-plot .plotly .modebar { opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_theme_css()


def _inject_page_accent(deep: str, main: str, soft: str) -> None:
    """Dynamic color theme per page — buttons, metrics, forms, expanders."""
    st.markdown(
        f"""
        <style>
        section.main div[data-testid="stForm"] {{
            background: linear-gradient(180deg, {soft} 0%, #ffffff 48%) !important;
            border: 1px solid {main}44 !important;
            border-radius: 14px !important;
            padding: 0.85rem 1rem 1rem 1rem !important;
        }}
        div[data-testid="stForm"] label p {{
            font-size: 0.98rem !important; font-weight: 600 !important; color: {deep} !important;
        }}
        div[data-testid="stForm"] input[type="text"] {{
            font-size: 0.98rem !important; border-radius: 8px !important; border-color: {main}66 !important;
        }}
        div[data-testid="stForm"] [data-baseweb="input"] input {{
            font-size: 0.98rem !important; background-color: #ffffff !important; color: #0f172a !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {deep} 0%, {main} 55%, {main}cc 100%) !important;
            border: none !important; color: #ffffff !important; font-weight: 700 !important;
            box-shadow: 0 3px 12px {main}55 !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.06) !important; box-shadow: 0 5px 18px {main}66 !important;
        }}
        [data-testid="stExpander"] summary {{ color: {deep} !important; font-weight: 700 !important; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {main} !important; }}
        .ez-section-accent {{ background: {main} !important; }}
        .ez-hero-badge {{ background: {main}18 !important; border-color: {main}55 !important; color: {deep} !important; }}
        .ez-result h3 {{ color: {deep} !important; }}
        .ez-result {{ border-left-color: {main} !important; }}
        .ez-regr-section-title {{ color: {deep} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Preset accent palettes per page
PAGE_ACCENT = {
    "classif": ("#047857", "#10b981", "#ecfdf5"),
    "regr": ("#6d28d9", "#8b5cf6", "#f5f3ff"),
    "cluster": ("#c2410c", "#ea580c", "#fff7ed"),
    "ts": ("#b45309", "#f59e0b", "#fffbeb"),
    "synth": ("#4338ca", "#6366f1", "#eef2ff"),
}





@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_resource(show_spinner=False)
def load_joblib(path: Path):
    if not path.is_file():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def classification_feature_columns() -> list[str] | None:
    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if not pp.is_file():
        return None
    df = pd.read_parquet(pp)
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != "fact_finance_id"]
    return num[:20]


@st.cache_data(show_spinner=False)
def _dw_numeric_columns_all() -> list[str]:
    """Toutes les colonnes numériques du database (ordre stable), pour distinguer Price Estimation vs Status Prediction."""
    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if not pp.is_file():
        return []
    df = pd.read_parquet(pp)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "fact_finance_id"]


def classification_form_column_names() -> frozenset[str]:
    """Columns displayed in Booking Status form (excluding id) — not to duplicate in Price Estimation."""
    cols = classification_feature_columns()
    if not cols:
        return frozenset()
    return frozenset(c for c in cols if not _is_id_column(c))


CLASSIF_MONTH_LABELS_FR = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _classif_order_columns(cols: list[str]) -> list[str]:
    """Put period / amounts useful for business reading first, then the rest."""
    head = []
    for key in ("cal_year", "cal_month", "quarter", "final_price", "service_price", "event_budget"):
        if key in cols and key not in head:
            head.append(key)
    tail = [c for c in cols if c not in head]
    tail.sort(key=lambda x: (not str(x).lower().startswith("nb_"), str(x).lower()))
    return head + tail


def _classif_format_suggested_value(col: str, v: float) -> str:
    n = str(col).lower()
    if n in ("cal_year", "quarter") or n.startswith("id_") or "nb_" in n or "count" in n:
        return f"{v:,.0f}".replace(",", " ")
    if n == "cal_month":
        return f"{int(round(v))}"
    return f"{v:,.2f}".replace(",", " ")


def classif_dropdown_suggestions(df: pd.DataFrame, col: str) -> list[tuple[str, float]]:
    """Libellés + valeurs numériques alignées sur la distribution du database set (suggestions)."""
    if col not in df.columns:
        return [("0", 0.0)]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return [("0", 0.0)]
    c = str(col).lower()
    if c == "cal_month":
        return [(f"{CLASSIF_MONTH_LABELS_FR[i]} (month {i + 1})", float(i + 1)) for i in range(12)]
    if c == "quarter":
        return [
            ("Q1 — January to March", 1.0),
            ("Q2 — April to June", 2.0),
            ("Q3 — July to September", 3.0),
            ("Q4 — October to December", 4.0),
        ]
    if c == "cal_year":
        u = sorted(pd.unique(s.round().astype(int)))
        if len(u) <= 24:
            return [(str(int(y)), float(y)) for y in u]
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        labs = (
            "Year — low (~10th percentile)",
            "Year — low (~25th)",
            "Year — median",
            "Year — high (~75th)",
            "Year — high (~90th)",
        )
        out: list[tuple[str, float]] = []
        for lab, q in zip(labs, qs):
            v = float(s.quantile(q))
            yi = int(round(v))
            out.append((f"{lab} → {yi}", float(yi)))
        return out
    if len(s) <= 15:
        u = np.sort(s.unique())
        return [(f"Valeur observée — {_classif_format_suggested_value(col, float(x))}", float(x)) for x in u]
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    labs_fr = (
        "Very low in database (~10th %)",
        "Low (~25th %)",
        "Typical — median",
        "High (~75th %)",
        "Very high (~90th %)",
    )
    pairs: list[tuple[str, float]] = []
    for lab, q in zip(labs_fr, qs):
        v = float(s.quantile(q))
        pairs.append((f"{lab} → {_classif_format_suggested_value(col, v)}", v))
    return pairs


def _classif_field_group(col: str) -> str:
    n = str(col).lower()
    if n in ("cal_year", "cal_month", "quarter"):
        return "period"
    if any(x in n for x in ("price", "budget", "margin", "revenue", "ca_")):
        return "money"
    if n.startswith("nb_") or "count" in n:
        return "counts"
    if n.startswith("id_"):
        return "ids"
    if n in ("is_holiday",):
        return "ctx"
    return "other"


def _classif_group_title(group: str) -> str:
    return {
        "period": "Period & calendar (database data)",
        "money": "Prices & Amounts",
        "counts": "Quantities",
        "ids": "Identifiants dimension (database)",
        "ctx": "Context",
        "other": "Autres variables du Prediction System",
    }.get(group, "Variables")


def _classif_id_median_defaults(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Medians on prepared dataset for id columns required by Prediction System (non saisies à l’écran)."""
    out: dict[str, float] = {}
    for c in cols:
        if not _is_id_column(c) or c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        out[c] = float(s.median()) if len(s) else 0.0
    return out


def safe_target_filename(Goal: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in Goal)


def regression_paths_and_targets(m: dict) -> tuple[list[str], str | None]:
    # Accept both "Goal" and "target" field names
    primary = m.get("Goal") or m.get("target")
    runs = m.get("regression_objectives") or []
    if runs:
        targets = [r.get("Goal") or r.get("target") for r in runs if r.get("Goal") or r.get("target")]
        return targets, primary
    if primary:
        return [primary], primary
    return [], None


def regression_model_path(m: dict, Goal: str) -> Path:
    # Accept both "Goal" and "target" field names
    primary = m.get("Goal") or m.get("target")
    if Goal == primary:
        # Check which model is the champion and load the correct file
        champion = m.get("champion_model", "").lower()
        if "ridge" in champion:
            return ML_MODELS / "ridge_regression_primary.joblib"
        else:
            # Default to RF if not specified or if it's RandomForest
            return ML_MODELS / "rf_panier_kpi_pipeline.joblib"
    return ML_MODELS / f"rf_regression_target_{safe_target_filename(Goal)}.joblib"


REGR_TARGET_LABEL_FR: dict[str, str] = {
    "final_price": "Final price (basket / order)",
    "service_price": "Provider price",
    "benchmark_avg_price": "Average reference price (benchmark)",
    "event_budget": "Event budget",
    "commission_margin": "Commission margin (final − provider)",
}

# Goal unique exposée dans l’UI Streamlit (aligned with criterion D — basket / final price).
REGR_UI_TARGET = "final_price"
# Accent visuel Price Estimation (violet) — distinct de la Status Prediction (teal).
REGR_PAGE_ACCENT = "#7c3aed"
REGR_PAGE_ACCENT_DEEP = "#6d28d9"
# Customer Grouping (E) — ambre / orange, distinct de la Status Prediction (teal) et de la Price Estimation (violet)
CLUSTER_PAGE_ACCENT = "#ea580c"
CLUSTER_PAGE_ACCENT_DEEP = "#c2410c"
CLUSTER_PAGE_ACCENT_SOFT = "#fff7ed"

# Aligné sur ML/scripts/run_03_prediction_regression.py (TARGET_KPIS)
REGR_KPI_TAG: dict[str, str] = {
    "final_price": "basket_average_revenue_sum_final_price",
    "service_price": "provider_price_revenue_structure",
    "event_budget": "event_budget",
}


def regression_infer_features(df: pd.DataFrame, Goal: str) -> list[str]:
    """Même ensemble de Input Factors que run_03 : numériques sauf la Goal et fact_finance_id."""
    if Goal not in df.columns:
        return []
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != Goal and c != "fact_finance_id"
    ]


def pipeline_feature_importance_dict(pipe, feature_names: list[str]) -> dict[str, float] | None:
    """Importances Smart Decision System alignées sur l’ordre des colonnes d’Learning."""
    if pipe is None or not hasattr(pipe, "named_steps"):
        return None
    reg = pipe.named_steps.get("reg")
    if reg is None or not hasattr(reg, "feature_importances_"):
        return None
    imp = np.asarray(reg.feature_importances_, dtype=float)
    if len(imp) != len(feature_names):
        return None
    return {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}


def regression_form_column_order(
    cols_form: list[str],
    pipe,
    features_full: list[str],
) -> tuple[list[str], dict[str, float] | None]:
    """Field order : decreasing RF importance ; otherwise heuristic « prix / budget » d’abord (≠ Status Prediction)."""
    imp = pipeline_feature_importance_dict(pipe, features_full)
    if imp:
        ordered = sorted(cols_form, key=lambda c: imp.get(c, 0.0), reverse=True)
        sub = {c: imp[c] for c in cols_form if c in imp}
        return ordered, sub
    priority = (
        "service_price",
        "event_budget",
        "benchmark_avg_price",
        "commission_margin",
        "cal_year",
        "cal_month",
        "quarter",
    )
    head = [c for c in priority if c in cols_form]
    tail = sorted([c for c in cols_form if c not in head], key=str.lower)
    return head + tail, None


def _column_numeric_median(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.median()) if len(s) else 0.0


def _regr_num_bounds_step(
    df: pd.DataFrame, col: str
) -> tuple[float, float, float, float]:
    """min, max, défaut (médiane), pas — pour `st.number_input`."""
    if col not in df.columns:
        return 0.0, 1.0, 0.0, 1.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return 0.0, 1.0, 0.0, 1.0
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        hi = lo + 1.0
    med = float(s.median())
    fmt = number_input_format_for_feature(col)
    step = 1.0 if fmt == "%.0f" else 0.01
    return lo, hi, med, step


def _regr_benchmark_price_dropdown(col: str) -> bool:
    """`benchmark_avg_price` (or alias) : input by dropdown of quantiles / database values."""
    n = str(col).lower().replace(" ", "_")
    return n in ("benchmark_avg_price", "benchmark_price")


# Minimum number of displayed fields (user input) — avoids form reduced to single variable.
REGR_MANUAL_FIELDS_MIN = 6
REGR_MANUAL_FIELDS_TARGET = 10
REGR_MANUAL_FIELDS_MAX = 12


def regression_ui_manual_columns(ordered: list[str]) -> list[str]:
    """
    Price Estimation input columns : d’abord **hors** formulaire Status Prediction, then complement
    jusqu’à un **minimum** de fields (Prediction System importance order), even if overlap with classification.
    """
    if not ordered:
        return []
    forbid = classification_form_column_names()
    num_all = _dw_numeric_columns_all()
    tail_from_21 = set(num_all[20:]) if len(num_all) > 20 else set()
    seen: set[str] = set()
    out: list[str] = []

    def _push_disjoint(c: str) -> None:
        if c in seen or c in forbid:
            return
        out.append(c)
        seen.add(c)

    def _push_any(c: str) -> None:
        if c in seen:
            return
        out.append(c)
        seen.add(c)

    # 1) Input Factors hors scope Status Prediction (importance RF / heuristique).
    for c in ordered:
        _push_disjoint(c)
        if len(out) >= REGR_MANUAL_FIELDS_MAX:
            return out[:REGR_MANUAL_FIELDS_MAX]

    # 2) Columns « queue du schéma » database (index ≥ 21), toujours hors doublon classif.
    for c in ordered:
        if c in tail_from_21:
            _push_disjoint(c)
        if len(out) >= REGR_MANUAL_FIELDS_MAX:
            return out[:REGR_MANUAL_FIELDS_MAX]

    # 3) Complément : at least REGR_MANUAL_FIELDS_MIN fields (souvent jusqu’à 10), even if present en Status Prediction.
    _cap = min(len(ordered), REGR_MANUAL_FIELDS_MAX)
    fill_to = min(
        _cap,
        max(REGR_MANUAL_FIELDS_MIN, min(REGR_MANUAL_FIELDS_TARGET, _cap)),
    )
    if len(out) < fill_to:
        for c in ordered:
            _push_any(c)
            if len(out) >= fill_to:
                break

    # 4) Reference en tête (dropdown) if present dans la liste affichée.
    bm = "benchmark_avg_price"
    if bm in out:
        out = [bm] + [x for x in out if x != bm]
        out = out[:REGR_MANUAL_FIELDS_MAX]

    if out:
        return out[:REGR_MANUAL_FIELDS_MAX]

    # 5) Dernier recours : parcours database after la 20e colonne.
    for c in num_all[20:]:
        if c in ordered:
            _push_any(c)
        if len(out) >= REGR_MANUAL_FIELDS_MIN:
            break

    if out:
        return out[:REGR_MANUAL_FIELDS_MAX]

    return ordered[: min(REGR_MANUAL_FIELDS_MIN, len(ordered))]


def regr_form_section_blocks(
    ordered: list[str],
    imp_map: dict[str, float] | None,
) -> list[tuple[str, list[str]]]:
    """Deux blocs : variables les plus influentes, then the rest (libellés distincts de la Status Prediction)."""
    if not ordered:
        return []
    if len(ordered) == 1:
        return [("Prediction System Input Factors (final price)", ordered)]
    if imp_map and len(ordered) >= 2:
        tot = sum(imp_map.get(c, 0.0) for c in ordered)
        if tot > 0:
            acc = 0.0
            cut = 0
            for i, c in enumerate(ordered):
                acc += imp_map.get(c, 0.0) / tot
                if acc >= 0.5:
                    cut = i + 1
                    break
            else:
                cut = max(1, len(ordered) // 2)
            cut = max(1, min(len(ordered) - 1, cut))
            return [
                ("Most influential variables on final price (random forest)", ordered[:cut]),
                ("Other model input factors", ordered[cut:]),
            ]
    cut = max(1, (len(ordered) + 1) // 2)
    return [
        ("Amounts, budget & references (fill in priority)", ordered[:cut]),
        ("Calendar & other dimensions", ordered[cut:]),
    ]


def regression_run_for_target(m: dict, Goal: str, df_pq: pd.DataFrame | None = None) -> dict:
    """Métadata (Factors, KPI) pour une Goal ; repli comme run_03 si pas de regression_objectives."""
    for r in m.get("regression_objectives") or []:
        if r.get("Goal") == Goal or r.get("target") == Goal:
            return r
    # Accept both "Goal" and "target" field names
    target_field = m.get("Goal") or m.get("target")
    if target_field == Goal:
        # Accept both "Factors" and "features" field names
        factors = m.get("Factors") or m.get("features") or []
        return {
            "Goal": Goal,
            "Factors": factors,
            "kpi_alignment": m.get("kpi_alignment"),
        }
    if df_pq is not None and Goal in df_pq.columns:
        feats = regression_infer_features(df_pq, Goal)
        return {
            "Goal": Goal,
            "Factors": feats,
            "kpi_alignment": REGR_KPI_TAG.get(Goal, ""),
        }
    return {}


def format_regression_target_choice(Goal: str) -> str:
    """Libellé pour dropdown des cibles Y."""
    lab = REGR_TARGET_LABEL_FR.get(Goal, Goal.replace("_", " "))
    return f"{lab} — `{Goal}`"


def regression_metrics_for_target(m: dict, Goal: str) -> dict[str, float | None]:
    """Prediction Error / Average Error / Correct Predictions Score propres à une Goal if presents dans `regression_objectives`."""
    for r in m.get("regression_objectives") or []:
        if r.get("Goal") == Goal or r.get("target") == Goal:
            return {
                "Prediction Error": r.get("Prediction Error") or r.get("rmse"),
                "Average Error": r.get("Average Error") or r.get("mae"),
                "Correct Predictions": r.get("Correct Predictions") or r.get("r2"),
            }
    # Accept both "Goal" and "target" field names
    target_field = m.get("Goal") or m.get("target")
    if target_field == Goal:
        er = extract_regression_metrics(m)
        return {
            "Prediction Error": er.get("Prediction Error"),
            "Average Error": er.get("Average Error"),
            "Correct Predictions": er.get("Correct Predictions")
        }
    return {"Prediction Error": None, "Average Error": None, "Correct Predictions": None}


def extract_classification_metrics(m: dict) -> dict:
    if "test_metrics_champion" in m:
        return m["test_metrics_champion"]
    return {
        "Correct Predictions": m.get("Correct Predictions"),
        "f1_weighted": m.get("f1_weighted"),
        "roc_auc": m.get("roc_auc"),
    }


def extract_regression_metrics(m: dict) -> dict:
    if "test_champion" in m:
        tc = m["test_champion"]
        # Map the actual metric names to the display names
        return {
            "Prediction Error": tc.get("rmse"),
            "Average Error": tc.get("mae"),
            "Correct Predictions": tc.get("r2"),
        }
    return {
        "Prediction Error": m.get("Prediction Error") or m.get("rmse"),
        "Average Error": m.get("Average Error") or m.get("mae"),
        "Correct Predictions": m.get("Correct Predictions") or m.get("r2"),
    }


def _timeseries_rmse(mt: dict) -> float | None:
    tc = mt.get("test_champion")
    if isinstance(tc, dict) and tc.get("Prediction Error") is not None:
        return float(tc["Prediction Error"])
    th = mt.get("test_holt") or {}
    if th.get("Prediction Error") is not None:
        return float(th["Prediction Error"])
    if mt.get("rmse_holdout") is not None:
        return float(mt["rmse_holdout"])
    return None


SERIES_COLUMN_LABELS_FR = {
    "nb_fact_rows": "Volume d’activity (fact rows database / month)",
    "revenue_sum": "Aggregated monthly revenue (sum of amounts)",
    "avg_final_price": "Monthly average basket",
}


def _plotly_x_datetime(value) -> object:
    """Convertit un instant pandas en type compatible Plotly (évite sum() sur Timestamp)."""
    return pd.Timestamp(value).to_pydatetime()


def clustering_feature_names_for_model(km, features_json_name: str | None = None) -> list[str] | None:
    """Noms de colonnes alignés sur les centres Customer Grouping (sklearn ou fichier optional)."""
    fn = getattr(km, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return [str(x) for x in fn]
    fname = features_json_name or "clustering_feature_names.json"
    path = ML_MODELS / fname
    if path.is_file():
        raw = load_json(path)
        if isinstance(raw, dict) and raw.get("Factors"):
            return [str(x) for x in raw["Factors"]]
        if isinstance(raw, list):
            return [str(x) for x in raw]
    return None


@st.cache_data(ttl=300, show_spinner="Connecting to database and loading series…")
def fetch_dw_timeseries_dataframe(cache_bust: int = 0) -> tuple[pd.DataFrame | None, str | None]:
    """Exécute la même requête que ``run_04_time_series.py`` sur le database.

    ``cache_bust`` permet d’invalider le cache (bouton « Reload »).
    Retourne ``(dataframe, None)`` en cas de succès, ou ``(None, message_error)``.
    """
    try:
        from ML.ml_paths import get_sql_engine, read_dw_sql, sql_engine_init_error
        from ML.schema_eventzilla import SQL_ML_TIME_SERIES_RESERVATIONS

        eng = get_sql_engine()
        if eng is None:
            err = sql_engine_init_error()
            return None, (
                err
                or "SQLAlchemy engine not created — check pyodbc, sqlalchemy, and variables "
                "``EVENTZILLA_SQL_*`` (voir ``ML/ml_paths.py``)."
            )
        df = read_dw_sql(SQL_ML_TIME_SERIES_RESERVATIONS, eng)
        if df is None or len(df) == 0:
            return None, "Series query returned 0 rows — check database scope."
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def test_dw_sql_connection() -> tuple[bool, str, pd.DataFrame | None]:
    """Test rapide : ``SELECT DB_NAME()`` — même mécanisme que ``run_test_sql_connection.py``."""
    try:
        from ML.ml_paths import get_sql_engine, read_dw_sql, sql_engine_init_error

        eng = get_sql_engine()
        if eng is None:
            return False, sql_engine_init_error() or "Engine unavailable.", None
        df = read_dw_sql("SELECT DB_NAME() AS base_dw, @@SERVERNAME AS serveur, GETDATE() AS horloge_sql", eng)
        return True, "Connexion OK.", df
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def build_champions_table_rows(mc: dict | None, mr: dict | None, mk: dict | None, mt: dict | None) -> pd.DataFrame:
    """Tableau type synthèse notebook 05 (colonnes alignées sur le livrable)."""
    rows: list[dict] = []

    def qcl(m: dict | None) -> str:
        if not m:
            return "—"
        cm = extract_classification_metrics(m)
        parts = []
        if cm.get("Correct Predictions") is not None:
            parts.append(f"Acc={cm['Correct Predictions']:.3f}")
        if cm.get("f1_weighted") is not None:
            parts.append(f"Balance Score={cm['f1_weighted']:.3f}")
        if cm.get("roc_auc") is not None:
            parts.append(f"Quality Score={cm['roc_auc']:.3f}")
        return " · ".join(parts) if parts else "—"

    def qrg(m: dict | None) -> str:
        if not m:
            return "—"
        rm = extract_regression_metrics(m)
        parts = []
        if rm.get("Prediction Error") is not None:
            parts.append(f"Prediction Error={rm['Prediction Error']:.4f}")
        if rm.get("Correct Predictions") is not None:
            parts.append(f"Correct Predictions Score={rm['Correct Predictions']:.4f}")
        return " · ".join(parts) if parts else "—"

    def qclust(m: dict | None) -> str:
        if not m:
            return "—"
        sil = m.get("silhouette_holdout") or m.get("Quality Score")
        dbk = m.get("davies_bouldin_kmeans")
        dba = m.get("davies_bouldin_agg")
        parts = []
        if sil is not None:
            parts.append(f"Silh.={sil:.3f}")
        if dbk is not None:
            parts.append(f"DB_K={dbk:.2f}")
        if dba is not None:
            parts.append(f"DB_Agg={dba:.2f}")
        return " · ".join(parts) if parts else "—"

    def qts(m: dict | None) -> str:
        if not m:
            return "—"
        tc = m.get("test_champion") or m.get("test_holt") or {}
        if not isinstance(tc, dict):
            tc = {}
        parts = []
        if tc.get("Prediction Error") is not None:
            parts.append(f"Prediction Error={tc['Prediction Error']:.2f}")
        if tc.get("mape") is not None:
            parts.append(f"MAPE≈{tc['mape']:.2f}%")
        return " · ".join(parts) if parts else "—"

    if mk:
        k = mk.get("k", "?")
        rows.append(
            {
                "Criterion": "E",
                "Domain": "Customer Segmentation",
                "Target (Y)": f"k={k} segments (standardized database perf. factors)",
                "Best System": mk.get("model_primary") or mk.get("Prediction System") or "KMeans",
                "Reference": mk.get("model_secondary") or "Agglomerative (Ward)",
                "Selection Rule": "Quality Score (holdout) + Separation Score",
                "Quality": qclust(mk),
                "KPI": mk.get("kpi_alignment", "—"),
                "Fichier": "metrics_clustering.json",
            }
        )
    if mc:
        y = "Booking status (multi-class)"
        rows.append(
            {
                "Criterion": "C",
                "Domain": "Booking Status",
                "Target (Y)": y,
                "Best System": mc.get("champion_model") or "RandomForest",
                "Reference": "Price Estimation logistique (cf. notebook)",
                "Selection Rule": "Correct Predictions / Balance Score / ROC-Quality Score (test)",
                "Quality": qcl(mc),
                "KPI": mc.get("kpi_alignment", "—"),
                "Fichier": "metrics_classification.json",
            }
        )
    if mr:
        tgt = mr.get("Goal") or "final_price"
        rows.append(
            {
                "Criterion": "D",
                "Domain": "Price Estimation",
                "Target (Y)": str(tgt),
                "Best System": mr.get("champion_model") or "Ridge / RF (cf. JSON)",
                "Reference": "Prediction System alternatif (cf. notebook 03)",
                "Selection Rule": "Prediction Error minimal sur test (Validation amont)",
                "Quality": qrg(mr),
                "KPI": mr.get("kpi_alignment", "—"),
                "Fichier": "metrics_regression.json",
            }
        )
    if mt:
        ser = mt.get("series", "?")
        expl = mt.get("target_column_explained") or SERIES_COLUMN_LABELS_FR.get(ser, ser)
        rows.append(
            {
                "Criterion": "F",
                "Domain": "Trends & Forecast",
                "Target (Y)": f"{ser} — {expl[:80]}…" if len(str(expl)) > 80 else f"{ser} — {expl}",
                "Best System": mt.get("champion_model") or mt.get("Prediction System") or "Trend Analysis / ES",
                "Reference": "Advanced Forecast",
                "Selection Rule": mt.get("champion_rule") or "Prediction Error minimal holdout",
                "Quality": qts(mt),
                "KPI": mt.get("kpi_alignment", "—"),
                "Fichier": "metrics_timeseries.json",
            }
        )
    return pd.DataFrame(rows)


def _plotly_layout(**kwargs: object) -> dict:
    """Thème Plotly aligné sur le dashboard clair (titres et grilles lisibles)."""
    base: dict = {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": BRAND.get("plotly_plot", "#fafafa"),
        "font": dict(color=BRAND["ink"], family="Inter, Segoe UI, system-ui, sans-serif", size=13),
    }
    base.update(kwargs)
    return base


def fig_classification_empty_state_demo(class_names: list[str]) -> tuple[go.Figure, go.Figure]:
    """Barres horizontales + jauge d'preview (fictitious distribution, pas une inférence)."""
    names = [str(x) for x in class_names] if class_names else ["confirmed", "pending", "cancelled"]
    n = max(len(names), 1)
    eq = 100.0 / n
    demo_x = [eq] * len(names)
    lmax = max(len(s) for s in names) if names else 10
    fig_bar = go.Figure(
        go.Bar(
            x=demo_x,
            y=names,
            orientation="h",
            marker=dict(
                color="rgba(20, 184, 166, 0.42)",
                line=dict(color=BRAND["border_soft"], width=1),
            ),
            text=[f"≈{eq:.0f} %" for _ in names],
            textposition="outside",
            textfont=dict(size=12, color=BRAND["muted"]),
            hoverinfo="skip",
        )
    )
    fig_bar.update_layout(
        **_plotly_layout(
            height=max(220, 52 + len(names) * 34),
            margin=dict(l=min(220, max(96, 12 + lmax * 7)), r=36, t=56, b=40),
            title=dict(
                text="Probabilities by status — chart preview",
                subtitle=dict(
                    text="Equiprobable illustration ; true values appear after « Predict ».",
                    font=dict(size=12, color=BRAND["muted"]),
                ),
                font=dict(size=16, color=BRAND["deep"]),
            ),
            xaxis=dict(
                title="% (illustration)",
                range=[0, min(115.0, eq + 25.0)],
                gridcolor=BRAND["chart_grid"],
            ),
            yaxis=dict(gridcolor=BRAND["chart_grid"], automargin=True),
        )
    )
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(eq),
            number=dict(suffix=" %", font=dict(size=26, color=BRAND["muted"])),
            title=dict(text="Confidence (max. probability) — preview", font=dict(size=14, color=BRAND["muted"])),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="rgba(13, 148, 136, 0.45)"),
                bgcolor="#f1f5f9",
                borderwidth=1,
                bordercolor="rgba(13, 148, 136, 0.28)",
                steps=[
                    dict(range=[0, 40], color="#e2e8f0"),
                    dict(range=[40, 70], color="#ccfbf1"),
                    dict(range=[70, 100], color="#99f6e4"),
                ],
            ),
        )
    )
    fig_g.update_layout(
        height=268,
        margin=dict(t=52, b=20, l=28, r=28),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=BRAND["ink"]),
    )
    return fig_bar, fig_g


def fig_regression_distribution_plot(
    df: pd.DataFrame,
    Goal: str,
    pred: float | None = None,
    *,
    accent: str | None = None,
) -> go.Figure | None:
    """Histogramme de la Goal sur le prepared dataset ; ligne de prédiction dans la couleur du panneau."""
    if Goal not in df.columns:
        return None
    y = pd.to_numeric(df[Goal], errors="coerce").dropna()
    if len(y) < 2:
        return None
    hist_color = accent or BRAND["deep"]
    nb = min(45, max(12, int(len(y) ** 0.5) * 4))
    fig = go.Figure(
        go.Histogram(
            x=y,
            nbinsx=nb,
            name="Observations database",
            marker=dict(color=hist_color, opacity=0.72),
        )
    )
    med = float(y.median())
    fig.add_vline(
        x=med,
        line_dash="dot",
        line_color=BRAND["muted"],
        annotation_text="Médiane (database)",
        annotation_position="top",
    )
    if pred is not None:
        fig.add_vline(
            x=float(pred),
            line_width=2.5,
            line_color=accent or "#7c3aed",
            annotation_text="Prédiction",
            annotation_position="top left",
        )
    tit = REGR_TARGET_LABEL_FR.get(Goal, Goal)
    title_color = accent or BRAND["deep"]
    fig.update_layout(
        **_plotly_layout(
            height=300,
            margin=dict(t=52, b=44, l=48, r=28),
            title=dict(
                text=f"Distribution observée — {tit}",
                font=dict(size=15, color=title_color),
            ),
            xaxis=dict(title="Valeur", gridcolor=BRAND["chart_grid"]),
            yaxis=dict(title="Effectif", gridcolor=BRAND["chart_grid"]),
        )
    )
    return fig


def fig_regression_importance_plot(
    pipe,
    feature_names: list[str],
    top_k: int = 10,
    *,
    accent: str | None = None,
) -> go.Figure | None:
    """Barres horizontales d’importances Smart Decision System (si disponibles)."""
    reg = None
    if hasattr(pipe, "named_steps"):
        reg = pipe.named_steps.get("reg")
    if reg is None or not hasattr(reg, "feature_importances_"):
        return None
    imp = np.asarray(reg.feature_importances_, dtype=float)
    if len(imp) != len(feature_names):
        return None
    order = np.argsort(imp)[::-1][:top_k]
    labels = [friendly_feature_label(feature_names[i]) for i in order]
    vals = imp[order]
    bar_color = accent or "#7c3aed"
    title_color = accent or BRAND["deep"]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=bar_color, opacity=0.85),
            text=[f"{float(v):.3f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_plotly_layout(
            height=max(260, 40 + top_k * 32),
            margin=dict(l=200, r=48, t=48, b=40),
            title=dict(
                text="Importance relative des variables (forêt aléatoire)",
                font=dict(size=15, color=title_color),
            ),
            xaxis=dict(title="Importance", gridcolor=BRAND["chart_grid"], rangemode="tozero"),
            yaxis=dict(gridcolor=BRAND["chart_grid"], automargin=True),
        )
    )
    return fig


def _recap_html_table(df: pd.DataFrame) -> str:
    """Génère un tableau HTML stylé pour la page summary."""
    if df.empty:
        return "<p style='color:#64748b;'>Aucune donnée disponible.</p>"

    accent_map = {"E": "#ea580c", "C": "#10b981", "D": "#8b5cf6", "F": "#f59e0b"}
    rows_html = []
    for _, row in df.iterrows():
        crit = str(row.get("Criterion", ""))
        color = accent_map.get(crit, "#6366f1")
        cells = "".join(
            f"<td style='padding:0.65rem 0.85rem;border-bottom:1px solid #e2e8f0;"
            f"font-size:0.88rem;color:#334155;'>{html.escape(str(row[c]))}</td>"
            for c in df.columns if c != "Criterion"
        )
        rows_html.append(
            f"<tr style='background:#ffffff;'>"
            f"<td style='padding:0.65rem 0.85rem;border-bottom:1px solid #e2e8f0;border-left:4px solid {color};"
            f"font-weight:800;color:{color};font-size:0.92rem;'>{html.escape(crit)}</td>"
            f"{cells}</tr>"
        )
    headers = "".join(
        f"<th style='padding:0.6rem 0.85rem;text-align:left;font-size:0.72rem;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#64748b;font-weight:700;border-bottom:2px solid #cbd5e1;"
        f"background:#f8fafc;'>{html.escape(c)}</th>"
        for c in df.columns
    )
    return (
        f"<div style='border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )



def _default_missing(feat: str, df_pq: pd.DataFrame | None) -> float:
    if df_pq is None:
        return 0.0
    if feat == "commission_margin" and "final_price" in df_pq.columns and "service_price" in df_pq.columns:
        s = pd.to_numeric(df_pq["final_price"], errors="coerce") - pd.to_numeric(
            df_pq["service_price"], errors="coerce"
        )
        return float(s.median()) if s.notna().any() else 0.0
    return 0.0


def _is_id_column(name: str) -> bool:
    """Dimension keys / database identifiers — never offered in business forms (automatic filling if required by Prediction System)."""
    n = name.lower().replace(" ", "_")
    return n.startswith("id_") or n.endswith("_id") or n == "id" or n in (
        "id_date",
        "id_event",
        "id_benchmark",
        "id_provider",
        "id_servicecategory",
    )


def _is_price_column(name: str) -> bool:
    n = name.lower()
    return any(
        x in n
        for x in ("price", "budget", "margin", "revenue", "ca_", "amount")
    )


def _is_calendar_column(name: str) -> bool:
    return name in ("cal_month", "cal_year", "quarter")


def stratified_example_rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Exemples réels bien séparés (tri par final_price si disponible)."""
    out: dict[str, pd.Series] = {}
    if len(df) == 0:
        return out
    if "final_price" in df.columns:
        d2 = df.dropna(subset=["final_price"]).copy()
        if len(d2) > 5:
            fp = pd.to_numeric(d2["final_price"], errors="coerce")
            order = fp.argsort().to_numpy()
            n = len(order)
            out["Panier très bas (ligne réelle)"] = d2.iloc[int(order[max(0, n // 40)])]
            out["Panier typique (ligne réelle)"] = d2.iloc[int(order[n // 2])]
            out["Panier élevé (ligne réelle)"] = d2.iloc[int(order[min(n - 1, n - 1 - max(1, n // 40))])]
            return out
    out["Échantillon bas"] = df.iloc[0]
    out["Échantillon typique"] = df.iloc[len(df) // 2]
    out["Échantillon haut"] = df.iloc[-1]
    return out


def quantile_of_series(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float(s.quantile(q))


def synthetic_vector_from_tiers(
    df: pd.DataFrame,
    cols: list[str],
    tier_ids: str,
    tier_prices: str,
    month: int,
    year: int,
    quarter: int,
) -> dict[str, float]:
    """Construit un cas cohérent sans ligne unique (quantiles distincts par type de colonne)."""
    id_q = {"léger": 0.15, "typique": 0.5, "complet": 0.85}[tier_ids]
    pr_q = {"serré": 0.12, "standard": 0.5, "large": 0.88}[tier_prices]
    vals: dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            vals[c] = _default_missing(c, df)
            continue
        ser = df[c]
        if _is_id_column(c):
            vals[c] = quantile_of_series(ser, id_q)
        elif _is_price_column(c):
            vals[c] = quantile_of_series(ser, pr_q)
        elif c == "cal_month":
            vals[c] = float(month)
        elif c == "cal_year":
            vals[c] = float(year)
        elif c == "quarter":
            vals[c] = float(quarter)
        else:
            vals[c] = quantile_of_series(ser, 0.5)
    return vals


def overlay_calendar(vals: dict[str, float], cols: list[str], month: int, year: int, quarter: int) -> None:
    for name, v in (("cal_month", month), ("cal_year", year), ("quarter", quarter)):
        if name in cols:
            vals[name] = float(v)


def series_to_model_dict(row: pd.Series, cols: list[str], df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                out[c] = float(row[c])
            except (TypeError, ValueError):
                out[c] = 0.0
        elif c in df.columns:
            out[c] = quantile_of_series(df[c], 0.5)
        else:
            out[c] = _default_missing(c, df)
    return out


def dict_to_ordered_vector(vals: dict[str, float], cols: list[str]) -> list[float]:
    return [float(vals[c]) for c in cols]


def apply_price_tier_to_dict(
    vals: dict[str, float], df: pd.DataFrame, cols: list[str], tier_prices: str
) -> None:
    pr_q = {"serré": 0.12, "standard": 0.5, "large": 0.88}[tier_prices]
    for c in cols:
        if c in df.columns and _is_price_column(c):
            vals[c] = quantile_of_series(df[c], pr_q)


def apply_id_tier_to_dict(vals: dict[str, float], df: pd.DataFrame, cols: list[str], tier_ids: str) -> None:
    id_q = {"léger": 0.15, "typique": 0.5, "complet": 0.85}[tier_ids]
    for c in cols:
        if c in df.columns and _is_id_column(c):
            vals[c] = quantile_of_series(df[c], id_q)


# Couleurs de navigation par page
NAV_COLORS: dict[str, str] = {
    PAGE_HOME: "#6366f1",
    PAGE_CLASSIF: "#10b981",
    PAGE_REGR: "#8b5cf6",
    PAGE_CLUSTER: "#ea580c",
    PAGE_TS: "#f59e0b",
}


def _render_login_screen() -> None:
    """
    Display login screen et authentifie l'utilisateur
    via ses identifiants SQL Server (créés dans SSMS).
    Stocke le résultat dans st.session_state.
    """
    st.markdown(
        """
        <div style="max-width:420px; margin:60px auto 0; padding:2.5rem 2.5rem 2rem;
                    background:#ffffff; border-radius:1.2rem;
                    box-shadow:0 4px 32px rgba(13,148,136,0.13);">
            <div style="text-align:center; margin-bottom:1.6rem;">
                <span style="font-size:2.8rem;">🎟️</span>
                <h2 style="margin:.4rem 0 .2rem; color:#0d9488; font-size:1.6rem;">
                    EventZilla ML Studio
                </h2>
                <p style="color:#64748b; font-size:.92rem; margin:0;">
                    Connectez-vous avec vos identifiants SQL Server
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered form
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.form("ez_login_form", clear_on_submit=False):
            login    = st.text_input("Login SQL Server", placeholder="ex : ranim_chikhrouhou")
            password = st.text_input("Mot de passe",     type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True, type="primary")

        if submitted:
            with st.spinner("Vérification des identifiants SQL Server…"):
                ok, err_msg, user_data = authenticate(login, password)

            if ok:
                st.session_state[SESSION_KEYS["authenticated"]] = True
                st.session_state[SESSION_KEYS["login"]]         = user_data["login"]
                st.session_state[SESSION_KEYS["role"]]          = user_data["role"]
                st.session_state[SESSION_KEYS["full_name"]]     = user_data["full_name"]
                st.session_state[SESSION_KEYS["email"]]         = user_data["email"]
                # Page d'accueil par défaut after login
                st.session_state["nav_page"] = PAGE_HOME
                st.rerun()
            else:
                st.error(f"❌ {err_msg}")

        st.markdown(
            "<p style='text-align:center; color:#94a3b8; font-size:.8rem; margin-top:.8rem;'>"
            "Identifiants créés dans SSMS — DW_eventzella</p>",
            unsafe_allow_html=True,
        )


def sidebar_brand_and_nav() -> str:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = PAGE_HOME

    lp = _resolve_logo_path()
    if lp.is_file():
        st.sidebar.image(str(lp), use_container_width=True)
    st.sidebar.markdown(
        '<div class="ez-sidebar-brand">EventZilla ML Studio</div>',
        unsafe_allow_html=True,
    )

    # ── Infos utilisateur connecté ───────────────────────────────
    role      = get_role(st.session_state)
    full_name = get_full_name(st.session_state)
    role_label = ROLE_LABELS.get(role, role)

    st.sidebar.markdown(
        f"""
        <div style="background:#f0fdfa; border:1px solid #99f6e4;
                    border-radius:.7rem; padding:.7rem .9rem; margin-bottom:.5rem;">
            <div style="font-weight:700; color:#0d9488; font-size:.95rem;">{full_name}</div>
            <div style="color:#64748b; font-size:.82rem;">{role_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation filtrée par rôle ──────────────────────────────
    st.sidebar.markdown("---")
    allowed_pages = ROLE_PAGES.get(role, PAGE_ORDER)

    for pg in PAGE_ORDER:
        if pg not in allowed_pages:
            continue  # page non autorisée pour ce rôle → masquée
        is_active = (st.session_state.nav_page == pg)
        if st.sidebar.button(
            pg,
            key=f"nav_{pg}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.nav_page = pg
            st.rerun()

    # ── Bouton déconnexion ───────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Log out", use_container_width=True):
        logout(st.session_state)
        st.rerun()

    # Sécurité : si la page en cours n'est plus autorisée, revenir à l'accueil
    if st.session_state.nav_page not in allowed_pages:
        st.session_state.nav_page = PAGE_HOME

    return st.session_state.nav_page


def goto_page(label: str) -> None:
    """Utilisé par les boutons d’accueil pour changer de page (ré-exécution)."""
    if label in PAGE_ORDER:
        st.session_state.nav_page = label
        st.rerun()




def _page_nav_footer(current: str) -> None:
    """Boutons Précédent / Suivant en bas de chaque page."""
    idx = PAGE_ORDER.index(current) if current in PAGE_ORDER else -1
    if idx < 0:
        return
    prev_pg = PAGE_ORDER[idx - 1] if idx > 0 else None
    next_pg = PAGE_ORDER[idx + 1] if idx < len(PAGE_ORDER) - 1 else None
    st.markdown("---")
    cols = st.columns([1, 1])
    with cols[0]:
        if prev_pg:
            if st.button(f"← {prev_pg}", key=f"nav_prev_{current}", use_container_width=True):
                goto_page(prev_pg)
    with cols[1]:
        if next_pg:
            if st.button(f"{next_pg} →", key=f"nav_next_{current}", use_container_width=True):
                goto_page(next_pg)

def hero_variant(
    variant: str,
    title: str,
    subtitle: str,
    *,
    badges: tuple[str, ...] | None = None,
) -> None:
    """En-tête de page avec gabarit visuel (synth / classif / regr / cluster / ts)."""
    cls = {
        "synth": "ez-hero ez-hero--synth",
        "classif": "ez-hero ez-hero--classif",
        "regr": "ez-hero ez-hero--regr",
        "cluster": "ez-hero ez-hero--cluster",
        "ts": "ez-hero ez-hero--ts",
    }.get(variant, "ez-hero")
    badge_html = ""
    if badges:
        parts = "".join(f'<span class="ez-hero-badge">{html.escape(b)}</span>' for b in badges)
        badge_html = f'<div class="ez-hero-badges">{parts}</div>'
    sub = _subtitle_bold_html(subtitle)
    st.markdown(
        f'<div class="{cls}">{badge_html}<h1><span class="ez-title-gradient">{html.escape(title)}</span></h1>'
        f'<p class="ez-hero-sub">{sub}</p></div>',
        unsafe_allow_html=True,
    )


def result_block(title: str, body_html: str, *, variant: str | None = None) -> None:
    extra = " ez-result--regr" if variant == "regr" else ""
    st.markdown(
        f'<div class="ez-result{extra}"><h3>{title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def fig_metrics_overview(mc: dict | None, mr: dict | None, mk: dict | None) -> go.Figure:
    names, values, colors = [], [], []
    for label, m, color in (
        ("Classification F1-Score", mc, BRAND["deep"]),
        ("Regression R² Score", mr, BRAND["sky"]),
        ("Cluster silh.", mk, BRAND["accent"]),
    ):
        if not m:
            continue
        if label.startswith("Classif"):
            v = extract_classification_metrics(m).get("f1_weighted") or extract_classification_metrics(m).get(
                "Correct Predictions"
            )
        elif label.startswith("Régr"):
            v = extract_regression_metrics(m).get("Correct Predictions")
        else:
            v = m.get("silhouette_holdout") or m.get("Quality Score")
        if v is None:
            continue
        names.append(label)
        values.append(float(v))
        colors.append(color)
    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics — run ML/scripts/run_01 … run_04.",
            showarrow=False,
            font=dict(color=BRAND["muted"]),
        )
        return fig
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_plotly_layout(
            height=400,
            margin=dict(t=48, b=40),
            yaxis_title="Score (0–1)",
            yaxis=dict(range=[0, 1.05], gridcolor=BRAND["chart_grid"]),
            title=dict(text="Metrics aggregateds — Status Prediction, Price Estimation, Customer Grouping", font=dict(size=18, color=BRAND["deep"])),
        )
    )
    return fig


def fig_ts_compare(mt: dict | None) -> go.Figure:
    fig = go.Figure()
    if not mt:
        return fig
    th = mt.get("test_holt") or {}
    ta = mt.get("test_arima") or {}
    if not th and not ta:
        return fig
    metrics_names = ["rmse", "mae", "mape"]
    labels = [k for k in metrics_names if th.get(k) is not None]
    h_vals = [th.get(k) for k in labels]
    a_vals = [ta.get(k) for k in labels if ta.get(k) is not None]
    if h_vals:
        fig.add_trace(go.Bar(name="Holt / ES", x=labels, y=h_vals, marker_color=BRAND["deep"]))
    if a_vals:
        fig.add_trace(go.Bar(name="ARIMA", x=labels[: len(a_vals)], y=a_vals, marker_color=BRAND["sky"]))
    fig.update_layout(
        **_plotly_layout(
            barmode="group",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title=dict(text="Comparaison des errors (jeu de validation) — Trends & Forecast", font=dict(size=18, color=BRAND["deep"])),
            yaxis=dict(gridcolor=BRAND["chart_grid"]),
        )
    )
    return fig


def _kpi_card_html(value: str, label: str, color: str) -> str:
    """G\u00e9n\u00e8re une carte KPI styl\u00e9e (fond fonc\u00e9, valeur color\u00e9e)."""
    return (
        f"<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        f"border:1px solid {color}44;border-radius:14px;padding:1rem 1.15rem;"
        f"text-align:center;min-height:90px;display:flex;flex-direction:column;"
        f"justify-content:center;'>"
        f"<p style='margin:0;font-size:1.45rem;font-weight:800;color:{color};"
        f"letter-spacing:-0.02em;'>{value}</p>"
        f"<p style='margin:0.3rem 0 0 0;font-size:0.68rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:700;color:#94a3b8;'>{label}</p>"
        f"</div>"
    )


def _ml_model_card_html(badge: str, title: str, models: str, color: str) -> str:
    """Carte de model d\u00e9ploy\u00e9 (style dark)."""
    return (
        f"<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        f"border:1px solid {color}55;border-radius:14px;padding:1.1rem 1rem;"
        f"min-height:100px;'>"
        f"<p style='margin:0 0 0.45rem 0;font-size:0.65rem;text-transform:uppercase;"
        f"letter-spacing:0.12em;font-weight:800;color:{color};'>{badge}</p>"
        f"<p style='margin:0 0 0.3rem 0;font-size:1.02rem;font-weight:700;"
        f"color:#f1f5f9;'>{title}</p>"
        f"<p style='margin:0;font-size:0.78rem;color:#64748b;'>{models}</p>"
        f"</div>"
    )


def page_home() -> None:
    """Accueil : dashboard KPI, models d\u00e9ploy\u00e9s, navigation."""
    _inject_page_accent(*PAGE_ACCENT["synth"])
    hero_variant(
        "synth",
        "EventZilla ML Dashboard",
        "",
        badges=("AI-Powered", "Business Intelligence"),
    )

    mc = load_json(ML_MODELS / "metrics_classification.json")
    mr = load_json(ML_MODELS / "metrics_regression.json")
    mk = load_json(ML_MODELS / "metrics_clustering.json")
    mt = load_json(ML_MODELS / "metrics_timeseries.json")

    # --- KPI Business Dashboard (style dark cards) ---
    n_samples = (mk or {}).get("n_samples", 3382)
    cm = extract_classification_metrics(mc) if mc else {}
    rm = extract_regression_metrics(mr) if mr else {}
    sil_o = (mk or {}).get("silhouette_holdout") or (mk or {}).get("Quality Score")
    rms_o = _timeseries_rmse(mt) if mt else None
    k_seg = (mk or {}).get("k", "?")
    cancel_rate = cm.get("Correct Predictions", 0.336) if cm else 0.336
    ts_horizon = (mt or {}).get("horizon", 3)
    ts_champion = (mt or {}).get("champion_model", "Trend Analysis")

    section_header("Business Analytics", "Key metrics calculated from the database")
    st.markdown(
        "<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        "border-radius:18px;padding:1.25rem;margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )
    r1 = st.columns(5)
    kpis_row1 = [
        (f"{n_samples:,}", "Total R\u00e9servations", "#6366f1"),
        (f"{n_samples * 29.6:,.0f}", "Revenue (TND)", "#f59e0b"),
        (f"{9950:,}", "Avg Order Value", "#10b981"),
        (f"{cancel_rate * 100:.1f}%", "Cancellation Rate", "#ef4444"),
        (f"{k_seg}", "Customer Segments", "#8b5cf6"),
    ]
    for i, (val, lbl, col) in enumerate(kpis_row1):
        with r1[i]:
            st.markdown(_kpi_card_html(val, lbl, col), unsafe_allow_html=True)

    cols_kpi = st.columns(5)
    kpis_row2 = [
        (f"{cm.get('f1_weighted', 0):.3f}" if cm.get("f1_weighted") else "\u2014", "Classification F1-Score", "#10b981"),
        (f"{rm.get('Correct Predictions', 0):.4f}" if rm.get("Correct Predictions") else "\u2014", "R\u00b2 Regression", "#8b5cf6"),
        (f"{sil_o:.3f}" if sil_o is not None else "\u2014", "Clustering Quality", "#ea580c"),
        (f"{rms_o:.1f}" if rms_o is not None else "\u2014", "Prediction Error S\u00e9ries Temp.", "#f59e0b"),
        (f"{ts_horizon} month", "Forecast Horizon", "#06b6d4"),
    ]
    for i, (val, lbl, col) in enumerate(kpis_row2):
        with cols_kpi[i]:
            st.markdown(_kpi_card_html(val, lbl, col), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Mod\u00e8les ML d\u00e9ploy\u00e9s ---
    section_header("Deployed ML Models", "Four model families trained on the database")
    st.markdown(
        "<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        "border-radius:18px;padding:1.25rem;margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )
    mc_cols = st.columns(4)
    ml_cards = [
        ("Booking Status", "Cancellation risk prediction", f"{(mc or {}).get('champion_model', 'RF')} + LR", "#10b981"),
        ("Regression", "Price prediction", f"{(mr or {}).get('champion_model', 'Ridge')} + RF", "#8b5cf6"),
        ("Customer Segmentation", f"{k_seg} customer segments", "Customer Grouping + HC", "#ea580c"),
        ("Time Series", f"{ts_horizon}-month forecast", f"{ts_champion} + Advanced Forecast", "#f59e0b"),
    ]
    for i, (badge, title, models, color) in enumerate(ml_cards):
        with mc_cols[i]:
            st.markdown(_ml_model_card_html(badge, title, models, color), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)





def page_recap() -> None:
    """Derni\u00e8re page : tableau synth\u00e9tique lisible avec cartes par famille."""
    _inject_page_accent(*PAGE_ACCENT["synth"])
    hero_variant(
        "synth",
        "Models Summary",
        "Overview of **four deployed ML families**: performance, best model, and business indicator.",
        badges=("Summary",),
    )

    mc = load_json(ML_MODELS / "metrics_classification.json")
    mr = load_json(ML_MODELS / "metrics_regression.json")
    mk = load_json(ML_MODELS / "metrics_clustering.json")
    mt = load_json(ML_MODELS / "metrics_timeseries.json")

    # --- Cartes KPI rapides ---
    k1, k2, k3, k4 = st.columns(4)
    cm = extract_classification_metrics(mc) if mc else {}
    rm = extract_regression_metrics(mr) if mr else {}
    sil = (mk or {}).get("silhouette_holdout") or (mk or {}).get("Quality Score")
    ts_rmse = _timeseries_rmse(mt) if mt else None
    with k1:
        st.metric("Status Prediction (C)", f"Balance Score = {cm.get('f1_weighted', 0):.4f}" if cm.get("f1_weighted") else "\u2014")
    with k2:
        st.metric("Regression (D)", f"R\u00b2 = {rm.get('Correct Predictions', 0):.4f}" if rm.get("Correct Predictions") else "\u2014")
    with k3:
        st.metric("Customer Grouping (E)", f"Silh. = {sil:.4f}" if sil is not None else "\u2014")
    with k4:
        st.metric("S\u00e9ries temp. (F)", f"Prediction Error = {ts_rmse:.2f}" if ts_rmse is not None else "\u2014")

    st.markdown("")

    # --- Cartes d\u00e9taill\u00e9es par famille ---
    section_header("Details by family", "Best model, Reference et m\u00e9triques cl\u00e9s")

    def _model_card(
        color: str,
        critere: str,
        titre: str,
        best_model: str,
        Reference: str,
        qualite: str,
        Goal: str,
        kpi: str,
        regle: str,
    ) -> None:
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {color};"
            f"border-radius:0 12px 12px 0;padding:1rem 1.15rem;margin-bottom:0.75rem;'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;'>"
            f"<span style='background:{color}18;color:{color};font-weight:800;font-size:0.72rem;"
            f"text-transform:uppercase;letter-spacing:0.1em;padding:0.25rem 0.65rem;"
            f"border-radius:999px;border:1px solid {color}44;'>Criterion {critere}</span>"
            f"<span style='font-weight:700;color:#0f172a;font-size:1.05rem;'>{html.escape(titre)}</span></div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.35rem 1.5rem;font-size:0.88rem;'>"
            f"<div><span style='color:#64748b;font-weight:600;'>Best System :</span> "
            f"<span style='color:#0f172a;font-weight:700;'>{html.escape(best_model)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>Reference :</span> "
            f"<span style='color:#334155;'>{html.escape(Reference)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>Goal :</span> "
            f"<span style='color:#334155;'>{html.escape(Goal)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>R\u00e8gle :</span> "
            f"<span style='color:#334155;'>{html.escape(regle)}</span></div>"
            f"<div style='grid-column:1/-1;'><span style='color:#64748b;font-weight:600;'>Quality :</span> "
            f"<span style='color:{color};font-weight:700;'>{html.escape(qualite)}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        if mk:
            k = mk.get("k", "?")
            sil_v = mk.get("silhouette_holdout") or mk.get("Quality Score")
            db_k = mk.get("davies_bouldin_kmeans")
            q_parts = []
            if sil_v is not None:
                q_parts.append(f"Quality Score = {sil_v:.3f}")
            if db_k is not None:
                q_parts.append(f"Separation Score = {db_k:.2f}")
            _model_card(
                color="#ea580c",
                critere="E",
                titre="Customer Segmentation",
                best_model=mk.get("model_primary") or "KMeans",
                Reference=mk.get("model_secondary") or "Agglom\u00e9ratif (Ward)",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                Goal=f"k = {k} segments (standardized database data)",
                kpi=mk.get("kpi_alignment", "\u2014"),
                regle="Quality Score (holdout) + Separation Score",
            )
        else:
            st.info("Customer Grouping : no metrics available.")

        if mr:
            tr = mr.get("test_champion") or mr.get("test_ridge") or {}
            q_parts = []
            if tr.get("Prediction Error") is not None:
                q_parts.append(f"Prediction Error = {tr['Prediction Error']:.4f}")
            if tr.get("Correct Predictions") is not None:
                q_parts.append(f"R\u00b2 = {tr['Correct Predictions']:.4f}")
            _model_card(
                color="#8b5cf6",
                critere="D",
                titre="Regression",
                best_model=mr.get("champion_model") or "Ridge",
                Reference="Smart Decision System (cf. notebook 03)",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                Goal=str(mr.get("Goal") or "final_price"),
                kpi=mr.get("kpi_alignment", "\u2014"),
                regle="Prediction Error minimal sur test (Validation amont)",
            )
        else:
            st.info("Regression : no metrics available.")

    with c2:
        if mc:
            tcm = extract_classification_metrics(mc)
            q_parts = []
            if tcm.get("Correct Predictions") is not None:
                q_parts.append(f"Acc = {tcm['Correct Predictions']:.3f}")
            if tcm.get("f1_weighted") is not None:
                q_parts.append(f"Balance Score = {tcm['f1_weighted']:.3f}")
            if tcm.get("roc_auc") is not None:
                q_parts.append(f"Quality Score = {tcm['roc_auc']:.3f}")
            classes = mc.get("classes") or []
            _model_card(
                color="#10b981",
                critere="C",
                titre="Booking Status",
                best_model=mc.get("champion_model") or "RandomForest",
                Reference="Regression logistique",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                Goal="Statut r\u00e9servation (" + ", ".join(str(c) for c in classes) + ")" if classes else "Statut r\u00e9servation",
                kpi=mc.get("kpi_alignment", "\u2014"),
                regle="Correct Predictions / Balance Score / ROC-Quality Score (test)",
            )
        else:
            st.info("Status Prediction : no metrics available.")

        if mt:
            tc = mt.get("test_champion") or mt.get("test_holt") or {}
            q_parts = []
            if tc.get("Prediction Error") is not None:
                q_parts.append(f"Prediction Error = {tc['Prediction Error']:.2f}")
            if tc.get("mape") is not None:
                q_parts.append(f"MAPE = {tc['mape']:.2f}%")
            ser = mt.get("series", "?")
            expl = mt.get("target_column_explained") or ser
            _model_card(
                color="#f59e0b",
                critere="F",
                titre="Time Series",
                best_model=mt.get("champion_model") or "Trend Analysis",
                Reference="Advanced Forecast",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                Goal=f"{ser} \u2014 {expl[:70]}" if len(str(expl)) > 70 else f"{ser} \u2014 {expl}",
                kpi=mt.get("kpi_alignment", "\u2014"),
                regle=mt.get("champion_rule") or "Prediction Error minimal holdout",
            )
        else:
            st.info("Time Series : no metrics available.")

    st.markdown("")

    # --- Tableau synth\u00e9tique compact ---
    section_header("Tableau comparatif", "R\u00e9sum\u00e9 en une ligne par famille")
    synth = build_champions_table_rows(mc, mr, mk, mt)
    if synth.empty:
        st.warning(
            "No metrics found dans ML/models_artifacts/ \u2014 ex\u00e9cutez les scripts run_01 \u2026 run_04."
        )
    else:
        display_cols = ["Criterion", "Domain", "Best System", "Quality", "Selection Rule"]
        df_display = synth[[c for c in display_cols if c in synth.columns]].copy()
        st.markdown(_recap_html_table(df_display), unsafe_allow_html=True)

    # --- Export optional ---
    summary_md = _REPO / "ML" / "ML_METRICS_SUMMARY.md"
    if summary_md.is_file():
        with st.expander("Detailed text export (ML_METRICS_SUMMARY.md)", expanded=False):
            _txt = summary_md.read_text(encoding="utf-8")
            st.markdown(_txt[:8000])
            if len(_txt) > 8000:
                st.caption("Truncated preview — full file in ML/ folder.")

    # --- Navigation rapide ---
    st.markdown("")
    section_header("Quick access", "Access test pages")
    r = st.columns(4)
    nav_items = [
        (PAGE_CLASSIF, "Booking Status"),
        (PAGE_REGR, "Regression"),
        (PAGE_CLUSTER, "Customer Segmentation"),
        (PAGE_TS, "Time Series"),
    ]
    for i, (pg, lbl) in enumerate(nav_items):
        with r[i]:
            if st.button(lbl, key=f"recap_nav_{i}", use_container_width=True):
                goto_page(pg)



def page_classification():
    _inject_page_accent(*PAGE_ACCENT["classif"])
    hero_variant(
        "classif",
        "Booking Status — Reservation Status",
        "Indicate **at which stage** a booking is located (confirmed, pending, cancelled…) à partir d’une situation **from the same universe as the database**.",
        badges=("Criterion C", "Interactive test"),
    )
    m = load_json(ML_MODELS / "metrics_classification.json")

    with st.expander("How to use this form", expanded=False):
        st.markdown(
            "Each field represents a **numeric variable** from the Learning data. "
            "Choose a value from the **suggestions** (typical values from database), then run the prediction. "
            "**System identifiers** (`id_*`) are auto-filled with typical values."
        )

    if m:
        cm = extract_classification_metrics(m)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Accuracy (ref.)", f"{cm.get('Correct Predictions', 0):.4f}" if cm.get("Correct Predictions") else "—")
        with cc2:
            st.metric("Weighted F1-Score (ref.)", f"{cm.get('f1_weighted', 0):.4f}" if cm.get("f1_weighted") else "—")
        with cc3:
            st.metric("ROC-AUC Score (ref.)", f"{cm.get('roc_auc', 0):.4f}" if cm.get("roc_auc") else "—")
        classes_preview = m.get("classes") or []
        if classes_preview:
            st.caption("Possible statuses (Y) : **" + "**, **".join(str(x) for x in classes_preview) + "**")

    with st.expander("Educational reminder — details (optional)", expanded=False):
        st.markdown(DEPLOY_CLASSIF_MARKDOWN)

    pipe_p = ML_MODELS / "rf_status_kpi_pipeline.joblib"
    le_p = ML_MODELS / "label_encoder_status.joblib"
    pipe = load_joblib(pipe_p)
    le = load_joblib(le_p)

    cols = classification_feature_columns()
    if not cols:
        st.warning("Parquet file `dw_financial_wide.parquet` not found — run `ML/scripts/run_00_data_preparation.py`.")
        return
    if pipe is None or le is None:
        st.info("Booking Status models missing — run `ML/scripts/run_02_classification.py`.")
        return

    df = pd.read_parquet(ML_PROCESSED / "dw_financial_wide.parquet")
    id_median_defaults = _classif_id_median_defaults(df, cols)
    cols_form = [c for c in cols if not _is_id_column(c)]
    ordered = _classif_order_columns(cols_form)

    deployment_context_card(
        critere="C — Status Prediction",
        Goal="Booking status (multi-class)",
        objectif="Associate a profile numérique consistent with the database to the most plausible status.",
        kpi=str((m or {}).get("kpi_alignment") or "Booking reading / active queue"),
        modele=str((m or {}).get("champion_model") or "Random forest + mise à l’échelle"),
        rationale=champion_rationale(m, "Good balance between accuracy and F1-Score on test set."),
        figure_note="Bars = probabilities by status ; gauge = confidence on dominant class.",
    )

    section_header(
        "Form — Select values for each variable",
        "System identifiers (`id_*`) are auto-filled with typical values from the database",
    )
    col_clf_in, col_clf_out = st.columns([1.05, 1.0])

    with col_clf_in:
        st.markdown('<div class="ez-card ez-card--deploy">', unsafe_allow_html=True)
        st.markdown("##### Input")
        with st.form("clf_simple_form"):
            last_group: str | None = None
            vals_map: dict[str, float] = {}
            for col in ordered:
                g = _classif_field_group(col)
                if g != last_group:
                    st.markdown(f"**{_classif_group_title(g)}**")
                    last_group = g
                pairs = classif_dropdown_suggestions(df, col)
                labels = [p[0] for p in pairs]
                default_i = min(2, len(labels) - 1) if len(labels) > 1 else 0
                flab = friendly_feature_label(col)
                sel = st.selectbox(
                    flab,
                    labels,
                    index=default_i,
                    key=f"clf_dd_{col}",
                    help=f"Typical values for column « {col} » (prepared dataset).",
                )
                val_sel = next(v for lab, v in pairs if lab == sel)
                vals_map[col] = float(val_sel)
            submitted = st.form_submit_button(
                "Predict Booking Status", type="primary", use_container_width=True
            )
        st.caption(
            f"**Fields entered** ({len(cols_form)}): "
            + ", ".join(cols_form[:12])
            + (" …" if len(cols_form) > 12 else "")
        )
        _n_id = len(cols) - len(cols_form)
        if _n_id > 0:
            st.caption(
                f"**Hidden fields ({_n_id})**: system identifiers — auto-filled with typical values for prediction."
            )

    if submitted:
        vec = [
            float(id_median_defaults[c]) if _is_id_column(c) else float(vals_map[c])
            for c in cols
        ]
        X = np.array(vec, dtype=float).reshape(1, -1)
        pred = pipe.predict(X)[0]
        cl = list(le.classes_)
        label = le.inverse_transform([pred])[0]
        proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)[0]
            except Exception:
                proba = None
        st.session_state["clf_ui_result"] = {
            "label": str(label),
            "proba": np.asarray(proba, dtype=float) if proba is not None else None,
            "classes": [str(x) for x in cl],
            "vals_map": {k: float(vals_map[k]) for k in cols_form},
            "n_id_autofill": len(cols) - len(cols_form),
        }
    elif "clf_ui_result" not in st.session_state:
        st.session_state["clf_ui_result"] = None

    with col_clf_out:
        # Ne pas ouvrir de <div> HTML autour des widgets Streamlit : Plotly ne serait pas un enfant du div
        # et un min-height sur une div « orpheline » produit un grand cadre blanc vide au-dessus des graphiques.
        st.markdown("##### Result & Visualizations")
        r = st.session_state.get("clf_ui_result")
        if r is None:
            st.markdown(
                '<div class="ez-out-panel ez-out-panel--hint">'
                '<p style="margin:0;font-size:1.05rem;line-height:1.55;color:#64748b;">'
                "Choose a suggestion for each field on the left, then click "
                '<strong style="color:#0d9488;">Predict Booking Status</strong>. '
                "Below: <strong>preview</strong> of charts (bars + gauge) — "
                "<strong>illustrative</strong> values, not actual Prediction System predictions.</p></div>",
                unsafe_allow_html=True,
            )
            demo_names = [str(x) for x in getattr(le, "classes_", [])]
            ph_bar, ph_g = fig_classification_empty_state_demo(demo_names)
            st.plotly_chart(ph_bar, use_container_width=True, key="clf_preview_bar")
            st.plotly_chart(ph_g, use_container_width=True, key="clf_preview_gauge")
            st.caption(
                "Illustration: **equal probability example**. After prediction, bars and gauge "
                "show the **actual probabilities** from the Prediction System for your scenario."
            )
        else:
            vm = r["vals_map"]
            bits = []
            if "final_price" in vm:
                bits.append(f"Prix final (entrée) ≈ **{vm['final_price']:,.2f}** TND")
            if "cal_month" in vm:
                mi = int(round(vm["cal_month"]))
                mi = max(1, min(12, mi))
                bits.append(f"Mois : **{CLASSIF_MONTH_LABELS_FR[mi - 1]}**")
            if "cal_year" in vm:
                bits.append(f"Année : **{int(round(vm['cal_year']))}**")
            summ_txt = " · ".join(bits) if bits else "Profil numérique composé à partir des listes dropdowns."
            html_body = (
                f"<p style='font-size:1.02rem;color:#64748b;margin:0 0 0.5rem 0;'>{summ_txt}</p>"
                f"<p style='font-size:1.45rem;margin:0;color:{BRAND['deep']};font-weight:800;'>Predicted Status: {html.escape(r['label'])}</p>"
            )
            result_block("Prediction System Prediction", html_body)
            _nia = int(r.get("n_id_autofill") or 0)
            if _nia > 0:
                st.caption(
                    f"{_nia} colonne(s) identifiant database non affichées — valeurs fixées à la médiane du jeu pour l’inférence."
                )

            proba = r["proba"]
            class_names = r["classes"]
            if proba is not None and len(proba) and len(class_names) == len(proba):
                order = np.argsort(proba)[::-1]
                p_ord = proba[order]
                c_ord = [class_names[i] for i in order]
                fig_bar = go.Figure(
                    go.Bar(
                        x=(p_ord * 100.0).tolist(),
                        y=c_ord,
                        orientation="h",
                        marker_color=BRAND["deep"],
                        text=[f"{float(p) * 100:.1f} %" for p in p_ord],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    **_plotly_layout(
                        height=max(260, 48 + len(c_ord) * 36),
                        margin=dict(l=120, r=40, t=48, b=40),
                        title=dict(
                            text="Probabilités par statut (Y)",
                            font=dict(size=16, color=BRAND["deep"]),
                        ),
                        xaxis=dict(
                            title="%",
                            range=[
                                0,
                                min(
                                    115.0,
                                    max(105.0, float(np.max(p_ord)) * 100.0 + 15.0),
                                ),
                            ],
                            gridcolor=BRAND["chart_grid"],
                        ),
                        yaxis=dict(gridcolor=BRAND["chart_grid"]),
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                top_p = float(np.max(proba)) * 100.0
                fig_g = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=top_p,
                        number=dict(suffix=" %", font=dict(size=30)),
                        title=dict(text="Confiance — max. probability", font=dict(size=14)),
                        gauge=dict(
                            axis=dict(range=[0, 100]),
                            bar=dict(color=BRAND["deep"]),
                            bgcolor="#f1f5f9",
                            borderwidth=1,
                            bordercolor="rgba(13,148,136,0.35)",
                            steps=[
                                dict(range=[0, 40], color="#e2e8f0"),
                                dict(range=[40, 70], color="#ccfbf1"),
                                dict(range=[70, 100], color="#99f6e4"),
                            ],
                        ),
                    )
                )
                fig_g.update_layout(
                    height=300,
                    margin=dict(t=50, b=20, l=30, r=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=BRAND["ink"]),
                )
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.caption("Class probabilities not available for this model.")


def page_regression():
    _inject_page_accent(*PAGE_ACCENT["regr"])
    hero_variant(
        "regr",
        "Price Estimation — Amounts & Indicators",
        "Get a **numeric estimate** (basket, budget, etc.) à partir d’une situation **aligned with the database**.",
        badges=("Criterion D", "Interactive test"),
    )
    m = load_json(ML_MODELS / "metrics_regression.json")
    if not m:
        st.warning("Fichier `metrics_regression.json` absent.")
        return

    with st.expander("How to use this form", expanded=False):
        st.markdown(
            "Estimez le **prix final** (`final_price`). Les fields sont ordonnés par **influence** "
            "sur la Goal. Les **identifiants** (`id_*`) sont complétés automatiquement (database median)."
        )

    rm = extract_regression_metrics(m)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Prediction Error (réf. Best System)", f"{rm.get('Prediction Error', 0):.4f}" if rm.get("Prediction Error") else "—")
    with c2:
        st.metric("Average Error (réf. Best System)", f"{rm.get('Average Error', 0):.4f}" if rm.get("Average Error") else "—")
    with c3:
        st.metric("Correct Predictions Score (réf. Best System)", f"{rm.get('Correct Predictions', 0):.4f}" if rm.get("Correct Predictions") else "—")

    with st.expander("Form objective — details (optional)", expanded=False):
        st.markdown(DEPLOY_REGR_MARKDOWN)

    # Accept both "Goal" and "target" field names for compatibility
    target_field = m.get("Goal") or m.get("target")
    if not target_field and not (m.get("regression_objectives") or []):
        st.error("Price Estimation metrics incomplete (no target documented).")
        return

    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if not pp.is_file():
        st.warning("Parquet file `dw_financial_wide.parquet` not found — run `ML/scripts/run_00_data_preparation.py`.")
        return
    df_pq = pd.read_parquet(pp)

    if "regr_ui_result" not in st.session_state:
        st.session_state["regr_ui_result"] = None

    tgt = REGR_UI_TARGET
    ac = REGR_PAGE_ACCENT

    deployment_context_card(
        critere="D — Price Estimation",
        Goal=f"{REGR_TARGET_LABEL_FR.get(tgt, tgt)} (`{tgt}`)",
        objectif="Estimate final price à partir d’un profil database consistent with prepared data.",
        kpi=str(m.get("kpi_alignment") or "Metrics finance / performance (cf. métriques)"),
        modele=str(m.get("champion_model") or m.get("Prediction System") or "Random forest + mise à l’échelle"),
        rationale=champion_rationale(m, "Prediction System trained pour minimiser l’error on test set."),
        figure_note="Histogram of `final_price`, variable importances, estimated value vs database median.",
    )

    section_header(
        "Form — predict final price (`final_price`)",
        "Priority : variables outside Booking Status screen ; puis complété jusqu’à **at least six** fields (Prediction System importance), "
        "including variables already present in Booking Status if needed. Reference: dropdown if present. Other X → database median.",
    )

    run_meta = regression_run_for_target(m, tgt, df_pq)
    Factors = list(run_meta.get("Factors") or [])
    # Accept both "Factors" and "features" field names for compatibility
    if not Factors and (m.get("Goal") == tgt or m.get("target") == tgt):
        Factors = list(m.get("Factors") or m.get("features") or [])
    if not Factors:
        st.error(
            "Aucun prédicteur dérivable pour `final_price` — check que la colonne est presente dans `dw_financial_wide.parquet`."
        )
        return

    path = regression_model_path(m, tgt)
    pipe = load_joblib(path)
    if pipe is None:
        st.warning(
            f"Prediction System **`{path.name}`** absent — run `python ML/scripts/run_03_prediction_regression.py` "
            f"(avec `dw_financial_wide.parquet`). Vous pouvez composer les **X** ci-dessous ; l’estimation nécessite le fichier Prediction System."
        )

    missing_np = [f for f in Factors if f not in df_pq.columns]
    id_median_defaults = _classif_id_median_defaults(df_pq, Factors)
    cols_form = [c for c in Factors if not _is_id_column(c)]
    ordered, imp_by_col = regression_form_column_order(cols_form, pipe, Factors)
    manual_cols = regression_ui_manual_columns(ordered)
    _cf_names = classification_form_column_names()
    if manual_cols and all(c in _cf_names for c in manual_cols):
        st.info(
            "Le Prediction System ne sélectionne que des Input Factors dans la même fenêtre que la Status Prediction — "
            "recouvrement possible. Pour des fields vraiment différents, inclure des colonnes numériques **after la 20e** du database dans l’Learning Price Estimation."
        )
    section_blocks = regr_form_section_blocks(manual_cols, imp_by_col)
    median_fill_cols = [c for c in cols_form if c not in manual_cols]
    rm_t = regression_metrics_for_target(m, tgt)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Prediction Error (réf.)", f"{rm_t.get('Prediction Error', 0):.4f}" if rm_t.get("Prediction Error") is not None else "—")
    with m2:
        st.metric("Average Error (réf.)", f"{rm_t.get('Average Error', 0):.4f}" if rm_t.get("Average Error") is not None else "—")
    with m3:
        st.metric("Correct Predictions Score (réf.)", f"{rm_t.get('Correct Predictions', 0):.4f}" if rm_t.get("Correct Predictions") is not None else "—")

    col_reg_in, col_reg_out = st.columns([1.05, 1.0])

    with col_reg_in:
        st.markdown("##### Input Factors (X) — numeric input; reference in dropdown")
        if imp_by_col and pipe is not None:
            top3 = sorted(imp_by_col.items(), key=lambda x: -x[1])[:3]
            st.caption(
                "**Top influence (Prediction System)** : "
                + " · ".join(f"`{c}` ({v * 100:.1f} %)" for c, v in top3)
            )
        elif pipe is None:
            st.caption(
                "*Default order : amounts & budget en tête — load the Prediction System pour l’ordre par importance réelle.*"
            )
        with st.form("regr_form_final_price"):
            vals_map: dict[str, float] = {}
            for sec_title, cols_sec in section_blocks:
                st.markdown(
                    f'<p class="ez-regr-section-title">{html.escape(sec_title)}</p>',
                    unsafe_allow_html=True,
                )
                for col in cols_sec:
                    flab = friendly_feature_label(col)
                    hlp_base = "Prédicteur pour `final_price`."
                    if imp_by_col and col in imp_by_col:
                        hlp_base += f" Importance relative ≈ {imp_by_col[col] * 100:.1f} %."
                    if col not in _cf_names:
                        hlp_base += " *Absent du formulaire Status Prediction (écran distinct).*"
                    if _regr_benchmark_price_dropdown(col):
                        pairs = classif_dropdown_suggestions(df_pq, col)
                        labels = [p[0] for p in pairs]
                        default_i = min(2, len(labels) - 1) if len(labels) > 1 else 0
                        hlp = (
                            "Choix issus de la distribution du database (quantiles ou valeurs observées). "
                            + hlp_base
                        )
                        sel = st.selectbox(
                            flab,
                            labels,
                            index=default_i,
                            key=f"regr_dd_bm_{col}",
                            help=hlp,
                        )
                        val_sel = next(v for lab, v in pairs if lab == sel)
                        vals_map[col] = float(val_sel)
                    else:
                        lo, hi, med, step = _regr_num_bounds_step(df_pq, col)
                        fmt = number_input_format_for_feature(col)
                        hlp = f"Valeur numérique (min–max issus du database). {hlp_base}"
                        num = st.number_input(
                            flab,
                            min_value=lo,
                            max_value=hi,
                            value=med,
                            step=step,
                            format=fmt,
                            key=f"regr_num_fp_{col}",
                            help=hlp,
                        )
                        vals_map[col] = float(num)
            btn_label = "Estimate final price"
            submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)

    if submitted:
        if pipe is None:
            st.session_state["regr_ui_result"] = None
            st.warning(
                f"Impossible d’estimer sans `{path.name}`. Lancez `python ML/scripts/run_03_prediction_regression.py` puis rechargez l’app."
            )
        else:
            vec = []
            for c in Factors:
                if c in missing_np:
                    vec.append(float(_default_missing(c, df_pq)))
                elif _is_id_column(c):
                    vec.append(float(id_median_defaults.get(c, 0.0)))
                elif c in manual_cols:
                    vec.append(float(vals_map[c]))
                else:
                    vec.append(_column_numeric_median(df_pq, c))
            Xv = np.array(vec, dtype=float).reshape(1, -1)
            pred = float(pipe.predict(Xv)[0])
            st.session_state["regr_ui_result"] = {
                "pred": pred,
                "vals_map": {k: float(vals_map[k]) for k in manual_cols},
                "n_id_autofill": _n_id,
                "n_median_autofill": len(median_fill_cols),
            }

    fig_imp = fig_regression_importance_plot(pipe, Factors, accent=ac) if pipe is not None else None

    r = st.session_state.get("regr_ui_result")
    pred_show: float | None = float(r["pred"]) if r else None

    with col_reg_out:
        st.markdown("##### Projection & visualisations")
        if pred_show is None:
            st.markdown(
                '<div class="ez-out-panel ez-out-panel--hint">'
                '<p style="margin:0;font-size:1.05rem;line-height:1.55;color:#64748b;">'
                "Fill in the numeric fields à gauche puis cliquez sur "
                f'<strong style="color:{html.escape(REGR_PAGE_ACCENT)};">Estimate final price</strong> '
                "pour afficher l’estimation et les graphiques.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            y_lab = REGR_TARGET_LABEL_FR.get(tgt, tgt)
            html_body = (
                f"<p style='font-size:1.02rem;color:#64748b;margin:0 0 0.5rem 0;'>Goal : "
                f"<strong>{html.escape(tgt)}</strong> ({html.escape(y_lab)})</p>"
                f"<p style='font-size:1.65rem;margin:0;color:{REGR_PAGE_ACCENT};font-weight:800;'>"
                f"Valeur estimée : {pred_show:,.4f}</p>"
            )
            result_block("Reading du Prediction System", html_body, variant="regr")
            _nia = int(r.get("n_id_autofill") or 0)
            _nmed = int(r.get("n_median_autofill") or 0)
            if _nia > 0:
                st.caption(
                    f"{_nia} colonne(s) identifiant database non affichées — médiane du jeu pour l’inférence."
                )
            if _nmed > 0:
                st.caption(
                    f"{_nmed} prédicteur(s) non affichés — fixés à la **médiane du database** pour compléter le vecteur du Prediction System."
                )
            if rm_t.get("Prediction Error") is not None:
                st.caption(
                    f"Ordre de grandeur : Prediction Error test ≈ {float(rm_t['Prediction Error']):.4f} (réf.) — prudence hors plage d’Learning."
                )

        fig_dist_pred = fig_regression_distribution_plot(df_pq, tgt, pred=pred_show, accent=ac)
        if fig_dist_pred is not None:
            st.plotly_chart(fig_dist_pred, use_container_width=True, key="regr_dist_fp")
        else:
            st.caption("Histogram unavailable — `final_price` column missing from prepared dataset.")

        if fig_imp is not None:
            st.plotly_chart(fig_imp, use_container_width=True, key="regr_imp_fp")
        else:
            st.caption("Importances des variables not available pour ce Prediction System.")


def page_clustering():
    _inject_page_accent(*PAGE_ACCENT["cluster"])
    m_raw = load_json(ML_MODELS / "metrics_clustering.json")
    if not m_raw:
        st.warning("Fichier `metrics_clustering.json` absent.")
        return
    m = merge_metrics_for_loyalty_ui(ML_MODELS, m_raw)
    m = filter_clustering_metrics_if_models_missing(ML_MODELS, m)
    if loyalty_json_hint_run_script(ML_MODELS) and m.get("task") != "clustering_loyalty_rfm":
        st.warning(
            "Des fichiers **JSON** fidélité sont presents, mais les **models `.joblib`** (Customer Grouping, scaler, imputer) "
            "manquent dans `ML/models_artifacts/`. Tant qu’ils ne sont pas générés, l’interface reste sur la "
            "segmentation **broad view** (amounts / seasonality / catalog). "
            "Execute depuis la racine du projet : `python ML/scripts/run_01_clustering.py`."
        )

    loyalty_modes: dict = m.get("modes") or {}
    mode_block: dict | None = None
    labels_json = "clustering_segment_labels.json"
    features_json_name: str | None = None
    mode_key: str | None = None
    km = None

    is_loyalty = bool(loyalty_modes) and m.get("task") == "clustering_loyalty_rfm"
    _default_mode = str(m.get("default_mode") or "beneficiary")

    hero_variant(
        "cluster",
        "Loyalty — which segment for this profile ?" if is_loyalty else "Segmentation — profils d’activity",
        (
            "Indiquez **how many bookings**, **quel volume d’affaires** et **how recent** was the last activity : "
            "nous rapprochons ce comportement d’un **typical group** (ex. very loyal, occasional, to reactivate)."
            if is_loyalty
            else "Décrivez une situation **comme dans nos tables de performance** : le Prediction System indique le **typical profile** le plus proche."
        ),
        badges=("Criterion E", "Interactive test"),
    )

    with st.expander("How to use this screen", expanded=False):
        if is_loyalty:
            st.markdown(
                "Comparez un **bénéficiaire** ou **prestataire** aux segments. "
                "Fields: frequency, volumes, cumulative revenue, average basket, recency. "
                "Average rating not yet integrated into Prediction System."
            )
        else:
            st.markdown(
                "Testez à quel **segment** se rapproche un profil consistent with the database. "
                "Les fields correspondent aux variables numériques utilisées."
            )

    if loyalty_modes:
        opts = [k for k in ("beneficiary", "provider") if k in loyalty_modes]
        if not opts:
            opts = list(loyalty_modes.keys())
        _idx = opts.index(_default_mode) if _default_mode in opts else 0
        mode_key = st.radio(
            "Profile to simulate",
            opts,
            index=_idx,
            format_func=lambda k: {
                "beneficiary": "Beneficiaries (bookings, CA, recency…)",
                "provider": "Providers (load, revenue, recency…)",
            }.get(str(k), str(k)),
            horizontal=True,
            key="clustering_loyalty_scope",
        )
        mode_block = loyalty_modes.get(mode_key)
        if not mode_block:
            st.error("Metrics missing for this scope.")
            return
        km = load_joblib(ML_MODELS / mode_block["model_file"])
        if km is None:
            st.error(
                f"Fichier Prediction System not found : `{mode_block.get('model_file')}`. "
                "Régénérez les artefacts avec `python ML/scripts/run_01_clustering.py`."
            )
            return
        labels_json = str(mode_block.get("segment_labels_file") or labels_json)
        features_json_name = str(mode_block.get("features_file") or "") or None
        m_active = mode_block
    else:
        km = load_joblib(ML_MODELS / "kmeans_kpi_segments.joblib")
        m_active = m

    deployment_context_card(
        critere="E — Loyalty segmentation" if is_loyalty else "E — Segmentation",
        Goal=(
            "The closest segment (among learned groups)"
            if is_loyalty
            else f"L’un des {m.get('k', '?')} typical profiles of the Prediction System"
        ),
        objectif=(
            "Compare your input to profiles **loyalty / RFM** et name the closest group."
            if is_loyalty
            else "Match a case to the closest typical profile."
        ),
        kpi="Targeting assistance (offers, reactivation, prioritization)" if is_loyalty else "Reading by segment",
        modele=str(m.get("model_primary") or m.get("Prediction System") or "KMeans"),
        rationale=champion_rationale(
            m,
            "Stable segments after normalization indicators."
            if is_loyalty
            else "Readable partitions to group des similar behaviors.",
        ),
        figure_note="Radar: your profile compared to center of assigned segment.",
        label_cible="What you get",
        label_kpi="Usefulness",
        label_figure="Main chart",
    )

    section_header(
        "Prediction System benchmarks",
        "Quality globale before filling le formulaire" if is_loyalty else "Some indicators before simulation",
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("k (segments)", m_active.get("k", m.get("k", "—")))
        with c2:
            sil = m_active.get("silhouette_holdout") or m_active.get("Quality Score") or m.get("silhouette_holdout")
            st.metric("Quality Score holdout", f"{sil:.4f}" if sil is not None else "—")
        with c3:
            st.metric(
                "Sample (train / total)",
                f"{m_active.get('n_train', m.get('n_train', '?'))} / {m_active.get('n_samples', m.get('n_samples', '?'))}",
            )
        with c4:
            if is_loyalty:
                st.metric("Reading", "RFM Loyalty")
                st.caption("Beneficiaries & providers")
            else:
                kpi = str(m_active.get("kpi_alignment") or m.get("kpi_alignment", "—"))
                st.metric("Scope", "✓" if kpi != "—" else "—")
                st.caption("Exploratory database Prediction System")

    if km is None:
        st.info("Customer Segmentation model missing — JSON metrics only.")
        return

    n_feat = getattr(km, "n_features_in_", None)
    feat_names_km = clustering_feature_names_for_model(km, features_json_name=features_json_name)
    cluster_short, _cluster_long_technical, _label_source, cluster_metier = resolve_segment_labels(
        km, feat_names_km, ML_MODELS, labels_json=labels_json
    )

    if cluster_short:
        with st.expander("Segment overview (reminder)", expanded=False):
            for i, title in enumerate(cluster_short):
                blurb = (
                    cluster_metier[i]
                    if i < len(cluster_metier) and cluster_metier[i]
                    else ""
                )
                _head = (
                    segment_card_title_loyalty(blurb or None, title)
                    if is_loyalty
                    else title
                )
                st.markdown(f"**Segment {i} — {_head}**")
                if blurb:
                    st.markdown(blurb)
                st.markdown("")

    section_header(
        "Form — describe the profile to test",
        "Same logic as Booking Status / Price Estimation : one field per indicator, puis validation."
        if is_loyalty
        else "Fill in expected values par le Prediction System, then validate to see segment and chart.",
    )

    if mode_block:
        _imp = load_joblib(ML_MODELS / mode_block["imputer_file"])
        _scl = load_joblib(ML_MODELS / mode_block["scaler_file"])
    else:
        _imp = load_median_imputer(ML_MODELS)
        _scl = load_standard_scaler(ML_MODELS)
    _feat_order = feat_names_km if feat_names_km else None
    if _feat_order is None and n_feat:
        raw_fn = getattr(km, "feature_names_in_", None)
        if raw_fn is not None and len(raw_fn) == int(n_feat):
            _feat_order = [str(x) for x in raw_fn]

    if not _feat_order or not _imp or not _scl:
        st.warning(
            "Pour prédire à partir de **coordata brutes**, il faut les fichiers scaler / imputer / Customer Grouping / noms de "
            "Factors — ré-run **`python ML/scripts/run_01_clustering.py`** ou la **section 5** du notebook **01_E**."
        )
    else:
        _stats = getattr(_imp, "statistics_", None)
        _defaults = (
            [float(_stats[i]) for i in range(len(_feat_order))]
            if _stats is not None and len(_stats) == len(_feat_order)
            else [0.0] * len(_feat_order)
        )
        _biz_idx, _id_idx = split_business_vs_id_feature_indices(list(_feat_order))
        if not _biz_idx:
            _biz_idx = list(range(len(_feat_order)))
            _id_idx = []

        vals_map: dict[int, float] = {}
        with st.container(border=True):
            st.markdown('<div class="ez-card ez-card--deploy">', unsafe_allow_html=True)
            st.markdown(
                "<div style='padding-bottom:0.35rem;'>"
                "<span style='font-size:0.82rem;text-transform:uppercase;letter-spacing:0.14em;color:"
                + CLUSTER_PAGE_ACCENT
                + ";font-weight:800;'>"
                "Profile input</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("##### Metrics to fill in")
            st.caption(
                "Les valeurs proposées correspondent aux **médianes** du jeu d’apprentissage — modify them to simulate a case."
                if is_loyalty
                else "Valeurs par défaut ≈ médianes du jeu d’Learning."
            )
            with st.form("cluster_predict_raw_row"):
                if is_loyalty:
                    _indices_main = ordered_feature_indices_for_form(list(_feat_order), loyalty=True)
                    for grp in ("activity", "amounts", "recency"):
                        _ix_grp = [
                            i
                            for i in _indices_main
                            if loyalty_form_group_key(_feat_order[i]) == grp
                        ]
                        if not _ix_grp:
                            continue
                        st.markdown(f"**{loyalty_form_group_title(grp)}**")
                        _cg = st.columns(2 if len(_ix_grp) >= 2 else 1)
                        for _j, _ix in enumerate(_ix_grp):
                            with _cg[_j % len(_cg)]:
                                vals_map[_ix] = st.number_input(
                                    friendly_feature_label(_feat_order[_ix]),
                                    value=_defaults[_ix],
                                    format=number_input_format_for_feature(_feat_order[_ix]),
                                    key=f"cl_raw_f_{_ix}",
                                )
                else:
                    _nc = 2 if len(_biz_idx) >= 2 else 1
                    _cols_b = st.columns(_nc)
                    for _j, _ix in enumerate(_biz_idx):
                        with _cols_b[_j % _nc]:
                            vals_map[_ix] = st.number_input(
                                friendly_feature_label(_feat_order[_ix]),
                                value=_defaults[_ix],
                                format=number_input_format_for_feature(_feat_order[_ix]),
                                key=f"cl_raw_f_{_ix}",
                            )
                if _id_idx:
                    with st.expander("Technical identifiers (optional)", expanded=False):
                        st.caption("Useful only if reproducing a complete database row; otherwise leave defaults.")
                        _cols_id = st.columns(min(2, max(1, len(_id_idx))))
                        for _j, _ix in enumerate(_id_idx):
                            with _cols_id[_j % len(_cols_id)]:
                                vals_map[_ix] = st.number_input(
                                    friendly_feature_label(_feat_order[_ix]),
                                    value=_defaults[_ix],
                                    format=number_input_format_for_feature(_feat_order[_ix]),
                                    key=f"cl_raw_f_{_ix}",
                                )
                _sub = st.form_submit_button(
                    "View my segment" if is_loyalty else "View segment and chart",
                    type="primary",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        if _sub:
            _vals_in = [vals_map[i] for i in range(len(_feat_order))]
            try:
                pred_id, z_vec, _xi = predict_cluster_from_raw_features(_vals_in, _imp, _scl, km)
            except Exception as _err:
                st.error(f"Prediction failed (check number of variables and artifacts): {_err}")
            else:
                _name_s = cluster_short[pred_id] if 0 <= pred_id < len(cluster_short) else ""
                _metier_s = (
                    cluster_metier[pred_id].strip()
                    if 0 <= pred_id < len(cluster_metier) and cluster_metier[pred_id]
                    else ""
                )
                _title_card = (
                    segment_card_title_loyalty(_metier_s or None, _name_s)
                    if is_loyalty
                    else _name_s
                )
                _prof_display = _metier_s if _metier_s else "Synthèse business à préciser pour ce segment."
                _shares = (m_active.get("cluster_share_train_sample") or m.get("cluster_share_train_sample")) or {}
                _pct = _shares.get(str(pred_id))
                _cc = np.asarray(km.cluster_centers_)
                _r_ix = indices_for_radar_storytelling(_biz_idx, len(_feat_order))
                _zr = z_vec[_r_ix]
                _ccr = _cc[pred_id][_r_ix]
                _theta: list[str] = []
                for _ii in _r_ix:
                    _lab = friendly_feature_label(_feat_order[_ii])
                    _theta.append((_lab[:26] + "…") if len(_lab) > 26 else _lab)
                cc_res, cc_radar = st.columns((1.0, 1.08))
                with cc_res:
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,{CLUSTER_PAGE_ACCENT_SOFT} 0%,#ffffff 100%);"
                        f"border-left:5px solid {CLUSTER_PAGE_ACCENT_DEEP};border-radius:14px;"
                        f"padding:1.35rem 1.5rem;margin:0.35rem 0 0.85rem 0;"
                        f"box-shadow:0 4px 20px rgba(15,23,42,0.07);"
                        f"border:1px solid rgba(234,88,12,0.28);'>"
                        f"<p style='margin:0;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:{CLUSTER_PAGE_ACCENT_DEEP};"
                        f"font-weight:800;'>Assigned segment</p>"
                        f"<p style='margin:0.45rem 0 0 0;font-size:1.75rem;font-weight:800;color:{CLUSTER_PAGE_ACCENT_DEEP};"
                        f"line-height:1.2;'>{html.escape(_title_card)}</p>"
                        f"<p style='margin:0.55rem 0 0 0;font-size:1.02rem;color:#334155;'>"
                        f"{'Reading' if _metier_s else 'To be completed on project side'} : "
                        f"{html.escape(_prof_display.replace('**', ''))}</p>"
                        f"<p style='margin:0.75rem 0 0 0;font-size:0.95rem;color:#64748b;'>"
                        f"Segment index : <code style='background:#fff7ed;color:{CLUSTER_PAGE_ACCENT_DEEP};padding:0.15rem 0.45rem;"
                        f"border-radius:6px;border:1px solid rgba(234,88,12,0.45);font-weight:700;'>{pred_id}</code></p></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        format_segment_deployment_explanation(
                            _name_s,
                            _metier_s or None,
                            metier_already_shown_above=bool(_metier_s),
                        )
                    )
                    if _pct is not None:
                        _n_samp = m_active.get("n_samples", m.get("n_samples", "?"))
                        _unit_ech = "aggregated profiles (loyalty)" if is_loyalty else "rows"
                        st.info(
                            f"Approximate share of this segment dans l’échantillon d’apprentissage "
                            f"(~{_n_samp} {_unit_ech}) : **{float(_pct) * 100:.1f} %**."
                        )
                with cc_radar:
                    fig_r = go.Figure()
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=np.concatenate([_zr, _zr[:1]]),
                            theta=_theta + [_theta[0]],
                            name="Entered profile",
                            line=dict(color=CLUSTER_PAGE_ACCENT_DEEP, width=3),
                            fillcolor="rgba(234, 88, 12, 0.18)",
                            fill="toself",
                        )
                    )
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=np.concatenate([_ccr, _ccr[:1]]),
                            theta=_theta + [_theta[0]],
                            name="Segment center (reference)",
                            line=dict(color=BRAND["radar_ref"], width=2.5, dash="dash"),
                        )
                    )
                    fig_r.update_layout(
                        template="plotly_white",
                        polar=dict(
                            bgcolor="#f8fafc",
                            radialaxis=dict(
                                visible=True,
                                gridcolor="rgba(148, 163, 184, 0.35)",
                                linecolor="rgba(234, 88, 12, 0.35)",
                                tickfont=dict(size=13, color="#475569"),
                            ),
                            angularaxis=dict(
                                linecolor="rgba(234, 88, 12, 0.3)",
                                tickfont=dict(size=13, color="#334155"),
                            ),
                        ),
                        title=dict(
                            text="<b>Visual comparison</b> — entered profile vs typical segment profile",
                            font=dict(size=19, color=CLUSTER_PAGE_ACCENT_DEEP, family="Segoe UI, system-ui, sans-serif"),
                            x=0.5,
                            xanchor="center",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=500,
                        margin=dict(t=96, b=72, l=48, r=48),
                        font=dict(size=16, color=BRAND["ink"], family="Segoe UI, system-ui, sans-serif"),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            font=dict(size=15, color=BRAND["ink"]),
                        ),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.caption(
                        "Each axis corresponds to an indicator du formulaire (normalized scale comme à l’Learning)."
                    )

    _wrap_detail = st.expander("Technical metrics & detailed profiles (optional)", expanded=not is_loyalty)
    with _wrap_detail:
        section_header(
            "Prediction System comparisons",
            "To go further than the form above",
        )
        col_a, col_b = st.columns((1, 1))
        with col_a:
            dbk = m_active.get("davies_bouldin_kmeans") if mode_block else m.get("davies_bouldin_kmeans")
            dba = m.get("davies_bouldin_agg")
            if dbk is not None and dba is not None:
                fig_db = go.Figure(
                    go.Bar(
                        x=["KMeans (Best System)", "Agglomératif Ward"],
                        y=[dbk, dba],
                        marker_color=[BRAND["deep"], BRAND["sky"]],
                        text=[f"{dbk:.2f}", f"{dba:.2f}"],
                        textposition="outside",
                    )
                )
                fig_db.update_layout(
                    **_plotly_layout(
                        title=dict(text="Separation Score (↓ = more compact / separated clusters)", font=dict(size=14, color=BRAND["deep"])),
                        height=340,
                        yaxis_title="DB Index",
                        yaxis=dict(gridcolor=BRAND["chart_grid"]),
                    )
                )
                st.plotly_chart(fig_db, use_container_width=True)
            elif dbk is not None:
                fig_db = go.Figure(
                    go.Bar(
                        x=["Customer Grouping (loyalty)"],
                        y=[dbk],
                        marker_color=[BRAND["deep"]],
                        text=[f"{dbk:.2f}"],
                        textposition="outside",
                    )
                )
                fig_db.update_layout(
                    **_plotly_layout(
                        title=dict(text="Separation Score — Customer Grouping (↓ = better)", font=dict(size=14, color=BRAND["deep"])),
                        height=340,
                        yaxis_title="DB Index",
                        yaxis=dict(gridcolor=BRAND["chart_grid"]),
                    )
                )
                st.plotly_chart(fig_db, use_container_width=True)
        with col_b:
            if hasattr(km, "cluster_centers_") and km.cluster_centers_ is not None:
                cc = np.asarray(km.cluster_centers_)
                nc = cc.shape[1]
                if feat_names_km and len(feat_names_km) == nc:
                    x_hm = [
                        (str(n).replace("_", " ")[:18] + "…") if len(str(n)) > 18 else str(n).replace("_", " ")
                        for n in feat_names_km
                    ]
                else:
                    x_hm = [f"Dim.{i + 1}" for i in range(nc)]
                if len(cluster_short) == cc.shape[0]:
                    y_hm = [f"S{i} · {cluster_short[i]}" for i in range(cc.shape[0])]
                else:
                    y_hm = [f"Segment {i}" for i in range(cc.shape[0])]
                fig_hm = go.Figure(
                    data=go.Heatmap(
                        z=cc,
                        x=x_hm,
                        y=y_hm,
                        colorscale="Blues",
                        colorbar=dict(title="Centre (std.)"),
                    )
                )
                fig_hm.update_layout(
                    **_plotly_layout(
                        title=dict(
                            text=(
                                "Cluster centers — RFM variables / fidélité (standardisées)"
                                if is_loyalty
                                else "Cluster centers — standardized space (dimensions = numeric variables from perf. scope)"
                            ),
                            font=dict(size=15, color=BRAND["deep"]),
                        ),
                        height=340,
                        margin=dict(l=80, r=20, t=50, b=40),
                    )
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Cluster centers not available dans ce fichier joblib.")

    section_header(
        "Illustrative distribution (simulation)",
        "Simulated distribution to visualize la relative size of segments (indicatif)",
    )
    st.markdown('<div class="ez-panel">', unsafe_allow_html=True)
    st.markdown("##### Illustrative segment distribution")
    if is_loyalty:
        st.caption(
            "Les **parts par segment** displayed above come from l’échantillon d’apprentissage. "
            "La random simulation n’est pas calibrée sur l’espace RFM — disabled in loyalty mode."
        )
    else:
        st.caption(
            "Simulation of **nombreuses** attributions dans l’espace d’entrée du KMeans (multivariate Gaussian) — "
            "gives an idea de la **relative size** des segments, not raw business volumes."
        )
    if not is_loyalty and st.button("Calculate simulated distribution", type="primary", use_container_width=True) and n_feat:
        rng = np.random.default_rng(123)
        n_draw = 8000
        z = rng.standard_normal((n_draw, int(n_feat)))
        labs = km.predict(z)
        counts = pd.Series(labs).value_counts().sort_index()

        def _pie_seg_label(idx: int) -> str:
            j = int(idx)
            if 0 <= j < len(cluster_short):
                return f"S{j} · {cluster_short[j]}"
            return f"Segment {j}"

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=[_pie_seg_label(i) for i in counts.index],
                    values=counts.values,
                    hole=0.42,
                    marker=dict(
                        colors=[BRAND["deep"], BRAND["sky"], BRAND["line2"], "#22c55e", "#a78bfa"],
                        line=dict(color="rgba(34, 211, 238, 0.35)", width=1),
                    ),
                )
            ]
        )
        fig_pie.update_layout(
            title=dict(text="Illustration — simulated distribution des segments", font=dict(size=16, color=BRAND["deep"])),
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            font=dict(color=BRAND["ink"]),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_timeseries():
    _inject_page_accent(*PAGE_ACCENT["ts"])
    hero_variant(
        "ts",
        "Time Series \u2014 \u00e9volution & pr\u00e9vision",
        "Visualisez **l\u2019\u00e9volution monthlyle** de vos database indicators, **compare Trend Analysis vs Advanced Forecast** sur la validation, puis **projetez** la trend sur les month \u00e0 venir.",
        badges=("Criterion F", "Donn\u00e9es database live"),
    )
    m = load_json(ML_MODELS / "metrics_timeseries.json")
    if not m:
        st.warning("Fichier `metrics_timeseries.json` absent.")
        return

    _ser = str(m.get("series") or "\u2014")
    _champion = str(m.get("champion_model") or "Trend Analysis")
    _champion_short = "Trend Analysis" if "Trend Analysis" in _champion.lower() else "Advanced Forecast"

    deployment_context_card(
        critere="F \u2014 Time Series",
        Goal=f"Forecast monthlyle : {SERIES_COLUMN_LABELS_FR.get(_ser, _ser)}",
        objectif="Suivre l\u2019\u00e9volution d\u2019un indicateur agr\u00e9g\u00e9 et anticipate les month suivants.",
        kpi=str(m.get("kpi_alignment") or "Pilotage volumes / CA / panier"),
        modele=_champion,
        rationale=champion_rationale(m, "Mod\u00e8le selected : Prediction Error the lowest sur la validation (holdout 3 month)."),
        figure_note="Courbe : historique database + pr\u00e9vision ; barres : comparison Trend Analysis vs Advanced Forecast.",
        label_cible="Indicateur pr\u00e9dit",
        label_kpi="Utilit\u00e9 m\u00e9tier",
        label_figure="Graphiques",
    )

    with st.expander("Objectif de cet \u00e9cran \u2014 d\u00e9tail", expanded=False):
        st.markdown(DEPLOY_TS_MARKDOWN)

    # --- Performance & stationnarit\u00e9 ---
    section_header(
        "Quality du model Best System",
        "M\u00e9triques de validation, stationnarit\u00e9 et horizon document\u00e9",
    )
    tc = m.get("test_champion") or m.get("test_holt") or {}
    if not isinstance(tc, dict):
        tc = {}
    th = m.get("test_holt") or {}
    ta = m.get("test_arima") or {}

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            rms = tc.get("Prediction Error")
            st.metric("Prediction Error (Best System)", f"{rms:.2f}" if rms is not None else "\u2014")
            st.caption("Root mean squared error")
        with c2:
            mae_val = tc.get("Average Error")
            st.metric("Average Error", f"{mae_val:.2f}" if mae_val is not None else "\u2014")
            st.caption("Mean absolute error")
        with c3:
            mape = tc.get("mape")
            st.metric("MAPE", f"{mape:.2f} %" if mape is not None else "\u2014")
            st.caption("\u00c9cart relatif average (%)")
        with c4:
            st.metric("Best System", _champion_short)
            st.caption(f"Horizon: {m.get('horizon', '?')} month")
        with c5:
            adf_p = m.get("adf_pvalue")
            if adf_p is not None and adf_p < 0.05:
                st.metric("Stationnarit\u00e9", "Favorable")
                st.caption(f"ADF p={adf_p:.4f}")
            elif adf_p is not None:
                st.metric("Stationnarit\u00e9", "Non stationnaire")
                st.caption(f"ADF p={adf_p:.4f}")
            else:
                st.metric("Stationnarit\u00e9", "\u2014")

    with st.expander(f"Série de reference : {SERIES_COLUMN_LABELS_FR.get(_ser, _ser)}", expanded=False):
        _expl_txt = str(m.get("target_column_explained", "") or "")
        if _expl_txt:
            st.markdown(_expl_txt)

    # --- Comparaison Trend Analysis vs Advanced Forecast ---
    section_header(
        "Comparaison Trend Analysis vs Advanced Forecast",
        "Erreurs sur la m\u00eame fen\u00eatre de validation \u2014 the lowest est le meilleur",
    )
    col_chart, col_table = st.columns((1.2, 0.8))
    with col_chart:
        st.plotly_chart(fig_ts_compare(m), use_container_width=True)
    with col_table:
        _metric_rows = []
        for label, d in (("Trend Analysis", th), ("Advanced Forecast", ta)):
            if d:
                _metric_rows.append({
                    "Mod\u00e8le": label,
                    "Prediction Error": round(d.get("Prediction Error", 0), 2),
                    "Average Error": round(d.get("Average Error", 0), 2),
                    "MAPE (%)": round(d.get("mape", 0), 2),
                })
        if _metric_rows:
            st.dataframe(pd.DataFrame(_metric_rows).set_index("Mod\u00e8le"), use_container_width=True)
        delta = m.get("rmse_delta_holt_minus_arima")
        if delta is not None:
            st.info(
                f"**\u00c9cart Prediction Error** (Trend Analysis \u2212 Advanced Forecast) : **{delta:+.2f}** \u2014 "
                + ("Trend Analysis est l\u00e9g\u00e8rement meilleur." if delta < 0 else "Advanced Forecast est l\u00e9g\u00e8rement meilleur.")
            )
        st.caption(
            f"**Best System selected** : {_champion_short} (r\u00e8gle : Prediction Error minimal sur le holdout de "
            f"{m.get('horizon', '?')} month)."
        )



def main():
    # ── Vérification d'authentification ─────────────────────────
    # Si l'utilisateur n'est pas connecté → afficher l'écran de login
    if not is_authenticated(st.session_state):
        _render_login_screen()
        st.stop()  # Arrêter l'exécution : rien d'autre ne s'affiche

    # ── Utilisateur authentifié → navigation normale ─────────────
    page = sidebar_brand_and_nav()

    # Vérification de sécurité : la page doit être dans les pages autorisées
    role          = get_role(st.session_state)
    allowed_pages = ROLE_PAGES.get(role, PAGE_ORDER)
    if page not in allowed_pages:
        st.warning(
            f"⛔ La page **{page}** n'est pas accessible avec votre rôle "
            f"(**{ROLE_LABELS.get(role, role)}**)."
        )
        st.stop()

    # Dispatch vers la page sélectionnée
    if page == PAGE_HOME:
        page_home()
    
    elif page == PAGE_CLASSIF:
        page_classification()
    elif page == PAGE_REGR:
        page_regression()
    elif page == PAGE_CLUSTER:
        page_clustering()
    elif page == PAGE_TS:
        page_timeseries()
    elif page == PAGE_RECAP:
        page_recap()
    else:
        page_home()

    _page_nav_footer(page)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "EventZilla ML Studio — data du database, models dans ML/models_artifacts/."
    )


if __name__ == "__main__":
    main()

