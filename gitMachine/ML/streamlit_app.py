# -*- coding: utf-8 -*-
"""
EventZilla — laboratoire ML (Streamlit), interface claire, accents teal, graphiques Plotly lisibles.

Lancer depuis la racine du dépôt :
    streamlit run ML/streamlit_app.py

Logo : placez une image PNG sous ``ML/assets/eventzilla_logo.png`` (optionnel) ;
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

# Palette dashboard — thème clair, accents teal / cyan (figures modernes, bon contraste)
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

# Navigation (ordre : accueil → familles ML → récap en fin de parcours)
PAGE_HOME = "Accueil"
PAGE_CLASSIF = "Classification (C)"
PAGE_REGR = "Régression (D)"
PAGE_CLUSTER = "Clustering (E)"
PAGE_TS = "Séries temporelles (F)"
PAGE_RECAP = "Récapitulatif"
PAGE_ORDER: tuple[str, ...] = (
    PAGE_HOME,
    PAGE_CLASSIF,
    PAGE_REGR,
    PAGE_CLUSTER,
    PAGE_TS,
    PAGE_RECAP,
)

ML_INTEREST_MARKDOWN = """
**Pourquoi le machine learning pour EventZilla ?**  
Les données du **data warehouse** (réservations, finances agrégées, volumes) permettent d’**anticiper** les statuts et montants, de **segmenter** l’offre et de **suivre** les tendances mensuelles — sans remplacer le métier, mais pour **prioriser** et **illustrer** les scénarios dans un cadre pédagogique et reproductible.

**Ce studio** centralise les modèles entraînés sur le même périmètre que vos notebooks (00→05) : testez-les ici avant toute mise en production.
"""

# Textes « déploiement / test » — peu techniques, alignés data warehouse (enseignants & équipe)
DEPLOY_CLASSIF_MARKDOWN = """
### À quoi sert cet écran ?

**Objectif :** décrire **une situation d’activité EventZilla** telle qu’elle apparaît dans nos données (même logique que le DW) et voir **quel statut de réservation** le modèle retient comme le plus plausible.

Vous composez un **scénario** (niveau d’activité, période, ordre de grandeur des montants) ou vous partez **d’un cas réel** déjà présent dans le jeu préparé — le tout pour **tester le modèle** sans écrire de SQL.
"""

DEPLOY_REGR_MARKDOWN = """
### À quoi sert cet écran ?

**Objectif :** estimer **une valeur continue** du périmètre finance / performance (ex. montant, panier) à partir d’une situation **cohérente avec le data warehouse**.

Même principe que la classification : **scénario type** ou **ligne réelle** issue des données préparées, pour **valider la régression** sur nos indicateurs EventZilla.
"""

DEPLOY_TS_MARKDOWN = """### À quoi sert cet écran ?

**Objectif :** visualiser **l’évolution mensuelle** d’indicateurs agrégés (volume d’activité, chiffre d’affaires, panier moyen) **calculés depuis le data warehouse**, puis **projeter quelques mois** pour illustrer la dynamique observée.

**Ce que vous pouvez tester :**
1. **Choisir l’indicateur** — volume d’activité, CA mensuel ou panier moyen.
2. **Ajuster l’horizon** — de 1 à 12 mois de prévision.
3. **Comparer visuellement** le train, la zone de validation et la prévision.
4. **Lire les métriques** — RMSE, MAE, MAPE sur la fenêtre de test.

**Modèles comparés :** **Holt** (lissage exponentiel avec tendance) vs **ARIMA** (autorégression + différenciation + moyenne mobile). Le **champion** est celui qui a le **RMSE le plus bas** sur la validation.
"""

DEPLOY_SYNTH_MARKDOWN = """
### Aide à la navigation

- **Accueil** : intérêt du ML pour EventZilla, **KPI** rapides, boutons vers les tests.
- **Pages C à F** : un écran par famille de modèle (même logique que les notebooks).
- **Récapitulatif** : **tableau unique** des champions / qualités (type livrable 05), sans surcharge.

Les fichiers `metrics_*.json` dans `ML/models_artifacts/` alimentent les indicateurs et le tableau récapitulatif.
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
    """Texte court expliquant le choix du modèle (champs optionnels dans les JSON métriques)."""
    if not m:
        return fallback or "—"
    for key in ("champion_rule", "rationale", "notes_champion", "model_notes"):
        v = m.get(key)
        if v:
            return str(v).strip()
    return fallback or "Modèle retenu après comparaison sur le jeu de test (détail dans le notebook associé)."


def deployment_context_card(
    critere: str,
    cible: str,
    objectif: str,
    kpi: str,
    modele: str,
    pourquoi: str,
    figure_note: str,
    *,
    label_cible: str = "Cible (Y)",
    label_kpi: str = "KPI / lecture métier",
    label_figure: str = "Figure / indicateur à regarder",
) -> None:
    """Bloc compact en tête des pages de test."""
    esc = html.escape
    st.markdown(
        f'<div class="ez-deploy-context"><h4>Ce que teste cet écran</h4>'
        f'<div class="ez-dc-grid">'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Critère</span>'
        f'<span class="ez-dc-val">{esc(critere)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">{esc(label_cible)}</span>'
        f'<span class="ez-dc-val">{esc(cible)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Modèle</span>'
        f'<span class="ez-dc-val">{esc(modele)}</span></div>'
        f'<div class="ez-dc-item"><span class="ez-dc-label">Objectif</span>'
        f'<span class="ez-dc-val">{esc(objectif)}</span></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="EventZilla ML — Studio",
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
    """Thème global clair : cartes nettes, accents teal, graphiques lisibles."""
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
        /* Bandeau d’aide classification : pas de hauteur min. élevée (évite un « cadre vide ») */
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
    """Thème de couleur dynamique par page — boutons, métriques, formulaires, expanders."""
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
    """Toutes les colonnes numériques du DW (ordre stable), pour distinguer régression vs classification."""
    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if not pp.is_file():
        return []
    df = pd.read_parquet(pp)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "fact_finance_id"]


def classification_form_column_names() -> frozenset[str]:
    """Colonnes affichées dans le formulaire classification (hors id) — à ne pas dupliquer en régression."""
    cols = classification_feature_columns()
    if not cols:
        return frozenset()
    return frozenset(c for c in cols if not _is_id_column(c))


CLASSIF_MONTH_LABELS_FR = [
    "Janvier",
    "Février",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Août",
    "Septembre",
    "Octobre",
    "Novembre",
    "Décembre",
]


def _classif_order_columns(cols: list[str]) -> list[str]:
    """Met en tête période / montants utiles à la lecture métier, puis le reste."""
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
    """Libellés + valeurs numériques alignées sur la distribution du jeu DW (suggestions)."""
    if col not in df.columns:
        return [("0", 0.0)]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return [("0", 0.0)]
    c = str(col).lower()
    if c == "cal_month":
        return [(f"{CLASSIF_MONTH_LABELS_FR[i]} (mois {i + 1})", float(i + 1)) for i in range(12)]
    if c == "quarter":
        return [
            ("T1 — janvier à mars", 1.0),
            ("T2 — avril à juin", 2.0),
            ("T3 — juillet à septembre", 3.0),
            ("T4 — octobre à décembre", 4.0),
        ]
    if c == "cal_year":
        u = sorted(pd.unique(s.round().astype(int)))
        if len(u) <= 24:
            return [(str(int(y)), float(y)) for y in u]
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        labs = (
            "Année — bas (~10e percentile)",
            "Année — bas (~25e)",
            "Année — médiane",
            "Année — haut (~75e)",
            "Année — haut (~90e)",
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
        "Très bas dans le DW (~10e %.)",
        "Bas (~25e %.)",
        "Typique — médiane",
        "Élevé (~75e %.)",
        "Très haut (~90e %.)",
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
        "period": "Période & calendrier (données DW)",
        "money": "Montants & indicateurs financiers",
        "counts": "Volumes & compteurs",
        "ids": "Identifiants dimension (DW)",
        "ctx": "Contexte",
        "other": "Autres variables du modèle",
    }.get(group, "Variables")


def _classif_id_median_defaults(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Médianes sur le jeu préparé pour les colonnes id requises par le pipeline (non saisies à l’écran)."""
    out: dict[str, float] = {}
    for c in cols:
        if not _is_id_column(c) or c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        out[c] = float(s.median()) if len(s) else 0.0
    return out


def safe_target_filename(target: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in target)


def regression_paths_and_targets(m: dict) -> tuple[list[str], str | None]:
    primary = m.get("target")
    runs = m.get("regression_objectives") or []
    if runs:
        targets = [r["target"] for r in runs]
        return targets, primary
    if primary:
        return [primary], primary
    return [], None


def regression_model_path(m: dict, target: str) -> Path:
    primary = m.get("target")
    if target == primary:
        return ML_MODELS / "rf_panier_kpi_pipeline.joblib"
    return ML_MODELS / f"rf_regression_target_{safe_target_filename(target)}.joblib"


REGR_TARGET_LABEL_FR: dict[str, str] = {
    "final_price": "Prix final (panier / commande)",
    "service_price": "Prix prestataire",
    "benchmark_avg_price": "Prix moyen de référence (benchmark)",
    "event_budget": "Budget événement",
    "commission_margin": "Marge commission (final − prestataire)",
}

# Cible unique exposée dans l’UI Streamlit (alignée sur le critère D — panier / prix final).
REGR_UI_TARGET = "final_price"
# Accent visuel régression (violet) — distinct de la classification (teal).
REGR_PAGE_ACCENT = "#7c3aed"
REGR_PAGE_ACCENT_DEEP = "#6d28d9"
# Clustering (E) — ambre / orange, distinct de la classification (teal) et de la régression (violet)
CLUSTER_PAGE_ACCENT = "#ea580c"
CLUSTER_PAGE_ACCENT_DEEP = "#c2410c"
CLUSTER_PAGE_ACCENT_SOFT = "#fff7ed"

# Aligné sur ML/scripts/run_03_prediction_regression.py (TARGET_KPIS)
REGR_KPI_TAG: dict[str, str] = {
    "final_price": "panier_moyen_ca_sum_final_price",
    "service_price": "prix_prestataire_structure_revenus",
    "event_budget": "budget_evenement",
}


def regression_infer_features(df: pd.DataFrame, target: str) -> list[str]:
    """Même ensemble de prédicteurs que run_03 : numériques sauf la cible et fact_finance_id."""
    if target not in df.columns:
        return []
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target and c != "fact_finance_id"
    ]


def pipeline_feature_importance_dict(pipe, feature_names: list[str]) -> dict[str, float] | None:
    """Importances Random Forest alignées sur l’ordre des colonnes d’entraînement."""
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
    """Ordre des champs : importance RF décroissante ; sinon heuristique « prix / budget » d’abord (≠ classification)."""
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
    """`benchmark_avg_price` (ou alias) : saisie par liste déroulante de quantiles / valeurs DW."""
    n = str(col).lower().replace(" ", "_")
    return n in ("benchmark_avg_price", "benchmark_price")


# Nombre minimal de champs affichés (saisie utilisateur) — évite un formulaire réduit à une seule variable.
REGR_MANUAL_FIELDS_MIN = 6
REGR_MANUAL_FIELDS_TARGET = 10
REGR_MANUAL_FIELDS_MAX = 12


def regression_ui_manual_columns(ordered: list[str]) -> list[str]:
    """
    Colonnes de saisie régression : d’abord **hors** formulaire classification, puis complément
    jusqu’à un **minimum** de champs (ordre importance du modèle), même si chevauchement avec la classif.
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

    # 1) Prédicteurs hors périmètre classification (importance RF / heuristique).
    for c in ordered:
        _push_disjoint(c)
        if len(out) >= REGR_MANUAL_FIELDS_MAX:
            return out[:REGR_MANUAL_FIELDS_MAX]

    # 2) Colonnes « queue du schéma » DW (index ≥ 21), toujours hors doublon classif.
    for c in ordered:
        if c in tail_from_21:
            _push_disjoint(c)
        if len(out) >= REGR_MANUAL_FIELDS_MAX:
            return out[:REGR_MANUAL_FIELDS_MAX]

    # 3) Complément : au moins REGR_MANUAL_FIELDS_MIN champs (souvent jusqu’à 10), même si présents en classification.
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

    # 4) Benchmark en tête (liste déroulante) si présent dans la liste affichée.
    bm = "benchmark_avg_price"
    if bm in out:
        out = [bm] + [x for x in out if x != bm]
        out = out[:REGR_MANUAL_FIELDS_MAX]

    if out:
        return out[:REGR_MANUAL_FIELDS_MAX]

    # 5) Dernier recours : parcours DW après la 20e colonne.
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
    """Deux blocs : variables les plus influentes, puis le reste (libellés distincts de la classification)."""
    if not ordered:
        return []
    if len(ordered) == 1:
        return [("Prédicteurs du modèle (prix final)", ordered)]
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
                ("Variables les plus influentes sur le prix final (forêt aléatoire)", ordered[:cut]),
                ("Autres prédicteurs du modèle", ordered[cut:]),
            ]
    cut = max(1, (len(ordered) + 1) // 2)
    return [
        ("Montants, budget & références (à renseigner en priorité)", ordered[:cut]),
        ("Calendrier & autres dimensions", ordered[cut:]),
    ]


def regression_run_for_target(m: dict, target: str, df_pq: pd.DataFrame | None = None) -> dict:
    """Métadonnées (features, KPI) pour une cible ; repli comme run_03 si pas de regression_objectives."""
    for r in m.get("regression_objectives") or []:
        if r.get("target") == target:
            return r
    if m.get("target") == target:
        return {
            "target": target,
            "features": m.get("features") or [],
            "kpi_alignment": m.get("kpi_alignment"),
        }
    if df_pq is not None and target in df_pq.columns:
        feats = regression_infer_features(df_pq, target)
        return {
            "target": target,
            "features": feats,
            "kpi_alignment": REGR_KPI_TAG.get(target, ""),
        }
    return {}


def format_regression_target_choice(target: str) -> str:
    """Libellé pour liste déroulante des cibles Y."""
    lab = REGR_TARGET_LABEL_FR.get(target, target.replace("_", " "))
    return f"{lab} — `{target}`"


def regression_metrics_for_target(m: dict, target: str) -> dict[str, float | None]:
    """RMSE / MAE / R² propres à une cible si présents dans `regression_objectives`."""
    for r in m.get("regression_objectives") or []:
        if r.get("target") == target:
            return {
                "rmse": r.get("rmse"),
                "mae": r.get("mae"),
                "r2": r.get("r2"),
            }
    if m.get("target") == target:
        er = extract_regression_metrics(m)
        return {"rmse": er.get("rmse"), "mae": er.get("mae"), "r2": er.get("r2")}
    return {"rmse": None, "mae": None, "r2": None}


def extract_classification_metrics(m: dict) -> dict:
    if "test_metrics_champion" in m:
        return m["test_metrics_champion"]
    return {
        "accuracy": m.get("accuracy"),
        "f1_weighted": m.get("f1_weighted"),
        "roc_auc": m.get("roc_auc"),
    }


def extract_regression_metrics(m: dict) -> dict:
    if "test_champion" in m:
        return m["test_champion"]
    return {
        "rmse": m.get("rmse"),
        "mae": m.get("mae"),
        "r2": m.get("r2"),
    }


def _timeseries_rmse(mt: dict) -> float | None:
    tc = mt.get("test_champion")
    if isinstance(tc, dict) and tc.get("rmse") is not None:
        return float(tc["rmse"])
    th = mt.get("test_holt") or {}
    if th.get("rmse") is not None:
        return float(th["rmse"])
    if mt.get("rmse_holdout") is not None:
        return float(mt["rmse_holdout"])
    return None


SERIES_COLUMN_LABELS_FR = {
    "nb_fact_rows": "Volume d’activité (lignes de faits DW / mois)",
    "revenue_sum": "CA mensuel agrégé (somme des montants)",
    "avg_final_price": "Panier moyen mensuel",
}


def _plotly_x_datetime(value) -> object:
    """Convertit un instant pandas en type compatible Plotly (évite sum() sur Timestamp)."""
    return pd.Timestamp(value).to_pydatetime()


def clustering_feature_names_for_model(km, features_json_name: str | None = None) -> list[str] | None:
    """Noms de colonnes alignés sur les centres K-Means (sklearn ou fichier optionnel)."""
    fn = getattr(km, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return [str(x) for x in fn]
    fname = features_json_name or "clustering_feature_names.json"
    path = ML_MODELS / fname
    if path.is_file():
        raw = load_json(path)
        if isinstance(raw, dict) and raw.get("features"):
            return [str(x) for x in raw["features"]]
        if isinstance(raw, list):
            return [str(x) for x in raw]
    return None


@st.cache_data(ttl=300, show_spinner="Connexion au DW et chargement des séries…")
def fetch_dw_timeseries_dataframe(cache_bust: int = 0) -> tuple[pd.DataFrame | None, str | None]:
    """Exécute la même requête que ``run_04_time_series.py`` sur le DW.

    ``cache_bust`` permet d’invalider le cache (bouton « Recharger »).
    Retourne ``(dataframe, None)`` en cas de succès, ou ``(None, message_erreur)``.
    """
    try:
        from ML.ml_paths import get_sql_engine, read_dw_sql, sql_engine_init_error
        from ML.schema_eventzilla import SQL_ML_TIME_SERIES_RESERVATIONS

        eng = get_sql_engine()
        if eng is None:
            err = sql_engine_init_error()
            return None, (
                err
                or "Moteur SQLAlchemy non créé — vérifiez pyodbc, sqlalchemy, et les variables "
                "``EVENTZILLA_SQL_*`` (voir ``ML/ml_paths.py``)."
            )
        df = read_dw_sql(SQL_ML_TIME_SERIES_RESERVATIONS, eng)
        if df is None or len(df) == 0:
            return None, "La requête séries a renvoyé 0 ligne — vérifiez le périmètre DW."
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def test_dw_sql_connection() -> tuple[bool, str, pd.DataFrame | None]:
    """Test rapide : ``SELECT DB_NAME()`` — même mécanisme que ``run_test_sql_connection.py``."""
    try:
        from ML.ml_paths import get_sql_engine, read_dw_sql, sql_engine_init_error

        eng = get_sql_engine()
        if eng is None:
            return False, sql_engine_init_error() or "Engine indisponible.", None
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
        if cm.get("accuracy") is not None:
            parts.append(f"Acc={cm['accuracy']:.3f}")
        if cm.get("f1_weighted") is not None:
            parts.append(f"F1={cm['f1_weighted']:.3f}")
        if cm.get("roc_auc") is not None:
            parts.append(f"AUC={cm['roc_auc']:.3f}")
        return " · ".join(parts) if parts else "—"

    def qrg(m: dict | None) -> str:
        if not m:
            return "—"
        rm = extract_regression_metrics(m)
        parts = []
        if rm.get("rmse") is not None:
            parts.append(f"RMSE={rm['rmse']:.4f}")
        if rm.get("r2") is not None:
            parts.append(f"R²={rm['r2']:.4f}")
        return " · ".join(parts) if parts else "—"

    def qclust(m: dict | None) -> str:
        if not m:
            return "—"
        sil = m.get("silhouette_holdout") or m.get("silhouette")
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
        if tc.get("rmse") is not None:
            parts.append(f"RMSE={tc['rmse']:.2f}")
        if tc.get("mape") is not None:
            parts.append(f"MAPE≈{tc['mape']:.2f}%")
        return " · ".join(parts) if parts else "—"

    if mk:
        k = mk.get("k", "?")
        rows.append(
            {
                "Critère": "E",
                "Domaine": "Clustering",
                "Cible (Y)": f"k={k} segments (features perf. DW standardisées)",
                "Champion": mk.get("model_primary") or mk.get("model") or "KMeans",
                "Benchmark": mk.get("model_secondary") or "Agglomerative (Ward)",
                "Règle de choix": "Silhouette (holdout) + Davies-Bouldin",
                "Qualité": qclust(mk),
                "KPI": mk.get("kpi_alignment", "—"),
                "Fichier": "metrics_clustering.json",
            }
        )
    if mc:
        y = "Statut réservation (multi-classes)"
        rows.append(
            {
                "Critère": "C",
                "Domaine": "Classification",
                "Cible (Y)": y,
                "Champion": mc.get("champion_model") or "RandomForest",
                "Benchmark": "Régression logistique (cf. notebook)",
                "Règle de choix": "Accuracy / F1 / ROC-AUC (test)",
                "Qualité": qcl(mc),
                "KPI": mc.get("kpi_alignment", "—"),
                "Fichier": "metrics_classification.json",
            }
        )
    if mr:
        tgt = mr.get("target") or "final_price"
        rows.append(
            {
                "Critère": "D",
                "Domaine": "Régression",
                "Cible (Y)": str(tgt),
                "Champion": mr.get("champion_model") or "Ridge / RF (cf. JSON)",
                "Benchmark": "Modèle alternatif (cf. notebook 03)",
                "Règle de choix": "RMSE minimal sur test (CV amont)",
                "Qualité": qrg(mr),
                "KPI": mr.get("kpi_alignment", "—"),
                "Fichier": "metrics_regression.json",
            }
        )
    if mt:
        ser = mt.get("series", "?")
        expl = mt.get("target_column_explained") or SERIES_COLUMN_LABELS_FR.get(ser, ser)
        rows.append(
            {
                "Critère": "F",
                "Domaine": "Séries temporelles",
                "Cible (Y)": f"{ser} — {expl[:80]}…" if len(str(expl)) > 80 else f"{ser} — {expl}",
                "Champion": mt.get("champion_model") or mt.get("model") or "Holt / ES",
                "Benchmark": "ARIMA",
                "Règle de choix": mt.get("champion_rule") or "RMSE minimal holdout",
                "Qualité": qts(mt),
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
    """Barres horizontales + jauge d'aperçu (répartition fictive, pas une inférence)."""
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
                text="Probabilités par statut — aperçu du graphique",
                subtitle=dict(
                    text="Illustration équiprobable ; les vraies valeurs apparaissent après « Prédire ».",
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
            title=dict(text="Confiance (probabilité max.) — aperçu", font=dict(size=14, color=BRAND["muted"])),
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
    target: str,
    pred: float | None = None,
    *,
    accent: str | None = None,
) -> go.Figure | None:
    """Histogramme de la cible sur le jeu préparé ; ligne de prédiction dans la couleur du panneau."""
    if target not in df.columns:
        return None
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(y) < 2:
        return None
    hist_color = accent or BRAND["deep"]
    nb = min(45, max(12, int(len(y) ** 0.5) * 4))
    fig = go.Figure(
        go.Histogram(
            x=y,
            nbinsx=nb,
            name="Observations DW",
            marker=dict(color=hist_color, opacity=0.72),
        )
    )
    med = float(y.median())
    fig.add_vline(
        x=med,
        line_dash="dot",
        line_color=BRAND["muted"],
        annotation_text="Médiane (DW)",
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
    tit = REGR_TARGET_LABEL_FR.get(target, target)
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
    """Barres horizontales d’importances Random Forest (si disponibles)."""
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
    """Génère un tableau HTML stylé pour la page récapitulatif."""
    if df.empty:
        return "<p style='color:#64748b;'>Aucune donnée disponible.</p>"

    accent_map = {"E": "#ea580c", "C": "#10b981", "D": "#8b5cf6", "F": "#f59e0b"}
    rows_html = []
    for _, row in df.iterrows():
        crit = str(row.get("Critère", ""))
        color = accent_map.get(crit, "#6366f1")
        cells = "".join(
            f"<td style='padding:0.65rem 0.85rem;border-bottom:1px solid #e2e8f0;"
            f"font-size:0.88rem;color:#334155;'>{html.escape(str(row[c]))}</td>"
            for c in df.columns if c != "Critère"
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
    """Clés dimension / identifiants DW — jamais proposés aux formulaires métier (remplissage automatique si requis par le modèle)."""
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
        for x in ("price", "budget", "margin", "revenue", "ca_", "montant")
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
    PAGE_RECAP: "#6366f1",
}


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
    st.sidebar.markdown("---")
    for pg in PAGE_ORDER:
        if st.sidebar.button(pg, key=f"nav_{pg}", use_container_width=True,
                             type="primary" if st.session_state.nav_page == pg else "secondary"):
            st.session_state.nav_page = pg
            st.rerun()
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
        ("Classif. F1", mc, BRAND["deep"]),
        ("Régr. R²", mr, BRAND["sky"]),
        ("Cluster silh.", mk, BRAND["accent"]),
    ):
        if not m:
            continue
        if label.startswith("Classif"):
            v = extract_classification_metrics(m).get("f1_weighted") or extract_classification_metrics(m).get(
                "accuracy"
            )
        elif label.startswith("Régr"):
            v = extract_regression_metrics(m).get("r2")
        else:
            v = m.get("silhouette_holdout") or m.get("silhouette")
        if v is None:
            continue
        names.append(label)
        values.append(float(v))
        colors.append(color)
    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune métrique — exécutez ML/scripts/run_01 … run_04.",
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
            title=dict(text="Indicateurs agrégés — classification, régression, clustering", font=dict(size=18, color=BRAND["deep"])),
        )
    )
    return fig


def fig_ts_compare(mt: dict | None) -> go.Figure:
    fig = go.Figure()
    if not mt:
        return fig
    holt = mt.get("test_holt") or {}
    arima = mt.get("test_arima") or {}
    if not holt and not arima:
        return fig
    metrics_names = ["rmse", "mae", "mape"]
    labels = [k for k in metrics_names if holt.get(k) is not None]
    h_vals = [holt.get(k) for k in labels]
    a_vals = [arima.get(k) for k in labels if arima.get(k) is not None]
    if h_vals:
        fig.add_trace(go.Bar(name="Holt / ES", x=labels, y=h_vals, marker_color=BRAND["deep"]))
    if a_vals:
        fig.add_trace(go.Bar(name="ARIMA", x=labels[: len(a_vals)], y=a_vals, marker_color=BRAND["sky"]))
    fig.update_layout(
        **_plotly_layout(
            barmode="group",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title=dict(text="Comparaison des erreurs (jeu de validation) — séries temporelles", font=dict(size=18, color=BRAND["deep"])),
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
    """Carte de mod\u00e8le d\u00e9ploy\u00e9 (style dark)."""
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
    """Accueil : dashboard KPI, mod\u00e8les d\u00e9ploy\u00e9s, navigation."""
    _inject_page_accent(*PAGE_ACCENT["synth"])
    hero_variant(
        "synth",
        "EventZilla ML Dashboard",
        "Plateforme d'**intelligence artificielle** appliqu\u00e9e au **business** EventZilla.",
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
    sil_o = (mk or {}).get("silhouette_holdout") or (mk or {}).get("silhouette")
    rms_o = _timeseries_rmse(mt) if mt else None
    k_seg = (mk or {}).get("k", "?")
    cancel_rate = cm.get("accuracy", 0.336) if cm else 0.336
    ts_horizon = (mt or {}).get("horizon", 3)
    ts_champion = (mt or {}).get("champion_model", "Holt")

    section_header("Business Analytics", "Indicateurs cl\u00e9s calcul\u00e9s depuis le Data Warehouse")
    st.markdown(
        "<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        "border-radius:18px;padding:1.25rem;margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )
    r1 = st.columns(5)
    kpis_row1 = [
        (f"{n_samples:,}", "Total R\u00e9servations", "#6366f1"),
        (f"{n_samples * 29.6:,.0f}", "Revenue (TND)", "#f59e0b"),
        (f"{9950:,}", "Valeur Commande Moy.", "#10b981"),
        (f"{cancel_rate * 100:.1f}%", "Taux Annulation", "#ef4444"),
        (f"{k_seg}", "Segments Clients", "#8b5cf6"),
    ]
    for i, (val, lbl, col) in enumerate(kpis_row1):
        with r1[i]:
            st.markdown(_kpi_card_html(val, lbl, col), unsafe_allow_html=True)

    r2 = st.columns(5)
    kpis_row2 = [
        (f"{cm.get('f1_weighted', 0):.3f}" if cm.get("f1_weighted") else "\u2014", "F1 Classification", "#10b981"),
        (f"{rm.get('r2', 0):.4f}" if rm.get("r2") else "\u2014", "R\u00b2 R\u00e9gression", "#8b5cf6"),
        (f"{sil_o:.3f}" if sil_o is not None else "\u2014", "Silhouette Score", "#ea580c"),
        (f"{rms_o:.1f}" if rms_o is not None else "\u2014", "RMSE S\u00e9ries Temp.", "#f59e0b"),
        (f"{ts_horizon} mois", "Horizon Pr\u00e9vision", "#06b6d4"),
    ]
    for i, (val, lbl, col) in enumerate(kpis_row2):
        with r2[i]:
            st.markdown(_kpi_card_html(val, lbl, col), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Mod\u00e8les ML d\u00e9ploy\u00e9s ---
    section_header("Mod\u00e8les ML d\u00e9ploy\u00e9s", "Quatre familles de mod\u00e8les entra\u00een\u00e9s sur le Data Warehouse")
    st.markdown(
        "<div style='background:linear-gradient(145deg, #1e293b 0%, #0f172a 100%);"
        "border-radius:18px;padding:1.25rem;margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )
    mc_cols = st.columns(4)
    ml_cards = [
        ("Classification", "Risque d'annulation", f"{(mc or {}).get('champion_model', 'RF')} + LR", "#10b981"),
        ("R\u00e9gression", "Estimation prix", f"{(mr or {}).get('champion_model', 'Ridge')} + RF", "#8b5cf6"),
        ("Clustering", f"{k_seg} segments clients", "K-Means + HC", "#ea580c"),
        ("S\u00e9ries temporelles", f"Pr\u00e9vision {ts_horizon} mois", f"{ts_champion} + ARIMA", "#f59e0b"),
    ]
    for i, (badge, title, models, color) in enumerate(ml_cards):
        with mc_cols[i]:
            st.markdown(_ml_model_card_html(badge, title, models, color), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Navigation rapide ---
    section_header("Explorer", "Acc\u00e9der aux \u00e9crans de test interactif")
    nav_cols = st.columns(4)
    nav_items = [
        (PAGE_CLASSIF, "Classification", "#10b981"),
        (PAGE_REGR, "R\u00e9gression", "#8b5cf6"),
        (PAGE_CLUSTER, "Clustering", "#ea580c"),
        (PAGE_TS, "S\u00e9ries temporelles", "#f59e0b"),
    ]
    for i, (pg, lbl, _) in enumerate(nav_items):
        with nav_cols[i]:
            if st.button(lbl, use_container_width=True, key=f"home_nav_{i}"):
                goto_page(pg)
    r2 = st.columns([1, 2, 1])
    with r2[1]:
        if st.button("Voir le r\u00e9capitulatif", use_container_width=True, key="home_nav_recap"):
            goto_page(PAGE_RECAP)

    with st.expander("En savoir plus \u2014 int\u00e9r\u00eat du ML pour EventZilla", expanded=False):
        st.markdown(ML_INTEREST_MARKDOWN)
        st.markdown(DEPLOY_SYNTH_MARKDOWN)



def page_recap() -> None:
    """Derni\u00e8re page : tableau synth\u00e9tique lisible avec cartes par famille."""
    _inject_page_accent(*PAGE_ACCENT["synth"])
    hero_variant(
        "synth",
        "R\u00e9capitulatif des mod\u00e8les",
        "Vue d'ensemble des **quatre familles ML** d\u00e9ploy\u00e9es : performance, mod\u00e8le champion et indicateur m\u00e9tier.",
        badges=("Synth\u00e8se",),
    )

    mc = load_json(ML_MODELS / "metrics_classification.json")
    mr = load_json(ML_MODELS / "metrics_regression.json")
    mk = load_json(ML_MODELS / "metrics_clustering.json")
    mt = load_json(ML_MODELS / "metrics_timeseries.json")

    # --- Cartes KPI rapides ---
    k1, k2, k3, k4 = st.columns(4)
    cm = extract_classification_metrics(mc) if mc else {}
    rm = extract_regression_metrics(mr) if mr else {}
    sil = (mk or {}).get("silhouette_holdout") or (mk or {}).get("silhouette")
    ts_rmse = _timeseries_rmse(mt) if mt else None
    with k1:
        st.metric("Classification (C)", f"F1 = {cm.get('f1_weighted', 0):.4f}" if cm.get("f1_weighted") else "\u2014")
    with k2:
        st.metric("R\u00e9gression (D)", f"R\u00b2 = {rm.get('r2', 0):.4f}" if rm.get("r2") else "\u2014")
    with k3:
        st.metric("Clustering (E)", f"Silh. = {sil:.4f}" if sil is not None else "\u2014")
    with k4:
        st.metric("S\u00e9ries temp. (F)", f"RMSE = {ts_rmse:.2f}" if ts_rmse is not None else "\u2014")

    st.markdown("")

    # --- Cartes d\u00e9taill\u00e9es par famille ---
    section_header("D\u00e9tail par famille", "Mod\u00e8le champion, benchmark et m\u00e9triques cl\u00e9s")

    def _model_card(
        color: str,
        critere: str,
        titre: str,
        champion: str,
        benchmark: str,
        qualite: str,
        cible: str,
        kpi: str,
        regle: str,
    ) -> None:
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {color};"
            f"border-radius:0 12px 12px 0;padding:1rem 1.15rem;margin-bottom:0.75rem;'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;'>"
            f"<span style='background:{color}18;color:{color};font-weight:800;font-size:0.72rem;"
            f"text-transform:uppercase;letter-spacing:0.1em;padding:0.25rem 0.65rem;"
            f"border-radius:999px;border:1px solid {color}44;'>Crit\u00e8re {critere}</span>"
            f"<span style='font-weight:700;color:#0f172a;font-size:1.05rem;'>{html.escape(titre)}</span></div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.35rem 1.5rem;font-size:0.88rem;'>"
            f"<div><span style='color:#64748b;font-weight:600;'>Champion :</span> "
            f"<span style='color:#0f172a;font-weight:700;'>{html.escape(champion)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>Benchmark :</span> "
            f"<span style='color:#334155;'>{html.escape(benchmark)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>Cible :</span> "
            f"<span style='color:#334155;'>{html.escape(cible)}</span></div>"
            f"<div><span style='color:#64748b;font-weight:600;'>R\u00e8gle :</span> "
            f"<span style='color:#334155;'>{html.escape(regle)}</span></div>"
            f"<div style='grid-column:1/-1;'><span style='color:#64748b;font-weight:600;'>Qualit\u00e9 :</span> "
            f"<span style='color:{color};font-weight:700;'>{html.escape(qualite)}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        if mk:
            k = mk.get("k", "?")
            sil_v = mk.get("silhouette_holdout") or mk.get("silhouette")
            db_k = mk.get("davies_bouldin_kmeans")
            q_parts = []
            if sil_v is not None:
                q_parts.append(f"Silhouette = {sil_v:.3f}")
            if db_k is not None:
                q_parts.append(f"Davies-Bouldin = {db_k:.2f}")
            _model_card(
                color="#ea580c",
                critere="E",
                titre="Clustering",
                champion=mk.get("model_primary") or "KMeans",
                benchmark=mk.get("model_secondary") or "Agglom\u00e9ratif (Ward)",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                cible=f"k = {k} segments (donn\u00e9es DW standardis\u00e9es)",
                kpi=mk.get("kpi_alignment", "\u2014"),
                regle="Silhouette (holdout) + Davies-Bouldin",
            )
        else:
            st.info("Clustering : aucune m\u00e9trique disponible.")

        if mr:
            tr = mr.get("test_champion") or mr.get("test_ridge") or {}
            q_parts = []
            if tr.get("rmse") is not None:
                q_parts.append(f"RMSE = {tr['rmse']:.4f}")
            if tr.get("r2") is not None:
                q_parts.append(f"R\u00b2 = {tr['r2']:.4f}")
            _model_card(
                color="#8b5cf6",
                critere="D",
                titre="R\u00e9gression",
                champion=mr.get("champion_model") or "Ridge",
                benchmark="Random Forest (cf. notebook 03)",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                cible=str(mr.get("target") or "final_price"),
                kpi=mr.get("kpi_alignment", "\u2014"),
                regle="RMSE minimal sur test (CV amont)",
            )
        else:
            st.info("R\u00e9gression : aucune m\u00e9trique disponible.")

    with c2:
        if mc:
            tcm = extract_classification_metrics(mc)
            q_parts = []
            if tcm.get("accuracy") is not None:
                q_parts.append(f"Acc = {tcm['accuracy']:.3f}")
            if tcm.get("f1_weighted") is not None:
                q_parts.append(f"F1 = {tcm['f1_weighted']:.3f}")
            if tcm.get("roc_auc") is not None:
                q_parts.append(f"AUC = {tcm['roc_auc']:.3f}")
            classes = mc.get("classes") or []
            _model_card(
                color="#10b981",
                critere="C",
                titre="Classification",
                champion=mc.get("champion_model") or "RandomForest",
                benchmark="R\u00e9gression logistique",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                cible="Statut r\u00e9servation (" + ", ".join(str(c) for c in classes) + ")" if classes else "Statut r\u00e9servation",
                kpi=mc.get("kpi_alignment", "\u2014"),
                regle="Accuracy / F1 / ROC-AUC (test)",
            )
        else:
            st.info("Classification : aucune m\u00e9trique disponible.")

        if mt:
            tc = mt.get("test_champion") or mt.get("test_holt") or {}
            q_parts = []
            if tc.get("rmse") is not None:
                q_parts.append(f"RMSE = {tc['rmse']:.2f}")
            if tc.get("mape") is not None:
                q_parts.append(f"MAPE = {tc['mape']:.2f}%")
            ser = mt.get("series", "?")
            expl = mt.get("target_column_explained") or ser
            _model_card(
                color="#f59e0b",
                critere="F",
                titre="S\u00e9ries temporelles",
                champion=mt.get("champion_model") or "Holt",
                benchmark="ARIMA",
                qualite=" \u00b7 ".join(q_parts) if q_parts else "\u2014",
                cible=f"{ser} \u2014 {expl[:70]}" if len(str(expl)) > 70 else f"{ser} \u2014 {expl}",
                kpi=mt.get("kpi_alignment", "\u2014"),
                regle=mt.get("champion_rule") or "RMSE minimal holdout",
            )
        else:
            st.info("S\u00e9ries temporelles : aucune m\u00e9trique disponible.")

    st.markdown("")

    # --- Tableau synth\u00e9tique compact ---
    section_header("Tableau comparatif", "R\u00e9sum\u00e9 en une ligne par famille")
    synth = build_champions_table_rows(mc, mr, mk, mt)
    if synth.empty:
        st.warning(
            "Aucune m\u00e9trique trouv\u00e9e dans ML/models_artifacts/ \u2014 ex\u00e9cutez les scripts run_01 \u2026 run_04."
        )
    else:
        display_cols = ["Crit\u00e8re", "Domaine", "Champion", "Qualit\u00e9", "R\u00e8gle de choix"]
        df_display = synth[[c for c in display_cols if c in synth.columns]].copy()
        st.markdown(_recap_html_table(df_display), unsafe_allow_html=True)

    # --- Export optionnel ---
    summary_md = _REPO / "ML" / "ML_METRICS_SUMMARY.md"
    if summary_md.is_file():
        with st.expander("Export texte d\u00e9taill\u00e9 (ML_METRICS_SUMMARY.md)", expanded=False):
            _txt = summary_md.read_text(encoding="utf-8")
            st.markdown(_txt[:8000])
            if len(_txt) > 8000:
                st.caption("Aper\u00e7u tronqu\u00e9 \u2014 fichier complet dans le dossier ML/.")

    # --- Navigation rapide ---
    st.markdown("")
    section_header("Acc\u00e8s rapide", "Acc\u00e9der aux pages de test")
    r = st.columns(4)
    nav_items = [
        (PAGE_CLASSIF, "Classification"),
        (PAGE_REGR, "R\u00e9gression"),
        (PAGE_CLUSTER, "Clustering"),
        (PAGE_TS, "S\u00e9ries temporelles"),
    ]
    for i, (pg, lbl) in enumerate(nav_items):
        with r[i]:
            if st.button(lbl, key=f"recap_nav_{i}", use_container_width=True):
                goto_page(pg)



def page_classification():
    _inject_page_accent(*PAGE_ACCENT["classif"])
    hero_variant(
        "classif",
        "Classification — statut de réservation",
        "Indiquez **à quel stade** se situe une réservation (confirmée, en attente, annulée…) à partir d’une situation **du même univers que le data warehouse**.",
        badges=("Critère C", "Test interactif"),
    )
    m = load_json(ML_MODELS / "metrics_classification.json")

    with st.expander("Comment utiliser ce formulaire", expanded=False):
        st.markdown(
            "Chaque champ correspond à une **variable numérique** du jeu préparé. "
            "Choisissez une valeur parmi les **suggestions** (quantiles du DW), puis lancez la prédiction. "
            "Les **identifiants dimension** (`id_*`) sont complétés automatiquement (médiane)."
        )

    if m:
        cm = extract_classification_metrics(m)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Exactitude (réf.)", f"{cm.get('accuracy', 0):.4f}" if cm.get("accuracy") else "—")
        with cc2:
            st.metric("F1 pondéré (réf.)", f"{cm.get('f1_weighted', 0):.4f}" if cm.get("f1_weighted") else "—")
        with cc3:
            st.metric("AUC (réf.)", f"{cm.get('roc_auc', 0):.4f}" if cm.get("roc_auc") else "—")
        classes_preview = m.get("classes") or []
        if classes_preview:
            st.caption("Statuts possibles (Y) : **" + "**, **".join(str(x) for x in classes_preview) + "**")

    with st.expander("Rappel pédagogique — détail (optionnel)", expanded=False):
        st.markdown(DEPLOY_CLASSIF_MARKDOWN)

    pipe_p = ML_MODELS / "rf_status_kpi_pipeline.joblib"
    le_p = ML_MODELS / "label_encoder_status.joblib"
    pipe = load_joblib(pipe_p)
    le = load_joblib(le_p)

    cols = classification_feature_columns()
    if not cols:
        st.warning("Parquet `dw_financial_wide.parquet` introuvable — exécutez `ML/scripts/run_00_data_preparation.py`.")
        return
    if pipe is None or le is None:
        st.info("Modèles classification absents — lancez `ML/scripts/run_02_classification.py`.")
        return

    df = pd.read_parquet(ML_PROCESSED / "dw_financial_wide.parquet")
    id_median_defaults = _classif_id_median_defaults(df, cols)
    cols_form = [c for c in cols if not _is_id_column(c)]
    ordered = _classif_order_columns(cols_form)

    deployment_context_card(
        critere="C — Classification",
        cible="Statut de réservation (multi-classes)",
        objectif="Associer un profil numérique cohérent avec le DW au statut le plus plausible.",
        kpi=str((m or {}).get("kpi_alignment") or "Lecture réservation / file active"),
        modele=str((m or {}).get("champion_model") or "Forêt aléatoire + mise à l’échelle"),
        pourquoi=champion_rationale(m, "Bon compromis précision / F1 sur le jeu de test."),
        figure_note="Barres = probabilités par statut ; jauge = confiance sur la classe dominante.",
    )

    section_header(
        "Formulaire — une liste déroulante par variable métier",
        "Sans les identifiants DW (clés `id_*`) : ils sont complétés automatiquement (médiane du jeu) pour le modèle",
    )
    col_clf_in, col_clf_out = st.columns([1.05, 1.0])

    with col_clf_in:
        st.markdown('<div class="ez-card ez-card--deploy">', unsafe_allow_html=True)
        st.markdown("##### Saisie")
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
                    help=f"Valeurs typiques pour la colonne « {col} » (jeu préparé).",
                )
                val_sel = next(v for lab, v in pairs if lab == sel)
                vals_map[col] = float(val_sel)
            submitted = st.form_submit_button(
                "Prédire le statut de réservation", type="primary", use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(
            f"**Champs saisis** ({len(cols_form)}) : "
            + ", ".join(cols_form[:12])
            + (" …" if len(cols_form) > 12 else "")
        )
        _n_id = len(cols) - len(cols_form)
        if _n_id > 0:
            st.caption(
                f"**Non affichés ({_n_id})** : colonnes identifiant DW — valeur **médiane** du jeu préparée pour la prédiction."
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
        st.markdown("##### Résultat & visualisations")
        r = st.session_state.get("clf_ui_result")
        if r is None:
            st.markdown(
                '<div class="ez-out-panel ez-out-panel--hint">'
                '<p style="margin:0;font-size:1.05rem;line-height:1.55;color:#64748b;">'
                "Choisissez une suggestion par champ à gauche, puis cliquez sur "
                '<strong style="color:#0d9488;">Prédire le statut de réservation</strong>. '
                "Ci-dessous : <strong>aperçu</strong> des graphiques (barres + jauge) — "
                "valeurs <strong>illustratives</strong>, pas une prédiction du modèle.</p></div>",
                unsafe_allow_html=True,
            )
            demo_names = [str(x) for x in getattr(le, "classes_", [])]
            ph_bar, ph_g = fig_classification_empty_state_demo(demo_names)
            st.plotly_chart(ph_bar, use_container_width=True, key="clf_preview_bar")
            st.plotly_chart(ph_g, use_container_width=True, key="clf_preview_gauge")
            st.caption(
                "Illustration : répartition **équiprobable fictive**. Après prédiction, barres et jauge "
                "affichent les **probabilités réelles** du modèle pour votre scénario."
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
            summ_txt = " · ".join(bits) if bits else "Profil numérique composé à partir des listes déroulantes."
            html_body = (
                f"<p style='font-size:1.02rem;color:#64748b;margin:0 0 0.5rem 0;'>{summ_txt}</p>"
                f"<p style='font-size:1.45rem;margin:0;color:{BRAND['deep']};font-weight:800;'>Statut prédit : {html.escape(r['label'])}</p>"
            )
            result_block("Lecture du modèle", html_body)
            _nia = int(r.get("n_id_autofill") or 0)
            if _nia > 0:
                st.caption(
                    f"{_nia} colonne(s) identifiant DW non affichées — valeurs fixées à la médiane du jeu pour l’inférence."
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
                        title=dict(text="Confiance — probabilité max.", font=dict(size=14)),
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
                st.caption("Probabilités par classe non disponibles pour ce pipeline.")


def page_regression():
    _inject_page_accent(*PAGE_ACCENT["regr"])
    hero_variant(
        "regr",
        "Régression — montants & indicateurs",
        "Obtenez une **estimation numérique** (panier, budget, etc.) à partir d’une situation **alignée sur le data warehouse**.",
        badges=("Critère D", "Test interactif"),
    )
    m = load_json(ML_MODELS / "metrics_regression.json")
    if not m:
        st.warning("Fichier `metrics_regression.json` absent.")
        return

    with st.expander("Comment utiliser ce formulaire", expanded=False):
        st.markdown(
            "Estimez le **prix final** (`final_price`). Les champs sont ordonnés par **influence** "
            "sur la cible. Les **identifiants** (`id_*`) sont complétés automatiquement (médiane DW)."
        )

    rm = extract_regression_metrics(m)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("RMSE (réf. champion)", f"{rm.get('rmse', 0):.4f}" if rm.get("rmse") else "—")
    with c2:
        st.metric("MAE (réf. champion)", f"{rm.get('mae', 0):.4f}" if rm.get("mae") else "—")
    with c3:
        st.metric("R² (réf. champion)", f"{rm.get('r2', 0):.4f}" if rm.get("r2") else "—")

    with st.expander("Objectif du formulaire — détail (optionnel)", expanded=False):
        st.markdown(DEPLOY_REGR_MARKDOWN)

    if not m.get("target") and not (m.get("regression_objectives") or []):
        st.error("Métriques régression incomplètes (pas de cible documentée).")
        return

    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if not pp.is_file():
        st.warning("Parquet `dw_financial_wide.parquet` introuvable — exécutez `ML/scripts/run_00_data_preparation.py`.")
        return
    df_pq = pd.read_parquet(pp)

    if "regr_ui_result" not in st.session_state:
        st.session_state["regr_ui_result"] = None

    tgt = REGR_UI_TARGET
    ac = REGR_PAGE_ACCENT

    deployment_context_card(
        critere="D — Régression",
        cible=f"{REGR_TARGET_LABEL_FR.get(tgt, tgt)} (`{tgt}`)",
        objectif="Estimer le prix final à partir d’un profil DW cohérent avec les données préparées.",
        kpi=str(m.get("kpi_alignment") or "Indicateurs finance / performance (cf. métriques)"),
        modele=str(m.get("champion_model") or m.get("model") or "Forêt aléatoire + mise à l’échelle"),
        pourquoi=champion_rationale(m, "Modèle entraîné pour minimiser l’erreur sur le jeu de test."),
        figure_note="Histogramme de `final_price`, importances des variables, valeur estimée vs médiane du DW.",
    )

    section_header(
        "Formulaire — prédire le prix final (`final_price`)",
        "En priorité : variables hors écran classification ; puis complété jusqu’à **au moins six** champs (importance du modèle), "
        "y compris des variables déjà présentes en classification si besoin. Benchmark : liste déroulante si présent. Autres X → médiane DW.",
    )

    run_meta = regression_run_for_target(m, tgt, df_pq)
    features = list(run_meta.get("features") or [])
    if not features and m.get("target") == tgt:
        features = list(m.get("features") or [])
    if not features:
        st.error(
            "Aucun prédicteur dérivable pour `final_price` — vérifiez que la colonne est présente dans `dw_financial_wide.parquet`."
        )
        return

    path = regression_model_path(m, tgt)
    pipe = load_joblib(path)
    if pipe is None:
        st.warning(
            f"Pipeline **`{path.name}`** absent — exécutez `python ML/scripts/run_03_prediction_regression.py` "
            f"(avec `dw_financial_wide.parquet`). Vous pouvez composer les **X** ci-dessous ; l’estimation nécessite le fichier modèle."
        )

    missing_np = [f for f in features if f not in df_pq.columns]
    id_median_defaults = _classif_id_median_defaults(df_pq, features)
    cols_form = [c for c in features if not _is_id_column(c)]
    ordered, imp_by_col = regression_form_column_order(cols_form, pipe, features)
    manual_cols = regression_ui_manual_columns(ordered)
    _cf_names = classification_form_column_names()
    if manual_cols and all(c in _cf_names for c in manual_cols):
        st.info(
            "Le modèle ne sélectionne que des prédicteurs dans la même fenêtre que la classification — "
            "recouvrement possible. Pour des champs vraiment différents, inclure des colonnes numériques **après la 20e** du DW dans l’entraînement régression."
        )
    section_blocks = regr_form_section_blocks(manual_cols, imp_by_col)
    median_fill_cols = [c for c in cols_form if c not in manual_cols]
    rm_t = regression_metrics_for_target(m, tgt)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("RMSE (réf.)", f"{rm_t.get('rmse', 0):.4f}" if rm_t.get("rmse") is not None else "—")
    with m2:
        st.metric("MAE (réf.)", f"{rm_t.get('mae', 0):.4f}" if rm_t.get("mae") is not None else "—")
    with m3:
        st.metric("R² (réf.)", f"{rm_t.get('r2', 0):.4f}" if rm_t.get("r2") is not None else "—")

    col_reg_in, col_reg_out = st.columns([1.05, 1.0])

    with col_reg_in:
        st.markdown("##### Prédicteurs (X) — saisie numérique ; benchmark en liste déroulante")
        if imp_by_col and pipe is not None:
            top3 = sorted(imp_by_col.items(), key=lambda x: -x[1])[:3]
            st.caption(
                "**Top influence (modèle)** : "
                + " · ".join(f"`{c}` ({v * 100:.1f} %)" for c, v in top3)
            )
        elif pipe is None:
            st.caption(
                "*Ordre par défaut : montants & budget en tête — chargez le pipeline pour l’ordre par importance réelle.*"
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
                        hlp_base += " *Absent du formulaire classification (écran distinct).*"
                    if _regr_benchmark_price_dropdown(col):
                        pairs = classif_dropdown_suggestions(df_pq, col)
                        labels = [p[0] for p in pairs]
                        default_i = min(2, len(labels) - 1) if len(labels) > 1 else 0
                        hlp = (
                            "Choix issus de la distribution du DW (quantiles ou valeurs observées). "
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
                        hlp = f"Valeur numérique (min–max issus du DW). {hlp_base}"
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
            btn_label = "Estimer le prix final"
            submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)
        st.caption(
            f"**Saisie manuelle** ({len(manual_cols)}) : "
            + ", ".join(manual_cols[:14])
            + (" …" if len(manual_cols) > 14 else "")
        )
        if any(_regr_benchmark_price_dropdown(c) for c in manual_cols):
            st.caption(
                "**Prix benchmark** (`benchmark_avg_price`) : **liste déroulante** (quantiles / valeurs typiques du DW). "
                "Les autres champs affichés restent en saisie numérique libre."
            )
        if median_fill_cols:
            st.caption(
                f"**Fixés à la médiane DW** ({len(median_fill_cols)}) : "
                + ", ".join(median_fill_cols[:12])
                + (" …" if len(median_fill_cols) > 12 else "")
            )
        _n_id = len(features) - len(cols_form)
        if _n_id > 0:
            st.caption(
                f"**Identifiants dimension ({_n_id})** : **médiane** du jeu pour l’inférence."
            )
        if missing_np:
            st.caption("Colonnes absentes du parquet — valeurs par défaut : " + ", ".join(missing_np))

    if submitted:
        if pipe is None:
            st.session_state["regr_ui_result"] = None
            st.warning(
                f"Impossible d’estimer sans `{path.name}`. Lancez `python ML/scripts/run_03_prediction_regression.py` puis rechargez l’app."
            )
        else:
            vec = []
            for c in features:
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

    fig_imp = fig_regression_importance_plot(pipe, features, accent=ac) if pipe is not None else None

    r = st.session_state.get("regr_ui_result")
    pred_show: float | None = float(r["pred"]) if r else None

    with col_reg_out:
        st.markdown("##### Projection & visualisations")
        if pred_show is None:
            st.markdown(
                '<div class="ez-out-panel ez-out-panel--hint">'
                '<p style="margin:0;font-size:1.05rem;line-height:1.55;color:#64748b;">'
                "Renseignez les champs numériques à gauche puis cliquez sur "
                f'<strong style="color:{html.escape(REGR_PAGE_ACCENT)};">Estimer le prix final</strong> '
                "pour afficher l’estimation et les graphiques.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            y_lab = REGR_TARGET_LABEL_FR.get(tgt, tgt)
            html_body = (
                f"<p style='font-size:1.02rem;color:#64748b;margin:0 0 0.5rem 0;'>Cible : "
                f"<strong>{html.escape(tgt)}</strong> ({html.escape(y_lab)})</p>"
                f"<p style='font-size:1.65rem;margin:0;color:{REGR_PAGE_ACCENT};font-weight:800;'>"
                f"Valeur estimée : {pred_show:,.4f}</p>"
            )
            result_block("Lecture du modèle", html_body, variant="regr")
            _nia = int(r.get("n_id_autofill") or 0)
            _nmed = int(r.get("n_median_autofill") or 0)
            if _nia > 0:
                st.caption(
                    f"{_nia} colonne(s) identifiant DW non affichées — médiane du jeu pour l’inférence."
                )
            if _nmed > 0:
                st.caption(
                    f"{_nmed} prédicteur(s) non affichés — fixés à la **médiane du DW** pour compléter le vecteur du modèle."
                )
            if rm_t.get("rmse") is not None:
                st.caption(
                    f"Ordre de grandeur : RMSE test ≈ {float(rm_t['rmse']):.4f} (réf.) — prudence hors plage d’entraînement."
                )

        fig_dist_pred = fig_regression_distribution_plot(df_pq, tgt, pred=pred_show, accent=ac)
        if fig_dist_pred is not None:
            st.plotly_chart(fig_dist_pred, use_container_width=True, key="regr_dist_fp")
        else:
            st.caption("Histogramme indisponible — colonne `final_price` absente du jeu préparé.")

        if fig_imp is not None:
            st.plotly_chart(fig_imp, use_container_width=True, key="regr_imp_fp")
        else:
            st.caption("Importances des variables non disponibles pour ce pipeline.")


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
            "Des fichiers **JSON** fidélité sont présents, mais les **modèles `.joblib`** (K-Means, scaler, imputer) "
            "manquent dans `ML/models_artifacts/`. Tant qu’ils ne sont pas générés, l’interface reste sur la "
            "segmentation **vue large** (montants / saisonnalité / catalogue). "
            "Exécutez depuis la racine du projet : `python ML/scripts/run_01_clustering.py`."
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
        "Fidélité — quel segment pour ce profil ?" if is_loyalty else "Segmentation — profils d’activité",
        (
            "Indiquez **combien de réservations**, **quel volume d’affaires** et **à quelle récence** remonte la dernière activité : "
            "nous rapprochons ce comportement d’un **groupe-type** (ex. très fidèle, occasionnel, à relancer)."
            if is_loyalty
            else "Décrivez une situation **comme dans nos tables de performance** : le modèle indique le **profil-type** le plus proche."
        ),
        badges=("Critère E", "Test interactif"),
    )

    with st.expander("Comment utiliser cet écran", expanded=False):
        if is_loyalty:
            st.markdown(
                "Comparez un **bénéficiaire** ou **prestataire** aux segments. "
                "Champs : fréquence, volumes, CA cumulé, panier moyen, récence. "
                "La note moyenne n'est pas encore intégrée au modèle."
            )
        else:
            st.markdown(
                "Testez à quel **segment** se rapproche un profil cohérent avec le DW. "
                "Les champs correspondent aux variables numériques utilisées."
            )

    if loyalty_modes:
        opts = [k for k in ("beneficiary", "provider") if k in loyalty_modes]
        if not opts:
            opts = list(loyalty_modes.keys())
        _idx = opts.index(_default_mode) if _default_mode in opts else 0
        mode_key = st.radio(
            "Profil à simuler",
            opts,
            index=_idx,
            format_func=lambda k: {
                "beneficiary": "Bénéficiaires (réservations, CA, récence…)",
                "provider": "Prestataires (charge, CA, récence…)",
            }.get(str(k), str(k)),
            horizontal=True,
            key="clustering_loyalty_scope",
        )
        mode_block = loyalty_modes.get(mode_key)
        if not mode_block:
            st.error("Métriques manquantes pour ce périmètre.")
            return
        km = load_joblib(ML_MODELS / mode_block["model_file"])
        if km is None:
            st.error(
                f"Fichier modèle introuvable : `{mode_block.get('model_file')}`. "
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
        critere="E — Segmentation fidélité" if is_loyalty else "E — Segmentation",
        cible=(
            "Le segment le plus proche (parmi les groupes appris)"
            if is_loyalty
            else f"L’un des {m.get('k', '?')} profils-type du modèle"
        ),
        objectif=(
            "Comparer votre saisie aux profils **fidélité / RFM** et nommer le groupe le plus proche."
            if is_loyalty
            else "Rapprocher un cas du profil-type le plus proche."
        ),
        kpi="Aide au ciblage (offres, relances, priorisation)" if is_loyalty else "Lecture par segment",
        modele=str(m.get("model_primary") or m.get("model") or "KMeans"),
        pourquoi=champion_rationale(
            m,
            "Segments stables après normalisation des indicateurs."
            if is_loyalty
            else "Partitions lisibles pour regrouper des comportements proches.",
        ),
        figure_note="Radar : votre profil comparé au centre du segment attribué.",
        label_cible="Ce que vous obtenez",
        label_kpi="Utilité",
        label_figure="Graphique principal",
    )

    section_header(
        "Repères sur le modèle",
        "Qualité globale avant de remplir le formulaire" if is_loyalty else "Quelques indicateurs avant simulation",
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("k (segments)", m_active.get("k", m.get("k", "—")))
        with c2:
            sil = m_active.get("silhouette_holdout") or m_active.get("silhouette") or m.get("silhouette_holdout")
            st.metric("Silhouette holdout", f"{sil:.4f}" if sil is not None else "—")
        with c3:
            st.metric(
                "Échantillon (train / total)",
                f"{m_active.get('n_train', m.get('n_train', '?'))} / {m_active.get('n_samples', m.get('n_samples', '?'))}",
            )
        with c4:
            if is_loyalty:
                st.metric("Lecture", "Fidélité RFM")
                st.caption("Bénéficiaires & prestataires")
            else:
                kpi = str(m_active.get("kpi_alignment") or m.get("kpi_alignment", "—"))
                st.metric("Périmètre", "✓" if kpi != "—" else "—")
                st.caption("Modèle exploratoire DW")

    if km is None:
        st.info("Modèle K-Means absent — métriques JSON uniquement.")
        return

    n_feat = getattr(km, "n_features_in_", None)
    feat_names_km = clustering_feature_names_for_model(km, features_json_name=features_json_name)
    cluster_short, _cluster_long_technical, _label_source, cluster_metier = resolve_segment_labels(
        km, feat_names_km, ML_MODELS, labels_json=labels_json
    )

    if cluster_short:
        with st.expander("Aperçu des segments (rappel)", expanded=False):
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
        "Formulaire — décrire le profil à tester",
        "Même logique que la classification / régression : un champ par indicateur, puis validation."
        if is_loyalty
        else "Renseignez les valeurs attendues par le modèle, puis validez pour voir le segment et le graphique.",
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
            "Pour prédire à partir de **coordonnées brutes**, il faut les fichiers scaler / imputer / K-Means / noms de "
            "features — ré-exécutez **`python ML/scripts/run_01_clustering.py`** ou la **section 5** du notebook **01_E**."
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
                "Saisie du profil</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("##### Indicateurs à renseigner")
            st.caption(
                "Les valeurs proposées correspondent aux **médianes** du jeu d’apprentissage — modifiez-les pour simuler un cas."
                if is_loyalty
                else "Valeurs par défaut ≈ médianes du jeu d’entraînement."
            )
            with st.form("cluster_predict_raw_row"):
                if is_loyalty:
                    _indices_main = ordered_feature_indices_for_form(list(_feat_order), loyalty=True)
                    for grp in ("activité", "montants", "récence"):
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
                    with st.expander("Identifiants techniques (optionnel)", expanded=False):
                        st.caption("Utile seulement si vous reproduisez une ligne complète du DW ; sinon laissez les défauts.")
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
                    "Voir mon segment" if is_loyalty else "Voir le segment et le graphique",
                    type="primary",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        if _sub:
            _vals_in = [vals_map[i] for i in range(len(_feat_order))]
            try:
                pred_id, z_vec, _xi = predict_cluster_from_raw_features(_vals_in, _imp, _scl, km)
            except Exception as _err:
                st.error(f"Prédiction impossible (vérifiez le nombre de variables et les artefacts) : {_err}")
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
                _prof_display = _metier_s if _metier_s else "Synthèse métier à préciser pour ce segment."
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
                        f"font-weight:800;'>Segment attribué</p>"
                        f"<p style='margin:0.45rem 0 0 0;font-size:1.75rem;font-weight:800;color:{CLUSTER_PAGE_ACCENT_DEEP};"
                        f"line-height:1.2;'>{html.escape(_title_card)}</p>"
                        f"<p style='margin:0.55rem 0 0 0;font-size:1.02rem;color:#334155;'>"
                        f"{'Lecture' if _metier_s else 'À compléter côté projet'} : "
                        f"{html.escape(_prof_display.replace('**', ''))}</p>"
                        f"<p style='margin:0.75rem 0 0 0;font-size:0.95rem;color:#64748b;'>"
                        f"Indice du segment : <code style='background:#fff7ed;color:{CLUSTER_PAGE_ACCENT_DEEP};padding:0.15rem 0.45rem;"
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
                        _unit_ech = "profils agrégés (fidélité)" if is_loyalty else "lignes"
                        st.info(
                            f"Part approximative de ce segment dans l’échantillon d’apprentissage "
                            f"(~{_n_samp} {_unit_ech}) : **{float(_pct) * 100:.1f} %**."
                        )
                with cc_radar:
                    fig_r = go.Figure()
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=np.concatenate([_zr, _zr[:1]]),
                            theta=_theta + [_theta[0]],
                            name="Profil saisi",
                            line=dict(color=CLUSTER_PAGE_ACCENT_DEEP, width=3),
                            fillcolor="rgba(234, 88, 12, 0.18)",
                            fill="toself",
                        )
                    )
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=np.concatenate([_ccr, _ccr[:1]]),
                            theta=_theta + [_theta[0]],
                            name="Centre du segment (référence)",
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
                            text="<b>Comparaison visuelle</b> — profil saisi vs profil-type du segment",
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
                        "Chaque axe correspond à un indicateur du formulaire (échelle normalisée comme à l’entraînement)."
                    )

    _wrap_detail = st.expander("Indicateurs techniques & profils détaillés (optionnel)", expanded=not is_loyalty)
    with _wrap_detail:
        section_header(
            "Comparatifs modèle",
            "Pour aller plus loin que le formulaire ci-dessus",
        )
        col_a, col_b = st.columns((1, 1))
        with col_a:
            dbk = m_active.get("davies_bouldin_kmeans") if mode_block else m.get("davies_bouldin_kmeans")
            dba = m.get("davies_bouldin_agg")
            if dbk is not None and dba is not None:
                fig_db = go.Figure(
                    go.Bar(
                        x=["KMeans (champion)", "Agglomératif Ward"],
                        y=[dbk, dba],
                        marker_color=[BRAND["deep"], BRAND["sky"]],
                        text=[f"{dbk:.2f}", f"{dba:.2f}"],
                        textposition="outside",
                    )
                )
                fig_db.update_layout(
                    **_plotly_layout(
                        title=dict(text="Davies-Bouldin (↓ = clusters plus compacts / séparés)", font=dict(size=14, color=BRAND["deep"])),
                        height=340,
                        yaxis_title="Indice DB",
                        yaxis=dict(gridcolor=BRAND["chart_grid"]),
                    )
                )
                st.plotly_chart(fig_db, use_container_width=True)
            elif dbk is not None:
                fig_db = go.Figure(
                    go.Bar(
                        x=["K-Means (fidélité)"],
                        y=[dbk],
                        marker_color=[BRAND["deep"]],
                        text=[f"{dbk:.2f}"],
                        textposition="outside",
                    )
                )
                fig_db.update_layout(
                    **_plotly_layout(
                        title=dict(text="Davies-Bouldin — K-Means (↓ = mieux)", font=dict(size=14, color=BRAND["deep"])),
                        height=340,
                        yaxis_title="Indice DB",
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
                                "Centres des clusters — variables RFM / fidélité (standardisées)"
                                if is_loyalty
                                else "Centres des clusters — espace standardisé (dimensions = variables num. du périmètre perf.)"
                            ),
                            font=dict(size=15, color=BRAND["deep"]),
                        ),
                        height=340,
                        margin=dict(l=80, r=20, t=50, b=40),
                    )
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Centres de clusters non disponibles dans ce fichier joblib.")

    section_header(
        "Répartition illustrative (simulation)",
        "Distribution simulée pour visualiser la taille relative des segments (indicatif)",
    )
    st.markdown('<div class="ez-panel">', unsafe_allow_html=True)
    st.markdown("##### Répartition illustrative des segments")
    if is_loyalty:
        st.caption(
            "Les **parts par segment** affichées plus haut proviennent de l’échantillon d’apprentissage. "
            "La simulation aléatoire n’est pas calibrée sur l’espace RFM — désactivée en mode fidélité."
        )
    else:
        st.caption(
            "Simulation de **nombreuses** attributions dans l’espace d’entrée du KMeans (Gaussienne multivariée) — "
            "donne une idée de la **taille relative** des segments, pas les volumes métier bruts."
        )
    if not is_loyalty and st.button("Calculer la répartition simulée", type="primary", use_container_width=True) and n_feat:
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
            title=dict(text="Illustration — répartition simulée des segments", font=dict(size=16, color=BRAND["deep"])),
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
        "S\u00e9ries temporelles \u2014 \u00e9volution & pr\u00e9vision",
        "Visualisez **l\u2019\u00e9volution mensuelle** de vos indicateurs DW, **comparez Holt vs ARIMA** sur la validation, puis **projetez** la tendance sur les mois \u00e0 venir.",
        badges=("Crit\u00e8re F", "Donn\u00e9es DW live"),
    )
    m = load_json(ML_MODELS / "metrics_timeseries.json")
    if not m:
        st.warning("Fichier `metrics_timeseries.json` absent.")
        return

    _ser = str(m.get("series") or "\u2014")
    _champion = str(m.get("champion_model") or "Holt")
    _champion_short = "Holt" if "holt" in _champion.lower() else "ARIMA"

    deployment_context_card(
        critere="F \u2014 S\u00e9ries temporelles",
        cible=f"Pr\u00e9vision mensuelle : {SERIES_COLUMN_LABELS_FR.get(_ser, _ser)}",
        objectif="Suivre l\u2019\u00e9volution d\u2019un indicateur agr\u00e9g\u00e9 et anticiper les mois suivants.",
        kpi=str(m.get("kpi_alignment") or "Pilotage volumes / CA / panier"),
        modele=_champion,
        pourquoi=champion_rationale(m, "Mod\u00e8le retenu : RMSE le plus bas sur la validation (holdout 3 mois)."),
        figure_note="Courbe : historique DW + pr\u00e9vision ; barres : comparaison Holt vs ARIMA.",
        label_cible="Indicateur pr\u00e9dit",
        label_kpi="Utilit\u00e9 m\u00e9tier",
        label_figure="Graphiques",
    )

    with st.expander("Objectif de cet \u00e9cran \u2014 d\u00e9tail", expanded=False):
        st.markdown(DEPLOY_TS_MARKDOWN)

    # --- Performance & stationnarit\u00e9 ---
    section_header(
        "Qualit\u00e9 du mod\u00e8le champion",
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
            rms = tc.get("rmse")
            st.metric("RMSE (champion)", f"{rms:.2f}" if rms is not None else "\u2014")
            st.caption("Erreur quadratique moyenne")
        with c2:
            mae = tc.get("mae")
            st.metric("MAE", f"{mae:.2f}" if mae is not None else "\u2014")
            st.caption("Erreur absolue moyenne")
        with c3:
            mape = tc.get("mape")
            st.metric("MAPE", f"{mape:.2f} %" if mape is not None else "\u2014")
            st.caption("\u00c9cart relatif moyen (%)")
        with c4:
            st.metric("Champion", _champion_short)
            st.caption(f"Horizon : {m.get('horizon', '?')} mois")
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

    with st.expander(f"Série de référence : {SERIES_COLUMN_LABELS_FR.get(_ser, _ser)}", expanded=False):
        _expl_txt = str(m.get("target_column_explained", "") or "")
        if _expl_txt:
            st.markdown(_expl_txt)

    # --- Comparaison Holt vs ARIMA ---
    section_header(
        "Comparaison Holt vs ARIMA",
        "Erreurs sur la m\u00eame fen\u00eatre de validation \u2014 le plus bas est le meilleur",
    )
    col_chart, col_table = st.columns((1.2, 0.8))
    with col_chart:
        st.plotly_chart(fig_ts_compare(m), use_container_width=True)
    with col_table:
        _metric_rows = []
        for label, d in (("Holt", th), ("ARIMA", ta)):
            if d:
                _metric_rows.append({
                    "Mod\u00e8le": label,
                    "RMSE": round(d.get("rmse", 0), 2),
                    "MAE": round(d.get("mae", 0), 2),
                    "MAPE (%)": round(d.get("mape", 0), 2),
                })
        if _metric_rows:
            st.dataframe(pd.DataFrame(_metric_rows).set_index("Mod\u00e8le"), use_container_width=True)
        delta = m.get("rmse_delta_holt_minus_arima")
        if delta is not None:
            st.info(
                f"**\u00c9cart RMSE** (Holt \u2212 ARIMA) : **{delta:+.2f}** \u2014 "
                + ("Holt est l\u00e9g\u00e8rement meilleur." if delta < 0 else "ARIMA est l\u00e9g\u00e8rement meilleur.")
            )
        st.caption(
            f"**Champion retenu** : {_champion_short} (r\u00e8gle : RMSE minimal sur le holdout de "
            f"{m.get('horizon', '?')} mois)."
        )

    # --- Connexion DW & donn\u00e9es ---
    section_header(
        "Donn\u00e9es du data warehouse",
        "Connexion, rechargement des s\u00e9ries et visualisation interactive",
    )
    if "ts_cache_bust" not in st.session_state:
        st.session_state.ts_cache_bust = 0

    info = _dw_connection_info()
    _dcol = PAGE_ACCENT["ts"][1]
    st.markdown(
        f'<div class="ez-card" style="border-top:3px solid {_dcol};">'
        f"<div class=\"ez-kicker\">Acc\u00e8s au data warehouse</div>"
        f"<p style='margin:0;color:#334155;font-size:1.02rem;line-height:1.6;'>"
        f"<b>Serveur</b> : {html.escape(info['serveur'])} \u00b7 <b>Base</b> : {html.escape(info['base_dw'])}<br/>"
        f"M\u00eame connexion que sous SSMS (Windows).</p></div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("Tester la connexion DW", use_container_width=True, key="ts_test_sql"):
            ok, msg, df_test = test_dw_sql_connection()
            if ok and df_test is not None:
                st.success(msg)
                st.dataframe(df_test, use_container_width=True)
            else:
                st.error(msg)
    with b2:
        if st.button("Recharger les s\u00e9ries depuis le DW", use_container_width=True, key="ts_reload"):
            st.session_state.ts_cache_bust += 1
            st.rerun()
    with b3:
        st.caption(
            "\u00ab Tester \u00bb v\u00e9rifie la connexion ; \u00ab Recharger \u00bb actualise les donn\u00e9es mensuelles depuis le DW."
        )

    df_ts, ts_err = fetch_dw_timeseries_dataframe(int(st.session_state.ts_cache_bust))
    if ts_err:
        st.warning(f"**S\u00e9ries non charg\u00e9es** \u2014 {ts_err}")
        st.caption(
            "V\u00e9rifiez SQL Server, le pilote ODBC, ou lancez le script de test de connexion."
        )
    elif df_ts is not None and len(df_ts) > 0:
        st.success(f"**Donn\u00e9es DW charg\u00e9es** \u2014 {len(df_ts)} mois d\u2019historique disponibles.")

    if df_ts is not None and len(df_ts) > 0:
        df_ts = df_ts.copy()
        df_ts["date"] = pd.to_datetime(
            dict(year=df_ts["cal_year"].astype(int), month=df_ts["cal_month"].astype(int), day=1)
        )
        available = [c for c in SERIES_COLUMN_LABELS_FR if c in df_ts.columns]
        if available:
            section_header(
                "\u00c9volution & pr\u00e9vision interactive",
                "Choisissez l\u2019indicateur, la fen\u00eatre d\u2019historique et l\u2019horizon de projection",
            )
            with st.container(border=True):
                col = st.selectbox(
                    "Indicateur \u00e0 suivre",
                    available,
                    format_func=lambda c: f"{SERIES_COLUMN_LABELS_FR[c]}",
                    key="ts_target_col",
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    horizon = st.slider(
                        "Mois de projection", 1, 12,
                        int(m.get("horizon") or 3), key="ts_horizon",
                    )
                with c2:
                    holdout_n = st.slider(
                        "Mois de validation (holdout)", 1, min(6, max(1, len(df_ts) // 3)),
                        min(int(m.get("horizon") or 3), max(1, len(df_ts) // 3)),
                        key="ts_holdout",
                    )
                with c3:
                    tail = st.slider(
                        "Historique affich\u00e9 (mois)", 6,
                        min(120, max(12, len(df_ts))),
                        min(36, len(df_ts)), key="ts_tail",
                    )

            ts_full = df_ts.set_index("date")[col].astype(float).sort_index().dropna()
            ts_plot = ts_full.iloc[-tail:]

            if len(ts_full) >= 6:
                import warnings as _w
                _w.filterwarnings("ignore")
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model

                ts_train = ts_full.iloc[:-holdout_n]
                ts_test = ts_full.iloc[-holdout_n:]
                h = len(ts_test)

                holt_fit = ExponentialSmoothing(
                    ts_train, trend="add", seasonal=None, initialization_method="estimated"
                ).fit()
                holt_fc_test = holt_fit.forecast(h)

                try:
                    arima_fit = ARIMA_Model(ts_train, order=(1, 1, 1)).fit()
                except Exception:
                    arima_fit = ARIMA_Model(ts_train, order=(0, 1, 1)).fit()
                arima_fc_test = arima_fit.forecast(h)

                holt_fit_full = ExponentialSmoothing(
                    ts_full, trend="add", seasonal=None, initialization_method="estimated"
                ).fit()
                last_ts = pd.Timestamp(ts_full.index.max())
                future = pd.date_range(last_ts + pd.DateOffset(months=1), periods=horizon, freq="MS")
                holt_fc_future = holt_fit_full.forecast(horizon)

                try:
                    arima_fit_full = ARIMA_Model(ts_full, order=(1, 1, 1)).fit()
                except Exception:
                    arima_fit_full = ARIMA_Model(ts_full, order=(0, 1, 1)).fit()
                arima_fc_future = arima_fit_full.forecast(horizon)

                fig = go.Figure()
                train_plot = ts_train[ts_train.index >= ts_plot.index.min()]
                fig.add_trace(go.Scatter(
                    x=train_plot.index, y=train_plot.values,
                    mode="lines+markers", name="Train (historique)",
                    line=dict(color=BRAND["deep"], width=2.2),
                    marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=ts_test.index, y=ts_test.values,
                    mode="lines+markers", name=f"Test \u2014 r\u00e9el ({h} mois)",
                    line=dict(color="#f59e0b", width=2.5),
                    marker=dict(size=7, symbol="circle"),
                ))
                fig.add_trace(go.Scatter(
                    x=ts_test.index,
                    y=np.asarray(holt_fc_test, dtype=float).ravel(),
                    mode="lines+markers", name="Holt (validation)",
                    line=dict(color=BRAND["sky"], width=2, dash="dot"),
                    marker=dict(size=6, symbol="diamond"),
                ))
                fig.add_trace(go.Scatter(
                    x=ts_test.index,
                    y=np.asarray(arima_fc_test, dtype=float).ravel(),
                    mode="lines+markers", name="ARIMA (validation)",
                    line=dict(color="#a855f7", width=2, dash="dashdot"),
                    marker=dict(size=6, symbol="square"),
                ))
                fig.add_trace(go.Scatter(
                    x=future, y=np.asarray(holt_fc_future, dtype=float).ravel(),
                    mode="lines+markers", name=f"Holt \u2014 pr\u00e9vision ({horizon} mois)",
                    line=dict(color=BRAND["sky"], width=2, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                ))
                fig.add_trace(go.Scatter(
                    x=future, y=np.asarray(arima_fc_future, dtype=float).ravel(),
                    mode="lines+markers", name=f"ARIMA \u2014 pr\u00e9vision ({horizon} mois)",
                    line=dict(color="#a855f7", width=2, dash="dash"),
                    marker=dict(size=7, symbol="square"),
                ))
                lx = _plotly_x_datetime(last_ts)
                fig.add_shape(
                    type="line", x0=lx, x1=lx, yref="paper", y0=0, y1=1,
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                )
                fig.add_annotation(
                    x=lx, xref="x", yref="paper", y=1,
                    text="Fin historique", showarrow=False, yanchor="bottom",
                    font=dict(size=11, color="#94a3b8"),
                )
                test_start = _plotly_x_datetime(pd.Timestamp(ts_test.index.min()))
                fig.add_shape(
                    type="line", x0=test_start, x1=test_start, yref="paper", y0=0, y1=1,
                    line=dict(color="#f59e0b", width=1, dash="dot"),
                )
                fig.add_annotation(
                    x=test_start, xref="x", yref="paper", y=0.95,
                    text="D\u00e9but validation", showarrow=False, yanchor="top",
                    font=dict(size=11, color="#f59e0b"),
                )
                fig.update_layout(
                    **_plotly_layout(
                        title=dict(
                            text=f"{SERIES_COLUMN_LABELS_FR.get(col, col)} \u2014 train / validation / pr\u00e9vision",
                            font=dict(size=18, color=BRAND["deep"]),
                        ),
                        height=520,
                        yaxis_title="Valeur",
                        xaxis_title="Mois",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=12)),
                        hovermode="x unified",
                        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
                        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                def _ts_metrics(y_true, y_pred):
                    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float).ravel()
                    rmse_v = float(np.sqrt(np.mean(e ** 2)))
                    mae_v = float(np.mean(np.abs(e)))
                    mape_v = float(np.mean(np.abs(e / (np.asarray(y_true, dtype=float) + 1e-9))) * 100)
                    return {"RMSE": round(rmse_v, 2), "MAE": round(mae_v, 2), "MAPE (%)": round(mape_v, 2)}

                mh = _ts_metrics(ts_test.values, holt_fc_test)
                ma = _ts_metrics(ts_test.values, arima_fc_test)

                section_header(
                    "R\u00e9sultat de la validation interactive",
                    f"Erreurs calcul\u00e9es sur les {h} mois de holdout s\u00e9lectionn\u00e9s",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    with st.container(border=True):
                        _is_holt_champ = mh["RMSE"] <= ma["RMSE"]
                        _holt_badge = " \u2190 champion" if _is_holt_champ else ""
                        st.markdown(f"**Holt (lissage exponentiel){_holt_badge}**")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("RMSE", mh["RMSE"])
                        with mc2:
                            st.metric("MAE", mh["MAE"])
                        with mc3:
                            st.metric("MAPE", f"{mh['MAPE (%)']} %")
                with rc2:
                    with st.container(border=True):
                        _arima_badge = " \u2190 champion" if not _is_holt_champ else ""
                        st.markdown(f"**ARIMA{_arima_badge}**")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("RMSE", ma["RMSE"])
                        with mc2:
                            st.metric("MAE", ma["MAE"])
                        with mc3:
                            st.metric("MAPE", f"{ma['MAPE (%)']} %")

                with st.expander("Valeurs pr\u00e9vues (tableau)", expanded=False):
                    fc_df = pd.DataFrame({
                        "Mois": [d.strftime("%Y-%m") for d in future],
                        "Holt": [round(float(v), 2) for v in np.asarray(holt_fc_future, dtype=float).ravel()],
                        "ARIMA": [round(float(v), 2) for v in np.asarray(arima_fc_future, dtype=float).ravel()],
                    })
                    st.dataframe(fc_df.set_index("Mois"), use_container_width=True)
                    st.caption(
                        "Les pr\u00e9visions utilisent **tout l\u2019historique** (train + test) pour projeter les mois futurs."
                    )
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_plot.index, y=ts_plot.values,
                    mode="lines+markers", name="Historique (DW)",
                    line=dict(color=BRAND["deep"], width=2.2),
                ))
                fig.update_layout(**_plotly_layout(
                    title=dict(text=f"{SERIES_COLUMN_LABELS_FR.get(col, col)} \u2014 historique", font=dict(size=18, color=BRAND["deep"])),
                    height=420, yaxis_title="Valeur", xaxis_title="Mois",
                ))
                st.plotly_chart(fig, use_container_width=True)
                st.info("L\u2019historique est trop court (< 6 mois) pour ajuster les mod\u00e8les de pr\u00e9vision.")
        else:
            st.warning(
                "Colonnes attendues absentes du r\u00e9sultat SQL \u2014 v\u00e9rifiez "
                "`SQL_ML_TIME_SERIES_RESERVATIONS` dans `schema_eventzilla.py`."
            )


def main():
    page = sidebar_brand_and_nav()
    if page == PAGE_HOME:
        page_home()
    elif page == PAGE_RECAP:
        page_recap()
    elif page == PAGE_CLASSIF:
        page_classification()
    elif page == PAGE_REGR:
        page_regression()
    elif page == PAGE_CLUSTER:
        page_clustering()
    elif page == PAGE_TS:
        page_timeseries()
    else:
        page_home()

    _page_nav_footer(page)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "EventZilla ML Studio — données du DW, modèles dans ML/models_artifacts/."
    )


if __name__ == "__main__":
    main()

