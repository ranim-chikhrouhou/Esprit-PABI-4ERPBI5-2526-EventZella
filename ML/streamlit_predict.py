# -*- coding: utf-8 -*-
"""
EventZilla — Interface Prédiction ML (Validation S12)
=====================================================
Web App Streamlit connectée à l'API FastAPI.
Appel complet : Interface → /auth/login → /predict/* → Résultat affiché

Lancement :
    streamlit run ML/streamlit_predict.py --server.port 8501

API cible (configurable) :
    http://localhost:8000   (local)
    http://fastapi:8000     (Docker)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
_METRICS_TS_PATH = _REPO_ROOT / "ML" / "models_artifacts" / "metrics_timeseries.json"


def _load_timeseries_metrics_local() -> dict:
    """Charge metrics_timeseries.json (params Holt / métriques champion)."""
    if _METRICS_TS_PATH.is_file():
        try:
            return json.loads(_METRICS_TS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _demo_monthly_volumes(n_months: int = 48, seed: int = 42) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Série mensuelle synthétique type nb_fact_rows (DW EventZilla)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months, dtype=float)
    level = 820 + 4.2 * t + rng.normal(0, 35, n_months)
    seasonal = 45 * np.sin(2 * np.pi * (t % 12) / 12)
    y = np.clip(level + seasonal + rng.normal(0, 28, n_months), 200, None)
    start = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start, periods=n_months, freq="MS")
    return dates, y.astype(float)


def _holt_numpy_forecast(y: np.ndarray, alpha: float, beta: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Holt additif (double lissage) ; fitted[t] = prévision 1-pas depuis t-1."""
    n = len(y)
    level = float(y[0])
    trend = float(y[1] - y[0]) if n > 1 else 0.0
    fitted = np.zeros(n)
    fitted[0] = level
    for t in range(1, n):
        prev_l, prev_b = level, trend
        forecast_1 = prev_l + prev_b
        fitted[t] = forecast_1
        level = alpha * y[t] + (1 - alpha) * forecast_1
        trend = beta * (level - prev_l) + (1 - beta) * prev_b
    last_l, last_b = level, trend
    future = np.array([last_l + (j + 1) * last_b for j in range(h)], dtype=float)
    return fitted, future


def _holt_statsmodels_fit_forecast(
    y: np.ndarray, alpha: float, beta: float, h: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            optimized=False,
        )
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        fc = np.asarray(fit.forecast(h), dtype=float)
        return fitted, fc
    except Exception:
        return None, None


def build_timeseries_chart_figure(
    dates: pd.DatetimeIndex,
    y_hist: np.ndarray,
    horizon: int,
    alpha: float,
    beta: float,
) -> go.Figure:
    """Historique + prévision Holt + zone prévision."""
    y_hist = np.asarray(y_hist, dtype=float)
    fitted_sm, future_sm = _holt_statsmodels_fit_forecast(y_hist, alpha, beta, horizon)
    if fitted_sm is None or future_sm is None:
        fitted_sm, future_sm = _holt_numpy_forecast(y_hist, alpha, beta, horizon)

    last_date = dates[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_hist,
            mode="lines+markers",
            name="Historique (nb_fact_rows)",
            line=dict(color="#0d9488", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fitted_sm,
            mode="lines",
            name="Ajustement Holt (in-sample)",
            line=dict(color="#06b6d4", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_sm,
            mode="lines+markers",
            name=f"Prévision (+{horizon} mois)",
            line=dict(color="#a855f7", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_vrect(
        x0=future_dates[0],
        x1=future_dates[-1],
        fillcolor="rgba(168, 85, 247, 0.08)",
        layer="below",
        line_width=0,
    )
    fig.update_layout(
        title="Volume mensuel — Holt (tendance additive)",
        xaxis_title="Mois",
        yaxis_title="Nb lignes de faits (volume)",
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=48, r=24, t=56, b=48),
    )
    return fig


def build_rmse_comparison_figure(holt_m: dict, arima_m: dict) -> go.Figure:
    """Barres RMSE / MAPE Holt vs ARIMA."""
    labels = ["RMSE", "MAPE (%)", "MAE"]
    holt_vals = [holt_m.get("rmse"), holt_m.get("mape"), holt_m.get("mae")]
    arima_vals = [arima_m.get("rmse"), arima_m.get("mape"), arima_m.get("mae")]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Holt (champion)", x=labels, y=holt_vals, marker_color="#0d9488")
    )
    fig.add_trace(
        go.Bar(name="ARIMA (baseline)", x=labels, y=arima_vals, marker_color="#94a3b8")
    )
    fig.update_layout(
        title="Comparaison métriques test — Holt vs ARIMA",
        barmode="group",
        template="plotly_white",
        height=380,
        margin=dict(l=48, r=24, t=56, b=48),
    )
    return fig

# ── Configuration ─────────────────────────────────────────────────────────
API_BASE = os.environ.get("EVENTZILLA_API_URL", "http://localhost:8000")

USERS = {
    "naima_sarraj":      {"password": "Naima@Finance2025!",    "role": "Finance"},
    "ranim_chikhrouhou": {"password": "Ranim@Marketing2025!",  "role": "Marketing"},
    "anas_allam":        {"password": "Anas@CRM2025!",         "role": "CRM"},
}

# ── Style ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EventZilla ML",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background: #f0fdf4;
        border: 1px solid #16a34a;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .error-box {
        background: #fef2f2;
        border: 1px solid #dc2626;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def api_login(login: str, password: str) -> str | None:
    """Retourne le JWT token ou None en cas d'erreur."""
    try:
        r = requests.post(
            f"{API_BASE}/auth/login",
            json={"login": login, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("access_token")
        st.error(f"Authentification échouée : {r.json().get('detail', r.text)}")
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Impossible de contacter l'API sur {API_BASE}\n"
                 "Vérifiez que l'API FastAPI tourne : `python run_fastapi.py`")
    return None


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def api_get(path: str, token: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=auth_header(token),
                         params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API {r.status_code} : {r.text[:300]}")
    except Exception as e:
        st.error(f"Erreur réseau : {e}")
    return None


def api_post(path: str, token: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", headers=auth_header(token),
                          json=body, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API {r.status_code} : {r.text[:300]}")
    except Exception as e:
        st.error(f"Erreur réseau : {e}")
    return None


# ── Sidebar — Login ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎫 EventZilla ML")
    st.markdown("---")

    if "token" not in st.session_state:
        st.session_state.token = None
        st.session_state.user_login = None
        st.session_state.user_role = None

    if st.session_state.token is None:
        st.markdown("#### Connexion")
        sel_user = st.selectbox("Utilisateur", list(USERS.keys()),
                                format_func=lambda u: f"{u} ({USERS[u]['role']})")
        pwd = st.text_input("Mot de passe", type="password",
                            value=USERS[sel_user]["password"])
        if st.button("🔐 Se connecter", use_container_width=True):
            token = api_login(sel_user, pwd)
            if token:
                st.session_state.token = token
                st.session_state.user_login = sel_user
                st.session_state.user_role = USERS[sel_user]["role"]
                st.success(f"Connecté en tant que **{sel_user}**")
                st.rerun()
    else:
        st.success(f"✅ **{st.session_state.user_login}**")
        st.info(f"Rôle : {st.session_state.user_role}")
        st.markdown(f"API : `{API_BASE}`")
        st.markdown("---")
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.token = None
            st.session_state.user_login = None
            st.session_state.user_role = None
            st.rerun()

    st.markdown("---")
    st.markdown("**Liens utiles**")
    st.markdown(f"[📖 Docs API]({API_BASE}/docs)")
    st.markdown("[📊 MLflow UI](http://localhost:5000)")


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2>🎫 EventZilla — Plateforme ML</h2>
    <p>Interface de prédiction connectée à l'API FastAPI · Validation S12 MLOps</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.token is None:
    st.info("👈 Veuillez vous connecter dans la barre latérale pour accéder aux prédictions.")
    st.markdown("### Architecture S12")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🧠 Modèles ML**")
        st.markdown("- Random Forest\n- Ridge Regression\n- K-Means\n- Holt TimeSeries")
    with col2:
        st.markdown("**🔌 API FastAPI**")
        st.markdown("- `/auth/login`\n- `/predict/classification`\n- `/predict/regression`\n- `/predict/timeseries`")
    with col3:
        st.markdown("**📊 MLflow**")
        st.markdown("- Tracking des runs\n- Comparaison\n- Versioning\n- Artefacts")
    with col4:
        st.markdown("**🐳 Docker**")
        st.markdown("- FastAPI :8000\n- MLflow :5000\n- Streamlit :8502\n- Docker Compose")
    st.stop()

# ── Navigation ────────────────────────────────────────────────────────────
token = st.session_state.token
tabs = st.tabs([
    "🏠 Santé API",
    "💰 Régression (Prix)",
    "🎯 Classification (Statut)",
    "👥 Segmentation (RFM)",
    "📈 Séries Temporelles",
    "📋 Métriques Modèles",
])


# ── Tab 0 : Health Check ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🏠 Santé de l'API")
    if st.button("🔄 Vérifier l'état de l'API"):
        try:
            r = requests.get(f"{API_BASE}/", timeout=5)
            data = r.json()
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.json(data)
            st.markdown('</div>', unsafe_allow_html=True)

            modeles = data.get("modeles_charges", {})
            st.markdown("#### État des modèles chargés")
            cols = st.columns(4)
            for i, (k, v) in enumerate(modeles.items()):
                with cols[i % 4]:
                    if v:
                        st.success(f"✅ {k}")
                    else:
                        st.error(f"❌ {k}")
        except Exception as e:
            st.error(f"L'API n'est pas joignable : {e}")


# ── Tab 1 : Régression ────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 💰 Prédiction du montant final (Ridge)")
    st.info("Modèle : **Ridge** — Prédit le prix final d'une réservation en TND")

    with st.form("form_regression"):
        col1, col2, col3 = st.columns(3)
        with col1:
            service_price    = st.number_input("Prix du service (TND)", 100.0, 50000.0, 1200.0, step=50.0)
            benchmark_avg    = st.number_input("Prix benchmark moyen (TND)", 100.0, 50000.0, 1300.0, step=50.0)
            event_budget     = st.number_input("Budget événement (TND)", 500.0, 100000.0, 5000.0, step=100.0)
        with col2:
            commission_margin = st.number_input("Marge commission (TND)", 0.0, 5000.0, 150.0, step=10.0)
            cal_month        = st.selectbox("Mois", list(range(1, 13)), index=3)
            cal_year         = st.selectbox("Année", [2023, 2024, 2025], index=1)
        with col3:
            quarter          = st.selectbox("Trimestre", [1, 2, 3, 4], index=1)
            id_event         = st.number_input("ID Événement", 1, 500, 42)
            id_provider      = st.number_input("ID Prestataire", 1, 200, 7)

        submitted = st.form_submit_button("🔮 Prédire le prix", use_container_width=True)

    if submitted:
        body = {
            "id_date": 1.0, "id_event": float(id_event),
            "id_servicecategory": 3.0, "id_benchmark": 2.0,
            "id_provider": float(id_provider),
            "service_price": service_price,
            "benchmark_avg_price": benchmark_avg,
            "event_budget": event_budget,
            "cal_month": float(cal_month),
            "cal_year": float(cal_year),
            "quarter": float(quarter),
            "commission_margin": commission_margin,
        }
        with st.spinner("Prédiction en cours..."):
            result = api_post("/predict/regression", token, body)
        if result:
            prix = result.get("montant_predit", "N/A")
            st.markdown(f"""
            <div class="result-box">
                <h3>✅ Prix prédit : <strong>{prix} TND</strong></h3>
                <p>Modèle : {result.get('modele','Ridge')}</p>
                <p>Utilisateur : {result.get('utilisateur','')}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Réponse JSON complète"):
                st.json(result)


# ── Tab 2 : Classification ────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🎯 Prédiction du statut de réservation (Random Forest)")
    st.info("Modèle : **RandomForest** — Prédit : confirmed / pending / cancelled")

    with st.form("form_classification"):
        col1, col2 = st.columns(2)
        with col1:
            final_price   = st.number_input("Prix final (TND)", 100.0, 50000.0, 1500.0, step=50.0)
            service_price2 = st.number_input("Prix du service (TND)", 100.0, 50000.0, 1200.0, step=50.0)
            benchmark_avg2 = st.number_input("Prix benchmark moyen (TND)", 100.0, 50000.0, 1300.0, step=50.0)
            event_budget2  = st.number_input("Budget événement (TND)", 500.0, 100000.0, 5000.0, step=100.0)
        with col2:
            cal_month2  = st.selectbox("Mois", list(range(1, 13)), index=3, key="clf_month")
            cal_year2   = st.selectbox("Année", [2023, 2024, 2025], index=1, key="clf_year")
            quarter2    = st.selectbox("Trimestre", [1, 2, 3, 4], index=1, key="clf_q")
            id_event2   = st.number_input("ID Événement", 1, 500, 42, key="clf_ev")
            id_prov2    = st.number_input("ID Prestataire", 1, 200, 7, key="clf_prov")

        submitted2 = st.form_submit_button("🔮 Prédire le statut", use_container_width=True)

    if submitted2:
        body2 = {
            "id_date": 1.0, "id_event": float(id_event2),
            "id_servicecategory": 3.0, "id_benchmark": 2.0,
            "id_provider": float(id_prov2),
            "final_price": final_price,
            "service_price": service_price2,
            "benchmark_avg_price": benchmark_avg2,
            "event_budget": event_budget2,
            "cal_month": float(cal_month2),
            "cal_year": float(cal_year2),
            "quarter": float(quarter2),
        }
        with st.spinner("Classification en cours..."):
            result2 = api_post("/predict/classification", token, body2)
        if result2:
            statut = result2.get("statut_predit", "N/A")
            color_map = {"confirmed": "#16a34a", "pending": "#d97706", "cancelled": "#dc2626"}
            color = color_map.get(statut, "#0d9488")
            st.markdown(f"""
            <div class="result-box">
                <h3>✅ Statut prédit : <strong style="color:{color}">{statut.upper()}</strong></h3>
                <p>Modèle : {result2.get('modele','RandomForest')}</p>
            </div>
            """, unsafe_allow_html=True)
            probs = result2.get("probabilites", {})
            if probs:
                st.markdown("#### Probabilités par classe")
                prob_cols = st.columns(len(probs))
                for i, (cls, prob) in enumerate(probs.items()):
                    with prob_cols[i]:
                        st.metric(cls, f"{prob*100:.1f}%")
            with st.expander("Réponse JSON complète"):
                st.json(result2)


# ── Tab 3 : Segmentation ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 👥 Segmentation fidélité RFM (K-Means)")
    st.info("Modèle : **K-Means** — Segmentation bénéficiaires / prestataires")

    entity_type = st.radio("Type d'entité", ["beneficiaire", "prestataire"], horizontal=True)

    with st.form("form_segmentation"):
        col1, col2 = st.columns(2)
        with col1:
            nb_res     = st.number_input("Nombre de réservations", 1, 200, 12)
            ca_total   = st.number_input("CA total (TND)", 100.0, 200000.0, 15000.0, step=500.0)
            panier_moy = st.number_input("Panier moyen (TND)", 100.0, 10000.0, 1250.0, step=100.0)
        with col2:
            recency    = st.number_input("Recency (jours depuis dernière résa)", 1, 500, 30)
            avg_visit  = st.number_input("Nb visiteurs moyen", 0.0, 500.0, 85.0)
            vol_site   = st.number_input("Volume réservations site", 0, 100, 5)

        submitted3 = st.form_submit_button("🔮 Segmenter", use_container_width=True)

    if submitted3:
        body3 = {
            "nb_reservations_loyalty": float(nb_res),
            "ca_total_loyalty": float(ca_total),
            "panier_moyen_loyalty": float(panier_moy),
            "recency_days_loyalty": float(recency),
            "avg_nb_visitors_loyalty": float(avg_visit),
            "volume_reservations_site_loyalty": float(vol_site),
        }
        with st.spinner("Segmentation en cours..."):
            result3 = api_post(f"/predict/segmentation/{entity_type}", token, body3)
        if result3:
            label = result3.get("segment_label", "N/A")
            seg_id = result3.get("segment_id", "?")
            st.markdown(f"""
            <div class="result-box">
                <h3>✅ Segment : <strong>{label}</strong> (ID {seg_id})</h3>
                <p>Entité : {result3.get('type_entite','')}</p>
                <p>Modèle : {result3.get('modele','K-Means')}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Réponse JSON complète"):
                st.json(result3)


# ── Tab 4 : Séries Temporelles ────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 📈 Prévision séries temporelles (Holt)")
    st.info(
        "Modèle champion : **Holt** — Graphiques : historique synthétique aligné au périmètre DW "
        "(nb_fact_rows mensuel), ajustement in-sample et prévision multi-mois."
    )

    local_ts = _load_timeseries_metrics_local()
    hp = local_ts.get("holt_params") or {}
    default_alpha = float(hp.get("smoothing_level", 0.35))
    default_beta = float(hp.get("smoothing_trend", 0.12))

    c_opts1, c_opts2 = st.columns(2)
    with c_opts1:
        horizon = st.slider("Horizon de prévision (mois)", 1, 12, 3)
    with c_opts2:
        alpha_bt = st.slider("α — niveau (smoothing_level)", 0.05, 0.95, default_alpha, 0.05)
        beta_bt = st.slider("β — tendance (smoothing_trend)", 0.05, 0.95, default_beta, 0.05)

    ts_submit = st.button("🔮 Actualiser graphiques et métriques API", key="btn_ts")

    if ts_submit:
        st.session_state["ts_show_charts"] = True

    if st.session_state.get("ts_show_charts"):
        result4 = None
        if ts_submit:
            result4 = api_get("/predict/timeseries", token, params={"horizon": horizon})
            st.session_state["ts_last_api"] = result4
        else:
            result4 = st.session_state.get("ts_last_api")

        dates_demo, y_demo = _demo_monthly_volumes(n_months=48, seed=42)

        col_m1, col_m2, col_m3 = st.columns(3)
        if result4:
            champ_metrics = result4.get("metriques_test", {})
            with col_m1:
                st.metric("RMSE (Holt · métriques enreg.)", f"{champ_metrics.get('rmse', 'N/A')}")
            with col_m2:
                st.metric("MAPE", f"{champ_metrics.get('mape', 'N/A')} %")
            with col_m3:
                st.metric("R²", f"{champ_metrics.get('r2', 'N/A')}")

            st.markdown(f"**Note API :** {result4.get('note', '')}")
        else:
            holt_loc = local_ts.get("test_holt", {})
            with col_m1:
                st.metric("RMSE (fichier local)", f"{holt_loc.get('rmse', 'N/A')}")
            with col_m2:
                st.metric("MAPE (local)", f"{holt_loc.get('mape', 'N/A')} %")
            with col_m3:
                st.metric("R² (local)", f"{holt_loc.get('r2', 'N/A')}")

        st.markdown("#### 📊 Vue série — historique & prévision")
        fig_main = build_timeseries_chart_figure(dates_demo, y_demo, horizon, alpha_bt, beta_bt)
        st.plotly_chart(fig_main, use_container_width=True)

        holt_json = result4.get("metriques_holt", {}) if result4 else local_ts.get("test_holt", {})
        arima_json = result4.get("metriques_arima", {}) if result4 else local_ts.get("test_arima", {})
        if holt_json and arima_json:
            st.markdown("#### 📊 Comparaison Holt vs ARIMA (métriques hold-out projet)")
            fig_cmp = build_rmse_comparison_figure(holt_json, arima_json)
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.caption(
            "La série affichée est une **démo cohérente** avec le scoring Champion/Baseline du projet ; "
            "les coefficients Holt peuvent être ajustés pour illustrer la sensibilité. "
            "Pour une série issue du DW réel, branchez une lecture SQL ou un parquet dans ce même graphique."
        )

        if result4:
            with st.expander("Réponse JSON API (/predict/timeseries)"):
                st.json(result4)
    else:
        st.markdown("*Cliquez sur **Actualiser** pour charger les métriques API et afficher les graphiques.*")


# ── Tab 5 : Métriques ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 📋 Métriques globales des modèles")
    if st.button("📥 Charger les métriques"):
        result5 = api_get("/models/metrics", token)
        if result5:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🎯 Classification (RandomForest)")
                m_clf = result5.get("classification", {})
                if m_clf:
                    st.metric("Accuracy", f"{m_clf.get('accuracy','N/A')}")
                    st.metric("F1 Weighted", f"{m_clf.get('f1_weighted', m_clf.get('f1_score','N/A'))}")
                    st.json(m_clf)

                st.markdown("#### 👥 Clustering (K-Means)")
                st.json(result5.get("clustering", {}))

            with col2:
                st.markdown("#### 💰 Régression (Ridge)")
                m_reg = result5.get("regression", {})
                if m_reg:
                    st.metric("R² Score", f"{m_reg.get('r2_score','N/A')}")
                    st.metric("RMSE", f"{m_reg.get('rmse','N/A')} TND")
                    st.json(m_reg)

                st.markdown("#### 📈 Séries Temporelles (Holt)")
                st.json(result5.get("timeseries", {}))

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#64748b;font-size:0.85rem'>"
    f"EventZilla ML API · S12 MLOps · API: <code>{API_BASE}</code>"
    f"</div>",
    unsafe_allow_html=True
)
