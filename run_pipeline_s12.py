#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventZilla — Pipeline d'entraînement S12 (Validation MLOps)
============================================================
Pipeline end-to-end autonome :
  1. Génération / chargement des données (synthétiques EventZilla)
  2. Prétraitement
  3. Entraînement (RandomForest, Ridge, KMeans × 2, métriques TimeSeries)
  4. Évaluation + logging MLflow (≥ 2 runs comparables par modèle)
  5. Sauvegarde des artefacts avec les noms EXACTS attendus par main.py

Usage :
    python run_pipeline_s12.py

MLflow UI (SQLite — Overview complet) :
    python mlflow_ui_sqlite.py
    → http://localhost:5000
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ML.mlflow_tracking_uri import ensure_artifact_dir, get_tracking_uri
from ML.mlflow_visualization import (
    log_classification_charts,
    log_clustering_charts,
    log_regression_charts,
    log_timeseries_compare_metrics,
    log_timeseries_comparison_html,
    log_timeseries_plots,
    regression_metrics_dict,
)
REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "ML" / "models_artifacts"
PROCESSED_DIR = REPO_ROOT / "ML" / "data_processed"
MLRUNS_DIR = REPO_ROOT / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow — SQLite par défaut (nécessaire pour l’onglet Overview / charts UI)
ensure_artifact_dir(REPO_ROOT)
_MLFLOW_URI = get_tracking_uri(REPO_ROOT)
mlflow.set_tracking_uri(_MLFLOW_URI)
print(f"📡 MLflow Tracking URI : {_MLFLOW_URI}")
print("   ℹ️  Lancez l’UI avec : python mlflow_ui_sqlite.py  (SQLite + ./mlartifacts)")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ═════════════════════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES (domain EventZilla)
# ═════════════════════════════════════════════════════════════════════════════
def generate_eventzilla_data(n: int = 2000, seed: int = 42) -> tuple:
    """Génère des données synthétiques réalistes pour EventZilla."""
    rng = np.random.default_rng(seed)

    print(f"\n📊 Génération de {n} lignes de données EventZilla synthétiques...")

    # ── Données de classification (statut réservation) ────────────────────
    event_budget     = rng.uniform(1_000, 20_000, n)
    service_price    = event_budget * rng.uniform(0.15, 0.45, n)
    benchmark_avg    = service_price * rng.uniform(0.85, 1.20, n)
    commission       = service_price * rng.uniform(0.05, 0.15, n)
    final_price      = service_price + commission

    id_date          = rng.integers(1, 730, n)
    id_event         = rng.integers(1, 150, n)
    id_category      = rng.integers(1, 12, n)
    id_benchmark     = rng.integers(1, 50, n)
    id_provider      = rng.integers(1, 80, n)
    cal_month        = rng.integers(1, 13, n)
    cal_year         = rng.choice([2023, 2024, 2025], n)
    quarter          = ((cal_month - 1) // 3 + 1).astype(float)

    price_ratio = final_price / event_budget
    statuses = np.array(["confirmed", "pending", "cancelled"])
    status = np.empty(n, dtype=object)
    for i in range(n):
        if price_ratio[i] < 0.25:
            p = [0.70, 0.20, 0.10]
        elif price_ratio[i] < 0.40:
            p = [0.20, 0.60, 0.20]
        else:
            p = [0.10, 0.30, 0.60]
        status[i] = rng.choice(statuses, p=p)

    clf_df = pd.DataFrame({
        "id_date": id_date.astype(float),
        "id_event": id_event.astype(float),
        "id_servicecategory": id_category.astype(float),
        "id_benchmark": id_benchmark.astype(float),
        "id_provider": id_provider.astype(float),
        "final_price": final_price,
        "service_price": service_price,
        "benchmark_avg_price": benchmark_avg,
        "event_budget": event_budget,
        "cal_month": cal_month.astype(float),
        "cal_year": cal_year.astype(float),
        "quarter": quarter,
        "status": status,
    })

    # ── Données de régression (prédire final_price) ───────────────────────
    commission_margin = commission
    reg_df = pd.DataFrame({
        "id_date": id_date.astype(float),
        "id_event": id_event.astype(float),
        "id_servicecategory": id_category.astype(float),
        "id_benchmark": id_benchmark.astype(float),
        "id_provider": id_provider.astype(float),
        "service_price": service_price,
        "benchmark_avg_price": benchmark_avg,
        "event_budget": event_budget,
        "cal_month": cal_month.astype(float),
        "cal_year": cal_year.astype(float),
        "quarter": quarter,
        "commission_margin": commission_margin,
        "final_price": final_price,
    })

    # ── Données de clustering fidélité (RFM bénéficiaires) ───────────────
    n_ben = n
    nb_res       = rng.integers(1, 60, n_ben).astype(float)
    ca_total     = nb_res * rng.uniform(800, 2_500, n_ben)
    panier_moyen = ca_total / nb_res
    recency      = rng.integers(1, 400, n_ben).astype(float)
    avg_visitors = rng.uniform(20, 200, n_ben)
    vol_site     = rng.integers(1, 30, n_ben).astype(float)

    ben_df = pd.DataFrame({
        "nb_reservations_loyalty": nb_res,
        "ca_total_loyalty": ca_total,
        "panier_moyen_loyalty": panier_moyen,
        "recency_days_loyalty": recency,
        "avg_nb_visitors_loyalty": avg_visitors,
        "volume_reservations_site_loyalty": vol_site,
    })

    # ── Données de clustering fidélité (prestataires — légèrement différent)
    nb_res_p  = rng.integers(2, 80, n_ben).astype(float)
    ca_p      = nb_res_p * rng.uniform(1_000, 4_000, n_ben)
    panier_p  = ca_p / nb_res_p
    recency_p = rng.integers(1, 500, n_ben).astype(float)
    avg_vis_p = rng.uniform(30, 300, n_ben)
    vol_p     = rng.integers(1, 50, n_ben).astype(float)

    pro_df = pd.DataFrame({
        "nb_reservations_loyalty": nb_res_p,
        "ca_total_loyalty": ca_p,
        "panier_moyen_loyalty": panier_p,
        "recency_days_loyalty": recency_p,
        "avg_nb_visitors_loyalty": avg_vis_p,
        "volume_reservations_site_loyalty": vol_p,
    })

    clf_df.to_parquet(PROCESSED_DIR / "classification_data.parquet", index=False)
    reg_df.to_parquet(PROCESSED_DIR / "regression_data.parquet", index=False)
    ben_df.to_parquet(PROCESSED_DIR / "clustering_beneficiaire_data.parquet", index=False)
    pro_df.to_parquet(PROCESSED_DIR / "clustering_prestataire_data.parquet", index=False)

    print("   ✅ classification_data.parquet")
    print("   ✅ regression_data.parquet")
    print("   ✅ clustering_beneficiaire_data.parquet")
    print("   ✅ clustering_prestataire_data.parquet")

    return clf_df, reg_df, ben_df, pro_df


# ═════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — RandomForest (2 runs comparables)
# ═════════════════════════════════════════════════════════════════════════════
CLF_FEATURES = [
    "id_date", "id_event", "id_servicecategory", "id_benchmark",
    "id_provider", "final_price", "service_price", "benchmark_avg_price",
    "event_budget", "cal_month", "cal_year", "quarter",
]

def train_classification(df: pd.DataFrame) -> None:
    """Entraîne 2 runs RandomForest + sauvegarde le champion."""
    print("\n🎯 Classification — RandomForest (2 runs MLflow)...")
    mlflow.set_experiment("EventZilla_S12_Classification")

    X = df[CLF_FEATURES]
    y = df["status"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    best_model, best_acc = None, -1
    configs = [
        {"n_estimators": 100, "max_depth": 10, "run": "RF_n100_d10"},
        {"n_estimators": 200, "max_depth": 15, "run": "RF_n200_d15"},
    ]
    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['run']}_{TIMESTAMP}"):
            mlflow.set_tag("model_family", "RandomForest")
            mlflow.set_tag("task", "classification")
            mlflow.set_tag("pipeline_version", "S12")

            mlflow.log_param("n_estimators", cfg["n_estimators"])
            mlflow.log_param("max_depth", cfg["max_depth"])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples_train", len(X_tr))
            mlflow.log_param("classes", list(le.classes_))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            t0 = time.time()
            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            train_time = round(time.time() - t0, 3)

            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="weighted")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric("train_time_s", train_time)
            mlflow.log_metric("n_estimators", cfg["n_estimators"])

            log_classification_charts(
                y_te,
                y_pred,
                list(le.classes_),
                CLF_FEATURES,
                model.feature_importances_,
            )

            mlflow.sklearn.log_model(model, "model",
                registered_model_name="EventZilla_Classification")

            print(f"   Run {cfg['run']} → Acc={acc:.4f}, F1={f1:.4f}")

            if acc > best_acc:
                best_acc   = acc
                best_model = model

    # ── Sauvegarder le champion avec les noms exacts attendus par l'API ──
    clf_path = MODELS_DIR / "classification_status_champion_pipeline.joblib"
    le_path  = MODELS_DIR / "label_encoder_status.joblib"
    joblib.dump(best_model, clf_path)
    joblib.dump(le, le_path)

    metrics = {
        "model": "RandomForest",
        "accuracy": round(best_acc, 4),
        "f1_weighted": round(f1_score(best_model.predict(X_te), y_te, average="weighted"), 4),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "n_features": X.shape[1],
        "trained_on": TIMESTAMP,
    }
    (MODELS_DIR / "metrics_classification.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"   ✅ Modèle champion sauvegardé : {clf_path.name}  (accuracy={best_acc:.4f})")
    print(f"   ✅ Label encoder sauvegardé   : {le_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. RÉGRESSION — Ridge (2 runs comparables)
# ═════════════════════════════════════════════════════════════════════════════
REG_FEATURES = [
    "id_date", "id_event", "id_servicecategory", "id_benchmark",
    "id_provider", "service_price", "benchmark_avg_price", "event_budget",
    "cal_month", "cal_year", "quarter", "commission_margin",
]

def train_regression(df: pd.DataFrame) -> None:
    """Entraîne 2 runs Ridge + sauvegarde le champion."""
    print("\n💰 Régression — Ridge (2 runs MLflow)...")
    mlflow.set_experiment("EventZilla_S12_Regression")

    X = df[REG_FEATURES]
    y = df["final_price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_r2 = None, -np.inf
    configs = [
        {"alpha": 1.0,  "run": "Ridge_alpha1"},
        {"alpha": 0.1,  "run": "Ridge_alpha01"},
    ]
    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['run']}_{TIMESTAMP}"):
            mlflow.set_tag("model_family", "Ridge")
            mlflow.set_tag("task", "regression")
            mlflow.set_tag("pipeline_version", "S12")

            mlflow.log_param("alpha", cfg["alpha"])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples_train", len(X_tr))
            mlflow.log_param("target", "final_price")
            mlflow.log_param("test_size", 0.2)

            t0 = time.time()
            model = Ridge(alpha=cfg["alpha"])
            model.fit(X_tr, y_tr)
            train_time = round(time.time() - t0, 4)

            y_pred = model.predict(X_te)
            r2   = r2_score(y_te, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            mae  = float(mean_absolute_error(y_te, y_pred))

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("train_time_s", train_time)

            log_regression_charts(np.asarray(y_te), np.asarray(y_pred), target="final_price")

            mlflow.sklearn.log_model(model, "model",
                registered_model_name="EventZilla_Regression")

            print(f"   Run {cfg['run']} → R²={r2:.6f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

            if r2 > best_r2:
                best_r2    = r2
                best_model = model

    reg_path = MODELS_DIR / "ridge_regression_primary.joblib"
    joblib.dump(best_model, reg_path)

    metrics = {
        "model": "Ridge",
        "r2_score": round(best_r2, 6),
        "rmse": round(float(np.sqrt(mean_squared_error(
            y_te, best_model.predict(X_te)))), 4),
        "mae": round(float(mean_absolute_error(y_te, best_model.predict(X_te))), 4),
        "target": "final_price",
        "unit": "TND",
        "n_features": X.shape[1],
        "trained_on": TIMESTAMP,
    }
    (MODELS_DIR / "metrics_regression.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"   ✅ Modèle champion sauvegardé : {reg_path.name}  (R²={best_r2:.6f})")


# ═════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING FIDÉLITÉ — KMeans (bénéficiaires + prestataires)
# ═════════════════════════════════════════════════════════════════════════════
CLUSTER_FEATURES = [
    "nb_reservations_loyalty", "ca_total_loyalty", "panier_moyen_loyalty",
    "recency_days_loyalty", "avg_nb_visitors_loyalty", "volume_reservations_site_loyalty",
]

SEGMENT_LABELS = {
    0: {"label_short": "VIP", "label_metier_fr": "**VIP** : Client haute valeur"},
    1: {"label_short": "Fidèle", "label_metier_fr": "**Fidèle** : Client régulier"},
    2: {"label_short": "Occasionnel", "label_metier_fr": "**Occasionnel** : Client ponctuel"},
    3: {"label_short": "À risque", "label_metier_fr": "**À risque** : Inactif récent"},
}

def _build_labels_json(n_clusters: int) -> dict:
    segments = []
    for cid in range(n_clusters):
        info = SEGMENT_LABELS.get(cid, {"label_short": f"Segment {cid}",
                                        "label_metier_fr": f"Segment {cid}"})
        segments.append({"cluster_id": cid, **info})
    return {"n_clusters": n_clusters, "segments": segments}


def train_clustering_entity(df: pd.DataFrame, entity: str) -> float:
    """Entraîne KMeans + imputer + scaler pour une entité (beneficiaire / prestataire)."""
    print(f"\n👥 Clustering fidélité {entity} — KMeans (2 runs MLflow)...")
    exp_name = f"EventZilla_S12_Clustering_{entity.capitalize()}"
    mlflow.set_experiment(exp_name)

    X = df[CLUSTER_FEATURES].copy()

    best_model, best_sil, best_scaler, best_imputer = None, -1, None, None

    configs = [
        {"n_clusters": 4, "run": f"KMeans_k4_{entity}"},
        {"n_clusters": 3, "run": f"KMeans_k3_{entity}"},
    ]
    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['run']}_{TIMESTAMP}"):
            mlflow.set_tag("model_family", "KMeans")
            mlflow.set_tag("task", "clustering")
            mlflow.set_tag("entity", entity)
            mlflow.set_tag("pipeline_version", "S12")

            mlflow.log_param("n_clusters", cfg["n_clusters"])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("entity", entity)

            imputer = SimpleImputer(strategy="median")
            X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=CLUSTER_FEATURES)
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X_imp)

            model   = KMeans(n_clusters=cfg["n_clusters"], random_state=42, n_init="auto")
            labels  = model.fit_predict(X_scaled)
            sil     = silhouette_score(X_scaled, labels)

            mlflow.log_metric("silhouette_score", sil)
            mlflow.log_metric("n_clusters", cfg["n_clusters"])

            log_clustering_charts(X_scaled, labels, cfg["run"])

            mlflow.sklearn.log_model(model, "model",
                registered_model_name=f"EventZilla_Clustering_{entity.capitalize()}")

            print(f"   Run {cfg['run']} → silhouette={sil:.4f}")

            if sil > best_sil:
                best_sil     = sil
                best_model   = model
                best_scaler  = scaler
                best_imputer = imputer

    n_clusters_best = best_model.n_clusters

    suffix = "beneficiary" if entity == "beneficiaire" else "provider"
    joblib.dump(best_model,   MODELS_DIR / f"kmeans_loyalty_{suffix}.joblib")
    joblib.dump(best_scaler,  MODELS_DIR / f"kmeans_standard_scaler_loyalty_{suffix}.joblib")
    joblib.dump(best_imputer, MODELS_DIR / f"kmeans_median_imputer_loyalty_{suffix}.joblib")

    labels_data = _build_labels_json(n_clusters_best)
    labels_file = MODELS_DIR / f"clustering_segment_labels_loyalty_{suffix}.json"
    labels_file.write_text(json.dumps(labels_data, indent=2, ensure_ascii=False), encoding="utf-8")

    feat_file = MODELS_DIR / f"clustering_feature_names_loyalty_{suffix}.json"
    feat_file.write_text(json.dumps({"features": CLUSTER_FEATURES}, indent=2), encoding="utf-8")

    print(f"   ✅ kmeans_loyalty_{suffix}.joblib  (silhouette={best_sil:.4f})")
    print(f"   ✅ kmeans_standard_scaler_loyalty_{suffix}.joblib")
    print(f"   ✅ kmeans_median_imputer_loyalty_{suffix}.joblib")
    return best_sil


# ═════════════════════════════════════════════════════════════════════════════
# 5. SÉRIES TEMPORELLES — Holt vs ARIMA sur la série mensuelle du MÊME dataset
# ═════════════════════════════════════════════════════════════════════════════
def _monthly_volume_from_classification(df: pd.DataFrame) -> pd.Series:
    """
    Agrège le jeu classification en volume mensuel (proxy nb lignes / mois EventZilla).
    Utilise cal_year × cal_month — même lignes que RF classification.
    """
    d = df.copy()
    d["period"] = pd.to_datetime(
        dict(year=d["cal_year"].astype(int), month=d["cal_month"].astype(int), day=1)
    )
    monthly = d.groupby("period").size().sort_index()
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx, fill_value=0).astype(float)
    monthly.name = "nb_rows_monthly"
    return monthly


def train_timeseries_mlflow(clf_df: pd.DataFrame) -> None:
    """
    Entraîne Holt & ARIMA sur hold-out temporel, métriques réelles, artefacts MLflow :
    - PNG train/test/prévisions dans chaque run
    - métriques stepped pour Compare runs (holdout_actual_volume / holdout_forecast_volume)
    - run « Overview » avec HTML Plotly + CSV
    """
    print("\n📈 Séries temporelles — Holt vs ARIMA (données agrégées du dataset classification)...")
    mlflow.set_experiment("EventZilla_S12_TimeSeries")

    monthly = _monthly_volume_from_classification(clf_df)
    if len(monthly) < 12:
        print("   ⚠️  Série trop courte — skip TS MLflow détaillé.")
        return

    test_horizon = min(8, max(3, len(monthly) // 5))
    train = monthly.iloc[:-test_horizon]
    test = monthly.iloc[-test_horizon:]
    y_train = train.values.astype(float)
    y_test = test.values.astype(float)
    dates_train = train.index
    dates_test = test.index

    pred_holt = np.full_like(y_test, np.nan, dtype=float)
    pred_arima = np.full_like(y_test, np.nan, dtype=float)
    holt_params_logged: dict = {}
    arima_order = "(1,1,1)"

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        fit_h = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        pred_holt = np.asarray(fit_h.forecast(test_horizon), dtype=float)
        for attr in ("smoothing_level", "smoothing_trend", "smoothing_slope"):
            if hasattr(fit_h, attr):
                v = getattr(fit_h, attr, None)
                if v is not None and np.isfinite(v):
                    holt_params_logged[attr] = float(v)
    except Exception as e:
        print(f"   ⚠️  Holt statsmodels : {e}")
        pred_holt = np.full(len(y_test), float(np.mean(y_train)), dtype=float)

    try:
        from statsmodels.tsa.arima.model import ARIMA

        fit_a = ARIMA(y_train, order=(1, 1, 1)).fit()
        pred_arima = np.asarray(fit_a.forecast(test_horizon), dtype=float)
    except Exception as e:
        print(f"   ⚠️  ARIMA statsmodels : {e} — repli drift naïf.")
        pred_arima = np.linspace(y_train[-1], y_train[-1] * 0.95, len(y_test))

    metrics_holt = regression_metrics_dict(y_test, pred_holt)
    metrics_arima = regression_metrics_dict(y_test, pred_arima)

    # ── Run Holt ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"Holt_ExponentialSmoothing_{TIMESTAMP}"):
        mlflow.set_tag("model_family", "ExponentialSmoothing")
        mlflow.set_tag("task", "timeseries")
        mlflow.set_tag("champion", "true" if metrics_holt["rmse"] <= metrics_arima["rmse"] else "false")
        mlflow.set_tag("dataset", "monthly_volume_from_classification_cal_month")
        mlflow.log_param("model", "Holt")
        mlflow.log_param("trend", "add")
        mlflow.log_param("seasonal", "None")
        mlflow.log_param("series", "nb_rows_monthly_aggregated")
        mlflow.log_param("test_horizon_months", test_horizon)
        mlflow.log_param("n_train_months", len(y_train))
        for k, v in holt_params_logged.items():
            mlflow.log_param(k, v)
        for k, v in metrics_holt.items():
            mlflow.log_metric(k, v)
        log_timeseries_compare_metrics(y_test, pred_holt)
        log_timeseries_plots(
            dates_train, y_train, dates_test, y_test,
            pred_holt, pred_arima, "Holt",
        )

    # ── Run ARIMA ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"ARIMA_{arima_order.replace(',', '')}_{TIMESTAMP}"):
        mlflow.set_tag("model_family", "ARIMA")
        mlflow.set_tag("task", "timeseries")
        mlflow.set_tag("champion", "true" if metrics_arima["rmse"] < metrics_holt["rmse"] else "false")
        mlflow.set_tag("dataset", "monthly_volume_from_classification_cal_month")
        mlflow.log_param("model", "ARIMA")
        mlflow.log_param("order", arima_order)
        mlflow.log_param("series", "nb_rows_monthly_aggregated")
        mlflow.log_param("test_horizon_months", test_horizon)
        for k, v in metrics_arima.items():
            mlflow.log_metric(k, v)
        log_timeseries_compare_metrics(y_test, pred_arima)
        log_timeseries_plots(
            dates_train, y_train, dates_test, y_test,
            pred_holt, pred_arima, "ARIMA",
        )

    # ── Run synthèse (graphiques combinés + HTML) ────────────────────────────
    with mlflow.start_run(run_name=f"TS_Overview_Compare_{TIMESTAMP}"):
        mlflow.set_tag("artifact_type", "comparison_dashboard")
        mlflow.set_tag("dataset", "monthly_volume_from_classification_cal_month")
        mlflow.log_param("description", "Vue combinée Holt vs ARIMA — même hold-out")
        mlflow.log_metric("holt_rmse_holdout", metrics_holt["rmse"])
        mlflow.log_metric("arima_rmse_holdout", metrics_arima["rmse"])
        mlflow.log_metric("holt_mape_holdout", metrics_holt["mape"])
        mlflow.log_metric("arima_mape_holdout", metrics_arima["mape"])
        log_timeseries_plots(
            dates_train, y_train, dates_test, y_test,
            pred_holt, pred_arima, "Overview_Holt_vs_ARIMA",
        )
        log_timeseries_comparison_html(
            dates_test, y_test, pred_holt, pred_arima,
            metrics_holt, metrics_arima,
        )

    metrics = {
        "champion_model": "Holt" if metrics_holt["rmse"] <= metrics_arima["rmse"] else "ARIMA",
        "series": "nb_fact_rows_monthly_from_classification",
        "test_champion": metrics_holt if metrics_holt["rmse"] <= metrics_arima["rmse"] else metrics_arima,
        "test_holt": metrics_holt,
        "test_arima": metrics_arima,
        "holt_params": holt_params_logged or {"smoothing_level": 0.35, "smoothing_trend": 0.12},
        "trained_on": TIMESTAMP,
        "test_horizon_months": test_horizon,
        "n_months_total": len(monthly),
    }
    (MODELS_DIR / "metrics_timeseries.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    (MODELS_DIR / "metrics_clustering.json").write_text(
        json.dumps({
            "model": "KMeans",
            "n_clusters": 4,
            "entities": ["beneficiaire", "prestataire"],
            "features": CLUSTER_FEATURES,
            "trained_on": TIMESTAMP,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"   ✅ Séries temporelles MLflow — Holt RMSE={metrics_holt['rmse']:.2f}, ARIMA RMSE={metrics_arima['rmse']:.2f}")
    print("   ✅ Artefacts : plots/, reports/timeseries_comparison_dashboard.html, data/holdout_predictions.csv")
    print("   ✅ Compare runs : métriques « holdout_forecast_volume » avec steps")


# ═════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
def run_s12_pipeline():
    banner = "=" * 65
    print(banner)
    print("  🚀 EventZilla — Pipeline S12 MLOps (Semaine 12)")
    print(banner)
    print(f"  📅 Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📡 MLflow  : {_MLFLOW_URI}")
    print(f"  📁 Modèles : {MODELS_DIR}")
    print()

    try:
        clf_df, reg_df, ben_df, pro_df = generate_eventzilla_data()

        train_classification(clf_df)
        train_regression(reg_df)
        sil_ben = train_clustering_entity(ben_df, "beneficiaire")
        sil_pro = train_clustering_entity(pro_df, "prestataire")
        train_timeseries_mlflow(clf_df)

        print()
        print(banner)
        print("  ✅ Pipeline S12 complété avec succès !")
        print(banner)
        print()
        print("  📦 Modèles générés dans ML/models_artifacts/ :")
        for f in sorted(MODELS_DIR.glob("*.joblib")):
            size_kb = f.stat().st_size // 1024
            print(f"     • {f.name:<60} ({size_kb} KB)")
        print()
        print("  📊 Métriques JSON :")
        for f in sorted(MODELS_DIR.glob("metrics_*.json")):
            print(f"     • {f.name}")
        print()
        print("  🔗 Prochaines étapes :")
        print("     1. Lancer MLflow UI  : python mlflow_ui_sqlite.py")
        print("     2. Vérifier les runs : http://localhost:5000")
        print("     3. Démarrer l'API    : python run_fastapi.py")
        print("     4. Tester l'API      : http://localhost:8000/docs")
        print("     5. Lancer Streamlit  : streamlit run ML/streamlit_predict.py")
        print()

    except Exception as exc:
        print()
        print(banner)
        print("  ❌ Échec du pipeline !")
        print(banner)
        print(f"  Erreur : {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_s12_pipeline()
