# -*- coding: utf-8 -*-
"""
Visualisations pour MLflow — artefacts PNG/HTML alignés EventZilla.

Les graphiques sont loggués avec mlflow.log_artifact() pour apparaître
dans l’UI MLflow (onglet Artifacts de chaque run). Les métriques séquentielles
(log_metric(..., step=)) permettent les courbes dans Compare runs.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _fig_to_mlflow(fig: plt.Figure, name: str, subdir: str = "plots") -> Path:
    """Sauvegarde une figure et retourne le chemin du fichier."""
    tmp = Path(tempfile.mkdtemp(prefix="ez_mlflow_"))
    out = tmp / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path=subdir)
    return out


def log_classification_charts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    feature_names: list[str] | None,
    importances: np.ndarray | None,
) -> None:
    """Matrice de confusion + importance des variables (Random Forest)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("EventZilla — Matrice de confusion (statut réservation)\nTest set — même jeu que entraînement")
    _fig_to_mlflow(fig, "confusion_matrix")

    if importances is not None and feature_names is not None:
        idx = np.argsort(importances)[-min(20, len(importances)) :]
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.barh(np.array(feature_names)[idx], importances[idx], color="#0d9488")
        ax2.set_xlabel("Importance")
        ax2.set_title("Random Forest — Top features (EventZilla)")
        ax2.invert_yaxis()
        _fig_to_mlflow(fig2, "feature_importance_top20")


def log_regression_charts(y_true: np.ndarray, y_pred: np.ndarray, target: str = "final_price") -> None:
    """Réel vs prédit + résidus."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_true, y_pred, alpha=0.35, c="#0d9488", edgecolors="none")
    lim = max(y_true.max(), y_pred.max())
    axes[0].plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    axes[0].set_xlabel(f"Réel — {target} (TND)")
    axes[0].set_ylabel(f"Prédit — {target} (TND)")
    axes[0].set_title("Régression Ridge — Test set")
    axes[0].legend()

    axes[1].hist(resid, bins=40, color="#06b6d4", edgecolor="white")
    axes[1].set_title("Résidus (réel − prédit)")
    axes[1].set_xlabel("TND")
    plt.tight_layout()
    _fig_to_mlflow(fig, "regression_actual_vs_pred_residuals")


def log_clustering_charts(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title_suffix: str,
) -> None:
    """Projection PCA 2D colorée par cluster."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", alpha=0.6, s=12)
    ax.set_title(f"K-Means — PCA 2D ({title_suffix})\nMême données fidélité que l’API")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    _fig_to_mlflow(fig, f"clusters_pca2d_{title_suffix.replace(' ', '_')}")


def log_timeseries_compare_metrics(actual: np.ndarray, predicted: np.ndarray) -> None:
    """
    Même nom de métrique sur chaque run pour que « Compare runs » trace plusieurs courbes.
    step = indice du mois dans le hold-out.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    for i, (a, pr) in enumerate(zip(actual, predicted)):
        mlflow.log_metric("holdout_actual_volume", float(a), step=i)
        mlflow.log_metric("holdout_forecast_volume", float(pr), step=i)


def log_timeseries_plots(
    dates_train: pd.DatetimeIndex | np.ndarray,
    y_train: np.ndarray,
    dates_test: pd.DatetimeIndex | np.ndarray,
    y_test: np.ndarray,
    pred_holt: np.ndarray,
    pred_arima: np.ndarray,
    model_name_for_single: str,
) -> None:
    """Figure une série : train + test + prévisions Holt & ARIMA (run courant)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    if hasattr(dates_train, "__len__"):
        ax.plot(dates_train, y_train, label="Train (volume mensuel agrégé)", color="#64748b", lw=1.5)
    ax.plot(dates_test, y_test, "o-", label="Réel — hold-out", color="#0f172a", lw=2, markersize=5)
    ax.plot(dates_test, pred_holt, "s--", label="Prédit — Holt", color="#0d9488", lw=2)
    ax.plot(dates_test, pred_arima, "^--", label="Prédit — ARIMA(1,1,1)", color="#c026d3", lw=2)
    ax.set_title(
        "EventZilla — nb lignes réservations / mois (agrégat du dataset classification)\n"
        f"Modèle mis en avant : {model_name_for_single}"
    )
    ax.set_xlabel("Mois")
    ax.set_ylabel("Volume (nombre de lignes)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    _fig_to_mlflow(fig, f"timeseries_train_test_forecast_{model_name_for_single.replace(' ', '_')}")

    # CSV joint pour téléchargement dans MLflow
    tmp = Path(tempfile.mkdtemp(prefix="ez_ts_csv_"))
    out_csv = tmp / "holdout_predictions.csv"
    dtest = pd.to_datetime(dates_test)
    pd.DataFrame(
        {
            "month": dtest.astype(str),
            "actual_volume": y_test,
            "pred_holt": pred_holt,
            "pred_arima": pred_arima,
        }
    ).to_csv(out_csv, index=False)
    mlflow.log_artifact(str(out_csv), artifact_path="data")


def log_timeseries_comparison_html(
    dates_test: pd.DatetimeIndex | np.ndarray,
    y_test: np.ndarray,
    pred_holt: np.ndarray,
    pred_arima: np.ndarray,
    metrics_holt: dict,
    metrics_arima: dict,
) -> None:
    """Rapport HTML interactif (Plotly) pour la vue Artifacts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dtest = pd.to_datetime(dates_test)
        fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.12)
        fig.add_trace(
            go.Scatter(x=dtest, y=y_test, name="Réel (hold-out)", mode="lines+markers", line=dict(color="#0f172a")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=dtest, y=pred_holt, name="Holt", mode="lines+markers", line=dict(color="#0d9488", dash="dash")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dtest, y=pred_arima, name="ARIMA", mode="lines+markers", line=dict(color="#c026d3", dash="dot")
            ),
            row=1,
            col=1,
        )
        err_h = np.abs(y_test - pred_holt)
        err_a = np.abs(y_test - pred_arima)
        fig.add_trace(go.Bar(x=dtest, y=err_h, name="|erreur| Holt", marker_color="#99f6e4"), row=2, col=1)
        fig.add_trace(go.Bar(x=dtest, y=err_a, name="|erreur| ARIMA", marker_color="#f5d0fe"), row=2, col=1)
        fig.update_layout(
            title_text="EventZilla — Comparaison Holt vs ARIMA (données projet agrégées mensuellement)",
            height=640,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(title_text="Mois", row=2, col=1)

        summary = {
            "holt_test_metrics": metrics_holt,
            "arima_test_metrics": metrics_arima,
            "note": "Série = nombre de lignes de réservations par mois (cal_year × cal_month) du même parquet que la classification.",
        }
        tmp = Path(tempfile.mkdtemp(prefix="ez_ts_html_"))
        html_path = tmp / "timeseries_comparison_dashboard.html"
        body = fig.to_html(include_plotlyjs="cdn", full_html=False)
        html_path.write_text(
            f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EventZilla TS — MLflow</title></head>
<body>
<h2>Comparaison Holt vs ARIMA (hold-out)</h2>
<pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>
{body}
</body></html>""",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(html_path), artifact_path="reports")
    except Exception:
        pass


def regression_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
