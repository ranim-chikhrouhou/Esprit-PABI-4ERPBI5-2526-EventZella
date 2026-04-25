# -*- coding: utf-8 -*-
"""
EventZilla ML API — Point d'entrée FastAPI.

Lancer :
    python -m uvicorn ML.api.main:app --reload --port 8000

Documentation interactive :
    http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Charger le fichier .env automatiquement au démarrage
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ML.api.auth_sql import (
    authenticate_sql_user,
    create_jwt_token,
    get_current_user,
    require_role,
)
from ML.ml_paths import ML_MODELS

# ── Application ──────────────────────────────────────────────────
app = FastAPI(
    title="EventZilla ML API",
    description="API de prédiction ML — authentification via logins SQL Server (SSMS).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:5678"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement des modèles au démarrage ──────────────────────────
_MODELS: dict = {}

def _load_models() -> None:
    """Charge tous les modèles .joblib disponibles au démarrage de l'API."""
    mapping = {
        # Classification — RandomForest champion
        "classification":    "classification_status_champion_pipeline.joblib",
        # LabelEncoder pour décoder 0/1/2 → cancelled/confirmed/pending
        "label_encoder_clf": "label_encoder_status.joblib",
        # Régression — Ridge champion (cible : final_price)
        "regression":        "ridge_regression_primary.joblib",
        # Clustering bénéficiaires (RFM fidélité)
        "clustering_ben": "kmeans_loyalty_beneficiary.joblib",
        # Clustering prestataires (RFM fidélité)
        "clustering_pro": "kmeans_loyalty_provider.joblib",
        # Scaler bénéficiaires
        "scaler_ben":     "kmeans_standard_scaler_loyalty_beneficiary.joblib",
        # Scaler prestataires
        "scaler_pro":     "kmeans_standard_scaler_loyalty_provider.joblib",
        # Imputer bénéficiaires
        "imputer_ben":    "kmeans_median_imputer_loyalty_beneficiary.joblib",
        # Imputer prestataires
        "imputer_pro":    "kmeans_median_imputer_loyalty_provider.joblib",
    }
    for key, filename in mapping.items():
        path = ML_MODELS / filename
        if path.is_file():
            _MODELS[key] = joblib.load(path)
        else:
            _MODELS[key] = None  # modèle absent — endpoint retournera un avertissement

    # Labels segments bénéficiaires
    labels_ben = ML_MODELS / "clustering_segment_labels_loyalty_beneficiary.json"
    labels_pro = ML_MODELS / "clustering_segment_labels_loyalty_provider.json"
    _MODELS["labels_ben"] = json.loads(labels_ben.read_text(encoding="utf-8")) if labels_ben.is_file() else {}
    _MODELS["labels_pro"] = json.loads(labels_pro.read_text(encoding="utf-8")) if labels_pro.is_file() else {}

    # Métriques (pour l'endpoint /metrics)
    for mkey, mfile in [
        ("metrics_classification", "metrics_classification.json"),
        ("metrics_regression",     "metrics_regression.json"),
        ("metrics_clustering",     "metrics_clustering.json"),
        ("metrics_timeseries",     "metrics_timeseries.json"),
    ]:
        mp = ML_MODELS / mfile
        _MODELS[mkey] = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}

_load_models()


# ── Schémas Pydantic ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    login:    str
    password: str
    model_config = {"json_schema_extra": {
        "example": {"login": "ranim_chikhrouhou", "password": "Ranim@Marketing2025!"}
    }}

class ClassificationInput(BaseModel):
    """
    Features pour prédire le statut d'une réservation.
    Correspond aux 12 colonnes numériques du dataset d'entraînement (notebook 02_C).
    """
    id_date:             float
    id_event:            float
    id_servicecategory:  float
    id_benchmark:        float
    id_provider:         float
    final_price:         float
    service_price:       float
    benchmark_avg_price: float
    event_budget:        float
    cal_month:           float = 1
    cal_year:            float = 2024
    quarter:             float = 1
    model_config = {"json_schema_extra": {"example": {
        "id_date": 1, "id_event": 42, "id_servicecategory": 3,
        "id_benchmark": 2, "id_provider": 7,
        "final_price": 1500, "service_price": 1200,
        "benchmark_avg_price": 1300, "event_budget": 2000,
        "cal_month": 4, "cal_year": 2024, "quarter": 2
    }}}

class RegressionInput(BaseModel):
    """Features pour prédire le montant final (final_price)."""
    id_date:            float
    id_event:           float
    id_servicecategory: float
    id_benchmark:       float
    id_provider:        float
    service_price:      float
    benchmark_avg_price: float
    event_budget:       float
    cal_month:          float
    cal_year:           float
    quarter:            float = 1
    commission_margin:  float = 0
    model_config = {"json_schema_extra": {"example": {
        "id_date": 1, "id_event": 42, "id_servicecategory": 3,
        "id_benchmark": 2, "id_provider": 7,
        "service_price": 1200, "benchmark_avg_price": 1300,
        "event_budget": 2000, "cal_month": 4, "cal_year": 2024,
        "quarter": 2, "commission_margin": 150
    }}}

class SegmentationInput(BaseModel):
    """Features RFM fidélité pour segmenter un bénéficiaire ou prestataire."""
    nb_reservations_loyalty:       float
    ca_total_loyalty:              float
    panier_moyen_loyalty:          float
    recency_days_loyalty:          float
    avg_nb_visitors_loyalty:       float = 0
    volume_reservations_site_loyalty: float = 0
    model_config = {"json_schema_extra": {"example": {
        "nb_reservations_loyalty": 12,
        "ca_total_loyalty": 15000,
        "panier_moyen_loyalty": 1250,
        "recency_days_loyalty": 30,
        "avg_nb_visitors_loyalty": 85,
        "volume_reservations_site_loyalty": 5
    }}}


# Endpoint minimal pour la validation S12 (/predict explicite)
@app.post("/predict", tags=["Validation S12"])
def predict(body: ClassificationInput):
    """Prédiction de statut sans JWT, utilisée pour la recette de validation."""
    model = _MODELS.get("classification")
    if model is None:
        raise HTTPException(503, "Modèle de classification non chargé.")

    features = [
        "id_date", "id_event", "id_servicecategory", "id_benchmark",
        "id_provider", "final_price", "service_price",
        "benchmark_avg_price", "event_budget", "cal_month", "cal_year", "quarter",
    ]
    df = pd.DataFrame([body.model_dump()])[features]
    pred_encoded = int(model.predict(df)[0])

    le = _MODELS.get("label_encoder_clf")
    statut = str(le.inverse_transform([pred_encoded])[0]) if le is not None else str(pred_encoded)
    return {"prediction": statut, "task": "classification", "endpoint": "/predict"}


# ── AUTH ─────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Authentification"])
def login(req: LoginRequest):
    """
    Connexion via identifiants SQL Server créés dans SSMS.
    Retourne un JWT Bearer token valable 8h.
    """
    user  = authenticate_sql_user(req.login, req.password)
    token = create_jwt_token(user)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "role":         user["role"],
        "full_name":    user["full_name"],
    }

@app.get("/auth/me", tags=["Authentification"])
def me(user: dict = Depends(get_current_user)):
    return user


# ── CLASSIFICATION — Marketing + Finance + CRM ───────────────────
@app.post("/predict/classification", tags=["Prédictions ML"])
def predict_classification(
    body: ClassificationInput,
    user: dict = Depends(require_role("marketing_manager", "financial_manager", "crm_manager")),
):
    """
    Prédit le statut d'une réservation (cancelled / confirmed / pending).
    Modèle champion : Random Forest — critère C.
    Accessible : Marketing, Finance, CRM.
    """
    model = _MODELS.get("classification")
    if model is None:
        raise HTTPException(503, "Modèle de classification non chargé.")

    features = ["id_date","id_event","id_servicecategory","id_benchmark",
                 "id_provider","final_price","service_price",
                 "benchmark_avg_price","event_budget","cal_month","cal_year","quarter"]
    df = pd.DataFrame([body.model_dump()])[features]

    pred_encoded = int(model.predict(df)[0])

    # Décoder l'entier → label métier (cancelled / confirmed / pending)
    le = _MODELS.get("label_encoder_clf")
    if le is not None:
        statut = str(le.inverse_transform([pred_encoded])[0])
        class_labels = list(le.classes_)
    else:
        statut = str(pred_encoded)
        class_labels = [str(c) for c in model.classes_]

    probas = {}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(df)[0]
        probas = {lbl: round(float(v), 3)
                  for lbl, v in zip(class_labels, p)}

    return {
        "utilisateur":   user["full_name"],
        "role":          user["role"],
        "statut_predit": statut,
        "probabilites":  probas,
        "modele":        "RandomForest (champion critère C)",
    }


# ── REGRESSION — Finance + Marketing ────────────────────────────
@app.post("/predict/regression", tags=["Prédictions ML"])
def predict_regression(
    body: RegressionInput,
    user: dict = Depends(require_role("financial_manager", "marketing_manager")),
):
    """
    Prédit le montant final d'une réservation (final_price en TND).
    Modèle champion : Ridge — critère D.
    Accessible : Finance, Marketing.
    """
    model = _MODELS.get("regression")
    if model is None:
        raise HTTPException(503, "Modèle de régression non chargé.")

    features = ["id_date","id_event","id_servicecategory","id_benchmark",
                 "id_provider","service_price","benchmark_avg_price",
                 "event_budget","cal_month","cal_year","quarter","commission_margin"]
    df = pd.DataFrame([body.model_dump()])[features]

    pred = float(model.predict(df)[0])
    return {
        "utilisateur":    user["full_name"],
        "role":           user["role"],
        "montant_predit": round(pred, 2),
        "unite":          "TND",
        "modele":         "Ridge (champion critère D, R²≈1.0)",
    }


# ── SEGMENTATION — Marketing + CRM ──────────────────────────────
@app.post("/predict/segmentation/{type_entite}", tags=["Prédictions ML"])
def predict_segmentation(
    type_entite: str,
    body: SegmentationInput,
    user: dict = Depends(require_role("marketing_manager", "crm_manager")),
):
    """
    Segmente un bénéficiaire ou prestataire selon ses métriques RFM.
    type_entite : 'beneficiaire' ou 'prestataire'
    Modèle : K-Means fidélité — critère E.
    Accessible : Marketing, CRM.
    """
    if type_entite == "beneficiaire":
        model, scaler, imputer = (_MODELS.get("clustering_ben"),
                                   _MODELS.get("scaler_ben"),
                                   _MODELS.get("imputer_ben"))
        labels_data = _MODELS.get("labels_ben", {})
    elif type_entite == "prestataire":
        model, scaler, imputer = (_MODELS.get("clustering_pro"),
                                   _MODELS.get("scaler_pro"),
                                   _MODELS.get("imputer_pro"))
        labels_data = _MODELS.get("labels_pro", {})
    else:
        raise HTTPException(400, "type_entite doit être 'beneficiaire' ou 'prestataire'.")

    if model is None:
        raise HTTPException(503, f"Modèle de clustering '{type_entite}' non chargé.")

    features = ["nb_reservations_loyalty","ca_total_loyalty","panier_moyen_loyalty",
                 "recency_days_loyalty","avg_nb_visitors_loyalty","volume_reservations_site_loyalty"]
    df = pd.DataFrame([body.model_dump()])[features]

    if imputer: df = pd.DataFrame(imputer.transform(df), columns=features)
    if scaler:  df = pd.DataFrame(scaler.transform(df),  columns=features)

    segment_id = int(model.predict(df)[0])

    # Récupérer le label métier du segment (VIP / Fidèle / Occasionnel / À risque)
    label = f"Segment {segment_id}"
    if labels_data and "segments" in labels_data:
        for s in labels_data["segments"]:
            if s.get("cluster_id") == segment_id:
                raw = s.get("label_metier_fr", s.get("label_short", label))
                # Extraire uniquement la catégorie avant ":" et nettoyer le markdown **...**
                categorie = raw.split(":")[0].replace("**", "").strip()
                label = categorie if categorie else label
                break

    return {
        "utilisateur":   user["full_name"],
        "role":          user["role"],
        "type_entite":   type_entite,
        "segment_id":    segment_id,
        "segment_label": label,
        "modele":        "K-Means fidélité (critère E)",
    }


# ── SÉRIES TEMPORELLES — Finance + Marketing ────────────────────
@app.get("/predict/timeseries", tags=["Prédictions ML"])
def predict_timeseries(
    horizon: int = 3,
    user: dict = Depends(require_role("financial_manager", "marketing_manager")),
):
    """
    Retourne les métriques du modèle Holt (champion critère F)
    et une prévision indicative basée sur les métriques enregistrées.
    Accessible : Finance, Marketing.
    """
    metrics = _MODELS.get("metrics_timeseries", {})
    if not metrics:
        raise HTTPException(503, "Métriques séries temporelles non disponibles.")

    return {
        "utilisateur":     user["full_name"],
        "role":            user["role"],
        "modele_champion": metrics.get("champion_model", "Holt"),
        "serie":           metrics.get("series", "nb_fact_rows"),
        "horizon_mois":    horizon,
        "metriques_test":  metrics.get("test_champion", {}),
        "metriques_holt":  metrics.get("test_holt", {}),
        "metriques_arima": metrics.get("test_arima", {}),
        "note": (
            "Prévision Holt entraînée sur volume mensuel DW. "
            f"RMSE={metrics.get('test_champion',{}).get('rmse','N/A'):.2f}, "
            f"MAPE={metrics.get('test_champion',{}).get('mape','N/A'):.2f}%"
        ),
    }


# ── ALERTES ERREURS n8n ──────────────────────────────────────────
class ErrorAlertRequest(BaseModel):
    workflow:  str
    node:      str = "inconnu"
    message:   str
    timestamp: str = ""

@app.post("/alert/error", tags=["Utilitaires"])
def alert_error(req: ErrorAlertRequest):
    """
    Reçu par le workflow Error Handler n8n.
    1. Sauvegarde le log dans n8n/results/error_log.jsonl
    2. Envoie un email d'alerte via Gmail SMTP (si configuré)
    """
    import json as _json, smtplib, os
    from datetime import datetime as _dt
    from email.mime.text import MIMEText

    results_dir = Path(__file__).resolve().parent.parent.parent / "n8n" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": req.timestamp or _dt.now().isoformat(),
        "workflow":  req.workflow,
        "node":      req.node,
        "message":   req.message,
    }

    # Append log
    log_file = results_dir / "error_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

    # Envoi email si credentials configurés
    gmail_user = os.environ.get("EVENTZILLA_GMAIL_USER", "")
    gmail_pass = os.environ.get("EVENTZILLA_GMAIL_PASS", "")
    email_sent = False
    email_error = ""

    if gmail_user and gmail_pass:
        try:
            # Lire les 5 dernières lignes du log
            derniers_logs = ""
            if log_file.is_file():
                lignes = log_file.read_text(encoding="utf-8").strip().splitlines()
                derniers_logs = "\n".join(lignes[-5:]) if lignes else "Aucun log"

            corps = (
                f"╔══════════════════════════════════════╗\n"
                f"║     ALERTE ERREUR — EventZilla ML    ║\n"
                f"╚══════════════════════════════════════╝\n\n"
                f"📋 DÉTAILS DE L'ERREUR\n"
                f"{'─'*40}\n"
                f"  Workflow : {req.workflow}\n"
                f"  Nœud     : {req.node}\n"
                f"  Message  : {req.message}\n"
                f"  Heure    : {entry['timestamp']}\n\n"
                f"📁 DERNIERS LOGS (error_log.jsonl)\n"
                f"{'─'*40}\n"
                f"{derniers_logs}\n\n"
                f"🔍 ACTION REQUISE\n"
                f"{'─'*40}\n"
                f"  1. Ouvrez n8n : http://localhost:5678/executions\n"
                f"  2. Vérifiez que FastAPI tourne : http://127.0.0.1:8000\n"
                f"  3. Consultez les logs : n8n/results/error_log.jsonl\n\n"
                f"─── EventZilla ML System · Alerte automatique ───"
            )
            msg = MIMEText(corps, "plain", "utf-8")
            msg["Subject"] = f"[ALERTE] EventZilla — Erreur dans {req.workflow} à {entry['timestamp'][:16]}"
            msg["From"]    = gmail_user
            msg["To"]      = "ranim.chikhrouhou@esprit.tn"

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(gmail_user, gmail_pass)
                smtp.send_message(msg)
            email_sent = True
        except Exception as e:
            email_error = str(e)

    return {
        "status":      "logged",
        "log_file":    "n8n/results/error_log.jsonl",
        "email_sent":  email_sent,
        "email_error": email_error or None,
    }


# ── SAUVEGARDE RÉSULTATS n8n ─────────────────────────────────────
class SaveResultRequest(BaseModel):
    workflow: str
    data:     dict

@app.post("/save_result", tags=["Utilitaires"])
def save_result(req: SaveResultRequest, user: dict = Depends(get_current_user)):
    """
    Sauvegarde les résultats d'un workflow n8n dans n8n/results/.
    Appelé par le nœud HTTP Request en fin de chaque pipeline.
    """
    import json as _json
    from datetime import datetime as _dt

    results_dir = Path(__file__).resolve().parent.parent.parent / "n8n" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    wf = req.workflow.lower()
    now = _dt.now()
    name_map = {
        "marketing": f"marketing_predictions_{now.strftime('%Y-%m-%d')}.json",
        "finance":   f"finance_predictions_{now.strftime('%Y-%m-%d')}.json",
        "crm":       f"crm_predictions_{now.strftime('%Y-%m-%d_%H-%M')}.json",
    }
    filename = name_map.get(wf, f"{wf}_{now.strftime('%Y-%m-%d_%H-%M')}.json")
    filepath = results_dir / filename

    filepath.write_text(
        _json.dumps(req.data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return {
        "status":   "saved",
        "file":     filename,
        "saved_by": user["full_name"],
    }


# ── MÉTRIQUES GLOBALES ───────────────────────────────────────────
@app.get("/metrics", tags=["Métriques"])
def get_metrics(user: dict = Depends(get_current_user)):
    """Retourne les métriques de tous les modèles champions."""
    return {
        "utilisateur":      user["full_name"],
        "classification":   _MODELS.get("metrics_classification", {}),
        "regression":       _MODELS.get("metrics_regression", {}),
        "clustering":       _MODELS.get("metrics_clustering", {}),
        "timeseries":       _MODELS.get("metrics_timeseries", {}),
    }


# ── HEALTH CHECK ─────────────────────────────────────────────────
@app.get("/", tags=["Santé"])
def health():
    modeles_charges = {k: (_MODELS[k] is not None)
                       for k in ["classification","regression","clustering_ben","clustering_pro"]}
    return {
        "status":          "ok",
        "app":             "EventZilla ML API v1.0",
        "modeles_charges": modeles_charges,
    }
