# -*- coding: utf-8 -*-
"""Étape 02 — Classification statut (KPI acceptation / annulation / entonnoir).

    python ML/scripts/run_02_classification.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

from ML.csv_local_fallback import (
    enrich_financial_wide_with_performance_reservation_status,
    financial_wide_has_status_column,
    load_reservation_dataframe,
    resolve_classification_status_column,
)
from ML.ml_paths import ML_MODELS, ML_PROCESSED, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_FINANCIAL_WIDE, build_sql_ml_financial_wide


def _load_features_frame() -> pd.DataFrame:
    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if pp.is_file():
        df = pd.read_parquet(pp)
        if financial_wide_has_status_column(df):
            return df
        eng = get_sql_engine()
        if eng is not None:
            try:
                df_dw = read_dw_sql(build_sql_ml_financial_wide(eng), eng)
                if financial_wide_has_status_column(df_dw):
                    return df_dw
            except Exception:
                pass
            try:
                df_en = enrich_financial_wide_with_performance_reservation_status(df, eng)
                if financial_wide_has_status_column(df_en):
                    return df_en
            except Exception:
                pass
        return df
    eng = get_sql_engine()
    if eng is None:
        if ml_sql_only():
            raise RuntimeError(
                "Données DW obligatoires (parquet ou SQL). EVENTZILLA_ML_SQL_ONLY=1."
            )
        return load_reservation_dataframe()
    try:
        return read_dw_sql(build_sql_ml_financial_wide(eng), eng)
    except Exception as e:
        print("DW classification (adaptatif):", e)
    try:
        return read_dw_sql(SQL_ML_FINANCIAL_WIDE, eng)
    except Exception as e2:
        if ml_sql_only():
            raise RuntimeError("DW inaccessible (dynamique + statique): " + str(e2)) from e2
        print("DW classification (statique):", e2)
    return load_reservation_dataframe()


def main() -> None:
    ensure_processed_dirs()
    ML_MODELS.mkdir(parents=True, exist_ok=True)

    df = _load_features_frame()
    status_col = resolve_classification_status_column(df)
    # Ne pas créer de classe artificielle « __missing__ » : les lignes sans statut réservation
    # (souvent sans jointure DimReservation) faussaient l’apprentissage et gonflaient la prédiction « missing ».
    sc = df[status_col]
    has_status = sc.notna()
    n_miss = int((~has_status).sum())
    if n_miss:
        print(f"[classification] Lignes sans statut exclues : {n_miss} (non utilisées pour l’entraînement).")
    df = df.loc[has_status].reset_index(drop=True)
    if len(df) < 20:
        raise RuntimeError(
            f"Après exclusion des statuts manquants, seulement {len(df)} ligne(s) — "
            "vérifiez la jointure réservation dans le DW ou le notebook 00."
        )
    y_raw = df[status_col].astype(str)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "fact_finance_id"][:20]
    X_df = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    strat = y if len(set(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X_df.values, y, test_size=0.25, random_state=42, stratify=strat
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="weighted")
    print("Accuracy:", round(acc, 4), "F1:", round(f1, 4))
    print(classification_report(yte, pred, target_names=[str(c) for c in le.classes_]))
    joblib.dump(pipe, ML_MODELS / "rf_status_kpi_pipeline.joblib")
    joblib.dump(le, ML_MODELS / "label_encoder_status.joblib")
    (ML_MODELS / "metrics_classification.json").write_text(
        json.dumps(
            {
                "task": "classification",
                "model": "RandomForest + StandardScaler",
                "accuracy": float(acc),
                "f1_weighted": float(f1),
                "classes": [str(c) for c in le.classes_],
                "kpi_alignment": "taux_acceptation_annulation_funnel",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
