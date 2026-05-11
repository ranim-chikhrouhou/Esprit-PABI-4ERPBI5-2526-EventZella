# -*- coding: utf-8 -*-
"""Étape 03 — Régression multi-cibles (KPI panier, prestataire, benchmark, budget, marge).

    python ML/scripts/run_03_prediction_regression.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ML.csv_local_fallback import load_reservation_dataframe
from ML.ml_paths import ML_MODELS, ML_PROCESSED, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_FINANCIAL_WIDE, build_sql_ml_financial_wide

MIN_ROWS = 30

TARGET_KPIS = [
    ("final_price", "panier_moyen_ca_sum_final_price"),
    ("service_price", "prix_prestataire_structure_revenus"),
    ("benchmark_avg_price", "positionnement_tarifaire_benchmark"),
    ("event_budget", "budget_evenement"),
    ("commission_margin", "marge_finale_moins_prestataire_commission"),
]


def _load_df() -> pd.DataFrame:
    pp = ML_PROCESSED / "dw_financial_wide.parquet"
    if pp.is_file():
        return pd.read_parquet(pp)
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
        print("DW régression (adaptatif):", e)
    try:
        return read_dw_sql(SQL_ML_FINANCIAL_WIDE, eng)
    except Exception as e2:
        if ml_sql_only():
            raise RuntimeError("DW inaccessible (dynamique + statique): " + str(e2)) from e2
        print("DW régression (statique):", e2)
    return load_reservation_dataframe()


def main() -> None:
    ensure_processed_dirs()
    ML_MODELS.mkdir(parents=True, exist_ok=True)

    df = _load_df()
    df_reg = df.copy()
    if (
        "final_price" in df_reg.columns
        and "service_price" in df_reg.columns
        and "commission_margin" not in df_reg.columns
    ):
        df_reg["commission_margin"] = (
            pd.to_numeric(df_reg["final_price"], errors="coerce")
            - pd.to_numeric(df_reg["service_price"], errors="coerce")
        )

    runs: list[dict] = []
    pipes: dict = {}

    for target, kpi_tag in TARGET_KPIS:
        if target not in df_reg.columns:
            continue
        yraw = pd.to_numeric(df_reg[target], errors="coerce")
        oth = [
            c
            for c in df_reg.select_dtypes(include=[np.number]).columns
            if c != target and c != "fact_finance_id"
        ]
        block = pd.concat([yraw, df_reg[oth]], axis=1).dropna()
        if len(block) < MIN_ROWS or block[target].std(skipna=True) == 0 or not oth:
            print("Skip", target)
            continue
        feat = oth
        y = block[target].values
        X = block[feat].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("reg", RandomForestRegressor(n_estimators=120, random_state=42)),
            ]
        )
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        run_rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        run_r2 = float(r2_score(yte, pred))
        run_mae = float(mean_absolute_error(yte, pred))
        print(target, "RMSE", run_rmse, "R2", run_r2, "n", len(block))
        runs.append(
            {
                "target": target,
                "rmse": run_rmse,
                "r2": run_r2,
                "mae": run_mae,
                "n_samples": int(len(block)),
                "features": feat,
                "kpi_alignment": kpi_tag,
            }
        )
        pipes[target] = pipe

    if not runs:
        raise SystemExit("Aucune cible régression exploitable.")

    priority = [t for t, _ in TARGET_KPIS]
    primary_target = next(
        (t for t in priority if any(r["target"] == t for r in runs)),
        runs[0]["target"],
    )
    primary_row = next(r for r in runs if r["target"] == primary_target)
    others = [r for r in runs if r["target"] != primary_target]

    joblib.dump(pipes[primary_target], ML_MODELS / "rf_panier_kpi_pipeline.joblib")
    for r in others:
        t = r["target"]
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in t)
        joblib.dump(pipes[t], ML_MODELS / f"rf_regression_target_{safe}.joblib")

    (ML_MODELS / "metrics_regression.json").write_text(
        json.dumps(
            {
                "task": "regression",
                "model": "RandomForestRegressor + StandardScaler",
                "target": primary_target,
                "rmse": float(primary_row["rmse"]),
                "r2": float(primary_row["r2"]),
                "mae": float(primary_row["mae"]),
                "features": primary_row["features"],
                "kpi_alignment": primary_row["kpi_alignment"],
                "regression_objectives": runs,
                "other_targets_count": len(others),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
