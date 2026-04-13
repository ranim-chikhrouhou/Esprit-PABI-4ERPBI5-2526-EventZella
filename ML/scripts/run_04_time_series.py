# -*- coding: utf-8 -*-
"""Étape 04 — Séries temporelles : volume, CA mensuel, panier moyen (si DW).

    python ML/scripts/run_04_time_series.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

from ML.csv_local_fallback import load_reservation_dataframe, monthly_series_from_reservation
from ML.ml_paths import ML_MODELS, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_TIME_SERIES_RESERVATIONS

SERIES_KPIS = [
    ("nb_fact_rows", "count_id_reservation_mensuel_anticipation"),
    ("revenue_sum", "ca_mensuel_sum_final_price_projection"),
    ("avg_final_price", "panier_moyen_mensuel_projection"),
]


def main() -> None:
    ensure_processed_dirs()
    ML_MODELS.mkdir(parents=True, exist_ok=True)

    df_ts = None
    eng = get_sql_engine()
    if eng is not None:
        try:
            df_ts = read_dw_sql(SQL_ML_TIME_SERIES_RESERVATIONS, eng)
        except Exception as e:
            if ml_sql_only():
                raise RuntimeError("Série temporelle : SQL DW obligatoire : " + str(e)) from e
            print("Série depuis DW impossible, agrégat local :", e)
    if df_ts is None:
        if ml_sql_only():
            raise RuntimeError(
                "Aucune série (SQL requis si EVENTZILLA_ML_SQL_ONLY=1)."
            )
        df_ts = monthly_series_from_reservation(load_reservation_dataframe())

    df_ts["date"] = pd.to_datetime(
        dict(year=df_ts["cal_year"].astype(int), month=df_ts["cal_month"].astype(int), day=1)
    )
    kpi_by_col = dict(SERIES_KPIS)
    available = [name for name, _ in SERIES_KPIS if name in df_ts.columns]
    if not available:
        raise ValueError("Aucune colonne séries dans df_ts")

    series_runs: list[dict] = []
    for col in available:
        ts = df_ts.set_index("date")[col].astype(float).sort_index()
        if len(ts) < 6:
            print("Skip", col, "(historique court)")
            continue
        train = ts.iloc[:-3]
        test = ts.iloc[-3:]
        model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        fc = fit.forecast(len(test))
        mape = float(np.mean(np.abs((test - fc) / (test + 1e-9))) * 100)
        rmse = float(np.sqrt(np.mean((test - fc) ** 2)))
        print(col, "RMSE", rmse, "MAPE~", round(mape, 2))
        series_runs.append(
            {
                "series": col,
                "rmse_holdout": rmse,
                "mape_approx": mape,
                "horizon": int(len(test)),
                "kpi_alignment": kpi_by_col[col],
            }
        )

    if not series_runs:
        raise ValueError(
            "Aucune série exploitable — enrichissez le DW (ou ML_SQL_ONLY=0 + Reservation)."
        )

    primary = next((r for r in series_runs if r["series"] == "nb_fact_rows"), series_runs[0])
    (ML_MODELS / "metrics_timeseries.json").write_text(
        json.dumps(
            {
                "task": "time_series",
                "model": "ExponentialSmoothing",
                "series": primary["series"],
                "rmse_holdout": float(primary["rmse_holdout"]),
                "mape_approx": float(primary["mape_approx"]),
                "horizon": int(primary["horizon"]),
                "kpi_alignment": primary["kpi_alignment"],
                "time_series_objectives": series_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
