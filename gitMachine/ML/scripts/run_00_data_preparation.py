# -*- coding: utf-8 -*-
"""Étape 00 — Préparation des données depuis le DW (SSMS).

    Par défaut (``EVENTZILLA_ML_SQL_ONLY=1``) : aucun repli Excel/CSV.
    Pour autoriser les fichiers locaux : ``set EVENTZILLA_ML_SQL_ONLY=0``.

    cd "chemin\\vers\\PI BI NEW"
    python ML/scripts/run_00_data_preparation.py
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

from ML.csv_local_fallback import csv_search_roots_hint, load_numeric_from_local_csvs  # noqa: E402
from ML.ml_paths import (  # noqa: E402
    DATABASE_DW,
    ML_PROCESSED,
    SQL_SERVER,
    backup_paths_status,
    ensure_processed_dirs,
    get_sql_engine,
    ml_sql_only,
    read_dw_sql,
    sql_engine_init_error,
)
from ML.schema_eventzilla import (  # noqa: E402
    SQL_LIST_TABLES,
    SQL_ML_FINANCIAL_WIDE,
    build_sql_ml_financial_wide,
)


def main() -> None:
    ensure_processed_dirs()
    print("Sauvegardes FilesMachine/DB (fichier sur disque, pas la connexion SQL):", backup_paths_status())
    print("Base DW:", DATABASE_DW, "| Serveur:", SQL_SERVER, "| ML_SQL_ONLY:", ml_sql_only())

    engine = get_sql_engine()
    if engine is None and ml_sql_only():
        raise SystemExit(
            "Connexion SQL indisponible alors que EVENTZILLA_ML_SQL_ONLY=1 (données SSMS uniquement)."
        )
    df_ml = None
    if engine is not None:
        try:
            tables = read_dw_sql(SQL_LIST_TABLES, engine)
            print("Tables DW (extrait):\n", tables.head(25))
            try:
                q_fin = build_sql_ml_financial_wide(engine)
                df_ml = read_dw_sql(q_fin, engine)
                print("Jeu ML financier (SQL adaptatif):", df_ml.shape)
            except Exception as e_dyn:
                print("SQL adaptatif indisponible, requête statique :", e_dyn)
                df_ml = read_dw_sql(SQL_ML_FINANCIAL_WIDE, engine)
                print("Jeu ML financier (SQL statique):", df_ml.shape)
            if df_ml is not None and len(df_ml) == 0:
                if ml_sql_only():
                    raise SystemExit(
                        "0 ligne renvoyée par le DW (vérifiez les faits peuplés et la jointure f.id_date = d.id_date_SK)."
                    )
                print("0 ligne DW — bascule fichiers locaux.")
                df_ml = None
            if df_ml is not None:
                df_ml.to_parquet(ML_PROCESSED / "dw_financial_wide.parquet", index=False)
        except Exception as e:
            if ml_sql_only():
                raise SystemExit(f"Lecture DW obligatoire (ML_SQL_ONLY) — échec : {e}") from e
            print("Lecture DW échouée — bascule CSV si besoin:", e)
            df_ml = None
    else:
        print("Engine SQL non disponible.")
        if sql_engine_init_error():
            print("Détail:", sql_engine_init_error())
        if not ml_sql_only():
            print("Installez sqlalchemy+pyodbc ; test: python ML/scripts/run_test_sql_connection.py")

    if df_ml is not None:
        num_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
        X_raw = df_ml[num_cols].copy()
    else:
        if ml_sql_only():
            raise SystemExit(
                "Aucune donnée DW chargée (ML_SQL_ONLY=1). Corrigez la connexion ou la requête SQL."
            )
        try:
            X_raw = load_numeric_from_local_csvs()
        except FileNotFoundError as e:
            raise SystemExit(
                "DW indisponible ou requête en échec, et aucun CSV exploitable. "
                f"Emplacements essayés : {csv_search_roots_hint()}. Détails : {e}"
            ) from e

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    print(X_raw.shape, list(X_raw.columns)[:12])

    np.save(ML_PROCESSED / "X_raw_numeric.npy", X_raw.to_numpy(dtype=np.float64))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num = imputer.fit_transform(X_raw)
    X_std = scaler.fit_transform(X_num)

    np.save(ML_PROCESSED / "X_standardized.npy", X_std)
    df_out = pd.DataFrame(X_std, columns=X_raw.columns)
    try:
        df_out.to_parquet(ML_PROCESSED / "features_matrix.parquet", index=False)
    except Exception:
        df_out.to_csv(ML_PROCESSED / "features_matrix.csv", index=False)

    joblib.dump(scaler, ML_PROCESSED / "standard_scaler.joblib")
    joblib.dump(imputer, ML_PROCESSED / "median_imputer.joblib")

    mm = MinMaxScaler()
    X_mm = mm.fit_transform(X_num)
    np.save(ML_PROCESSED / "X_minmax.npy", X_mm)
    joblib.dump(mm, ML_PROCESSED / "minmax_scaler.joblib")

    with open(ML_PROCESSED / "numeric_feature_list.json", "w", encoding="utf-8") as f:
        json.dump(list(X_raw.columns), f, indent=2)

    print("Ecrit:", ML_PROCESSED.resolve())


if __name__ == "__main__":
    main()
