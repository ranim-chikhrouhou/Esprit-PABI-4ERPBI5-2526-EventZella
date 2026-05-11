# -*- coding: utf-8 -*-
"""Étape 01 — Clustering K-Means sur profils **fidélité / RFM** (bénéficiaires & prestataires).

**Différence avec le notebook** ``01_E_clustering_segmentation.ipynb`` : celui-ci segmente la **vue large**
(``SQL_ML_PERFORMANCE_WIDE``) — une **ligne = un fait / réservation**. Ce script travaille sur des **agrégats**
par ``id_beneficiary`` ou ``id_provider`` (variables ``*_loyalty``) : une **ligne = un acteur** pour le scoring
fidélité / Streamlit.

Agrégations par ``id_beneficiary`` ou ``id_provider`` : fréquence, CA, panier moyen, récence, volumes.

    python ML/scripts/run_01_clustering.py

Sorties (``ML/models_artifacts/``) :
- ``kmeans_loyalty_beneficiary.joblib`` (+ scaler / imputer / feature names / segment labels)
- Idem ``loyalty_provider``
- ``loyalty_segment_distribution.csv`` — effectifs et parts par segment et par entité (si entraînement OK)
- ``metrics_clustering.json`` (bloc ``modes.beneficiary`` / ``modes.provider``)
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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ML.cluster_labels import cluster_labels_from_centers
from ML.ml_paths import ML_MODELS, ML_PROCESSED, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import (
    CLUSTERING_NUMERIC_DROP,
    SQL_ML_CLUSTERING_LOYALTY_BENEFICIARY,
    SQL_ML_CLUSTERING_LOYALTY_PROVIDER,
    SQL_ML_PERFORMANCE_WIDE,
)

K_LOYALTY = 4

LOYALTY_ENTITY_DROP: dict[str, tuple[str, ...]] = {
    "beneficiary": ("id_beneficiary",),
    "provider": ("id_provider",),
}


def _numeric_feature_matrix(df: pd.DataFrame, extra_drop: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    drop_cols = [c for c in CLUSTERING_NUMERIC_DROP if c in df.columns]
    drop_cols.extend([c for c in extra_drop if c in df.columns])
    num = df.select_dtypes(include=[np.number]).columns.drop(labels=drop_cols, errors="ignore")
    if len(num) == 0:
        raise ValueError("0 colonne numérique utilisable après exclusion des clés")
    X_raw = df[list(num)].replace([np.inf, -np.inf], np.nan).values
    return X_raw, [str(c) for c in num]


def loyalty_aggregate_pandas(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Même logique métier que les requêtes SQL (repli sans agrégation SQL)."""
    col = "id_beneficiary" if entity == "beneficiary" else "id_provider"
    if col not in df.columns:
        raise ValueError(f"Colonne {col} absente du jeu performance large")
    df = df.copy()
    df["_fp"] = pd.to_numeric(df["final_price"], errors="coerce")
    if "full_date" in df.columns:
        df["_dt"] = pd.to_datetime(df["full_date"], errors="coerce")
    else:
        df["_dt"] = pd.NaT
    if "nb_visitors" in df.columns:
        nv = pd.to_numeric(df["nb_visitors"], errors="coerce").fillna(0)
    else:
        nv = pd.Series(0.0, index=df.index)
    if "nb_reservations_site" in df.columns:
        ns = pd.to_numeric(df["nb_reservations_site"], errors="coerce").fillna(0)
    else:
        ns = pd.Series(0.0, index=df.index)
    df["_nv"], df["_ns"] = nv, ns
    parts: list[dict] = []
    now = pd.Timestamp.now().normalize()
    for key, sub in df.groupby(col):
        if pd.isna(key):
            continue
        nb = int(len(sub))
        ca = float(np.nansum(sub["_fp"].values))
        pm = float(np.nanmean(sub["_fp"].values)) if nb else 0.0
        last = sub["_dt"].max()
        if pd.notna(last):
            rec = float((now - last).days)
        else:
            rec = 9999.0
        parts.append(
            {
                col: key,
                "nb_reservations_loyalty": float(nb),
                "ca_total_loyalty": ca,
                "panier_moyen_loyalty": pm,
                "recency_days_loyalty": rec,
                "avg_nb_visitors_loyalty": float(np.nanmean(sub["_nv"].values)),
                "volume_reservations_site_loyalty": float(np.nansum(sub["_ns"].values)),
            }
        )
    return pd.DataFrame(parts)


def _load_loyalty_frame(engine, sql: str, entity: str) -> pd.DataFrame | None:
    try:
        df = read_dw_sql(sql, engine)
        if df is None or len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"Lecture SQL fidélité ({entity}) impossible:", e)
        return None


def _fallback_performance_wide(engine) -> pd.DataFrame | None:
    try:
        df = read_dw_sql(SQL_ML_PERFORMANCE_WIDE, engine)
        return df if df is not None and len(df) else None
    except Exception as e:
        print("Repli performance wide:", e)
        return None


def train_loyalty_cluster(
    df: pd.DataFrame,
    entity: str,
    prefix: str,
    models_dir: Path,
) -> dict | None:
    """Entraîne K-Means, exporte joblib + JSON ; retourne le bloc métrique."""
    extra = LOYALTY_ENTITY_DROP.get(entity)
    if extra is None:
        return None
    try:
        X_raw, feat_names = _numeric_feature_matrix(df, extra)
    except ValueError as e:
        print(f"[{prefix}] {e}")
        return None

    n_cap = min(12000, len(X_raw))
    X_work = X_raw[:n_cap]
    k = min(K_LOYALTY, len(X_work))
    if k < 2 or len(feat_names) == 0:
        print(f"[{prefix}] Pas assez de lignes ou de features (n={len(X_work)}, p={len(feat_names)})")
        return None

    idx_all = np.arange(len(X_work))
    idx_train, idx_hold = train_test_split(idx_all, test_size=0.2, random_state=42)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xm = scaler.fit_transform(imp.fit_transform(X_work))
    X_train_m = Xm[idx_train]
    X_hold_m = Xm[idx_hold]

    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_train_m)
    lab_tr = km_temp.predict(X_train_m)
    lab_ho = km_temp.predict(X_hold_m)
    sil_train = float(silhouette_score(X_train_m, lab_tr)) if len(np.unique(lab_tr)) > 1 else 0.0
    sil_hold = float(silhouette_score(X_hold_m, lab_ho)) if len(np.unique(lab_ho)) > 1 else 0.0

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xm)
    labels = km.predict(Xm)
    db_km = float(davies_bouldin_score(Xm, labels))

    cluster_label_short, cluster_label_long = cluster_labels_from_centers(
        np.asarray(km.cluster_centers_), feat_names
    )

    joblib.dump(km, models_dir / f"kmeans_{prefix}.joblib")
    joblib.dump(scaler, models_dir / f"kmeans_standard_scaler_{prefix}.joblib")
    joblib.dump(imp, models_dir / f"kmeans_median_imputer_{prefix}.joblib")
    (models_dir / f"clustering_feature_names_{prefix}.json").write_text(
        json.dumps({"features": feat_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    uniq_l, counts = np.unique(labels, return_counts=True)
    _share = pd.Series(labels).value_counts(normalize=True).sort_index()
    distribution_rows = [
        {
            "entity": entity,
            "artifact_prefix": prefix,
            "cluster_id": int(cid),
            "n_rows": int(n),
            "share_sample": float(_share.get(int(cid), 0.0)),
        }
        for cid, n in zip(uniq_l, counts)
    ]
    seg_doc = {
        "k": k,
        "method": "loyalty_rfm_kmeans",
        "entity": entity,
        "description_fr": "Segments fidélité (RFM simplifié) — libellés dérivés des centres K-Means.",
        "segments": [
            {
                "cluster_id": i,
                "label_short": cluster_label_short[i],
                "label_long_plain": cluster_label_long[i].replace("**", ""),
                "label_metier_fr": _default_loyalty_story(entity, i, k),
                "share_train_sample": float(_share.get(i, 0.0)),
            }
            for i in range(k)
        ],
    }
    (models_dir / f"clustering_segment_labels_{prefix}.json").write_text(
        json.dumps(seg_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "entity": entity,
        "artifact_prefix": prefix,
        "k": k,
        "silhouette": sil_hold,
        "silhouette_train": sil_train,
        "silhouette_holdout": sil_hold,
        "davies_bouldin_kmeans": db_km,
        "n_samples": int(len(X_work)),
        "n_train": int(len(idx_train)),
        "n_holdout": int(len(idx_hold)),
        "kpi_alignment": f"fidelite_{entity}s_rfm",
        "features_file": f"clustering_feature_names_{prefix}.json",
        "segment_labels_file": f"clustering_segment_labels_{prefix}.json",
        "model_file": f"kmeans_{prefix}.joblib",
        "scaler_file": f"kmeans_standard_scaler_{prefix}.joblib",
        "imputer_file": f"kmeans_median_imputer_{prefix}.joblib",
        "cluster_share_train_sample": {str(int(i)): float(_share.get(i, 0.0)) for i in range(k)},
        "distribution_rows": distribution_rows,
    }


def _default_loyalty_story(entity: str, cluster_id: int, k: int) -> str:
    """Narration métier indicative (démo) — à affiner avec les centres réels."""
    stories_ben = {
        0: "Profil type **VIP** : volume et CA élevés, récence faible — prioriser offres exclusives et relation privilégiée.",
        1: "**Fidèle** : fréquence régulière, valeur saine — programmes de fidélité et avantages récurrents.",
        2: "**Occasionnel** : achats irréguliers ou panier modéré — campagnes de réactivation ciblées.",
        3: "**À risque / inactif** : récence élevée ou faible engagement — offres de réactivation et enquêtes sortie.",
    }
    stories_prov = {
        0: "Prestataire **stratégique** : fort volume et CA — partenariats prioritaires.",
        1: "Prestataire **actif** : flux régulier — maintien des conditions commerciales.",
        2: "Prestataire **ponctuel** : opportunités de développement.",
        3: "Prestataire **faible rotation** — suivi commercial ou réorientation.",
    }
    d = stories_ben if entity == "beneficiary" else stories_prov
    return d.get(cluster_id, f"Segment {cluster_id + 1}/{k} — profil à qualifier via heatmap des centres.")


def main() -> None:
    ensure_processed_dirs()
    ML_MODELS.mkdir(parents=True, exist_ok=True)

    print(
        "[run_01_clustering] Périmètre : fidélité RFM (agrégats par bénéficiaire / prestataire, *_loyalty). "
        "Le notebook 01_E_clustering_segmentation.ipynb = vue LARGE (fait par ligne), autre jeu et autres artefacts."
    )

    engine = get_sql_engine()
    modes_out: dict[str, dict] = {}
    distribution_rows_all: list[dict] = []

    specs = (
        ("beneficiary", "loyalty_beneficiary", SQL_ML_CLUSTERING_LOYALTY_BENEFICIARY),
        ("provider", "loyalty_provider", SQL_ML_CLUSTERING_LOYALTY_PROVIDER),
    )

    for entity, prefix, sql in specs:
        df = None
        if engine is not None:
            df = _load_loyalty_frame(engine, sql, entity)
            if df is None and entity == "beneficiary":
                wide = _fallback_performance_wide(engine)
                if wide is not None:
                    try:
                        df = loyalty_aggregate_pandas(wide, entity)
                        print(f"[{prefix}] Agrégation pandas (repli SQL) — lignes:", len(df))
                    except Exception as e:
                        print(f"[{prefix}] Agrégation pandas impossible:", e)
            elif df is None and entity == "provider":
                wide = _fallback_performance_wide(engine)
                if wide is not None:
                    try:
                        df = loyalty_aggregate_pandas(wide, entity)
                        print(f"[{prefix}] Agrégation pandas (repli SQL) — lignes:", len(df))
                    except Exception as e:
                        print(f"[{prefix}] Agrégation pandas impossible:", e)

        if df is None and not ml_sql_only():
            # Repli local : parquet performance si un jour disponible
            pass

        if df is not None and len(df) >= K_LOYALTY:
            block = train_loyalty_cluster(df, entity, prefix, ML_MODELS)
            if block:
                modes_out[entity] = block
                distribution_rows_all.extend(block.get("distribution_rows", []))
                print("OK", prefix, "silhouette_holdout=", round(block["silhouette_holdout"], 4))

    if not modes_out:
        if ml_sql_only():
            raise SystemExit(
                "Clustering fidélité : aucune donnée (SQL DW obligatoire). Vérifiez la connexion et les requêtes."
            )
        raw_fp = ML_PROCESSED / "X_raw_numeric.npy"
        if not raw_fp.is_file():
            raise SystemExit("Exécutez run_00_data_preparation.py (X_raw_numeric.npy manquant) ou configurez le DW.")
        X_raw = np.load(raw_fp)
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_for_cluster = scaler.fit_transform(imp.fit_transform(X_raw))
        n_cap = min(8000, len(X_for_cluster))
        X_work = X_for_cluster[:n_cap]
        k = K_LOYALTY
        idx_all = np.arange(len(X_work))
        idx_train, idx_hold = train_test_split(idx_all, test_size=0.2, random_state=42)
        X_train = X_work[idx_train]
        X_hold = X_work[idx_hold]
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_train)
        sil_hold = float(silhouette_score(X_hold, km_temp.predict(X_hold)))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_work)
        feat_names = [f"dim_{i}" for i in range(X_work.shape[1])]
        cluster_label_short, cluster_label_long = cluster_labels_from_centers(
            np.asarray(km.cluster_centers_), feat_names
        )
        joblib.dump(km, ML_MODELS / "kmeans_kpi_segments.joblib")
        joblib.dump(scaler, ML_MODELS / "kmeans_standard_scaler.joblib")
        joblib.dump(imp, ML_MODELS / "kmeans_median_imputer.joblib")
        labels = km.predict(X_work)
        _, counts = np.unique(labels, return_counts=True)
        _share = pd.Series(labels).value_counts(normalize=True).sort_index()
        metrics = {
            "task": "clustering",
            "model": "KMeans",
            "k": k,
            "silhouette": sil_hold,
            "silhouette_holdout": sil_hold,
            "n_samples": int(len(X_work)),
            "kpi_alignment": "diversite_offre_segments_fallback",
            "cluster_segment_labels_file": "clustering_segment_labels.json",
            "cluster_feature_names_file": "clustering_feature_names.json",
            "cluster_share_train_sample": {str(int(i)): float(_share.get(i, 0.0)) for i in range(k)},
            "modes": {},
        }
        (ML_MODELS / "metrics_clustering.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("OK fallback legacy kmeans_kpi_segments (pas de SQL fidélité)")
        return

    # Référence « champion » : bénéficiaires si dispo, sinon prestataires
    primary = "beneficiary" if "beneficiary" in modes_out else "provider"
    primary_block = modes_out[primary]

    metrics = {
        "task": "clustering_loyalty_rfm",
        "model": "KMeans",
        "model_primary": "KMeans",
        "k": primary_block["k"],
        "silhouette": primary_block["silhouette_holdout"],
        "silhouette_holdout": primary_block["silhouette_holdout"],
        "silhouette_train": primary_block["silhouette_train"],
        "n_samples": primary_block["n_samples"],
        "n_train": primary_block["n_train"],
        "n_holdout": primary_block["n_holdout"],
        "kpi_alignment": "fidelite_beneficiaires_et_prestataires",
        "default_mode": primary,
        "modes": modes_out,
        "davies_bouldin_kmeans": modes_out.get(primary, {}).get("davies_bouldin_kmeans"),
    }
    (ML_MODELS / "metrics_clustering.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if distribution_rows_all:
        dist_fp = ML_MODELS / "loyalty_segment_distribution.csv"
        pd.DataFrame(distribution_rows_all).to_csv(dist_fp, index=False, encoding="utf-8")
        print("OK", dist_fp, "lignes=", len(distribution_rows_all))
    print("OK", ML_MODELS / "metrics_clustering.json", "modes:", list(modes_out.keys()))


if __name__ == "__main__":
    main()
