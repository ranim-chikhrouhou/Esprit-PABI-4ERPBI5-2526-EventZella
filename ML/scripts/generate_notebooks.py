# -*- coding: utf-8 -*-
"""Génère les 6 notebooks ML (critères A, E, C, D, F + synthèse).

    python ML/scripts/generate_notebooks.py

Sortie : ``00_A_*.ipynb`` … ``05_synthese_*.ipynb`` sous ``ML/notebooks/``.
Références métier : ``docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md``,
``ML/EventZilla_Dashboards_Improved.pdf``.
"""
from __future__ import annotations

import json
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
ROOT = ML_DIR / "notebooks"
META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

REF_KPI_MD = "docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md"
REF_PDF = "ML/EventZilla_Dashboards_Improved.pdf"

# Notebooks : noms de fichiers alignés sur la grille d’évaluation (critères A, E, C, D, F) + synthèse
NB_A = "00_A_preparation_donnees_feature_engineering.ipynb"
NB_E = "01_E_clustering_segmentation.ipynb"
NB_C = "02_C_classification_statut_reservation.ipynb"
NB_D = "03_D_regression_montants_KPI.ipynb"
NB_F = "04_F_series_temporelles.ipynb"
NB_SYNTHESE = "05_synthese_metriques_validation.ipynb"


def _split_code_cell_lines(s: str) -> list[str]:
    """Découpe sur les sauts de ligne hors des chaînes \"...\" ou '...'.

    Évite le piège ``split('\\n')`` : un vrai saut de ligne *dans* un littéral
    Python (ex. ``print(\"a:\\n\", x)``) ne doit pas couper la cellule Jupyter.
    """
    lines: list[str] = []
    buf: list[str] = []
    n = len(s)
    i = 0
    in_dq = False
    in_sq = False
    escaped = False
    while i < n:
        c = s[i]
        if escaped:
            buf.append(c)
            escaped = False
            i += 1
            continue
        if in_dq:
            if c == "\\":
                escaped = True
                buf.append(c)
            elif c == '"':
                in_dq = False
                buf.append(c)
            else:
                buf.append(c)
            i += 1
            continue
        if in_sq:
            if c == "\\":
                escaped = True
                buf.append(c)
            elif c == "'":
                in_sq = False
                buf.append(c)
            else:
                buf.append(c)
            i += 1
            continue
        if c == '"':
            in_dq = True
            buf.append(c)
        elif c == "'":
            in_sq = True
            buf.append(c)
        elif c == "\n":
            lines.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        lines.append("".join(buf))
    return lines


def md(s: str) -> dict:
    lines = s.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(s: str) -> dict:
    lines = _split_code_cell_lines(s.strip())
    src = [ln + "\n" for ln in lines]
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


def save(name: str, cells: list) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": META, "cells": cells}
    (ROOT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("OK", ROOT / name)


REPO_FIND = """
from pathlib import Path
import sys
REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "ML" / "ml_paths.py").is_file():
        break
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "ML" / "ml_paths.py").is_file():
    raise FileNotFoundError("Répertoire de travail attendu : racine du dépôt (dossier contenant ML/).")
sys.path.insert(0, str(REPO_ROOT))
"""

# --- Connexion DW (même serveur / base que dans SSMS) — aligné avec ML/notebook_enrich_shared.py
SSMS_CONN_MD = """## 🔌 Connexion au Data Warehouse (SQL Server / SSMS)

L’accès aux données repose sur le **même environnement** que sous **SSMS** : base **`DW_eventzella`** (ou `EVENTZILLA_SQL_DW`), via **SQLAlchemy** et **pyodbc**, authentification Windows.

La cellule suivante affiche serveur, base, extrait de chaîne de connexion, ainsi qu’un test `SELECT DB_NAME()` / `SERVERPROPERTY`.

> En cas d’anomalie (service SQL, driver ODBC, paramètres `EVENTZILLA_SQL_*`), la configuration est documentée dans `ML/ml_paths.py`."""

SSMS_CONN_CODE = (
    REPO_FIND
    + """
# Connexion DW — diagnostic (même serveur / base que SSMS)
from ML.ml_paths import (
    DATABASE_DW,
    SQL_SERVER,
    SQL_DRIVER,
    SQL_PORT,
    build_windows_auth_uri,
    get_sql_engine,
    ml_sql_only,
    read_dw_sql,
    sql_engine_init_error,
)

print("=" * 62)
print(" EVENTZILLA — Connexion DW (équivalent accès SSMS)")
print("=" * 62)
print("  Serveur SQL      :", SQL_SERVER + (":" + str(SQL_PORT) if SQL_PORT else ""))
print("  Base DW cible    :", DATABASE_DW)
print("  Driver ODBC      :", SQL_DRIVER)
print("  Mode DW seul     :", ml_sql_only(), "(EVENTZILLA_ML_SQL_ONLY=1 → pas de Excel/CSV)")
try:
    _uri = build_windows_auth_uri()
    print("  Chaîne (extrait) :", (_uri[:88] + "…") if len(_uri) > 88 else _uri)
except Exception as _uerr:
    print("  Chaîne URI       : erreur", _uerr)
_eng = get_sql_engine()
if _eng is not None:
    try:
        _chk = read_dw_sql(
            "SELECT DB_NAME() AS base_active, CAST(SERVERPROPERTY('ServerName') AS NVARCHAR(128)) AS serveur",
            _eng,
        )
        print("  Test SQL         : OK — même base que sous SSMS si base_active =", DATABASE_DW)
        print(_chk.to_string(index=False))
    except Exception as _qerr:
        print("  Test SQL         : ÉCHEC —", _qerr)
else:
    print("  Engine           : ABSENT —", sql_engine_init_error() or "voir pip sqlalchemy pyodbc")
print("=" * 62)
"""
)

FIGURES_GUIDE_MD = """## Où apparaissent les figures (graphiques) ?

Les **graphiques Matplotlib / Seaborn** s’affichent **sous la cellule** qui appelle `plt.show()` (rendu **inline** dans Jupyter / VS Code).

| Problème | Piste |
|----------|--------|
| Aucune image | Exécuter les cellules **dans l’ordre** depuis le haut ; vérifier que la cellule avec `plt.show()` a bien été exécutée. |
| Toujours vide | Ajouter une fois en tête de notebook : `%matplotlib inline` (déjà présent dans les notebooks avec figures). |

**Liste des graphiques :** voir le tableau **Plan** dans ce notebook (chaque figure apparaît juste après la cellule qui la génère)."""

# --------------------------------------------------------------------------- 00
save(
    NB_A,
    [
        md(
            f"""# Critère **A** — Préparation des données et feature engineering

## Contenu attendu pour la validation
**Pas de modèle ML** : extraction depuis le **DW** (connexion type SSMS), nettoyage (infini, manquants), **imputation**, **mise à l’échelle** ; sorties consommées par les notebooks **E, C, D, F**.

## Références
- `{REF_PDF}`
- `{REF_KPI_MD}`"""
        ),
        md(
            """## Grille d’évaluation — **A — Data Preparation & Feature Engineering**

### 1. Data cleaning

| Sous-thème | Traitement dans ce notebook | Justification |
|------------|----------------------------|---------------|
| **Valeurs manquantes** | Diagnostic (% par colonne) ; imputation par **médiane** (`SimpleImputer`) après visualisation | La médiane est **robuste aux asymétries** et aux quelques extrêmes ; adaptée aux montants et identifiants numériques mélangés. |
| **Outliers (valeurs aberrantes)** | **Pas d’élimination automatique** des lignes : préservation du volume métier et des cas réels (gros paniers, etc.). Les **boxplots** et le **taux IQR** documentent l’ampleur des extrêmes ; la mise à l’échelle réduit leur **dominance numérique** sans les censurer. |
| **Encodage** | Les variables **catégorielles** (ex. libellés de statut) ne sont pas dans `X_raw` : elles sont gérées dans le notebook **C** (classification). Ici : **features déjà numériques** (DW / jointures). |
| **Scaling / normalisation** | **StandardScaler** (z-score) comme sortie principale ; **MinMaxScaler** [0,1] en variante exportée | Voir section *« Choix du scaling »* ci-dessous. |

### 2. Feature engineering & sélection

| Approche (cours) | Utilisation ici |
|------------------|-----------------|
| **Filter** (filtre) | Sélection **implicite** : uniquement colonnes **numériques** du DataFrame ; possibilité d’exclure plus tard des colonnes à **variance nulle** (affichée en annexe). Pas de seuil de corrélation appliqué ici pour garder un maximum d’information pour les notebooks aval. |
| **Wrapper** (ex. RFE) | **Non utilisé** à cette étape : coûteux et lié à un modèle superviseur ; réservé aux pipelines **C / D** si besoin. |
| **Embedded** (ex. Lasso, importances RF) | **Non à l’étape A** ; présent dans les notebooks **C / D** après choix du modèle. |
| **Domain-based (métier)** | **Approche principale** : variables issues du **schéma en étoile** EventZilla (`Fact_*`, `Dim*`) via `build_sql_ml_financial_wide` — alignement **KPI** (prix, réservation, événement, date). |

### Complément **B** (compréhension en aval)

Pas d’entraînement dans ce notebook ; les hypothèses des algorithmes (distances, pénalités L2, etc.) sont détaillées dans **E, C, D, F**."""
        ),
        md(
            """## Choix du **StandardScaler** vs **MinMaxScaler** vs **RobustScaler**

- **StandardScaler** (μ=0, σ=1) : **référence** pour ce projet — adapté au **clustering** (distances), à la **régression Ridge/Lasso** et à la **régression logistique** (pénalités comparables entre variables). Les coefficients restent comparables en ordre de grandeur relatif.
- **MinMaxScaler** [0,1] : produit en **complément** pour algorithmes ou visualisations qui exigent des **bornes fixes** (certaines NN, interprétation « proportion »).
- **RobustScaler** (médiane et IQR) : **non retenu par défaut** ici car nous imputons déjà à la médiane ; pourrait être testé si des **outliers extrêmes** dominaient encore après analyse (non montré pour limiter la redondance avec l’imputation médiane).

**Pourquoi standardiser ?** Les échelles diffèrent fortement (ex. `id_*` vs `final_price`) : sans scaling, une variable à grande variance **impose** sa métrique aux méthodes basées sur la distance ou la pénalisation L2."""
        ),
        md(
            """## Périmètre et livrables

| Livré | Non couvert (autres notebooks) |
|-------|--------------------------------|
| `X_raw_numeric.npy`, `X_standardized.npy`, `X_minmax.npy`, `features_matrix.*`, `numeric_feature_list.json` | Entraînement de modèles |
| Parquet `dw_financial_wide.parquet` | Prédiction / scoring |

## Chaîne aval (KPI EventZilla)

| Critère | Usage des sorties du critère A |
|---------|-------------------------------|
| **E** | Matrice numérique pour clustering |
| **C** / **D** | `dw_financial_wide.parquet` et matrices |
| **F** | Agrégats SQL (séries) depuis le DW |"""
        ),
        md(
            """## Déroulé

1. Connexion **SSMS / DW**
2. Chargement des tables et extraction financière large → `dw_financial_wide.parquet`
3. Construction de **`X_raw`**, diagnostic **manquants**, **figures EDA** (manquants, boxplots, corrélation, IQR)
4. **Imputation** + **StandardScaler** / **MinMaxScaler**, export disque
5. **Figure** : distributions **avant / après** standardisation (exemple sur variables métier)

**Données :** jointures `f.id_*` → `Dim*.*_SK` ; `EVENTZILLA_ML_SQL_ONLY=1` impose le DW par défaut."""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        code(
            REPO_FIND
            + '''
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

warnings.filterwarnings("ignore")

from ML.ml_paths import (
    ML_PROCESSED,
    SQL_SERVER,
    backup_paths_status,
    ensure_processed_dirs,
    get_sql_engine,
    ml_sql_only,
    read_dw_sql,
    sql_engine_init_error,
    DATABASE_DW,
)
from ML.schema_eventzilla import (
    SQL_LIST_TABLES,
    SQL_ML_FINANCIAL_WIDE,
    build_sql_ml_financial_wide,
)

ensure_processed_dirs()
print("[1] Backups FilesMachine/DB (présence fichier) :", backup_paths_status())

engine = get_sql_engine()
df_ml = None
if engine is None and ml_sql_only():
    raise RuntimeError("[1] Connexion SQL obligatoire (EVENTZILLA_ML_SQL_ONLY=1).")
if engine is not None:
    try:
        tables = read_dw_sql(SQL_LIST_TABLES, engine)
        print("[1] Tables DW (aperçu):")
        print(tables.head(25))
        try:
            q_fin = build_sql_ml_financial_wide(engine)
            df_ml = read_dw_sql(q_fin, engine)
            print("[1] Chargement DW : requête financière adaptée (jointures id_date → id_date_SK, etc.).")
        except Exception as e_dyn:
            print("[1] Requête dynamique indisponible, essai requête statique :", e_dyn)
            df_ml = read_dw_sql(SQL_ML_FINANCIAL_WIDE, engine)
        if df_ml is not None and len(df_ml) == 0:
            if ml_sql_only():
                raise RuntimeError("[1] 0 ligne depuis le DW — vérifiez les faits peuplés et id_date = id_date_SK.")
            print("[1] 0 ligne DW — bascule données locales (ML_SQL_ONLY=0).")
            df_ml = None
        if df_ml is not None:
            print("[1] Jeu large financier (lignes, cols) :", df_ml.shape)
            df_ml.to_parquet(ML_PROCESSED / "dw_financial_wide.parquet", index=False)
            print("[1] Parquet sauvegardé : dw_financial_wide.parquet")
    except Exception as e:
        if ml_sql_only():
            raise RuntimeError("[1] Lecture DW requise : " + str(e)) from e
        print("[1] Échec lecture DW — repli fichiers locaux si ML_SQL_ONLY=0 :", e)
        df_ml = None
else:
    print("[1] Pas d’engine SQL.")
    _err = sql_engine_init_error()
    if _err:
        print("[1] Détail technique :", _err)
    if not ml_sql_only():
        print("[1] Avec ML_SQL_ONLY=0 : repli Excel possible. Sinon : pip sqlalchemy pyodbc, ``python ML/scripts/run_test_sql_connection.py``.")'''
        ),
        md(
            """### Matrice numérique brute (`X_raw`)

- Si `df_ml` est disponible : **colonnes numériques** du DW (types pandas `number`).
- Si ``EVENTZILLA_ML_SQL_ONLY=0`` uniquement : repli CSV/Excel via `load_numeric_from_local_csvs`.

**Figures produites dans cette section** (dossier `ML/processed/`, préfixe `A_`) :

| Figure | Rôle pédagogique |
|--------|------------------|
| `A_missing_percent_bar.png` | Visualiser les colonnes les plus incomplètes → prioriser l’imputation et la confiance métier. |
| `A_boxplots_numeric.png` | Repérer asymétries, queues lourdes et ordres de grandeur comparés entre variables. |
| `A_correlation_heatmap.png` | Détecter redondances fortes (corrélations) sans encore supprimer de variables (filter « soft »). |
| `A_iqr_outlier_rate.png` | Quantifier les points **hors moustaches IQR** par colonne — **documentation** des extrêmes sans suppression de lignes. |"""
        ),
        code(
            """%matplotlib inline
from ML.csv_local_fallback import csv_search_roots_hint, load_numeric_from_local_csvs
from ML.ml_paths import ML_PROCESSED, ml_sql_only
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

if df_ml is not None:
    num_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    X_raw = df_ml[num_cols].copy()
    print("[2] Source : DW — colonnes numériques :", len(num_cols))
else:
    if ml_sql_only():
        raise RuntimeError("[2] Pas de données DW — vérifier connexion SSMS, faits peuplés, jointures id_date → id_date_SK.")
    try:
        X_raw = load_numeric_from_local_csvs()
        print("[2] Source : fichiers locaux (ML_SQL_ONLY=0).")
    except FileNotFoundError as e:
        raise SystemExit(
            "[2] Aucune donnée exploitable. Essayez : "
            f"{csv_search_roots_hint()}. Détail : {e}"
        ) from e

X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
print("[2] Taille X_raw :", X_raw.shape)
print("[2] Exemples de colonnes :", list(X_raw.columns)[:12])

miss_pct = (X_raw.isna().sum() / max(len(X_raw), 1) * 100).sort_values(ascending=False)
print("[2] Critère A — % manquants (top 8 colonnes) :")
print(miss_pct.head(8).round(2).to_string())

EDA_DIR = ML_PROCESSED
EDA_DIR.mkdir(parents=True, exist_ok=True)

top_n = min(15, len(miss_pct))
if top_n > 0 and float(miss_pct.iloc[0]) > 0:
    plt.figure(figsize=(10, 4))
    miss_pct.head(top_n).plot(kind="bar", color="steelblue", edgecolor="black", linewidth=0.3)
    plt.ylabel("% de valeurs manquantes")
    plt.title("Top colonnes par part de manquants (critère A)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "A_missing_percent_bar.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("[2] Figure manquants : aucun manquant ou jeu vide — graphique omis.")

plot_cols = []
for c in X_raw.columns:
    s = pd.to_numeric(X_raw[c], errors="coerce").dropna()
    if len(s) >= 30 and s.nunique() > 1 and np.isfinite(s.to_numpy(dtype=float)).all():
        plot_cols.append(c)
    if len(plot_cols) >= 6:
        break
if plot_cols:
    melt = (
        X_raw[plot_cols]
        .apply(pd.to_numeric, errors="coerce")
        .melt(var_name="variable", value_name="valeur")
    )
    plt.figure(figsize=(10, 4.5))
    sns.boxplot(data=melt.dropna(subset=["valeur"]), x="variable", y="valeur", palette="Set2")
    plt.xticks(rotation=35, ha="right")
    plt.title("Distribution (boxplots) — colonnes numériques sélectionnées")
    plt.ylabel("Valeur")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "A_boxplots_numeric.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("[2] Boxplots : pas assez de colonnes exploitables — graphique omis.")

num_df = X_raw.select_dtypes(include=[np.number])
if num_df.shape[1] >= 2:
    k = min(14, num_df.shape[1])
    sub = num_df.iloc[:, :k]
    if len(sub.dropna(how="all")) >= 5:
        cm = sub.corr(numeric_only=True)
        plt.figure(figsize=(max(6, k * 0.45), max(5, k * 0.42)))
        sns.heatmap(
            cm,
            annot=(k <= 10),
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=False,
            linewidths=0.2,
            cbar_kws={"shrink": 0.7},
        )
        plt.title("Matrice de corrélation (Pearson) — sous-ensemble de colonnes numériques")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "A_correlation_heatmap.png", dpi=120, bbox_inches="tight")
        plt.show()
    else:
        print("[2] Heatmap : trop peu de lignes utiles.")
else:
    print("[2] Heatmap : moins de 2 colonnes numériques.")

iqr_rows = []
for c in X_raw.columns[: min(20, X_raw.shape[1])]:
    s = pd.to_numeric(X_raw[c], errors="coerce").dropna()
    if len(s) < 20 or s.nunique() < 2:
        continue
    try:
        arr = np.asarray(s, dtype=np.float64)
    except (TypeError, ValueError):
        continue
    if arr.size < 20 or not np.all(np.isfinite(arr)):
        continue
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0 or not np.isfinite(iqr):
        continue
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_pct = float(((arr < lo) | (arr > hi)).mean() * 100)
    iqr_rows.append((c, out_pct))
if iqr_rows:
    iqr_df = (
        pd.DataFrame(iqr_rows, columns=["colonne", "pct_outliers_IQR"])
        .sort_values("pct_outliers_IQR", ascending=False)
        .head(12)
    )
    plt.figure(figsize=(9, 4))
    sns.barplot(data=iqr_df, x="colonne", y="pct_outliers_IQR", color="coral")
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("% de points hors moustaches IQR")
    plt.title("Documentation des extrêmes (aucune ligne supprimée)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "A_iqr_outlier_rate.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("[2] Taux IQR : pas assez de données par colonne — graphique omis.")

np.save(ML_PROCESSED / "X_raw_numeric.npy", X_raw.to_numpy(dtype=np.float64))
print("[2] Sauvegardé : X_raw_numeric.npy")""",
        ),
        md(
            """### Imputation et mise à l’échelle (critère A)

- **Médiane** (`SimpleImputer`) : complète les trous sans tirer la distribution vers une queue comme le ferait une forte asymétrie avec la moyenne.
- **StandardScaler** : z-score sur la matrice **déjà imputée** — c’est la sortie principale `features_matrix` / `X_standardized.npy` pour les notebooks qui utilisent des **distances** ou des **pénalités L2** sur des coefficients comparables.
- **MinMaxScaler** : variante [0,1] exportée pour les cas où des **bornes fixes** sont requises ; ne remplace pas le z-score comme référence du projet.
- **Extrêmes** : toujours **conservés** en lignes ; leur impact numérique est atténué par le scaling, pas par censure (voir boxplots et IQR ci-dessus).

**Figure** : histogrammes **après imputation (médiane) sans scaling** vs **après StandardScaler** sur une ou deux variables — illustre le z-score (centrage, échelle commune). Fichiers `A_hist_before_after_*.png` dans `ML/processed/`."""
        ),
        code(
            """%matplotlib inline
import matplotlib.pyplot as plt

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

# Histogrammes avant (imputé, non scalé) / après StandardScaler — jusqu’à 2 variables avec variance
demo_idx = []
for j, c in enumerate(X_raw.columns):
    col = X_num[:, j]
    if np.nanstd(col) > 1e-12:
        demo_idx.append((j, c))
    if len(demo_idx) >= 2:
        break
for k, (j, cname) in enumerate(demo_idx):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(X_num[:, j], bins=40, color="gray", alpha=0.88, edgecolor="white")
    axes[0].set_title("Après imputation (médiane), non scalé\\n" + str(cname)[:48])
    axes[0].set_xlabel("Valeur")
    axes[1].hist(X_std[:, j], bins=40, color="steelblue", alpha=0.88, edgecolor="white")
    axes[1].set_title("Après StandardScaler (μ=0, σ=1)\\n" + str(cname)[:48])
    axes[1].set_xlabel("Valeur centrée-réduite")
    plt.tight_layout()
    plt.savefig(ML_PROCESSED / f"A_hist_before_after_{k}.png", dpi=120, bbox_inches="tight")
    plt.show()

print("[3] Terminé. Dossier :", ML_PROCESSED.resolve())""",
        ),
    ],
)

# --------------------------------------------------------------------------- 01
save(
    NB_E,
    [
        md(
            f"""# Critère **E** — Clustering : comparaison de modèles et évaluation

## Contenu attendu pour la validation
Segmentation **non supervisée** : **K-Means** vs **clustering agglomératif** ; **coude**, **silhouette**, **Davies-Bouldin** ; figures **PCA 2D** et **heatmap** des centres ; lien KPI *diversité d’offre*.

## Critère **E** (projet intégré) — ce que ce notebook démontre

| Exigence | Implémentation |
|----------|----------------|
| **≥ 2 modèles** | **K-Means** (partitions convexes, minimisation inertie) **vs** **Clustering hiérarchique agglomératif** (`AgglomerativeClustering`, liaison *ward* — tend à former des groupes compacts). |
| **Évaluation** | **Silhouette**, **indice de Davies-Bouldin** (plus bas = mieux), **méthode du coude** (inertie vs *k*). |
| **Interprétation & viz** | **PCA 2D** pour projection visuelle ; **heatmap** des centres K-Means (profilage par segment). |
| **B — Compréhension** | *K-Means* : hypothèse de **clusters sphériques**, sensible à l’échelle (d’où `StandardScaler`) ; *Agglomératif* : pas de centres explicites, structure **hiérarchique**, coût plus élevé sur très grands *n*. |"""
        ),
        md(
            """## Si `import pandas` échoue (DLL / Smart App Control)

Message du type *« Une stratégie de contrôle d’application a bloqué ce fichier »* ou *`DLL load failed ... ccalendar`* : **Windows** bloque une bibliothèque native de pandas. **Aucune modification du notebook ne corrige ceci.**

**Pistes :** installer **Python 3.11 ou 3.12** (ex. [python.org](https://www.python.org/downloads/)) et en faire le **kernel** Jupyter ; demander une **exception** IT pour Python / `site-packages` ; ou exécuter le projet sous **WSL2** (Ubuntu). **Python 3.14** est très récent : les extensions compilées sont plus souvent bloquées."""
        ),
        md(
            f"""## Objectifs métier (rappel)

| Objectif métier (*descriptif*) | Ce que le clustering peut illustrer |
|--------------------------------|-------------------------------------|
| **Typologie commerciale** | Nombre de **profils** distincts (choix de *k* via coude + silhouette) |
| **Diversité d’offre / mix** | Segments qui **concentrent** certains services / événements |
| **Positionnement tarifaire** | Groupes prix vs benchmark |

**Lien KPI :** `{REF_KPI_MD}` et PDF *Improved*."""
        ),
        md(
            f"""## Données & prétraitement
1. **`SQL_ML_PERFORMANCE_WIDE`** (DW) ou **`X_raw_numeric.npy`** (critère **A**) si `ML_SQL_ONLY=0`.  
2. **Standardisation** obligatoire avant distances.

Références : `{REF_PDF}`, `{REF_KPI_MD}`"""
        ),
        md(
            """## Structure du notebook et figures

| Section | Contenu | Figures (`plt.show()`) |
|---------|---------|-------------------------|
| Connexion DW | Identique SSMS | — |
| Données | `SQL_ML_PERFORMANCE_WIDE` ou repli critère A | — |
| Matrice | Standardisation | — |
| Choix de *k* & modèles | Coude, silhouette, K-Means vs agglomératif | Inertie / silhouette |
| Interprétation | PCA 2D, heatmap centres | Nuages + heatmap |
| Livrables | `.joblib`, `metrics_clustering.json` | — |"""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        md(FIGURES_GUIDE_MD),
        code(
            REPO_FIND
            + '''
%matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ML.ml_paths import ML_PROCESSED, ML_MODELS, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_PERFORMANCE_WIDE, CLUSTERING_NUMERIC_DROP

sns.set_theme(style="whitegrid")
ensure_processed_dirs()
ML_MODELS.mkdir(parents=True, exist_ok=True)
feat_names: list[str] = []
print("[1] Prêt pour le clustering (critère E : K-Means vs Agglomératif). ML_SQL_ONLY:", ml_sql_only())''',
        ),
        md(
            """### Jeu de données pour le clustering (DW)

Requête **`SQL_ML_PERFORMANCE_WIDE`** sur la connexion SSMS ci-dessus. Noms de colonnes utilisés pour les **étiquettes** de la heatmap."""
        ),
        code(
            """X_for_cluster = None
engine = get_sql_engine()
if engine is not None:
    try:
        df = read_dw_sql(SQL_ML_PERFORMANCE_WIDE, engine)
        if len(df) == 0:
            raise ValueError("0 ligne (performance DW)")
        drop_cols = [c for c in CLUSTERING_NUMERIC_DROP if c in df.columns]
        num = df.select_dtypes(include=[np.number]).columns.drop(labels=drop_cols, errors="ignore")
        if len(num) == 0:
            raise ValueError("0 colonne numérique utilisable (performance DW)")
        feat_names = [str(c) for c in num]
        X_raw_mat = df[num].replace([np.inf, -np.inf], np.nan).values
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_for_cluster = scaler.fit_transform(imp.fit_transform(X_raw_mat))
        print("[2] Source DW — forme :", X_for_cluster.shape)
    except Exception as e:
        print("[2] DW indisponible :", e)

if X_for_cluster is None:
    if ml_sql_only():
        raise RuntimeError(
            "[2] SQL DW obligatoire (EVENTZILLA_ML_SQL_ONLY=1) ou ML_SQL_ONLY=0 + 00."
        )
    raw_fp = ML_PROCESSED / "X_raw_numeric.npy"
    if not raw_fp.is_file():
        raise FileNotFoundError("[2] Exécuter d'abord le critère A (`00_A_preparation_donnees_feature_engineering.ipynb`) — X_raw_numeric.npy manquant.")
    X_raw_mat = np.load(raw_fp)
    feat_names = [f"dim_{i}" for i in range(X_raw_mat.shape[1])]
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_for_cluster = scaler.fit_transform(imp.fit_transform(X_raw_mat))
    print("[2] Source 00 — forme :", X_for_cluster.shape)

N_CAP = min(8000, len(X_for_cluster))
X_work = X_for_cluster[:N_CAP]
print("[2] Sous-échantillon X_work :", X_work.shape)""",
        ),
        md(
            """### Choix de *k*, comparaison K-Means / agglomératif

**Graphiques suivants :** inertie vs *k* (coude), silhouette vs *k*. **Métriques :** silhouette (proche de 1), Davies-Bouldin (plus bas = mieux). **Holdout** sur K-Means : stabilité des segments."""
        ),
        code(
            """K_hi = min(10, max(3, len(X_work) // 200))
K_range = range(2, K_hi + 1)
inertias = []
sil_by_k = []
for k in K_range:
    km_i = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab_i = km_i.fit_predict(X_work)
    inertias.append(float(km_i.inertia_))
    sil_by_k.append(float(silhouette_score(X_work, lab_i)))
k_best = int(list(K_range)[int(np.argmax(sil_by_k))])
print("[3] k retenu (max silhouette sur plage) :", k_best, "| plage :", list(K_range))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(K_range), inertias, "o-")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertie (WCSS)")
ax[0].set_title("Critère E — méthode du coude")
ax[1].plot(list(K_range), sil_by_k, "o-", color="green")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette")
ax[1].set_title("Silhouette vs k")
plt.tight_layout()
plt.show()

km_model = KMeans(n_clusters=k_best, random_state=42, n_init=10)
labels_km = km_model.fit_predict(X_work)
agg = AgglomerativeClustering(n_clusters=k_best, linkage="ward")
labels_agg = agg.fit_predict(X_work)

sil_km = float(silhouette_score(X_work, labels_km))
sil_agg = float(silhouette_score(X_work, labels_agg))
db_km = float(davies_bouldin_score(X_work, labels_km))
db_agg = float(davies_bouldin_score(X_work, labels_agg))
print("[3] Comparaison (même k=%d) :" % k_best)
print("     Silhouette  K-Means:", round(sil_km, 4), "| Agglomératif:", round(sil_agg, 4))
print("     Davies-Bouldin (↓ mieux) K-Means:", round(db_km, 4), "| Agglomératif:", round(db_agg, 4))

idx_all = np.arange(len(X_work))
idx_train, idx_hold = train_test_split(idx_all, test_size=0.2, random_state=42)
X_train = X_work[idx_train]
X_hold = X_work[idx_hold]
km_temp = KMeans(n_clusters=k_best, random_state=42, n_init=10)
km_temp.fit(X_train)
lab_tr = km_temp.predict(X_train)
lab_ho = km_temp.predict(X_hold)
sil_train = float(silhouette_score(X_train, lab_tr))
sil_hold = float(silhouette_score(X_hold, lab_ho))
print("[3] K-Means stabilité — Silhouette train:", round(sil_train, 4), "| holdout:", round(sil_hold, 4))

km = KMeans(n_clusters=k_best, random_state=42, n_init=10)
km.fit(X_work)
labels = km.predict(X_work)
n = int(len(X_work))
sil = sil_hold
km_final = km""",
        ),
        md(
            """### Visualisations : PCA et profilage des segments

**PCA (2 axes) :** lecture géométrique des clusters (variance partielle). **Heatmap des centres K-Means** : profil moyen par segment (variables standardisées)."""
        ),
        code(
            """pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_work)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sc0 = ax[0].scatter(X2[:, 0], X2[:, 1], c=labels_km, cmap="tab10", s=10, alpha=0.75)
ax[0].set_title("PCA 2D — K-Means")
plt.colorbar(sc0, ax=ax[0], label="cluster")
sc1 = ax[1].scatter(X2[:, 0], X2[:, 1], c=labels_agg, cmap="tab10", s=10, alpha=0.75)
ax[1].set_title("PCA 2D — Agglomératif")
plt.colorbar(sc1, ax=ax[1], label="cluster")
plt.tight_layout()
plt.show()

n_show = min(20, km.cluster_centers_.shape[1])
cols = feat_names[:n_show] if len(feat_names) >= n_show else [f"f{i}" for i in range(n_show)]
hm = km.cluster_centers_[:, :n_show]
plt.figure(figsize=(10, max(3, k_best * 0.4)))
sns.heatmap(hm, annot=False, cmap="coolwarm", center=0, xticklabels=cols, yticklabels=[f"C{i}" for i in range(k_best)])
plt.title("Profilage clusters — centres K-Means (espace standardisé)")
plt.tight_layout()
plt.show()""",
        ),
        md(
            """### Artefacts produits (modèle principal : K-Means)"""
        ),
        code(
            """joblib.dump(km_final, ML_MODELS / "kmeans_kpi_segments.joblib")
joblib.dump(scaler, ML_MODELS / "kmeans_standard_scaler.joblib")
metrics = {
    "task": "clustering",
    "model_primary": "KMeans",
    "model_secondary": "AgglomerativeClustering_ward",
    "k": k_best,
    "silhouette": float(sil),
    "silhouette_train": float(sil_train),
    "silhouette_holdout": float(sil_hold),
    "silhouette_kmeans_full": sil_km,
    "silhouette_agg_full": sil_agg,
    "davies_bouldin_kmeans": db_km,
    "davies_bouldin_agg": db_agg,
    "n_samples": n,
    "n_train": int(len(X_train)),
    "n_holdout": int(len(X_hold)),
    "kpi_alignment": "diversite_offre_segments_critere_E",
}
(ML_MODELS / "metrics_clustering.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("[5] kmeans_kpi_segments.joblib + metrics_clustering.json (comparaison 2 modèles).")""",
        ),
    ],
)

# --------------------------------------------------------------------------- 02
save(
    NB_C,
    [
        md(
            f"""# Critère **C** — Classification supervisée : statut de réservation

## Contenu attendu pour la validation
**Deux modèles** (Random Forest vs régression logistique), **pipelines** + **GridSearchCV**, **validation croisée stratifiée**, métriques (**Accuracy, Precision, Recall, F1, ROC-AUC**), **matrice de confusion**, **ROC** (binaire), **importances** (RF) ; gestion du déséquilibre (`class_weight`).

## Critère **C** — exigences couvertes

| Exigence | Réalisation |
|----------|-------------|
| **≥ 2 modèles** | **Forêt aléatoire** (non linéaire, agrège des arbres ; `class_weight` pour déséquilibre) **vs** **régression logistique multinomiale** (linéaire en probit après *softmax* ; interprétabilité des coefficients). |
| **Pipeline + tuning** | `Pipeline(StandardScaler → estimateur)` + **`GridSearchCV`** (`StratifiedKFold`, score `f1_weighted`). |
| **Train/test + CV** | Holdout 25 % **stratifié** ; chaque modèle optimisé par **5-fold** sur le train. |
| **Déséquilibre** | `class_weight="balanced"` sur les deux estimateurs. |
| **Métriques** | Accuracy, **Precision / Recall / F1** (weighted), **ROC-AUC** (multiclasse *one-vs-rest*, moyenne pondérée). |
| **Interprétation** | Matrice de confusion, **importance des variables** (RF), courbe **ROC** si **binaire**. |
| **B — Compréhension** | RF : peu d’hypothèses de forme, risque de sur-apprentissage si peu de données ; LR : suppose une **frontière linéaire** dans l’espace transformé (après scaling). |

Références : `{REF_PDF}`, `{REF_KPI_MD}`"""
        ),
        md(
            """## Objectifs métier (rappel)

Le statut de réservation relie les KPI *taux d’acceptation*, *taux d’annulation*, entonnoir conversion — voir le PDF *Improved* et `EventZilla_Dashboards_KPIs_Objectifs.md`."""
        ),
        md(
            """## Structure et figures

| Section | Contenu | Figures |
|---------|---------|---------|
| Connexion DW | SSMS | — |
| Données | `dw_financial_wide.parquet` ou SQL | — |
| Cible / features | *y* = statut, split stratifié | — |
| Modèles & validation | GridSearch RF/LR, métriques test | Confusion, ROC (binaire), importances RF |
| Livrables | Pipeline champion, JSON | — |"""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        md(FIGURES_GUIDE_MD),
        code(
            REPO_FIND
            + '''
%matplotlib inline
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from ML.csv_local_fallback import (
    enrich_financial_wide_with_performance_reservation_status,
    financial_wide_has_status_column,
    load_reservation_dataframe,
    resolve_classification_status_column,
)
from ML.ml_paths import ML_PROCESSED, ML_MODELS, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_FINANCIAL_WIDE, build_sql_ml_financial_wide

warnings.filterwarnings("ignore")
ensure_processed_dirs()
ML_MODELS.mkdir(parents=True, exist_ok=True)
print("[1] Classification critère C — RF vs LogisticRegression. ML_SQL_ONLY:", ml_sql_only())''',
        ),
        code(
            """pp = ML_PROCESSED / "dw_financial_wide.parquet"
df = None
if pp.is_file():
    df = pd.read_parquet(pp)
    print("[2] Source : dw_financial_wide.parquet")
    if not financial_wide_has_status_column(df):
        eng = get_sql_engine()
        if eng is not None:
            try:
                q_fin = build_sql_ml_financial_wide(eng)
                df_dw = read_dw_sql(q_fin, eng)
                if financial_wide_has_status_column(df_dw):
                    df = df_dw
                    print("[2] Parquet sans statut — rechargement DW avec jointure DimReservation (``r.status`` → reservation_status).")
                else:
                    print("[2] Avertissement : requête rentabilité sans statut — essai pont Fact_Performance.")
            except Exception as e:
                print("[2] Rechargement DW pour ajouter le statut : échec —", e)
else:
    eng = get_sql_engine()
    if eng is None:
        if ml_sql_only():
            raise RuntimeError("[2] Pas de parquet DW et pas de SQL (EVENTZILLA_ML_SQL_ONLY=1). Exécuter le critère A ou vérifier la connexion SSMS.")
        df = load_reservation_dataframe()
        print("[2] Source : Reservation (local, ML_SQL_ONLY=0)")
    else:
        try:
            q_fin = build_sql_ml_financial_wide(eng)
            df = read_dw_sql(q_fin, eng)
            print("[2] Source : SQL DW (requête adaptée, jointures *_SK)")
        except Exception as e:
            print("[2] SQL dynamique indisponible, essai statique :", e)
            try:
                df = read_dw_sql(SQL_ML_FINANCIAL_WIDE, eng)
                print("[2] Source : SQL DW (statique)")
            except Exception as e2:
                if ml_sql_only():
                    raise RuntimeError("[2] DW inaccessible (dynamique + statique) : " + str(e2)) from e2
                print("[2] SQL statique échoué, local :", e2)
                df = load_reservation_dataframe()
if df is None:
    raise RuntimeError("[2] Aucun DataFrame chargé.")
if not financial_wide_has_status_column(df):
    _eng = get_sql_engine()
    if _eng is not None:
        df_try = enrich_financial_wide_with_performance_reservation_status(df, _eng)
        if financial_wide_has_status_column(df_try):
            df = df_try
            print("[2] Pont Fact_Performance + DimReservation : ``reservation_status`` fusionné sur (id_date, id_event).")
print("[2] Dimensions :", df.shape)""",
        ),
        code(
            """status_col = resolve_classification_status_column(df)
y_raw = df[status_col].map(lambda x: "__missing__" if pd.isna(x) else str(x))
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "fact_finance_id"][:20]
X_df = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
print("[3] Cible :", status_col, "| Classes :", y_raw.nunique())
print("[3] Features numériques (max 20) :", list(X_df.columns))

le = LabelEncoder()
y = le.fit_transform(y_raw)
strat = y if len(set(y)) > 1 else None
Xtr, Xte, ytr, yte = train_test_split(
    X_df.values, y, test_size=0.25, random_state=42, stratify=strat
)
print("[3] Train / test :", Xtr.shape[0], "/", Xte.shape[0])""",
        ),
        md(
            """### Comparaison des modèles, métriques et graphiques de validation

**Graphiques :** matrice de confusion (champion), ROC si classification binaire, importances variables si RF. **Interprétation :** privilégier les métriques sur le **jeu test** et le **F1 pondéré** en cas de classes déséquilibrées."""
        ),
        code(
            """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe_rf = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
])
grid_rf = {
    "clf__n_estimators": [80, 120],
    "clf__max_depth": [None, 12],
}
gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv, scoring="f1_weighted", n_jobs=-1)
gs_rf.fit(Xtr, ytr)
best_rf = gs_rf.best_estimator_
pred_rf = best_rf.predict(Xte)
proba_rf = best_rf.predict_proba(Xte)

pipe_lr = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=3000, class_weight="balanced")),
])
grid_lr = {"clf__C": [0.1, 1.0, 10.0]}
gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring="f1_weighted", n_jobs=-1)
gs_lr.fit(Xtr, ytr)
best_lr = gs_lr.best_estimator_
pred_lr = best_lr.predict(Xte)
proba_lr = best_lr.predict_proba(Xte)

class_labels = [str(c) for c in le.classes_]


def report_block(name, pred, proba):
    acc = accuracy_score(yte, pred)
    pr_w = precision_score(yte, pred, average="weighted", zero_division=0)
    rc_w = recall_score(yte, pred, average="weighted", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    if len(le.classes_) > 2:
        auc = roc_auc_score(yte, proba, multi_class="ovr", average="weighted")
    else:
        auc = roc_auc_score(yte, proba[:, 1])
    print(f"=== {name} (test) ===")
    print(" Accuracy:", round(acc, 4), "| Precision w:", round(pr_w, 4), "| Recall w:", round(rc_w, 4), "| F1 w:", round(f1w, 4), "| ROC-AUC:", round(auc, 4))
    print(classification_report(yte, pred, target_names=class_labels, zero_division=0))
    return {"acc": acc, "pr": pr_w, "rc": rc_w, "f1": f1w, "auc": auc}


m_rf = report_block("RandomForest (GridSearch)", pred_rf, proba_rf)
m_lr = report_block("LogisticRegression (GridSearch)", pred_lr, proba_lr)

if m_rf["f1"] >= m_lr["f1"]:
    champion, champ_name = best_rf, "RandomForest"
    pred_ch, proba_ch = pred_rf, proba_rf
else:
    champion, champ_name = best_lr, "LogisticRegression"
    pred_ch, proba_ch = pred_lr, proba_lr
print("[4] Champion (F1 pondéré test) :", champ_name)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay.from_predictions(yte, pred_ch, display_labels=class_labels, ax=ax[0], xticks_rotation=45, colorbar=False)
ax[0].set_title("Matrice de confusion — " + champ_name)
if len(le.classes_) == 2:
    RocCurveDisplay.from_predictions(yte, proba_ch[:, 1], ax=ax[1])
    ax[1].set_title("ROC — " + champ_name)
else:
    ax[1].axis("off")
    ax[1].text(0.1, 0.5, "ROC-AUC multiclasse (OvR weighted) déjà dans les métriques ;\\ncourbes par classe optionnelles en extension.", fontsize=10)
plt.tight_layout()
plt.show()

if champ_name == "RandomForest":
    imp = champion.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(8, 4))
    plt.barh(np.array(X_df.columns)[order][::-1], imp[order][::-1])
    plt.title("Importance des variables (RandomForest champion)")
    plt.tight_layout()
    plt.show()

acc = float(accuracy_score(yte, pred_ch))
f1 = float(f1_score(yte, pred_ch, average="weighted", zero_division=0))
prec = float(precision_score(yte, pred_ch, average="weighted", zero_division=0))
rec = float(recall_score(yte, pred_ch, average="weighted", zero_division=0))
if len(le.classes_) > 2:
    roc_auc = float(roc_auc_score(yte, proba_ch, multi_class="ovr", average="weighted"))
else:
    roc_auc = float(roc_auc_score(yte, proba_ch[:, 1]))""",
        ),
        code(
            """joblib.dump(champion, ML_MODELS / "classification_status_champion_pipeline.joblib")
joblib.dump(le, ML_MODELS / "label_encoder_status.joblib")
# Compat anciens scripts : dupliquer sous l’ancien nom si RF
if champ_name == "RandomForest":
    joblib.dump(champion, ML_MODELS / "rf_status_kpi_pipeline.joblib")

(ML_MODELS / "metrics_classification.json").write_text(
    json.dumps({
        "task": "classification",
        "criterion": "C",
        "champion_model": champ_name,
        "gridsearch_rf_best_params": gs_rf.best_params_,
        "gridsearch_lr_best_params": gs_lr.best_params_,
        "test_metrics_champion": {
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "f1_weighted": f1,
            "roc_auc": roc_auc,
        },
        "test_metrics_rf": {
            "accuracy": float(m_rf["acc"]),
            "precision_weighted": float(m_rf["pr"]),
            "recall_weighted": float(m_rf["rc"]),
            "f1_weighted": float(m_rf["f1"]),
            "roc_auc": float(m_rf["auc"]),
        },
        "test_metrics_lr": {
            "accuracy": float(m_lr["acc"]),
            "precision_weighted": float(m_lr["pr"]),
            "recall_weighted": float(m_lr["rc"]),
            "f1_weighted": float(m_lr["f1"]),
            "roc_auc": float(m_lr["auc"]),
        },
        "classes": class_labels,
        "kpi_alignment": "taux_acceptation_annulation_funnel_critere_C",
    }, indent=2),
    encoding="utf-8",
)
print("[5] Pipelines + metrics_classification.json (comparaison 2 modèles + GridSearch).")""",
        ),
    ],
)

# --------------------------------------------------------------------------- 03
save(
    NB_D,
    [
        md(
            f"""# Critère **D** — Régression : comparaison Ridge / Random Forest

## Contenu attendu pour la validation
**Deux modèles** (Ridge, RF), **K-Fold** sur le train, métriques **MSE, RMSE, MAE, R²** sur le test ; graphiques **réel vs prédit**, **résidus**, **coefficients Ridge**, **importances RF** ; lien KPI montants (panier, budget, etc.).

## Critère **D**

| Exigence | Réalisation |
|----------|-------------|
| **≥ 2 modèles** | **Ridge** (L2, coefficients interprétables après scaling) **vs** **`RandomForestRegressor`** (non linéaire, importance des variables). |
| **Validation** | **K-Fold (k=5)** sur le **train** pour estimer RMSE / R² ; **test** final 25 % pour généralisation. |
| **Métriques** | **MSE**, **RMSE**, **MAE**, **R²** (test). |
| **Interprétation** | **Résidus** (test − prédit), **réel vs prédit**, **coefficients Ridge** (valeurs β après scaling), **importances RF**. |
| **B** | Ridge suppose une relation **approximativement linéaire** après mise à l’échelle ; RF capture des **interactions** mais moins d’extrapolation hors support. |

Références : `{REF_PDF}`, `{REF_KPI_MD}`"""
        ),
        md(
            """## Cible principale

On prend la **première** variable disponible dans l’ordre : `final_price` → `service_price` → `benchmark_avg_price` → `event_budget` → `commission_margin` (avec dérivation possible `final_price - service_price`). Les autres colonnes KPI peuvent être traitées en répliquant la même méthode."""
        ),
        md(
            """## Structure et figures

| Section | Contenu | Figures |
|---------|---------|---------|
| Connexion DW | SSMS | — |
| Données | Parquet critère A ou requête SQL | — |
| Préparation | Cible, *X*, *y*, split | — |
| Modèles | CV 5-fold, test | Réel vs prédit, résidus, coef., importances |
| Livrables | `.joblib`, JSON | — |"""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        md(FIGURES_GUIDE_MD),
        code(
            REPO_FIND
            + '''
%matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ML.csv_local_fallback import load_reservation_dataframe
from ML.ml_paths import ML_PROCESSED, ML_MODELS, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_FINANCIAL_WIDE, build_sql_ml_financial_wide

ensure_processed_dirs()
ML_MODELS.mkdir(parents=True, exist_ok=True)
print("[1] Régression critère D — Ridge vs RandomForest. ML_SQL_ONLY:", ml_sql_only())''',
        ),
        code(
            """pp = ML_PROCESSED / "dw_financial_wide.parquet"
if pp.is_file():
    df = pd.read_parquet(pp)
    print("[2] Source : dw_financial_wide.parquet (critère A, après connexion SSMS)")
else:
    eng = get_sql_engine()
    if eng is None:
        if ml_sql_only():
            raise RuntimeError("[2] Pas de parquet DW et pas de SQL (EVENTZILLA_ML_SQL_ONLY=1). Exécuter le critère A (`00_A_preparation_donnees_feature_engineering.ipynb`).")
        df = load_reservation_dataframe()
        print("[2] Source : Reservation (local, ML_SQL_ONLY=0)")
    else:
        try:
            q_fin = build_sql_ml_financial_wide(eng)
            df = read_dw_sql(q_fin, eng)
            print("[2] Source : SQL DW (requête adaptée)")
        except Exception as e:
            print("[2] SQL dynamique indisponible, essai statique :", e)
            try:
                df = read_dw_sql(SQL_ML_FINANCIAL_WIDE, eng)
                print("[2] Source : SQL DW (statique)")
            except Exception as e2:
                if ml_sql_only():
                    raise RuntimeError("[2] DW inaccessible (dynamique + statique) : " + str(e2)) from e2
                print("[2] SQL statique échoué, local :", e2)
                df = load_reservation_dataframe()
print("[2] Dimensions :", df.shape)""",
        ),
        code(
            """MIN_ROWS = 30
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
TARGET_ORDER = [
    "final_price",
    "service_price",
    "benchmark_avg_price",
    "event_budget",
    "commission_margin",
]
primary_target = None
for t in TARGET_ORDER:
    if t in df_reg.columns:
        primary_target = t
        break
if primary_target is None:
    raise ValueError("[3] Aucune colonne cible régression reconnue.")
yraw = pd.to_numeric(df_reg[primary_target], errors="coerce")
oth = [c for c in df_reg.select_dtypes(include=[np.number]).columns if c != primary_target and c != "fact_finance_id"]
block = pd.concat([yraw, df_reg[oth]], axis=1).dropna()
if len(block) < MIN_ROWS or block[primary_target].std(skipna=True) == 0 or not oth:
    raise ValueError("[3] Pas assez de données ou variance nulle pour " + primary_target)
feat_names = list(oth)
y = block[primary_target].values
X = block[oth].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
print("[3] Cible :", primary_target, "| n=", len(block), "| features=", len(feat_names))
kpi_tag = {
    "final_price": "panier_moyen_ca_sum_final_price",
    "service_price": "prix_prestataire_structure_revenus",
    "benchmark_avg_price": "positionnement_tarifaire_benchmark",
    "event_budget": "budget_evenement",
    "commission_margin": "marge_finale_moins_prestataire_commission",
}.get(primary_target, "regression_kpi")""",
        ),
        md(
            """### Validation croisée, généralisation et graphiques d’analyse

**CV (5-fold)** sur le train (stabilité) ; **jeu test** pour **MSE, RMSE, MAE, R²**. **Graphiques :** réel vs prédit, résidus, coefficients Ridge, importances RF."""
        ),
        code(
            """cv = KFold(n_splits=5, shuffle=True, random_state=42)
pipe_ridge = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
pipe_rf = Pipeline([
    ("scale", StandardScaler()),
    ("reg", RandomForestRegressor(n_estimators=120, random_state=42)),
])


def cv_scores(name, pipe):
    mse = -cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_mean_squared_error")
    r2s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2")
    neg_mae = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error")
    return {
        "model": name,
        "cv_rmse_mean": float(np.sqrt(mse.mean())),
        "cv_rmse_std": float(np.sqrt(mse.std())),
        "cv_r2_mean": float(r2s.mean()),
        "cv_mae_mean": float(-neg_mae.mean()),
    }


print("[4] Validation croisée (5-fold) sur TRAIN uniquement :")
sr = cv_scores("Ridge", pipe_ridge)
sf = cv_scores("RandomForest", pipe_rf)
print("    Ridge   — CV RMSE:", round(sr["cv_rmse_mean"], 4), "| CV R²:", round(sr["cv_r2_mean"], 4))
print("    RF      — CV RMSE:", round(sf["cv_rmse_mean"], 4), "| CV R²:", round(sf["cv_r2_mean"], 4))

pipe_ridge.fit(Xtr, ytr)
pipe_rf.fit(Xtr, ytr)
pr_r = pipe_ridge.predict(Xte)
pr_f = pipe_rf.predict(Xte)


def test_metrics(pred):
    mse = mean_squared_error(yte, pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
    }


tr = test_metrics(pr_r)
tf = test_metrics(pr_f)
print("[4] TEST Ridge   :", {k: round(v, 5) for k, v in tr.items()})
print("[4] TEST RF      :", {k: round(v, 5) for k, v in tf.items()})

if tr["rmse"] <= tf["rmse"]:
    champion, champ_name, pred_ch = pipe_ridge, "Ridge", pr_r
else:
    champion, champ_name, pred_ch = pipe_rf, "RandomForest", pr_f
print("[4] Champion (RMSE test minimal) :", champ_name)

fig, ax = plt.subplots(2, 2, figsize=(10, 9))
ax[0, 0].scatter(yte, pr_r, alpha=0.5, s=12)
ax[0, 0].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--", lw=1)
ax[0, 0].set_xlabel("Réel")
ax[0, 0].set_ylabel("Prédit")
ax[0, 0].set_title("Ridge — réel vs prédit (test)")
ax[0, 1].scatter(yte, pr_f, alpha=0.5, s=12, color="green")
ax[0, 1].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--", lw=1)
ax[0, 1].set_title("RandomForest — réel vs prédit (test)")
res_r = yte - pr_r
res_f = yte - pr_f
ax[1, 0].scatter(pr_r, res_r, alpha=0.5, s=12)
ax[1, 0].axhline(0, color="r", ls="--")
ax[1, 0].set_xlabel("Prédit Ridge")
ax[1, 0].set_ylabel("Résidu")
ax[1, 0].set_title("Résidus Ridge")
ax[1, 1].scatter(pr_f, res_f, alpha=0.5, s=12, color="green")
ax[1, 1].axhline(0, color="r", ls="--")
ax[1, 1].set_xlabel("Prédit RF")
ax[1, 1].set_title("Résidus RF")
plt.tight_layout()
plt.show()

coef = pipe_ridge.named_steps["reg"].coef_
idx = np.argsort(np.abs(coef))[-min(15, len(coef)) :]
plt.figure(figsize=(8, 4))
plt.barh(np.array(feat_names)[idx], coef[idx])
plt.title("Coefficients Ridge (variables standardisées)")
plt.tight_layout()
plt.show()

imp = pipe_rf.named_steps["reg"].feature_importances_
idx2 = np.argsort(imp)[::-1][:15]
plt.figure(figsize=(8, 4))
plt.barh(np.array(feat_names)[idx2][::-1], imp[idx2][::-1])
plt.title("Importance des variables — RandomForest")
plt.tight_layout()
plt.show()""",
        ),
        code(
            """joblib.dump(pipe_ridge, ML_MODELS / "ridge_regression_primary.joblib")
joblib.dump(pipe_rf, ML_MODELS / "rf_regression_primary.joblib")
joblib.dump(champion, ML_MODELS / "rf_panier_kpi_pipeline.joblib")

(ML_MODELS / "metrics_regression.json").write_text(
    json.dumps({
        "task": "regression",
        "criterion": "D",
        "champion_model": champ_name,
        "target": primary_target,
        "kpi_alignment": kpi_tag,
        "features": feat_names,
        "cv_ridge": sr,
        "cv_random_forest": sf,
        "test_ridge": tr,
        "test_random_forest": tf,
        "test_champion": tr if champ_name == "Ridge" else tf,
    }, indent=2),
    encoding="utf-8",
)
print("[5] ridge_regression_primary.joblib, rf_regression_primary.joblib, rf_panier_kpi_pipeline (=champion), metrics_regression.json")""",
        ),
    ],
)

# --------------------------------------------------------------------------- 04
save(
    NB_F,
    [
        md(
            f"""# Critère **F** — Séries temporelles : analyse et prévision

## Contenu attendu pour la validation
Tests **ADF / KPSS**, **décomposition** ; **deux modèles** (Holt, ARIMA) ; **RMSE, MAE, MAPE** sur holdout ; figure **série / prévisions** ; données via **`SQL_ML_TIME_SERIES_RESERVATIONS`** (DW).

## Critère **F**

| Exigence | Réalisation |
|----------|-------------|
| **Analyse** | Tests **ADF** & **KPSS** (stationnarité) ; **décomposition** additive (tendance + saisonnier si assez d’observations). |
| **≥ 2 modèles** | **Holt** (`ExponentialSmoothing`, tendance) **vs** **ARIMA** (ordre (1,1,1) ou repli (0,1,1)). |
| **Évaluation** | **RMSE**, **MAE**, **MAPE** sur fenêtre test (derniers mois). |
| **B** | Holt suppose une **tendance lisse** ; ARIMA suppose une structure **AR/I/MA** sur la série différenciée — limites si saisonnalité forte non captée (extension : SARIMA). |

Références : `{REF_PDF}`, `{REF_KPI_MD}`"""
        ),
        md(
            """## Données

Requête **`SQL_ML_TIME_SERIES_RESERVATIONS`** (ou Excel si `ML_SQL_ONLY=0`). Colonnes typiques : `nb_fact_rows`, `revenue_sum`, `avg_final_price`."""
        ),
        md(
            """## Structure et figures

| Section | Contenu | Figures |
|---------|---------|---------|
| Connexion DW | SSMS | — |
| Série mensuelle | Agrégat SQL ou repli local | — |
| Stationnarité | ADF, KPSS, décomposition | 4 panneaux |
| Modèles | Holt vs ARIMA, holdout | Série + prévisions |
| Livrables | `metrics_timeseries.json` | — |"""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        md(FIGURES_GUIDE_MD),
        code(
            REPO_FIND
            + '''
%matplotlib inline
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

from ML.csv_local_fallback import load_reservation_dataframe, monthly_series_from_reservation
from ML.ml_paths import ML_PROCESSED, ML_MODELS, ensure_processed_dirs, get_sql_engine, ml_sql_only, read_dw_sql
from ML.schema_eventzilla import SQL_ML_TIME_SERIES_RESERVATIONS

ensure_processed_dirs()
ML_MODELS.mkdir(parents=True, exist_ok=True)
print("[1] Séries temporelles critère F. ML_SQL_ONLY:", ml_sql_only())''',
        ),
        code(
            """df_ts = None
eng = get_sql_engine()
if eng is not None:
    try:
        df_ts = read_dw_sql(SQL_ML_TIME_SERIES_RESERVATIONS, eng)
        print("[2] Source : agrégat SQL mensuel (SSMS).")
    except Exception as e:
        if ml_sql_only():
            raise RuntimeError("[2] Lecture SQL obligatoire : " + str(e)) from e
        print("[2] SQL indisponible, local :", e)
if df_ts is None:
    if ml_sql_only():
        raise RuntimeError("[2] Aucune donnée (EVENTZILLA_ML_SQL_ONLY=1).")
    df_ts = monthly_series_from_reservation(load_reservation_dataframe())
    print("[2] Source : Reservation (ML_SQL_ONLY=0).")
df_ts["date"] = pd.to_datetime(
    dict(year=df_ts["cal_year"].astype(int), month=df_ts["cal_month"].astype(int), day=1)
)
print(df_ts.head())""",
        ),
        md(
            """### Stationnarité (ADF, KPSS) et décomposition

**ADF / KPSS** : lecture conjointe des *p*-values. **Décomposition** : observé, tendance, saison, résidus (période adaptée à la longueur de série)."""
        ),
        code(
            """SERIES_KPIS = [
    ("nb_fact_rows", "count_id_reservation_mensuel_anticipation"),
    ("revenue_sum", "ca_mensuel_sum_final_price_projection"),
    ("avg_final_price", "panier_moyen_mensuel_projection"),
]
col_main = next((c for c, _ in SERIES_KPIS if c in df_ts.columns), None)
if col_main is None:
    raise ValueError("[3] Aucune colonne séries reconnue.")
ts = df_ts.set_index("date")[col_main].astype(float).sort_index()
if len(ts) < 8:
    raise ValueError("[3] Historique trop court pour ADF/KPSS/prévision fiable.")

adf_res = adfuller(ts.dropna(), autolag="AIC")
adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
print("[3] ADF — stat:", round(adf_stat, 4), "| p-value:", round(adf_p, 4), "| (p<0.05 → rejet racine unitaire)")

try:
    kpss_res = kpss(ts.dropna(), regression="c", nlags="auto")
    kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
    print("[3] KPSS — stat:", round(kpss_stat, 4), "| p-value:", round(kpss_p, 4), "| (p<0.05 → stationnarité rejetée)")
except Exception as _e:
    kpss_stat, kpss_p = None, None
    print("[3] KPSS indisponible :", _e)

period = min(12, max(2, len(ts) // 3))
try:
    decomp = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    decomp.observed.plot(ax=ax[0], title="Série " + col_main)
    decomp.trend.plot(ax=ax[1], title="Tendance")
    decomp.seasonal.plot(ax=ax[2], title="Saison (période=" + str(period) + ")")
    decomp.resid.plot(ax=ax[3], title="Résidus")
    plt.tight_layout()
    plt.show()
except Exception as ex:
    print("[3] Décomposition ignorée :", ex)""",
        ),
        md(
            """### Holt vs ARIMA (holdout et métriques)

Holdout : **3 derniers mois** ; train = reste. Courbe **réel vs prévisions**. **RMSE, MAE, MAPE** (MAPE fragile si niveaux proches de 0)."""
        ),
        code(
            """

def metrics_ts(y_true, y_pred):
    e = y_true - y_pred
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / (y_true + 1e-9))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


train = ts.iloc[:-3]
test = ts.iloc[-3:]
h = len(test)

fit_holt = ExponentialSmoothing(
    train, trend="add", seasonal=None, initialization_method="estimated"
).fit()
fc_holt = fit_holt.forecast(h)
m_holt = metrics_ts(test.values, fc_holt.values)

try:
    arima_fit = ARIMA(train, order=(1, 1, 1)).fit()
except Exception:
    arima_fit = ARIMA(train, order=(0, 1, 1)).fit()
fc_arima = arima_fit.forecast(h)
m_arima = metrics_ts(test.values, np.asarray(fc_arima, dtype=float).ravel())

print("[4] TEST", col_main, "| Holt   :", {k: round(v, 4) for k, v in m_holt.items()})
print("[4] TEST", col_main, "| ARIMA  :", {k: round(v, 4) for k, v in m_arima.items()})

if m_holt["rmse"] <= m_arima["rmse"]:
    primary_model, m_primary = "Holt_ExponentialSmoothing", m_holt
else:
    primary_model, m_primary = "ARIMA", m_arima

fig, ax = plt.subplots(figsize=(9, 4))
train.plot(ax=ax, label="Train", color="C0")
test.plot(ax=ax, label="Test", color="C1")
pd.Series(fc_holt.values, index=test.index).plot(ax=ax, label="Holt", ls="--")
pd.Series(np.asarray(fc_arima).ravel(), index=test.index).plot(ax=ax, label="ARIMA", ls=":")
ax.legend()
ax.set_title("Prévisions vs réel — " + col_main)
plt.tight_layout()
plt.show()""",
        ),
        code(
            """kpi_main = dict(SERIES_KPIS)[col_main]
(ML_MODELS / "metrics_timeseries.json").write_text(
    json.dumps({
        "task": "time_series",
        "criterion": "F",
        "series": col_main,
        "champion_model": primary_model,
        "adf_pvalue": float(adf_p),
        "kpss_pvalue": None if kpss_p is None else float(kpss_p),
        "decomposition_period_used": int(period),
        "test_holt": m_holt,
        "test_arima": m_arima,
        "test_champion": m_primary,
        "horizon": int(h),
        "kpi_alignment": kpi_main,
    }, indent=2),
    encoding="utf-8",
)
print("[5] metrics_timeseries.json (Holt vs ARIMA, critère F).")""",
        ),
    ],
)

# --------------------------------------------------------------------------- 05
save(
    NB_SYNTHESE,
    [
        md(
            f"""# Synthèse — Métriques des modèles et alignement KPI

## Contenu attendu pour la validation
Ce notebook **n’entraîne aucun modèle** : il **agrège** les fichiers **`metrics_*.json`** produits par les critères **E, C, D, F**, affiche un tableau et exporte un rapport **`ML/ML_METRICS_SUMMARY.md`** (lisible correcteurs + CSV).

## Chaîne d’exécution recommandée
1. **`{NB_A}`** — préparation / features (`ML/processed/`, pas de `metrics_*.json`).
2. **`{NB_E}`**, **`{NB_C}`**, **`{NB_D}`**, **`{NB_F}`** — entraînement et écriture des JSON sous `ML/models_artifacts/`.
3. **Ce notebook (`{NB_SYNTHESE}`)** — synthèse finale.

## Correspondance notebooks → fichiers métriques

| Notebook | Critère | Fichier JSON (défaut) |
|----------|---------|------------------------|
| `{NB_A}` | **A** — Data prep | *(aucun ; artefacts `.npy` / `.parquet` dans `processed/`)* |
| `{NB_E}` | **E** — Clustering | `metrics_clustering.json` |
| `{NB_C}` | **C** — Classification | `metrics_classification.json` |
| `{NB_D}` | **D** — Régression | `metrics_regression.json` |
| `{NB_F}` | **F** — Séries temporelles | `metrics_timeseries.json` |

**Alignement KPI** : champ `kpi_alignment` dans chaque JSON ; référence `{REF_KPI_MD}` et `{REF_PDF}`."""
        ),
        md(SSMS_CONN_MD),
        code(SSMS_CONN_CODE),
        md(
            """### Portée de ce notebook

Ce notebook **n’entraîne pas de modèles** : il agrège uniquement les `metrics_*.json` sous `ML/models_artifacts/` (critères **E, C, D, F**). La cellule **connexion** ci-dessus documente l’environnement SQL (cohérence du livrable, comme les autres notebooks).

La synthèse reflète l’état des `metrics_*.json` au moment de l’exécution ; elle est alignée avec les notebooks **E / C / D / F** lorsque ceux-ci ont produit ou mis à jour ces fichiers."""
        ),
        md(
            """## Bonus (barème)

- **Versionnement Git** : le dépôt inclut les notebooks et `ML/requirements.txt` ; des commits par phase (data prep, modèles, figures) facilitent la relecture et la traçabilité.
- **Déploiement** (optionnel) : exposer un modèle via une petite API ou app web ; non requis pour valider les critères **A–F** dans les notebooks."""
        ),
        md(
            """## 📋 Déroulé — étapes couvertes

1. Présence attendue des `metrics_*.json` sous `ML/models_artifacts/`
2. Tableau agrégé + aperçu « par critère »
3. Export `ML/ML_METRICS_SUMMARY.md` (sections + CSV)"""
        ),
        code(
            REPO_FIND
            + """
import json
from datetime import datetime

import pandas as pd
from ML.ml_paths import ML_MODELS

EXPECTED = {
    "metrics_clustering.json": "E — Clustering",
    "metrics_classification.json": "C — Classification",
    "metrics_regression.json": "D — Régression",
    "metrics_timeseries.json": "F — Séries temporelles",
}
paths = sorted(ML_MODELS.glob("metrics_*.json"))
missing = [fn for fn in EXPECTED if not (ML_MODELS / fn).is_file()]
if missing:
    print("[0] JSON manquants (notebooks E/C/D/F non exécutés ou chemins différents) :", missing)
if not paths:
    raise FileNotFoundError(
        "Aucun fichier metrics_*.json dans ML/models_artifacts/. "
        "Ces fichiers proviennent des notebooks E, C, D, F (après la préparation 00)."
    )
rows = []
for p in paths:
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_source_file"] = p.name
    rows.append(d)
df = pd.DataFrame(rows)
print("[1] Fichiers lus :", [p.name for p in paths])
df""",
        ),
        code(
            """import json
from datetime import datetime

from ML.ml_paths import ML_MODELS, ML_PROCESSED

out_md = ML_PROCESSED.parent / "ML_METRICS_SUMMARY.md"
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
parts = [
    "# EventZilla — Synthèse métriques ML\\n\\n",
    f"*Généré : {ts}*\\n\\n",
    "## Rappel chaîne d’exécution\\n\\n",
    "- **Critère A** : pas de `metrics_*.json` — voir `ML/processed/` (matrices, parquets).\\n",
    "- **E, C, D, F** : un fichier `metrics_*.json` par tâche dans `ML/models_artifacts/`.\\n\\n",
    "## Contenu JSON (copie intégrale par fichier)\\n\\n",
]
for p in sorted(ML_MODELS.glob("metrics_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    parts.append(f"### {p.name}\\n\\n")
    parts.append("```json\\n")
    parts.append(json.dumps(d, indent=2, ensure_ascii=False))
    parts.append("\\n```\\n\\n")

parts.append("## Tableau agrégé (CSV)\\n\\n")
parts.append("Les colonnes diffèrent selon les tâches ; le JSON ci-dessus reste la référence.\\n\\n```csv\\n")
_drop = [c for c in df.columns if str(c).startswith("_")]
parts.append(df.drop(columns=_drop, errors="ignore").to_csv(index=False))
parts.append("\\n```\\n")
out_md.write_text("".join(parts), encoding="utf-8")
print("[2] Écrit :", out_md.resolve())""",
        ),
    ],
)

if __name__ == "__main__":
    print("Terminé.")
