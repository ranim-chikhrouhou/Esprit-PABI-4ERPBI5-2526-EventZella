# -*- coding: utf-8 -*-
"""
Réorganise les notebooks 01–04 : intro unifiée (objectif global, cible technique,
2 modèles + justification, sommaire), sections séparées, emojis.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [x if x.endswith("\n") else x + "\n" for x in s.splitlines(True)]


# --- Contenus markdown (intro + référence) ---

INTRO_01 = """# 📊 Segmentation par clustering — EventZilla (critère **E**)

## 🎯 Objectif global
Regrouper les **lignes du DW** (faits performance / réservations) en **segments homogènes** sans étiquette supervisée, pour illustrer une **typologie** utile au pilotage (mix d’offre, volumes, montants) — aligné sur le KPI *diversité d’offre* / profils.

## 🎯 Objectif technique (pas de variable « cible » supervisée)
- **Entrées** : matrice numérique issue de **`SQL_ML_PERFORMANCE_WIDE`** (`Fact_PerformanceCommerciale` + `DimReservation` + `DimDate` : montants, ids, attributs calendaires) ou repli **`X_raw_numeric.npy`** (notebook 00).
- **Sortie** : étiquettes de **cluster** (0 … *k*−1) + métriques de qualité de partition.

## 🔀 Les deux modèles comparés (et pourquoi)
| Modèle | Intérêt pour nos données |
|--------|-------------------------|
| **K-Means** | Partitions convexes, **centres** interprétables (heatmap des profils) ; rapide ; adapté après **StandardScaler**. |
| **Clustering agglomératif (Ward)** | Hiérarchie **sans supposer** des grappes sphériques ; utile pour **contraster** avec K-Means sur la même grille *k*. |

## 📑 Sommaire
1. 🔌 Connexion DW — même environnement que SSMS.
2. 📥 Chargement & standardisation des features numériques.
3. 📐 Choix de *k* (coude, silhouette) puis **Modèle 1 — K-Means** et **Modèle 2 — Agglomératif** (métriques comparées).
4. 📊 Holdout sur K-Means (stabilité) — **PCA 2D** et **heatmap** des centres.
5. 💾 Sauvegarde `kmeans_kpi_segments.joblib` + `metrics_clustering.json`.

## ✅ Résultats attendus (validation E)
Silhouette, Davies-Bouldin, méthode du coude ; **comparaison explicite** des deux modèles ; visualisations **PCA** et **profilage** (heatmap).

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`
"""

REF_01 = """## 📌 Référence — critère **E** (rappel)

| Thème | Ce notebook |
|-------|-------------|
| ≥ 2 modèles | K-Means vs Agglomératif (Ward) |
| Évaluation | Silhouette ↑, Davies-Bouldin ↓, coude (inertie) |
| Viz | PCA 2D, heatmap des centres |

*Hypothèses (B)* : K-Means → clusters plutôt compacts ; agglomératif → structure hiérarchique, coût plus élevé sur très grands *n*.
"""

INTRO_02 = """# 📊 Classification supervisée — statut de réservation (critère **C**)

## 🎯 Objectif global
Construire un modèle qui **prédit le statut d’une réservation** à partir des informations financières et dimensionnelles du DW EventZilla, pour relier les KPI *taux d’acceptation / annulation* et l’entonnoir de conversion.

## 🎯 Objectif technique (cible supervisée)
- **Variable cible *y*** : **`reservation_status`** (ou colonne résolue équivalente : `status` / `statut` sur `DimReservation`), encodée en classes entières.
- **Features *X*** : jusqu’à **20 colonnes numériques** du jeu large (`final_price`, `event_budget`, identifiants dimensionnels, etc.) issues de `dw_financial_wide.parquet` ou requête SQL (`build_sql_ml_financial_wide`).
- **Découpage** : **train / test 75 % / 25 % stratifié** sur *y* ; **GridSearchCV** en **5-fold stratifié** **sur le train uniquement**.

## 🔀 Les deux modèles comparés (et pourquoi)
| Modèle | Justification métier / données |
|--------|--------------------------------|
| **Random Forest** | Capture des **non-linéarités** et interactions ; `class_weight='balanced'` pour le **déséquilibre** des statuts ; importances de variables pour l’explicabilité. |
| **Régression logistique (multinomiale)** | **Baseline** forte : frontières **linéaires** en log-odds après scaling ; coefficients interprétables ; même pondération de classes. |

## 📑 Sommaire
1. 🔌 Connexion DW.
2. 📥 Chargement du DataFrame + présence du statut réservation.
3. 🧪 Préparation *X*, *y*, **split stratifié**.
4. 🌲 **Modèle 1 — Random Forest** (pipeline + GridSearch).
5. 📐 **Modèle 2 — Régression logistique** (pipeline + GridSearch).
6. 📊 **Comparaison** : Accuracy, Precision / Recall / F1 (pondérés), ROC-AUC — matrice de confusion, ROC (binaire), importances (RF).
7. 💾 Pipeline champion + `metrics_classification.json`.

## ✅ Métriques de validation (critère C)
Accuracy, Precision, Recall, F1-score (pondérés), **ROC-AUC** ; matrice de confusion ; courbe ROC si **binaire** ; importances (RF).

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`
"""

REF_02 = """## 📌 Référence — grille **C**
Pipeline `StandardScaler → estimateur`, `GridSearchCV`, `StratifiedKFold`, `class_weight='balanced'`, métriques sur le **jeu test** après sélection d’hyperparamètres sur le train.
"""

INTRO_03 = """# 📈 Régression supervisée — montants & KPI (critère **D**)

## 🎯 Objectif global
Estimer les **montants clés** (panier, budget, benchmark…) à partir des autres variables numériques du fait rentabilité / dimensions, pour le pilotage financier et les projections du dashboard.

## 🎯 Objectif technique (cible de régression)
- **Cible *y*** : la **première colonne disponible** dans l’ordre : `final_price` → `service_price` → `benchmark_avg_price` → `event_budget` → `commission_margin` (dérivée `final_price - service_price` si besoin).
- **Features *X*** : autres colonnes **numériques** du même jeu (`dw_financial_wide.parquet` ou SQL), hors `fact_finance_id`.
- **Découpage** : **train / test 75 % / 25 %** ; **validation croisée 5-fold** sur le **train** pour comparer les modèles avant le score **test** final.

## 🔀 Les deux modèles comparés (et pourquoi)
| Modèle | Justification |
|--------|---------------|
| **Ridge (L2)** | Relations **approximativement linéaires** après `StandardScaler` ; **coefficients** comparables pour l’explicabilité ; régularisation pour le multicolinéarité. |
| **Random Forest régresseur** | Capture **non-linéarités** et interactions ; **importances** pour prioriser les leviers métier. |

## 📑 Sommaire
1. 🔌 Connexion DW.
2. 📥 Chargement des données.
3. 🧪 Construction *X*, *y*, **split train/test**.
4. 📉 **Modèle 1 — Ridge** : CV 5-fold puis ajustement test.
5. 🌲 **Modèle 2 — Random Forest** : idem.
6. 📊 **Comparaison** : MSE, RMSE, MAE, R² — graphiques réel vs prédit, résidus, coefficients Ridge, importances RF.
7. 💾 Pipelines + `metrics_regression.json`.

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`
"""

REF_03 = """## 📌 Référence — grille **D**
Métriques : **MSE, RMSE, MAE, R²** sur le test ; **K-Fold** sur le train pour la stabilité ; résidus et explicabilité (coefficients / importances).
"""

INTRO_04 = """# 📉 Séries temporelles — agrégats mensuels & prévision (critère **F**)

## 🎯 Objectif global
Analyser l’**évolution temporelle** des agrégats EventZilla (volume de faits, chiffre d’affaires, panier moyen) et produire des **prévisions** sur un horizon court pour le pilotage.

## 🎯 Objectif technique (série et cible de prévision)
- **Données** : requête **`SQL_ML_TIME_SERIES_RESERVATIONS`** (`Fact_RentabiliteFinanciere` + `DimDate`) — séries **`nb_fact_rows`**, **`revenue_sum`**, **`avg_final_price`** (priorité à la première disponible).
- **Prévision** : on compare deux modèles sur les **3 derniers mois** (holdout) ; le reste sert d’**entraînement**.

## 🔀 Les deux modèles comparés (et pourquoi)
| Modèle | Justification |
|--------|---------------|
| **Lissage exponentiel de Holt** | Tendance **lisse** ; léger et robuste sur séries courtes mensuelles. |
| **ARIMA** | Structure **AR/I/MA** sur série stationnarisée (différenciation) ; référence classique pour comparaison. |

## 📑 Sommaire
1. 🔌 Connexion DW.
2. 📥 Série mensuelle + choix de l’indicateur.
3. 🔬 **ADF / KPSS** et **décomposition** saisonnière.
4. 📉 **Modèle 1 — Holt** : entraînement / prévision holdout.
5. 📈 **Modèle 2 — ARIMA** : idem.
6. 📊 **Comparaison** : RMSE, MAE, MAPE + courbe réel vs prévisions.
7. 💾 `metrics_timeseries.json`.

## ✅ Analyse requise (critère F)
Stationnarité (ADF, KPSS), décomposition ; métriques **MAPE, RMSE, MAE** sur le test temporel.

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`
"""

REF_04 = """## 📌 Référence — grille **F**
Holdout temporel ; pas de fuite : le modèle ne voit pas les mois test à l’entraînement.
"""


def patch_notebook_01(nb: dict) -> None:
    cells = nb["cells"]
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO_01), "id": cells[0].get("id", "")}
    cells[1] = {"cell_type": "markdown", "metadata": {}, "source": lines(REF_01), "id": cells[1].get("id", "")}
    # Supprimer les anciennes cellules 2–3 (objectifs + structure) — fusionnés dans l’intro
    del cells[2:4]
    # Insérer avant la section "### 🧮 Choix de *k*..." (le sommaire de l’intro contient aussi « Choix de *k* » sans ce titre exact)
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "### 🧮 Choix de *k*" in "".join(c.get("source", [])):
            insert_at = i
            break
    else:
        insert_at = 11
    cells.insert(
        insert_at,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### ✂️ Train / validation (stabilité K-Means)\n\n"
                "Un **holdout 80 % / 20 %** sur les indices de `X_work` permet de mesurer la **stabilité** de la silhouette K-Means (train vs holdout). "
                "Les deux algorithmes (K-Means et agglomératif) sont ensuite comparés sur **tout** `X_work` avec le même *k*.\n"
            ),
        },
    )
    # Après insertion, trouver la grande cellule code avec "K_hi = min"
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "K_hi = min" in "".join(c.get("source", [])):
            big = i
            break
    else:
        return
    src = "".join(cells[big]["source"])
    # Découper en 3 blocs code (marqueurs exacts dans la cellule d’origine)
    m1 = "km_model = KMeans(n_clusters=k_best"
    m2 = "idx_all = np.arange(len(X_work))"
    i1 = src.find(m1)
    i2 = src.find(m2)
    if i1 == -1 or i2 == -1:
        return
    part_a = src[:i1].rstrip() + "\n"
    part_b = src[i1:i2].rstrip() + "\n"
    part_c = src[i2:].rstrip() + "\n"

    new_cells: list = []
    new_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📐 Grille de *k* : **coude** et **silhouette**\n\n"
                "Parcours des *k*, inertie (WCSS), score silhouette par *k* ; choix de `k_best`.\n"
            ),
        }
    )
    new_cells.append({"cell_type": "code", "metadata": {}, "source": lines(part_a), "execution_count": None, "outputs": []})
    new_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 🧩 K-Means vs agglomératif (Ward) — même *k*\n\n"
                "**Modèle 1 — K-Means** : centres + `labels_km`. **Modèle 2 — Agglomératif** : `labels_agg`. "
                "Comparaison **Silhouette** (↑) et **Davies-Bouldin** (↓).\n"
            ),
        }
    )
    new_cells.append({"cell_type": "code", "metadata": {}, "source": lines(part_b), "execution_count": None, "outputs": []})
    new_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📊 Holdout & modèle K-Means final\n\n"
                "Stabilité train/holdout ; ré-entraînement K-Means sur `X_work` pour export.\n"
            ),
        }
    )
    new_cells.append({"cell_type": "code", "metadata": {}, "source": lines(part_c), "execution_count": None, "outputs": []})

    cells[big : big + 1] = new_cells


def patch_notebook_02(nb: dict) -> None:
    cells = nb["cells"]
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO_02), "id": cells[0].get("id", "")}
    cells[1] = {"cell_type": "markdown", "metadata": {}, "source": lines(REF_02), "id": cells[1].get("id", "")}
    del cells[2:4]

    # Trouver cellule GridSearch RF+LR
    gi = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "gs_rf = GridSearchCV" in "".join(c.get("source", [])):
            gi = i
            break
    if gi is None:
        return
    src = "".join(cells[gi]["source"])
    m_lr = "pipe_lr = Pipeline"
    m_rep = "class_labels = [str(c) for c in le.classes_]"
    i_lr = src.find(m_lr)
    i_rep = src.find(m_rep)
    if i_lr == -1 or i_rep == -1:
        return
    part_rf = src[:i_lr].rstrip() + "\n"
    part_lr = src[i_lr:i_rep].rstrip() + "\n"
    part_rest = src[i_rep:].rstrip() + "\n"

    replacement = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 🌲 Modèle 1 — Random Forest (pipeline + GridSearchCV)\n\n"
                "Recherche sur grille (`n_estimators`, `max_depth`) avec **5-fold stratifié** et score **F1 pondéré**.\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(part_rf), "execution_count": None, "outputs": []},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📐 Modèle 2 — Régression logistique multinomiale\n\n"
                "Grille sur **`C`** ; même procédure de validation croisée sur le **train**.\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(part_lr), "execution_count": None, "outputs": []},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📊 Comparaison des modèles — jeu test & figures\n\n"
                "Métriques **Accuracy, Precision, Recall, F1, ROC-AUC** ; **champion** = meilleur F1 pondéré sur le test ; matrice de confusion et ROC (si binaire) ; importances si RF champion.\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(part_rest), "execution_count": None, "outputs": []},
    ]
    cells[gi : gi + 1] = replacement


def patch_notebook_03(nb: dict) -> None:
    cells = nb["cells"]
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO_03), "id": cells[0].get("id", "")}
    cells[1] = {"cell_type": "markdown", "metadata": {}, "source": lines(REF_03), "id": cells[1].get("id", "")}
    del cells[2:4]

    gi = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "pipe_ridge = Pipeline" in "".join(c.get("source", [])):
            gi = i
            break
    if gi is None:
        return
    src = "".join(cells[gi]["source"])

    i_fit = src.find("pipe_ridge.fit(Xtr, ytr)")
    if i_fit == -1:
        return
    block_defs_and_cv = src[:i_fit].rstrip() + "\n"
    block_fit_onwards = src[i_fit:].rstrip() + "\n"

    j = block_defs_and_cv.find("sf = cv_scores")
    if j == -1:
        return
    ridge_cv_block = block_defs_and_cv[:j].rstrip() + "\n"
    rf_cv_block = block_defs_and_cv[j:].rstrip() + "\n"

    replacement = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📉 Modèle 1 — Ridge (L2)\n\n"
                "Pipelines + fonction `cv_scores` ; **validation croisée 5-fold** sur le **train** (Ridge uniquement dans cette cellule).\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(ridge_cv_block), "execution_count": None, "outputs": []},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 🌲 Modèle 2 — Random Forest régresseur\n\n"
                "**CV 5-fold** sur le **train** pour la forêt aléatoire ; affichage des scores CV Ridge + RF.\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(rf_cv_block), "execution_count": None, "outputs": []},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "### 📊 Comparaison test — MSE / RMSE / MAE / R² & graphiques\n\n"
                "Ajustement sur tout le **train**, métriques sur le **test** ; **champion** = RMSE test minimal ; résidus, coefficients Ridge, importances RF.\n"
            ),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(block_fit_onwards), "execution_count": None, "outputs": []},
    ]
    cells[gi : gi + 1] = replacement


def patch_notebook_04(nb: dict) -> None:
    cells = nb["cells"]
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO_04), "id": cells[0].get("id", "")}
    cells[1] = {"cell_type": "markdown", "metadata": {}, "source": lines(REF_04), "id": cells[1].get("id", "")}
    del cells[2:4]

    gi = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "def metrics_ts" in "".join(c.get("source", [])):
            gi = i
            break
    if gi is None:
        return
    src = "".join(cells[gi]["source"])
    m_fc = "fc_holt = fit_holt.forecast"
    m_ar = "try:\n    arima_fit"
    i_h = src.find(m_fc)
    i_a = src.find("m_holt = metrics_ts")
    i_ar = src.find("try:\n    arima_fit")
    if i_h == -1 or i_ar == -1:
        # try without newline
        i_ar = src.find("arima_fit = ARIMA")
    if i_ar == -1:
        return
    # Split: train/test + holt fit + forecast + m_holt
    j = src.find("try:\n    arima_fit")
    if j == -1:
        j = src.find("arima_fit = ARIMA")
    part_holt = src[:j].rstrip() + "\n"
    part_arima_and_rest = src[j:].rstrip() + "\n"

    replacement = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines("### 📉 Modèle 1 — Holt (lissage exponentiel, tendance)\n\nHoldout = **3 derniers mois**.\n"),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(part_holt), "execution_count": None, "outputs": []},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines("### 📈 Modèle 2 — ARIMA\n\nOrdre (1,1,1) ou repli (0,1,1).\n"),
        },
        {"cell_type": "code", "metadata": {}, "source": lines(part_arima_and_rest), "execution_count": None, "outputs": []},
    ]
    cells[gi : gi + 1] = replacement


def main() -> None:
    p01 = ROOT / "01_E_clustering_segmentation.ipynb"
    p02 = ROOT / "02_C_classification_statut_reservation.ipynb"
    p03 = ROOT / "03_D_regression_montants_KPI.ipynb"
    p04 = ROOT / "04_F_series_temporelles.ipynb"

    for path, fn in [
        (p01, patch_notebook_01),
        (p02, patch_notebook_02),
        (p03, patch_notebook_03),
        (p04, patch_notebook_04),
    ]:
        nb = json.loads(path.read_text(encoding="utf-8"))
        fn(nb)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Patched", path.name)


if __name__ == "__main__":
    main()
