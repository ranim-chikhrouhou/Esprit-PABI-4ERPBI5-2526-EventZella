# -*- coding: utf-8 -*-
"""Réorganise le markdown des notebooks 00, 01, 03, 04 (titres, ---, espacements)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"


def L(s: str) -> list[str]:
    s = s.rstrip() + "\n"
    return [line + "\n" for line in s.splitlines()]


def save(nb_path: Path, nb: dict) -> None:
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_00() -> None:
    p = ROOT / "00_A_preparation_donnees_feature_engineering.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    c = nb["cells"]

    c[0]["source"] = L(
        """# Préparation des données et feature engineering (critère **A**)

*Notebook ML — EventZilla (DW, même périmètre que SSMS / Power BI).*

---

## 🎯 Objectif global

Ce notebook transforme les données du **Data Warehouse** EventZilla (même périmètre que sous SSMS / Power BI) en **matrice numérique** prête pour le machine learning : extraction SQL depuis les faits et dimensions (`Fact_RentabiliteFinanciere`, `DimDate`, `DimReservation` lorsque la jointure est résolue), traçabilité des colonnes, encodage des champs catégoriels, exploration de la qualité, **illustration** des familles de sélection de variables attendues au critère **A**, puis imputation et mise à l’échelle. **Aucun modèle final n’est entraîné ici** — les notebooks classification, régression, clustering et séries temporelles réutilisent les fichiers produits dans `ML/processed/`.

**📎 Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`

---

## 📑 Sommaire (ordre d’exécution)

1. 🔌 **Connexion au Data Warehouse** — Vérifier serveur, base active (`DATABASE_DW`), driver ODBC et test SQL ; même logique de confiance qu’un accès SSMS.
2. 💾 **Extraction du jeu financier large** — Liste des tables, requête `build_sql_ml_financial_wide`, sauvegarde `dw_financial_wide.parquet`, lineage **colonne → table** (`A_dw_columns_lineage.json`).
3. 🧮 **Matrice de features et analyse exploratoire** — Construction de `X_raw` (numériques du fait + calendrier + one-hot des catégories), figures `A_*.png`, détail d’encodage dans `A_encoding_onehot_summary.json` et `A_xraw_column_lineage.json`.
4. 🎯 **Illustration de la sélection de variables** — Filter (variance, corrélation), wrapper (RFE sur `final_price`), embedded (LassoCV) → `A_feature_selection_summary.json` (les matrices exportées restent **complètes** pour les notebooks aval).
5. **Imputation, mise à l’échelle et exports** — `SimpleImputer(median)`, `StandardScaler` et `MinMaxScaler`, transformateurs `.joblib`, matrices `X_*.npy` et `features_matrix.*` pour les étapes E / C / D / F.

---

## ✅ Métriques et livrables (critère **A**)

Valeurs manquantes, outliers documentés, encodage, scaling, sélection illustrée — sans entraînement de modèle final dans ce notebook."""
    )

    c[1]["source"] = L(
        """### Référence — critère **A**, livrables et environnement

### Alignement avec le critère A (mail d’évaluation)

| Thème | Traitement dans ce notebook | Données EventZilla |
|------|-----------------------------|---------------------|
| **Valeurs manquantes** | Diagnostic puis imputation **médiane** (étape « Imputation ») | Montants (`final_price`, `event_budget`…), identifiants, modalités 0/1 après one-hot |
| **Outliers** | Aucune suppression ; boxplots + taux IQR | Gros montants ou pics d’activité possibles — documentés, pas censurés |
| **Encodage** | `get_dummies` sur colonnes catégorielles (`full_date`, statuts, etc.) | Colonnes issues surtout de `DimDate` / `DimReservation` selon le SELECT |
| **Scaling** | `StandardScaler` (principal) + `MinMaxScaler` | Même liste de colonnes que `X_raw` pour comparabilité (€, IDs, 0/1) |
| **Sélection** | Filter / wrapper / embedded | Cible `final_price` sur `Fact_RentabiliteFinanciere` si présente ; JSON documentaire |

Les **pipelines complets** (grilles, métriques finales) sont dans les notebooks **C** et **D**. Le **critère B** (hypothèses des modèles) est détaillé dans E / C / D / F.

---

### Fichiers produits (`ML/processed/`)

| Fichier | Rôle |
|---------|------|
| `dw_financial_wide.parquet` | Table large issue du DW |
| `X_raw_numeric.npy`, `X_standardized.npy`, `X_minmax.npy` | Matrices pour notebooks aval |
| `features_matrix.parquet` (ou `.csv`) | Données standardisées |
| `numeric_feature_list.json` | Noms des colonnes (brut / scalé) |
| `A_encoding_onehot_summary.json` | Sources catégorielles + dummies |
| `A_dw_columns_lineage.json` | Colonnes `df_ml` → table DW |
| `A_xraw_column_lineage.json` | Colonnes `X_raw` → table DW |
| `A_feature_selection_summary.json` | Synthèse filter / wrapper / embedded |

---

### Scaling (rappel)

- **StandardScaler** : référence pour distances, Ridge/Lasso, coefficients comparables.
- **MinMaxScaler** : bornes fixes [0, 1] si un algorithme ou une viz l’exige.
- **RobustScaler** : non utilisé par défaut ; à tester si des extrêmes dominent encore après analyse.

---

### Environnement

`EVENTZILLA_ML_SQL_ONLY=1` impose la lecture depuis le DW (pas de repli Excel/CSV). Variables serveur et base : `ML/ml_paths.py` et préfixes `EVENTZILLA_SQL_*`."""
    )

    c[2]["source"] = L(
        """---

## Connexion au Data Warehouse

🎯 **Objectif** : confirmer que Python cible le **même serveur et la même base** que votre session SSMS (reproductibilité des extractions).

✅ **Résultats attendus** : affichage de `SQL_SERVER`, `DATABASE_DW`, driver ODBC ; test `SELECT DB_NAME()` réussi. En cas d’échec : service SQL, pilote ODBC ou configuration dans `ml_paths.py`."""
    )

    c[4]["source"] = L(
        """---

## Extraction du jeu financier large

🎯 **Objectif** : matérialiser le SELECT métier (fait rentabilité + calendrier ± réservation) dans un fichier **Parquet** réutilisable, avec **traçabilité** de chaque colonne vers `dbo.Fact_RentabiliteFinanciere`, `dbo.DimDate` ou `dbo.DimReservation`.

✅ **Résultats attendus** : aperçu des tables `INFORMATION_SCHEMA`, jeu `df_ml` non vide, fichier `dw_financial_wide.parquet`, console listant chaque colonne avec sa table DW probable, et `A_dw_columns_lineage.json`."""
    )

    c[6]["source"] = L(
        """---

## Matrice `X_raw`, encodage et analyse exploratoire (EDA)

🎯 **Objectif** : passer d’un `df_ml` mélangé (numériques + catégories) à une matrice **100 % numérique** : colonnes du fait et de `DimDate` conservées en numérique, catégories en **one-hot** (`drop_first=True`). Documenter quelles colonnes du DW ont produit quelles modalités.

✅ **Résultats attendus** : `X_raw` sauvegardé (`X_raw_numeric.npy`), `A_encoding_onehot_summary.json`, `A_xraw_column_lineage.json`, figures `A_missing_percent_bar.png`, `A_boxplots_numeric.png`, `A_correlation_heatmap.png`, `A_iqr_outlier_rate.png`.

#### 🔍 Lecture des figures (après exécution de la cellule de code)

| Figure | Lecture utile pour votre DW |
|--------|----------------------------|
| `A_missing_percent_bar.png` | Colonnes où l’imputation médiane pèsera le plus (fiabilité) |
| `A_boxplots_numeric.png` | Asymétrie des montants / scores, extrêmes possibles |
| `A_correlation_heatmap.png` | Redondance (ex. temps : `cal_month` vs `quarter`) |
| `A_iqr_outlier_rate.png` | Part des points au-delà des moustaches — pas de suppression ici |"""
    )

    c[8]["source"] = L(
        """---

## Illustration de la sélection de variables (filter · wrapper · embedded)

🎯 **Objectif** : montrer **trois familles** de méthodes du critère A sur **votre** matrice (cible `final_price` du fait rentabilité lorsqu’elle est présente), sans réduire les fichiers matriciels exportés ensuite.

✅ **Résultats attendus** : `A_feature_selection_summary.json` (variance quasi nulle, paires corrélées, RFE, coefficients Lasso). Les exports `X_*.npy` du notebook restent basés sur **toutes** les colonnes de `X_raw` après imputation.

**À retenir** : le filtre liste des colonnes quasi constantes et des paires |r|>0,95 ; le RFE peut mettre en avant des modalités `full_date_*` si le one-hot date est riche — effet de représentation, pas seulement « insight métier » ; le Lasso indique des variables dont le coefficient reste non nul sous L1."""
    )

    c[10]["source"] = L(
        """---

## Imputation, mise à l’échelle et exports pour les notebooks aval

🎯 **Objectif** : remplacer les NaN par la **médiane** par colonne, puis appliquer **StandardScaler** et **MinMaxScaler** sur l’ensemble des features (montants, IDs, dummy 0/1), et sauvegarder matrices + transformateurs pour réappliquer les mêmes règles au scoring.

✅ **Résultats attendus** : `median_imputer.joblib`, `standard_scaler.joblib`, `minmax_scaler.joblib`, `X_standardized.npy`, `X_minmax.npy`, `features_matrix.parquet` ou `.csv`, `numeric_feature_list.json`. Histogrammes avant/après sur un ou deux exemples de colonnes (console : nombre de NaN imputés par colonne illustrée).

#### 📊 Histogrammes avant / après (après exécution)

- **Panneau gauche** : distribution après imputation, **échelle métier** (comme dans le DW après encodage).
- **Panneau droit** : **z-scores** — comparabilité entre colonnes pour Ridge/Lasso, k-means, etc.
- **MinMax** : variantes dans `X_minmax.npy` pour les usages qui exigent [0, 1].

---

*Fin de la préparation — enchaîner avec les notebooks de modélisation (B→F) en réutilisant `ML/processed/`.*"""
    )

    save(p, nb)


def patch_01() -> None:
    p = ROOT / "01_E_clustering_segmentation.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    c = nb["cells"]

    c[0]["source"] = L(
        """# Segmentation par clustering (critère **E**)

*Notebook ML — EventZilla (DW, même périmètre que SSMS / Power BI).*

---

## 🎯 Objectif global

Regrouper les **lignes du DW** (faits performance / réservations) en **segments homogènes** sans étiquette supervisée, pour illustrer une **typologie** utile au pilotage (mix d’offre, volumes, montants) — aligné sur le KPI *diversité d’offre* / profils.

---

## 🎯 Objectif technique (pas de cible supervisée)

- **Entrées** : matrice numérique issue de **`SQL_ML_PERFORMANCE_WIDE`** (`Fact_PerformanceCommerciale` + `DimReservation` + `DimDate` : montants, ids, attributs calendaires) ou repli **`X_raw_numeric.npy`** (notebook 00).
- **Sortie** : étiquettes de **cluster** (0 … *k*−1) + métriques de qualité de partition.

---

## Les deux modèles comparés (et pourquoi)

| Modèle | Intérêt pour nos données |
|--------|-------------------------|
| **K-Means** | Partitions convexes, **centres** interprétables (heatmap des profils) ; rapide ; adapté après **StandardScaler**. |
| **Clustering agglomératif (Ward)** | Hiérarchie **sans supposer** des grappes sphériques ; utile pour **contraster** avec K-Means sur la même grille *k*. |

---

## 📑 Sommaire

1. 🔌 Connexion DW — même environnement que SSMS.
2. 📥 Chargement & standardisation des features numériques.
3. 📐 Choix de *k* (coude, silhouette) puis **Modèle 1 — K-Means** et **Modèle 2 — Agglomératif** (métriques comparées).
4. 📊 Holdout sur K-Means (stabilité) — **PCA 2D** et **heatmap** des centres.
5. 💾 Sauvegarde `kmeans_kpi_segments.joblib` + `metrics_clustering.json`.

---

## ✅ Résultats attendus (validation **E**)

Silhouette, Davies-Bouldin, méthode du coude ; **comparaison explicite** des deux modèles ; visualisations **PCA** et **profilage** (heatmap).

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`"""
    )

    c[1]["source"] = L(
        """### Référence — critères de validation (**E**)

| Thème | Ce notebook |
|-------|-------------|
| ≥ 2 modèles | K-Means vs Agglomératif (Ward) |
| Évaluation | Silhouette ↑, Davies-Bouldin ↓, coude (inertie) |
| Viz | PCA 2D, heatmap des centres |

*Hypothèses (B)* : K-Means → clusters plutôt compacts ; agglomératif → structure hiérarchique, coût plus élevé sur très grands *n*."""
    )

    c[2]["source"] = L(
        """---

## Connexion au Data Warehouse (SQL Server / SSMS)

Les données ML proviennent du **même environnement** que sous **SSMS** : base **`DW_eventzella`** (ou `EVENTZILLA_SQL_DW`), via **SQLAlchemy** + **pyodbc**, authentification Windows.

Exécuter la cellule ci-dessous **en premier** : serveur, base, extrait de chaîne de connexion, test `SELECT DB_NAME()` / `SERVERPROPERTY`.

> Si le message indique une erreur : service SQL arrêté, driver ODBC manquant, ou mauvais nom de serveur — ajuster `EVENTZILLA_SQL_SERVER` / `EVENTZILLA_SQL_DW` ou l’URI `EVENTZILLA_SQL_URI` dans l’environnement (voir `ML/ml_paths.py`).

🎯 **Objectif** : confirmer que Python cible le **même serveur et la même base** que SSMS (reproductibilité des extractions clustering).

✅ **Résultats attendus** : affichage `SQL_SERVER`, `DATABASE_DW`, driver ODBC ; test `SELECT DB_NAME()` / `SERVERPROPERTY` réussi ; message d’erreur explicite si l’engine est absent."""
    )

    c[4]["source"] = L(
        """### Après exécution — lecture (connexion)

- **À vérifier** : le message `Test SQL : OK` et que `base_active` correspond à **`DW_eventzella`** (ou votre base DW).
- **Si engine absent** : installer / configurer **pyodbc** + **SQLAlchemy** (`ML/ml_paths.py`) — sans connexion, le chargement SQL des cellules suivantes échouera.
- **Conclusion** : ce bloc valide que le notebook pointe vers le **même environnement** que SSMS avant tout calcul ML."""
    )

    c[5]["source"] = L(
        """---

### Rendu des figures *(Matplotlib / Seaborn)*

Les **graphiques Matplotlib / Seaborn** s’affichent **sous la cellule** qui appelle `plt.show()` (rendu **inline** dans Jupyter / VS Code).

| Problème | Piste |
|----------|--------|
| Aucune image | Exécuter les cellules **dans l’ordre** depuis le haut ; vérifier que la cellule avec `plt.show()` a bien été exécutée. |
| Toujours vide | Ajouter une fois en tête de notebook : `%matplotlib inline` (déjà présent dans les notebooks avec figures). |

**Liste des graphiques :** voir le tableau **Plan** dans ce notebook (chaque figure apparaît juste après la cellule qui la génère).

---

## Imports et configuration

🎯 **Objectif** : charger les **bibliothèques** (NumPy, scikit-learn, Seaborn), activer le rendu **inline** des figures et préparer les dossiers `processed/` / `models/`.

✅ **Résultats attendus** : ligne `[1] Prêt pour le clustering` ; valeur de `ML_SQL_ONLY` affichée pour savoir si le repli fichier est autorisé."""
    )

    c[7]["source"] = L(
        """### Après exécution — lecture (imports)

- La ligne `[1] Prêt pour le clustering` confirme que les **bibliothèques** (sklearn, seaborn, etc.) sont chargées et que le dossier `ML/models/` existe.
- **`ML_SQL_ONLY`** : si `True`, vous **devez** avoir des données DW ; si `False`, un repli vers `X_raw_numeric.npy` (critère A) est possible plus bas.
- **Conclusion** : l’environnement technique est prêt pour **standardisation** et **clustering**."""
    )

    c[8]["source"] = L(
        """---

## Jeu de données pour le clustering (DW)

Requête **`SQL_ML_PERFORMANCE_WIDE`** sur la connexion SSMS ci-dessus. Noms de colonnes utilisés pour les **étiquettes** de la heatmap.

🎯 **Objectif** : construire la matrice **standardisée** pour le clustering : requête `SQL_ML_PERFORMANCE_WIDE` ou repli `X_raw_numeric.npy` (critère A), imputation médiane, `StandardScaler`, sous-échantillon `X_work`.

✅ **Résultats attendus** : messages `[2]` avec formes `(n, p)` ; noms de colonnes dans `feat_names` pour la heatmap ; erreur claire si DW indisponible en mode SQL-only."""
    )

    c[10]["source"] = L(
        """### Après exécution — lecture (chargement)

- **`Source DW — forme (n, p)`** : *n* = nombre de lignes (réservations / faits), *p* = variables numériques retenues après exclusion des colonnes bruit (`CLUSTERING_NUMERIC_DROP`).
- **`X_work`** : sous-échantillon (plafond 8000) pour garder des temps de calcul **raisonnables** sur le clustering.
- **Standardisation** : les distances (K-Means, *ward*) sont calculées sur une **échelle commune** — indispensable pour ne pas laisser une variable à grande variance dominer.
- **Conclusion** : vous avez une **matrice propre et comparable** pour segmenter."""
    )

    c[11]["source"] = L(
        """---

## Train / validation (stabilité K-Means)

Un **holdout 80 % / 20 %** sur les indices de `X_work` permet de mesurer la **stabilité** de la silhouette K-Means (train vs holdout). Les deux algorithmes (K-Means et agglomératif) sont ensuite comparés sur **tout** `X_work` avec le même *k*."""
    )

    c[12]["source"] = L(
        """---

## Choix de *k*, comparaison K-Means / agglomératif

**Graphiques suivants :** inertie vs *k* (coude), silhouette vs *k*. **Métriques :** silhouette (proche de 1), Davies-Bouldin (plus bas = mieux). **Holdout** sur K-Means : stabilité des segments."""
    )

    c[13]["source"] = L(
        """---

## Grille de *k* : coude et silhouette

Parcours des *k*, inertie (WCSS), score silhouette par *k* ; choix de `k_best`.

🎯 **Objectif** : parcourir les valeurs de *k*, tracer **coude** (inertie WCSS) et **silhouette vs k**, retenir `k_best` (max silhouette sur la plage).

✅ **Résultats attendus** : graphiques coude + silhouette ; valeur `k_best` imprimée en console.

📊 **Visualisations** : **deux graphiques** : inertie vs *k*, silhouette vs *k* (`plt.show()`)."""
    )

    c[15]["source"] = L(
        """---

## K-Means vs agglomératif (Ward) — même *k*

**Modèle 1 — K-Means** : centres + `labels_km`. **Modèle 2 — Agglomératif** : `labels_agg`. Comparaison **Silhouette** (↑) et **Davies-Bouldin** (↓).

🎯 **Objectif** : entraîner **K-Means** et **clustering agglomératif (Ward)** avec le même `k_best` et **comparer** silhouette et indice de Davies-Bouldin.

✅ **Résultats attendus** : scores **Silhouette** et **Davies-Bouldin** pour les deux modèles (même *k*) ; interprétation : plus la silhouette est élevée et plus le Davies-Bouldin est bas, meilleure est la partition."""
    )

    c[17]["source"] = L(
        """---

## Holdout & modèle K-Means final

Stabilité train/holdout ; ré-entraînement K-Means sur `X_work` pour export.

🎯 **Objectif** : mesurer la **stabilité** du K-Means via un holdout **80 % / 20 %** sur les indices (silhouette train vs holdout), puis ré-entraîner K-Means sur tout `X_work` pour les étapes suivantes.

✅ **Résultats attendus** : silhouettes train / holdout ; objets `km`, `labels`, `km_final` prêts pour PCA et export."""
    )

    c[19]["source"] = L(
        """### Après exécution — lecture (choix de *k* et modèles)

- **Méthode du coude** : cherchez un **coude** sur la courbe inertie vs *k* — au-delà, ajouter des clusters **réduit peu** l’inertie (gain marginal).
- **Silhouette vs *k*** : le *k* retenu maximise la silhouette **sur la grille** (heuristique) ; comparez visuellement avec le coude pour éviter un *k* trop grand.
- **K-Means vs agglomératif** : même *k* — **Silhouette** plus haute = points mieux assortis à leur groupe ; **Davies-Bouldin** plus bas = clusters plus **compacts et séparés**.
- **Train / holdout (K-Means)** : silhouettes proches → segmentation **stable** ; écart fort → risque de sur-apprentissage du découpage sur cet échantillon.
- **Conclusion** : vous justifiez **numériquement** le nombre de segments et vous **comparez** deux familles d’algorithmes (critère **E**)."""
    )

    c[20]["source"] = L(
        """---

## Visualisations : PCA et profilage des segments

**PCA (2 axes) :** lecture géométrique des clusters (variance partielle). **Heatmap des centres K-Means** : profil moyen par segment (variables standardisées).

🎯 **Objectif** : projeter les données en **PCA 2D** pour visualiser K-Means vs agglomératif et tracer la **heatmap** des centres K-Means (profilage des segments).

✅ **Résultats attendus** : variance expliquée PC1–PC2 en console ; nuages de points par modèle ; heatmap des z-scores par segment.

📊 **Visualisations** : **PCA** : deux sous-graphiques (K-Means / agglomératif) ; **heatmap** Seaborn des centres de clusters."""
    )

    c[22]["source"] = L(
        """### Après exécution — lecture (PCA & heatmap)

- **PCA 2D** : projection sur **2 axes** = partie seulement de la variance totale (la **variance expliquée** PC1–PC2 est affichée en console : ligne `[4] PCA — variance expliquée…`). Les **nuages colorés** permettent de **voir** la séparation des segments, pas de valider statistiquement une vérité terrain.
- **Deux panneaux** : même projection, **étiquettes** K-Means vs agglomératif — les frontières peuvent différer même avec le même *k*.
- **Heatmap des centres K-Means** : chaque **ligne** = profil moyen d’un segment (z-score). Couleurs **chaudes/froides** = au-dessus / en-dessous de la moyenne globale sur la variable.
- **Conclusion** : vous reliez les segments à des **profils métier** (prix, volumes, etc.) et au **KPI diversité d’offre** (typologies distinctes)."""
    )

    c[23]["source"] = L(
        """---

## Artefacts produits (modèle principal : K-Means)

🎯 **Objectif** : sauvegarder le modèle K-Means final, les métriques (**E**) et les livrables JSON pour traçabilité.

✅ **Résultats attendus** : fichiers `.joblib` sous `ML/models/` et `metrics_clustering.json` dans `ML/processed/` (ou équivalent) ; message `[5]` de confirmation."""
    )

    c[25]["source"] = L(
        """### Après exécution — lecture (sauvegarde)

- **`kmeans_kpi_segments.joblib`** : modèle K-Means final (pour **attribuer** un segment à de nouvelles lignes après **même** `StandardScaler`).
- **`kmeans_standard_scaler.joblib`** : transformations **médiane + z-score** cohérentes avec l’entraînement.
- **`metrics_clustering.json`** : synthèse **silhouette**, **Davies-Bouldin**, effectifs train/holdout — utile pour le **rapport** et la grille **E**.
- **Conclusion** : livrables **reproductibles** ; pour la soutenance, résumez quel modèle a le meilleur **compromis** silhouette / DB et pourquoi le **K-Means** sert de modèle principal (centres interprétables)."""
    )

    c[26]["source"] = L(
        """---

## Synthèse

- **Segmentation** : typologie **non supervisée** sur données **standardisées** (performance DW ou critère A).
- **Choix de *k*** : **coude** + **silhouette** ; comparaison **K-Means** vs **agglomératif** (silhouette ↑, Davies-Bouldin ↓).
- **Lecture** : **PCA 2D** pour visualiser, **heatmap** pour le **profil métier** des segments.
- **Livrables** : modèle **K-Means** + `metrics_clustering.json` pour la **traçabilité** des métriques."""
    )

    save(p, nb)


def patch_03() -> None:
    p = ROOT / "03_D_regression_montants_KPI.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    c = nb["cells"]

    c[0]["source"] = L(
        """# Régression supervisée — montants & KPI (critère **D**)

*Notebook ML — EventZilla (DW, même périmètre que SSMS / Power BI).*

---

## 🎯 Objectif global

Estimer les **montants clés** (panier, budget, benchmark…) à partir des autres variables numériques du fait rentabilité / dimensions, pour le pilotage financier et les projections du dashboard.

---

## 🎯 Objectif technique (cible de régression)

- **Cible *y*** : la **première colonne disponible** dans l’ordre : `final_price` → `service_price` → `benchmark_avg_price` → `event_budget` → `commission_margin` (dérivée `final_price - service_price` si besoin).
- **Features *X*** : autres colonnes **numériques** du même jeu (`dw_financial_wide.parquet` ou SQL), hors `fact_finance_id`.
- **Découpage** : **train / test 75 % / 25 %** ; **validation croisée 5-fold** sur le **train** pour comparer les modèles avant le score **test** final.

---

## Les deux modèles comparés (et pourquoi)

| Modèle | Justification |
|--------|---------------|
| **Ridge (L2)** | Relations **approximativement linéaires** après `StandardScaler` ; **coefficients** comparables pour l’explicabilité ; régularisation pour le multicolinéarité. |
| **Random Forest régresseur** | Capture **non-linéarités** et interactions ; **importances** pour prioriser les leviers métier. |

---

## 📑 Sommaire

1. 🔌 Connexion DW.
2. 📥 Chargement des données.
3. 🧪 Construction *X*, *y*, **split train/test**.
4. 📉 **Modèle 1 — Ridge** : CV 5-fold puis ajustement test.
5. 🌲 **Modèle 2 — Random Forest** : idem.
6. 📊 **Comparaison** : MSE, RMSE, MAE, R² — graphiques réel vs prédit, résidus, coefficients Ridge, importances RF.
7. 💾 Pipelines + `metrics_regression.json`.

---

## ✅ Métriques de validation (critère **D**)

MSE, RMSE, MAE, R² sur le test ; validation croisée sur le train ; résidus et explicabilité (coefficients / importances).

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`"""
    )

    c[1]["source"] = L(
        """### Référence — grille **D**

Métriques : **MSE, RMSE, MAE, R²** sur le test ; **K-Fold** sur le train pour la stabilité ; résidus et explicabilité (coefficients / importances)."""
    )

    c[2]["source"] = L(
        """---

## Connexion au Data Warehouse (SQL Server / SSMS)

L’accès aux données repose sur le **même environnement** que sous **SSMS** : base **`DW_eventzella`** (ou `EVENTZILLA_SQL_DW`), via **SQLAlchemy** et **pyodbc**, authentification Windows.

La cellule suivante affiche serveur, base, extrait de chaîne de connexion, ainsi qu’un test `SELECT DB_NAME()` / `SERVERPROPERTY`.

> En cas d’anomalie (service SQL, driver ODBC, paramètres `EVENTZILLA_SQL_*`), la configuration est documentée dans `ML/ml_paths.py`.

🎯 **Objectif** : valider la **connexion DW** (même logique SSMS) avant la régression.

✅ **Résultats attendus** : diagnostic serveur / base / test SQL."""
    )

    c[4]["source"] = L(
        """### Après exécution — lecture (connexion)

Même logique que les autres notebooks : pointage **serveur / base** pour l’alignement avec **SSMS**."""
    )

    c[5]["source"] = L(
        """---

### Rendu des figures *(Matplotlib / Seaborn)*

Les graphiques apparaissent **sous** la cellule qui appelle `plt.show()` (mode **inline** dans Jupyter / VS Code).

| Comportement observé | Piste de diagnostic |
|---------------------|---------------------|
| Aucun graphique | Cellules non exécutées dans le flux logique du notebook, ou absence d’exécution de la cellule contenant `plt.show()`. |
| Zone d’affichage vide | La directive `%matplotlib inline` figure en tête des notebooks qui produisent des figures. |

Chaque figure s’affiche juste après la cellule qui la génère.

---

## Imports et configuration

🎯 **Objectif** : charger les dépendances **régression** (Ridge, RandomForestRegressor, métriques MSE/MAE/R², pipelines).

✅ **Résultats attendus** : message `[1]` avec état `ML_SQL_ONLY`."""
    )

    c[7]["source"] = L(
        """### Après exécution — lecture (imports)

Préparation des pipelines **Ridge / RF** et des métriques de régression.

🎯 **Objectif** : charger `dw_financial_wide.parquet` ou équivalent SQL / local pour le fait rentabilité.

✅ **Résultats attendus** : dimensions `[2]` et source indiquée."""
    )

    c[9]["source"] = L(
        """### Après exécution — lecture (chargement)

Source **parquet** ou **SQL** ; dimensions affichées en `[2]`.

🎯 **Objectif** : choisir la **cible** `y` selon la priorité KPI (`final_price`, etc.), construire `X` / `y` et **split train/test** 75 % / 25 %.

✅ **Résultats attendus** : nom de la cible, `n` et nombre de features ; bloc prêt pour la CV."""
    )

    c[11]["source"] = L(
        """### Après exécution — lecture (matrice de régression)

Choix de la **cible** dans l’ordre KPI ; construction de `X` / `y` et split **train/test**."""
    )

    c[12]["source"] = L(
        """---

## Entraînement, validation croisée et graphiques d’analyse

**CV (5-fold)** sur le train (stabilité) ; **jeu test** pour **MSE, RMSE, MAE, R²**. **Graphiques :** réel vs prédit, résidus, coefficients Ridge, importances RF."""
    )

    c[13]["source"] = L(
        """### Modèle 1 — Ridge (L2)

Pipelines + fonction `cv_scores` ; **validation croisée 5-fold** sur le **train** (Ridge uniquement dans cette cellule).

🎯 **Objectif** : définir les pipelines **Ridge** et **RF**, la fonction `cv_scores`, et lancer la **validation croisée 5-fold** sur le train pour **Ridge** uniquement.

✅ **Résultats attendus** : moyennes **CV RMSE**, **R²**, **MAE** pour Ridge (ligne Ridge dans la sortie `[4]` après la cellule RF)."""
    )

    c[15]["source"] = L(
        """### Modèle 2 — Random Forest régresseur

**CV 5-fold** sur le **train** pour la forêt aléatoire ; affichage des scores CV Ridge + RF.

🎯 **Objectif** : poursuivre la CV sur le **Random Forest** et afficher les scores CV **Ridge + RF** côte à côte.

✅ **Résultats attendus** : lignes console comparant RMSE / R² en CV pour les deux modèles."""
    )

    c[17]["source"] = L(
        """### Comparaison sur le jeu test — MSE / RMSE / MAE / R² & graphiques

Ajustement sur tout le **train**, métriques sur le **test** ; **champion** = RMSE test minimal ; résidus, coefficients Ridge, importances RF.

🎯 **Objectif** : ajuster les deux pipelines sur **tout le train**, évaluer sur le **test**, désigner le **champion** (RMSE minimal), tracer **réel vs prédit**, **résidus**, **coefficients Ridge**, **importances RF**.

✅ **Résultats attendus** : MSE, RMSE, MAE, R² sur le test pour Ridge et RF ; graphiques d’erreurs et d’explicabilité.

📊 **Visualisations** : **quatre panneaux** scatter réel/prédit + résidus ; barres coefficients Ridge ; barres importances RF."""
    )

    c[19]["source"] = L(
        """### Après exécution — lecture (validation et graphiques)

**CV** sur le train : stabilité des scores ; **test** : **MSE, RMSE, MAE, R²**. Graphiques **réel vs prédit** et **résidus** : qualité d’ajustement et biais éventuel ; **coefficients Ridge** et **importances RF** : lecture métier.

---

## Sauvegarde des artefacts

🎯 **Objectif** : sauvegarder les pipelines et le **JSON** de métriques (critère **D**).

✅ **Résultats attendus** : fichiers `.joblib` + `metrics_regression.json` ; message `[5]`."""
    )

    c[21]["source"] = L(
        """### Après exécution — lecture (fichiers produits)

Modèles sérialisés et **JSON** des métriques pour la synthèse projet."""
    )

    save(p, nb)


def patch_04() -> None:
    p = ROOT / "04_F_series_temporelles.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    c = nb["cells"]

    c[0]["source"] = L(
        """# Séries temporelles — agrégats mensuels & prévision (critère **F**)

*Notebook ML — EventZilla (DW, même périmètre que SSMS / Power BI).*

---

## 🎯 Objectif global

Analyser l’**évolution temporelle** des agrégats EventZilla (volume de faits, chiffre d’affaires, panier moyen) et produire des **prévisions** sur un horizon court pour le pilotage.

---

## 🎯 Objectif technique (série et cible de prévision)

- **Données** : requête **`SQL_ML_TIME_SERIES_RESERVATIONS`** (`Fact_RentabiliteFinanciere` + `DimDate`) — séries **`nb_fact_rows`**, **`revenue_sum`**, **`avg_final_price`** (priorité à la première disponible).
- **Prévision** : on compare deux modèles sur les **3 derniers mois** (holdout) ; le reste sert d’**entraînement**.

---

## Les deux modèles comparés (et pourquoi)

| Modèle | Justification |
|--------|---------------|
| **Lissage exponentiel de Holt** | Tendance **lisse** ; léger et robuste sur séries courtes mensuelles. |
| **ARIMA** | Structure **AR/I/MA** sur série stationnarisée (différenciation) ; référence classique pour comparaison. |

---

## 📑 Sommaire

1. 🔌 Connexion DW.
2. 📥 Série mensuelle + choix de l’indicateur.
3. 🔬 **ADF / KPSS** et **décomposition** saisonnière.
4. 📉 **Modèle 1 — Holt** : entraînement / prévision holdout.
5. 📈 **Modèle 2 — ARIMA** : idem.
6. 📊 **Comparaison** : RMSE, MAE, MAPE + courbe réel vs prévisions.
7. 💾 `metrics_timeseries.json`.

---

## ✅ Analyse requise (critère **F**)

Stationnarité (ADF, KPSS), décomposition ; métriques **MAPE, RMSE, MAE** sur le test temporel.

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`"""
    )

    c[1]["source"] = L(
        """### Référence — grille **F**

Holdout temporel ; pas de fuite : le modèle ne voit pas les mois test à l’entraînement.

---

## Connexion DW (rappel avant la cellule de code)

🎯 **Objectif** : valider la **connexion DW** pour les séries temporelles (alignement SSMS).

✅ **Résultats attendus** : bannière de diagnostic + test SQL."""
    )

    c[3]["source"] = L(
        """---

### Rendu des figures *(Matplotlib / Seaborn)*

Les **graphiques Matplotlib / Seaborn** s’affichent **sous la cellule** qui appelle `plt.show()` (rendu **inline** dans Jupyter / VS Code).

| Problème | Piste |
|----------|--------|
| Aucune image | Exécuter les cellules **dans l’ordre** depuis le haut ; vérifier que la cellule avec `plt.show()` a bien été exécutée. |
| Toujours vide | Ajouter une fois en tête de notebook : `%matplotlib inline` (déjà présent dans les notebooks avec figures). |

**Liste des graphiques :** voir le tableau **Plan** dans ce notebook (chaque figure apparaît juste après la cellule qui la génère)."""
    )

    c[4]["source"] = L(
        """---

## Imports et bibliothèques — critère **F**

🎯 **Objectif** : charger **statsmodels** (ARIMA, Holt, ADF, KPSS, décomposition), **pandas**, **matplotlib** et les chemins `ML/` pour la suite (séries, prévisions).

✅ **Résultats attendus** : message `[1]` avec état `ML_SQL_ONLY` ; dossiers `processed/` / `models/` prêts."""
    )

    c[7]["source"] = L(
        """---

## Stationnarité (ADF, KPSS) et décomposition

**ADF / KPSS** : lecture conjointe des *p*-values. **Décomposition** : observé, tendance, saison, résidus (période adaptée à la longueur de série).

🎯 **Objectif** : analyser la **stationnarité** (ADF, KPSS) et la **décomposition** (tendance / saisonnalité) de la série retenue.

✅ **Résultats attendus** : sorties de tests et graphiques de décomposition interprétables pour le critère **F**.

📊 **Visualisations** : courbes de la série, décomposition additive/multiplicative selon le code."""
    )

    c[8]["source"] = L(
        """---

## Sélection de la série et lien KPI dashboard

🎯 **Objectif** : associer chaque **colonne disponible** dans `df_ts` à un **identifiant KPI** (`SERIES_KPIS`) et construire la série `ts` indexée par mois pour l’analyse.

✅ **Résultats attendus** : choix de `col_main`, série `ts` indexée par mois ; préparation aux tests ADF/KPSS et aux prévisions."""
    )

    c[10]["source"] = L(
        """---

## Holt vs ARIMA (holdout et métriques)

Holdout : **3 derniers mois** ; train = reste. Courbe **réel vs prévisions**. **RMSE, MAE, MAPE** (MAPE fragile si niveaux proches de 0)."""
    )

    c[11]["source"] = L(
        """### Modèle 1 — Holt (lissage exponentiel, tendance)

Holdout = **3 derniers mois**.

🎯 **Objectif** : définir les métriques **MAPE / RMSE / MAE**, découper **train** / **test** (holdout derniers mois), ajuster **Holt** et produire les prévisions.

✅ **Résultats attendus** : métriques sur le holdout pour Holt ; objets de prévision pour le graphique comparatif.

📊 **Visualisations** : série **train / test** et courbe de prévision Holt (voir cellule suivante pour ARIMA)."""
    )

    c[13]["source"] = L(
        """### Modèle 2 — ARIMA

Ordre (1,1,1) ou repli (0,1,1).

🎯 **Objectif** : ajuster un **ARIMA** ((1,1,1) ou repli (0,1,1)), prévoir le holdout et **comparer** à Holt (métriques + figure).

✅ **Résultats attendus** : MAPE, RMSE, MAE pour ARIMA ; comparaison imprimée ; choix d’un modèle **primaire** selon RMSE.

📊 **Visualisations** : graphique **réel vs Holt vs ARIMA** sur la fenêtre test."""
    )

    save(p, nb)


def main() -> None:
    patch_00()
    print("OK 00")
    patch_01()
    print("OK 01")
    patch_03()
    print("OK 03")
    patch_04()
    print("OK 04")


if __name__ == "__main__":
    main()
