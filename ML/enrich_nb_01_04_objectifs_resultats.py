# -*- coding: utf-8 -*-
"""
Ajoute (comme le notebook 00) avant chaque bloc de code des notebooks 01–04 :
  🎯 **Objectif**
  ✅ **Résultats attendus**
  📊 **Visualisations** (si figures)
Sans dupliquer si le markdown précédent contient déjà « 🎯 **Objectif** ».
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"

MARK_START = "🎯 **Objectif**"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [x if x.endswith("\n") else x + "\n" for x in s.splitlines(True)]


def block(obj: str, res: str, viz: str | None = None) -> str:
    parts = [f"\n{MARK_START} : {obj}\n", f"\n✅ **Résultats attendus** : {res}\n"]
    if viz:
        parts.append(f"\n📊 **Visualisations** : {viz}\n")
    return "".join(parts)


def ensure_markdown_before_code(nb: dict, code_idx: int, text: str) -> bool:
    """Ajoute le bloc Objectif/Résultats au markdown précédent, ou insère une cellule markdown si la précédente est du code."""
    if code_idx == 0:
        return False
    prev = nb["cells"][code_idx - 1]
    cur = "".join(prev.get("source", [])) if prev["cell_type"] == "markdown" else ""
    if prev["cell_type"] == "markdown":
        if MARK_START in cur:
            return False
        prev["source"] = lines(cur.rstrip() + text)
        return True
    # Code après code : insérer une cellule markdown dédiée avant ce bloc
    nb["cells"].insert(
        code_idx,
        {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip() + "\n")},
    )
    return True


def classify_01(src: str) -> str | None:
    # Tirets / espaces insécables : éviter la dépendance à un seul caractère « — »
    if "Connexion DW" in src and "EVENTZILLA" in src and "read_dw_sql" in src:
        return block(
            "confirmer que Python cible le **même serveur et la même base** que SSMS (reproductibilité des extractions clustering).",
            "affichage `SQL_SERVER`, `DATABASE_DW`, driver ODBC ; test `SELECT DB_NAME()` / `SERVERPROPERTY` réussi ; message d’erreur explicite si l’engine est absent.",
            None,
        )
    if "Prêt pour le clustering" in src or (
        "%matplotlib inline" in src and "AgglomerativeClustering" in src
    ):
        return block(
            "charger les **bibliothèques** (NumPy, scikit-learn, Seaborn), activer le rendu **inline** des figures et préparer les dossiers `processed/` / `models/`.",
            "ligne `[1] Prêt pour le clustering` ; valeur de `ML_SQL_ONLY` affichée pour savoir si le repli fichier est autorisé.",
            None,
        )
    if "X_for_cluster = None" in src and "SQL_ML_PERFORMANCE_WIDE" in src:
        return block(
            "construire la matrice **standardisée** pour le clustering : requête `SQL_ML_PERFORMANCE_WIDE` ou repli `X_raw_numeric.npy` (critère A), imputation médiane, `StandardScaler`, sous-échantillon `X_work`.",
            "messages `[2]` avec formes `(n, p)` ; noms de colonnes dans `feat_names` pour la heatmap ; erreur claire si DW indisponible en mode SQL-only.",
            None,
        )
    if src.strip().startswith("K_hi = min"):
        return block(
            "parcourir les valeurs de *k*, tracer **coude** (inertie WCSS) et **silhouette vs k**, retenir `k_best` (max silhouette sur la plage).",
            "graphiques coude + silhouette ; valeur `k_best` imprimée en console.",
            "**Deux graphiques** : inertie vs *k*, silhouette vs *k* (`plt.show()`).",
        )
    if "km_model = KMeans" in src and "AgglomerativeClustering" in src:
        return block(
            "entraîner **K-Means** et **clustering agglomératif (Ward)** avec le même `k_best` et **comparer** silhouette et indice de Davies-Bouldin.",
            "scores **Silhouette** et **Davies-Bouldin** pour les deux modèles (même *k*) ; interprétation : plus la silhouette est élevée et plus le Davies-Bouldin est bas, meilleure est la partition.",
            None,
        )
    if "idx_all = np.arange" in src and "train_test_split" in src:
        return block(
            "mesurer la **stabilité** du K-Means via un holdout **80 % / 20 %** sur les indices (silhouette train vs holdout), puis ré-entraîner K-Means sur tout `X_work` pour les étapes suivantes.",
            "silhouettes train / holdout ; objets `km`, `labels`, `km_final` prêts pour PCA et export.",
            None,
        )
    if "pca = PCA" in src and "sns.heatmap" in src:
        return block(
            "projeter les données en **PCA 2D** pour visualiser K-Means vs agglomératif et tracer la **heatmap** des centres K-Means (profilage des segments).",
            "variance expliquée PC1–PC2 en console ; nuages de points par modèle ; heatmap des z-scores par segment.",
            "**PCA** : deux sous-graphiques (K-Means / agglomératif) ; **heatmap** Seaborn des centres de clusters.",
        )
    if "joblib.dump(km_final" in src or "metrics_clustering.json" in src:
        return block(
            "sauvegarder le modèle K-Means final, les métriques (**E**) et les livrables JSON pour traçabilité.",
            "fichiers `.joblib` sous `ML/models/` et `metrics_clustering.json` dans `ML/processed/` (ou équivalent) ; message `[5]` de confirmation.",
            None,
        )
    return None


def classify_02(src: str) -> str | None:
    if "Connexion DW" in src and "EVENTZILLA" in src and "read_dw_sql" in src:
        return block(
            "valider la **connexion DW** alignée avec SSMS avant chargement des données de classification.",
            "bloc ASCII de diagnostic + test SQL `DB_NAME()` / serveur ; erreur si engine absent.",
            None,
        )
    if "Classification critère C" in src and "%matplotlib inline" in src:
        return block(
            "importer scikit-learn, pipelines, métriques **C** (accuracy, F1, ROC…) et préparer `ML/models/`.",
            "message `[1]` confirmant le mode `ML_SQL_ONLY` et les imports.",
            None,
        )
    if "pp = ML_PROCESSED" in src and "dw_financial_wide.parquet" in src:
        return block(
            "charger le jeu **large** (`dw_financial_wide.parquet` ou SQL) et garantir la présence d’une colonne **statut réservation** (y compris via ponts DW).",
            "dimensions `[2]` ; messages indiquant la source (parquet, SQL, enrichissement statut).",
            None,
        )
    if "resolve_classification_status_column" in src and "train_test_split" in src:
        return block(
            "définir la **cible** `y` (statut encodé), les **features** numériques (max 20) et le **split stratifié** 75 % / 25 %.",
            "liste des features ; nombre de classes ; tailles train / test affichées en `[3]`.",
            None,
        )
    if "gs_rf = GridSearchCV" in src:
        return block(
            "optimiser **Random Forest** par **GridSearchCV** (5-fold **stratifié**, score F1 pondéré) sur le **train** uniquement.",
            "meilleurs hyperparamètres `gs_rf.best_params_` ; modèle `best_rf` prêt pour le test.",
            None,
        )
    if "pipe_lr = Pipeline" in src and "gs_lr = GridSearchCV" in src:
        return block(
            "optimiser la **régression logistique** (pipeline `StandardScaler` → multinomial) sur la même grille de validation.",
            "meilleur estimateur `best_lr` ; comparabilité avec la RF via le même protocole CV.",
            None,
        )
    if "class_labels = " in src and "classification_report" in src:
        return block(
            "**Comparer** RF et LR sur le **jeu test** : métriques globales, **matrice de confusion**, **ROC** (si binaire), **importances** (RF) ; désigner le **champion**.",
            "Accuracy, Precision / Recall / F1 pondérés, ROC-AUC ; figures ; nom du champion.",
            "**Matrice de confusion**, **courbe ROC** (si applicable), barres d’importance des variables (RF).",
        )
    if "joblib.dump(champion" in src and "metrics_classification.json" in src:
        return block(
            "persister le **pipeline champion** et les métriques détaillées pour le critère **C**.",
            "fichiers `.joblib` et `metrics_classification.json` ; message de confirmation.",
            None,
        )
    return None


def classify_03(src: str) -> str | None:
    if "Connexion DW" in src and "EVENTZILLA" in src and "read_dw_sql" in src:
        return block(
            "valider la **connexion DW** (même logique SSMS) avant la régression.",
            "diagnostic serveur / base / test SQL.",
            None,
        )
    if "Régression critère D" in src and "%matplotlib inline" in src:
        return block(
            "charger les dépendances **régression** (Ridge, RandomForestRegressor, métriques MSE/MAE/R², pipelines).",
            "message `[1]` avec état `ML_SQL_ONLY`.",
            None,
        )
    if "pp = ML_PROCESSED" in src and "read_parquet" in src:
        return block(
            "charger `dw_financial_wide.parquet` ou équivalent SQL / local pour le fait rentabilité.",
            "dimensions `[2]` et source indiquée.",
            None,
        )
    if "MIN_ROWS" in src and "TARGET_ORDER" in src and "train_test_split" in src:
        return block(
            "choisir la **cible** `y` selon la priorité KPI (`final_price`, etc.), construire `X` / `y` et **split train/test** 75 % / 25 %.",
            "nom de la cible, `n` et nombre de features ; bloc prêt pour la CV.",
            None,
        )
    if "pipe_ridge = Pipeline" in src and "def cv_scores" in src and "sf = cv_scores" not in src:
        return block(
            "définir les pipelines **Ridge** et **RF**, la fonction `cv_scores`, et lancer la **validation croisée 5-fold** sur le train pour **Ridge** uniquement.",
            "moyennes **CV RMSE**, **R²**, **MAE** pour Ridge (ligne Ridge dans la sortie `[4]` après la cellule RF).",
            None,
        )
    if "sf = cv_scores" in src and "pipe_ridge.fit" not in src:
        return block(
            "poursuivre la CV sur le **Random Forest** et afficher les scores CV **Ridge + RF** côte à côte.",
            "lignes console comparant RMSE / R² en CV pour les deux modèles.",
            None,
        )
    if "pipe_ridge.fit(Xtr" in src:
        return block(
            "ajuster les deux pipelines sur **tout le train**, évaluer sur le **test**, désigner le **champion** (RMSE minimal), tracer **réel vs prédit**, **résidus**, **coefficients Ridge**, **importances RF**.",
            "MSE, RMSE, MAE, R² sur le test pour Ridge et RF ; graphiques d’erreurs et d’explicabilité.",
            "**Quatre panneaux** scatter réel/prédit + résidus ; barres coefficients Ridge ; barres importances RF.",
        )
    if "joblib.dump(pipe_ridge" in src or "metrics_regression.json" in src:
        return block(
            "sauvegarder les pipelines et le **JSON** de métriques (critère **D**).",
            "fichiers `.joblib` + `metrics_regression.json` ; message `[5]`.",
            None,
        )
    return None


def classify_04(src: str) -> str | None:
    if "Connexion DW" in src and "EVENTZILLA" in src and "read_dw_sql" in src:
        return block(
            "valider la **connexion DW** pour les séries temporelles (alignement SSMS).",
            "bannière de diagnostic + test SQL.",
            None,
        )
    if "Séries temporelles critère F" in src and "%matplotlib inline" in src:
        return block(
            "charger **statsmodels** / **matplotlib** et préparer le contexte **F** (séries, prévisions).",
            "message `[1]` avec état `ML_SQL_ONLY`.",
            None,
        )
    if "SERIES_KPIS = " in src and "[" in src:
        return block(
            "associer chaque **colonne de série** (SQL) à un **identifiant KPI** du dashboard (traçabilité métrique ↔ donnée).",
            "dictionnaire `SERIES_KPIS` prêt pour l’export JSON final et la lecture métier.",
            None,
        )
    if "df_ts = None" in src and "SQL_ML_TIME_SERIES_RESERVATIONS" in src:
        return block(
            "charger l’**agrégat mensuel** depuis le DW (`SQL_ML_TIME_SERIES_RESERVATIONS`) ou indiquer l’échec pour basculement éventuel.",
            "messages `[2]` ; `df_ts` rempli ou erreur documentée.",
            None,
        )
    if "adfuller" in src and "seasonal_decompose" in src:
        return block(
            "analyser la **stationnarité** (ADF, KPSS) et la **décomposition** (tendance / saisonnalité) de la série retenue.",
            "sorties de tests et graphiques de décomposition interprétables pour le critère **F**.",
            "courbes de la série, décomposition additive/multiplicative selon le code.",
        )
    if "def metrics_ts" in src and "ExponentialSmoothing" in src:
        return block(
            "définir les métriques **MAPE / RMSE / MAE**, découper **train** / **test** (holdout derniers mois), ajuster **Holt** et produire les prévisions.",
            "métriques sur le holdout pour Holt ; objets de prévision pour le graphique comparatif.",
            "série **train / test** et courbe de prévision Holt (voir cellule suivante pour ARIMA).",
        )
    if "try:" in src and "arima_fit = ARIMA" in src and "fc_arima" in src:
        return block(
            "ajuster un **ARIMA** ((1,1,1) ou repli (0,1,1)), prévoir le holdout et **comparer** à Holt (métriques + figure).",
            "MAPE, RMSE, MAE pour ARIMA ; comparaison imprimée ; choix d’un modèle **primaire** selon RMSE.",
            "graphique **réel vs Holt vs ARIMA** sur la fenêtre test.",
        )
    if "metrics_timeseries.json" in src and "kpi_main" in src:
        return block(
            "relier la série au **libellé KPI** (`kpi_main`), écrire **`metrics_timeseries.json`** et clôturer la comparaison **F**.",
            "fichier JSON avec critère F, série, métriques Holt/ARIMA ; confirmation en console.",
            None,
        )
    return None


def classify(name: str, src: str) -> str | None:
    if name.startswith("01"):
        return classify_01(src)
    if name.startswith("02"):
        return classify_02(src)
    if name.startswith("03"):
        return classify_03(src)
    if name.startswith("04"):
        return classify_04(src)
    return None


def patch_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n_added = 0
    code_indices = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and "".join(c.get("source", [])).strip()
    ]
    for i in reversed(code_indices):
        src = "".join(nb["cells"][i].get("source", []))
        extra = classify(path.name, src)
        if not extra:
            continue
        if ensure_markdown_before_code(nb, i, extra):
            n_added += 1
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return n_added


def main() -> None:
    files = [
        ROOT / "01_E_clustering_segmentation.ipynb",
        ROOT / "02_C_classification_statut_reservation.ipynb",
        ROOT / "03_D_regression_montants_KPI.ipynb",
        ROOT / "04_F_series_temporelles.ipynb",
    ]
    for p in files:
        n = patch_notebook(p)
        print(p.name, "- markdown enrichis:", n)


if __name__ == "__main__":
    main()
