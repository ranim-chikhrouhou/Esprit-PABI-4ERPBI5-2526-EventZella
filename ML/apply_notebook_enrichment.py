# -*- coding: utf-8 -*-
"""
Enrichissement des notebooks ML : ton professionnel, guides critères, interprétations post-bloc.
Exécution : python ML/apply_notebook_enrichment.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from notebook_enrich_shared import soften_imperatives_markdown, replace_connexion_figures_in_markdown

ROOT = Path(__file__).resolve().parent / "notebooks"


def md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def split_lines(s: str) -> list[str]:
    """Format Jupyter : une chaîne par ligne avec \\n."""
    if not s.endswith("\n"):
        s += "\n"
    return [line + "\n" for line in s.split("\n")[:-1]] + ([s.split("\n")[-1]] if s.split("\n")[-1] else [])


def md_lines(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1] + "\n")
    return {"cell_type": "markdown", "metadata": {}, "source": out}


def process_markdown_cells(nb: dict) -> None:
    for c in nb["cells"]:
        if c.get("cell_type") != "markdown":
            continue
        src = "".join(c.get("source", []))
        src = replace_connexion_figures_in_markdown(src)
        src = soften_imperatives_markdown(src)
        c["source"] = split_lines(src)


def insert_after_fingerprint(
    cells: list, fingerprint: str, new_cell: dict, marker: str = "### 💬"
) -> bool:
    for i in range(len(cells) - 1, -1, -1):
        if cells[i].get("cell_type") != "code":
            continue
        body = "".join(cells[i].get("source", []))
        if fingerprint not in body:
            continue
        if i + 1 < len(cells):
            nxt = "".join(cells[i + 1].get("source", []))
            # Déjà inséré (titres « Lecture » ou « Après » selon versions)
            if marker in nxt:
                return False
        cells.insert(i + 1, new_cell)
        return True
    return False


GUIDE_C = """## 📖 Référence — critères **C** et **B** (classification)

### Alignement sur la grille officielle (**C**)

| Thème | Traitement dans ce notebook |
|-------|----------------------------|
| **≥ 2 modèles + comparaison** | **Random Forest** et **régression logistique multinomiale**, chacun avec **GridSearchCV** (recherche sur grille, équivalent utile à *RandomizedSearch* pour l’exploration d’hyperparamètres). |
| **Pipeline + réglage** | `Pipeline(StandardScaler → estimateur)` ; validation croisée **stratifiée** (5 plis), score **F1 pondéré**. |
| **Train / test & CV** | Holdout **25 %** stratifié ; **StratifiedKFold** sur le jeu d’entraînement pour la sélection d’hyperparamètres. |
| **Déséquilibre des classes** | `class_weight="balanced"` sur les deux estimateurs. |
| **Métriques** | Accuracy, Precision / Recall / F1 (pondérés), **ROC-AUC** (binaire ou OvR pondéré en multiclasse). |
| **Interprétation** | Matrice de confusion, courbe **ROC** (cas binaire), **importances** (Random Forest). |

### Compréhension des modèles (**B**)

- **Random Forest** — *Intuition* : agrégation d’arbres sur sous-échantillons ; *Paramètres* : profondeur, nombre d’arbres (grille) ; *Hypothèses* : peu de contraintes de forme ; *Limites* : interprétabilité partielle, sensibilité au déséquilibre sans pondération.
- **Régression logistique** — *Intuition* : frontières linéaires dans l’espace des features (après mise à l’échelle) ; *Paramètres* : `C` (régularisation implicite) ; *Hypothèses* : linéarité des log-odds ; *Limites* : interactions non modélisées sans extension.

*Choix liés au problème EventZilla* : prédire un **statut de réservation** à partir de variables numériques DW ; la comparaison **RF / LR** couvre une hypothèse **non linéaire** vs une hypothèse **linéaire**, ce qui répond à l’exigence de **justification** des modèles.
"""

INTERP_C = {
    "conn": "### 💬 Lecture — connexion DW\n\nLa sortie résume **serveur**, **base** et le test SQL : cohérence avec l’environnement **SSMS** pour la reproductibilité des extractions.\n",
    "imports": "### 💬 Lecture — imports et configuration\n\nLes bibliothèques **scikit-learn** et le répertoire `ML/models/` sont prêts ; le mode `ML_SQL_ONLY` conditionne la suite (DW strict ou repli local).\n",
    "data": "### 💬 Lecture — jeu de données\n\nLes dimensions `(lignes, colonnes)` et la présence du **statut** conditionnent la faisabilité de la classification ; le repli SQL ou parquet est documenté dans les messages `[2]`.\n",
    "prep": "### 💬 Lecture — cible et découpage\n\nLe **split stratifié** préserve la proportion des classes sur le train/test ; le nombre de classes et de features numériques fixe la complexité du problème.\n",
    "models": "### 💬 Lecture — entraînement, métriques et figures\n\nLes blocs **GridSearch** indiquent les **meilleurs hyperparamètres** par validation croisée. Les **métriques test** comparent RF et LR ; le **champion** retient le meilleur **F1 pondéré**. La **matrice de confusion** localise les confusions entre statuts ; la **ROC** (binaire) ou le texte (multiclasse) complète l’**AUC**. Les **barres d’importance** (RF) hiérarchisent les variables pour l’analyse métier.\n",
    "save": "### 💬 Lecture — artefacts produits\n\nLes fichiers **`.joblib`** et **`metrics_classification.json`** consolident le pipeline retenu et les métriques des deux modèles pour le **rapport** et la synthèse projet.\n",
}

GUIDE_D = """## 📖 Référence — critères **D** et **B** (régression)

### Grille officielle (**D**)

| Thème | Traitement dans ce notebook |
|-------|----------------------------|
| **≥ 2 modèles** | **Ridge** (pénalisation L2, coefficients interprétables après scaling) et **Random Forest régresseur** (non linéaire, importances). |
| **Validation** | **K-Fold (k=5)** sur le train pour scores moyens ; **jeu test** pour **MSE, RMSE, MAE, R²**. |
| **Interprétation** | **Résidus**, **réel vs prédit**, **coefficients Ridge**, **importances RF**. |

### Compréhension (**B**)

- **Ridge** — relation **approximativement linéaire** après `StandardScaler` ; coefficients comparables ; sensibilité aux corrélations fortes entre variables.
- **Random Forest** — **interactions** et non-linéarités ; moindre extrapolation hors du domaine observé des données.

*Justification* : prédire des **montants / marges** KPI ; Ridge offre une **lecture coefficient**, RF une **carte d’importance** pour le pilotage.
"""

INTERP_D = {
    "conn": "### 💬 Lecture — connexion DW\n\nMême logique que les autres notebooks : pointage **serveur / base** pour l’alignement avec **SSMS**.\n",
    "imports": "### 💬 Lecture — imports\n\nPréparation des pipelines **Ridge / RF** et des métriques de régression.\n",
    "data": "### 💬 Lecture — chargement\n\nSource **parquet** ou **SQL** ; dimensions affichées en `[2]`.\n",
    "prep": "### 💬 Lecture — matrice de régression\n\nChoix de la **cible** dans l’ordre KPI ; construction de `X` / `y` et split **train/test**.\n",
    "models": "### 💬 Lecture — validation et graphiques\n\n**CV** sur le train : stabilité des scores ; **test** : **MSE, RMSE, MAE, R²**. Graphiques **réel vs prédit** et **résidus** : qualité d’ajustement et biais éventuel ; **coefficients Ridge** et **importances RF** : lecture métier.\n",
    "save": "### 💬 Lecture — sauvegarde\n\nModèles sérialisés et **JSON** des métriques pour la synthèse projet.\n",
}

GUIDE_F = """## 📖 Référence — critères **F** et **B** (séries temporelles)

### Grille officielle (**F**)

| Thème | Traitement |
|-------|------------|
| **Analyse** | Tests **ADF** et **KPSS** (stationnarité) ; **décomposition** additive. |
| **≥ 2 modèles** | **Holt** (lissage exponentiel avec tendance) et **ARIMA** (structure AR/I/MA). |
| **Évaluation** | **RMSE**, **MAE**, **MAPE** sur fenêtre **holdout** (derniers mois). |

### Compréhension (**B**)

- **Holt** — tendance **locale lisse** ; peu de paramètres.
- **ARIMA** — structure sur série **différenciée** ; limite : saisonnalité forte non prise en compte sans **SARIMA** (piste d’extension).

*Justification* : séries **mensuelles agrégées** (revenus, volumes) — modèles classiques de référence pour la prévision court / moyen terme.
"""

INTERP_F = {
    "conn": "### 💬 Lecture — connexion\n\nDiagnostic d’accès DW identique aux autres notebooks.\n",
    "imports": "### 💬 Lecture — imports séries\n\n**statsmodels** pour ADF/KPSS, décomposition, Holt, ARIMA.\n",
    "data": "### 💬 Lecture — série mensuelle\n\nAgrégat SQL ou repli local ; aperçu `head()` pour contrôler les colonnes temporelles.\n",
    "station": "### 💬 Lecture — stationnarité et décomposition\n\n**ADF / KPSS** : cohérence recherchée entre tests (stationnarité vs tendance). **Décomposition** : partage tendance / saison / résidu pour la lecture métier.\n",
    "models": "### 💬 Lecture — Holt vs ARIMA\n\nComparaison sur **holdout** : **RMSE, MAE, MAPE** ; graphique **réel vs prévu** pour apprécier les erreurs en niveau.\n",
    "save": "### 💬 Lecture — métriques exportées\n\nFichier **JSON** récapitulatif pour la synthèse et le critère **F**.\n",
}

GUIDE_05 = """## 📖 Rôle de ce notebook — synthèse projet

Ce fichier **agrège** les métriques produits par les notebooks **A, E, C, D, F** (fichiers `metrics_*.json` sous `ML/models_artifacts/`), sans nouvel entraînement.

| Critère | Contenu typiquement agrégé |
|---------|----------------------------|
| **A** | Préparation (fichiers matrice / parquet) — indirect via les pipelines aval |
| **E** | Clustering — silhouette, Davies-Bouldin |
| **C** | Classification — accuracy, F1, ROC-AUC |
| **D** | Régression — RMSE, R², etc. |
| **F** | Séries — RMSE, MAE, MAPE |

La synthèse Markdown exportée sert au **rapport** et à la **traçabilité** des résultats pour l’équipe projet et l’évaluation.
"""


def enrich_file(path: Path, guide: str | None, interp: dict, fingerprints: list[tuple[str, str]]) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    process_markdown_cells(nb)

    if guide and not any("📖 Référence — critères" in "".join(c.get("source", [])) or "📖 Rôle de ce notebook" in "".join(c.get("source", [])) for c in cells[:4]):
        cells.insert(1, md_lines(guide))

    for key, fp in fingerprints:
        if key not in interp:
            continue
        insert_after_fingerprint(cells, fp, md_lines(interp[key]), "### 💬")

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK", path.name, "cells", len(cells))


def main() -> None:
    # --- C ---
    enrich_file(
        ROOT / "02_C_classification_statut_reservation.ipynb",
        GUIDE_C,
        INTERP_C,
        [
            ("conn", "Connexion DW — diagnostic"),
            ("imports", "[1] Classification critère C"),
            ("data", 'print("[2] Dimensions :'),
            ("prep", 'print("[3] Train / test :'),
            ("models", "cv = StratifiedKFold"),
            ("save", 'print("[5] Pipelines'),
        ],
    )

    # --- D ---
    enrich_file(
        ROOT / "03_D_regression_montants_KPI.ipynb",
        GUIDE_D,
        INTERP_D,
        [
            ("conn", "Connexion DW — diagnostic"),
            ("imports", "[1] Régression critère D"),
            ("data", 'print("[2] Dimensions :'),
            ("prep", "TARGET_ORDER"),
            ("models", "cv = KFold"),
            ("save", "metrics_regression.json"),
        ],
    )

    enrich_file(
        ROOT / "04_F_series_temporelles.ipynb",
        GUIDE_F,
        INTERP_F,
        [
            ("conn", "Connexion DW — diagnostic"),
            ("imports", "[1] Séries temporelles critère F"),
            ("data", "[2] Source : agrégat SQL"),
            ("station", "SERIES_KPIS = ["),
            ("models", "fit_holt = ExponentialSmoothing"),
            ("save", "metrics_timeseries.json"),
        ],
    )

    # 05 synthèse
    p = ROOT / "05_synthese_metriques_validation.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    process_markdown_cells(nb)
    cells = nb["cells"]
    if not any("📖 Rôle de ce notebook — synthèse" in "".join(c.get("source", [])) for c in cells[:4]):
        cells.insert(1, md_lines(GUIDE_05))
    # Interprétation après agrégation code (last code cell with ML_METRICS_SUMMARY)
    insert_after_fingerprint(
        cells,
        "ML_METRICS_SUMMARY.md",
        md_lines(
            "### 💬 Lecture — tableau agrégé\n\nLe récapitulatif consolidé permet de **contrôler** la cohérence des métriques entre critères et de préparer les **formulations** pour le rapport (forces / limites par tâche ML).\n"
        ),
        "### 💬",
    )
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK", p.name, "cells", len(cells))

    # A & E : ton uniquement (pas de double guide)
    for name in ("00_A_preparation_donnees_feature_engineering.ipynb", "01_E_clustering_segmentation.ipynb"):
        p2 = ROOT / name
        nb2 = json.loads(p2.read_text(encoding="utf-8"))
        process_markdown_cells(nb2)
        p2.write_text(json.dumps(nb2, ensure_ascii=False, indent=1), encoding="utf-8")
        print("tone", name)


if __name__ == "__main__":
    main()
