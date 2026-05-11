# -*- coding: utf-8 -*-
"""Réorganise le markdown du notebook 02 (titres, espacements, séparateurs)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"
NB = ROOT / "02_C_classification_statut_reservation.ipynb"


def lines(*parts: str) -> list[str]:
    out: list[str] = []
    for p in parts:
        if not p.endswith("\n"):
            p += "\n"
        for ln in p.splitlines():
            out.append(ln + "\n" if not ln.endswith("\n") else ln)
    return out


def md_cell(src: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def replace_first_match(cells: list, needle: str, new_source: list[str]) -> bool:
    for c in cells:
        if c.get("cell_type") != "markdown":
            continue
        s = "".join(c.get("source", []))
        if needle in s:
            c["source"] = new_source
            return True
    return False


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- Cellule 0 : introduction (un seul titre #) ---
    cells[0]["source"] = lines(
        """# Classification supervisée — statut de réservation (critère **C**)

*Notebook ML — EventZilla (DW, même périmètre que SSMS / Power BI).*

---

## 🎯 Objectif global

Construire un modèle qui **prédit le statut d’une réservation** à partir des informations financières et dimensionnelles du DW EventZilla, pour relier les KPI *taux d’acceptation / annulation* et l’entonnoir de conversion.

---

## 🎯 Objectif technique (cible supervisée)

- **Variable cible *y*** : **`reservation_status`** (ou colonne résolue équivalente : `status` / `statut` sur `DimReservation`), encodée en classes entières.
- **Features *X*** : jusqu’à **20 colonnes numériques** du jeu large (`final_price`, `event_budget`, identifiants dimensionnels, etc.) issues de `dw_financial_wide.parquet` ou requête SQL (`build_sql_ml_financial_wide`).
- **Découpage** : **train / test 75 % / 25 % stratifié** sur *y* ; **GridSearchCV** en **5-fold stratifié** **sur le train uniquement**.

---

## Les deux modèles comparés (et pourquoi)

| Modèle | Justification métier / données |
|--------|--------------------------------|
| **Random Forest** | Capture des **non-linéarités** et interactions ; `class_weight='balanced'` pour le **déséquilibre** des statuts ; importances de variables pour l’explicabilité. |
| **Régression logistique (multinomiale)** | **Baseline** forte : frontières **linéaires** en log-odds après scaling ; coefficients interprétables ; même pondération de classes. |

---

## 📑 Sommaire

1. 🔌 Connexion DW.
2. 📥 Chargement du DataFrame + présence du statut réservation.
3. 🧪 Préparation *X*, *y*, **split stratifié**.
4. 🌲 **Modèle 1 — Random Forest** (pipeline + GridSearch).
5. 📐 **Modèle 2 — Régression logistique** (pipeline + GridSearch).
6. 📊 **Comparaison** : Accuracy, Precision / Recall / F1 (pondérés), ROC-AUC — matrice de confusion, ROC (binaire), importances (RF).
7. 💾 Pipeline champion + `metrics_classification.json`.

---

## ✅ Métriques de validation (critère **C**)

Accuracy, Precision, Recall, F1-score (pondérés), **ROC-AUC** ; matrice de confusion ; courbe ROC si **binaire** ; importances (RF).

**Références** : `ML/EventZilla_Dashboards_Improved.pdf`, `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`
"""
    )

    cells[1]["source"] = lines(
        """### Référence — grille **C**

Pipeline `StandardScaler → estimateur`, `GridSearchCV`, `StratifiedKFold`, `class_weight='balanced'`, métriques sur le **jeu test** après sélection d’hyperparamètres sur le train.
"""
    )

    cells[2]["source"] = lines(
        """---

## Connexion au Data Warehouse (SQL Server / SSMS)

L’accès aux données repose sur le **même environnement** que sous **SSMS** : base **`DW_eventzella`** (ou `EVENTZILLA_SQL_DW`), via **SQLAlchemy** et **pyodbc**, authentification Windows.

La cellule suivante affiche serveur, base, extrait de chaîne de connexion, ainsi qu’un test `SELECT DB_NAME()` / `SERVERPROPERTY`.

> En cas d’anomalie (service SQL, driver ODBC, paramètres `EVENTZILLA_SQL_*`), la configuration est documentée dans `ML/ml_paths.py`.

🎯 **Objectif** : valider la **connexion DW** alignée avec SSMS avant chargement des données de classification.

✅ **Résultats attendus** : bloc ASCII de diagnostic + test SQL `DB_NAME()` / serveur ; erreur si engine absent.
"""
    )

    cells[4]["source"] = lines(
        """### Après exécution — lecture (connexion)

La sortie résume **serveur**, **base** et le test SQL : cohérence avec l’environnement **SSMS** pour la reproductibilité des extractions.
"""
    )

    cells[5]["source"] = lines(
        """---

### Rendu des figures *(Matplotlib / Seaborn)*

Les graphiques apparaissent **sous** la cellule qui appelle `plt.show()` (mode **inline** dans Jupyter / VS Code).

| Comportement observé | Piste de diagnostic |
|---------------------|---------------------|
| Aucun graphique | Cellules non exécutées dans le flux logique du notebook, ou absence d’exécution de la cellule contenant `plt.show()`. |
| Zone d’affichage vide | La directive `%matplotlib inline` figure en tête des notebooks qui produisent des figures. |

Chaque figure s’affiche juste après la cellule qui la génère.
"""
    )

    # Inserts en indices décroissants (notebook d’origine : 22 cellules, 0–21)
    cells.insert(
        10,
        md_cell(
            lines(
                """---

## Cible, features et découpage train / test

🎯 **Objectif** : définir la **cible** `y` (statut encodé), les **features** numériques (max 20) et le **split stratifié** 75 % / 25 %.

✅ **Résultats attendus** : liste des features ; nombre de classes ; tailles train / test affichées en `[3]`.
"""
            )
        ),
    )
    cells.insert(
        8,
        md_cell(
            lines(
                """---

## Chargement du jeu de données

🎯 **Objectif** : charger le jeu **large** (`dw_financial_wide.parquet` ou SQL) et garantir la présence d’une colonne **statut réservation** (y compris via ponts DW).

✅ **Résultats attendus** : dimensions `[2]` ; messages indiquant la source (parquet, SQL, enrichissement statut).

### Avant la cellule suivante — rappel

Vérifiez dans la sortie `[2]` que le DataFrame contient bien une colonne de statut exploitable pour la suite.
"""
            )
        ),
    )
    cells.insert(
        6,
        md_cell(
            lines(
                """---

## Imports et configuration

🎯 **Objectif** : importer scikit-learn, pipelines, métriques **C** (accuracy, F1, ROC…) et préparer `ML/models/`.

✅ **Résultats attendus** : message `[1]` confirmant le mode `ML_SQL_ONLY` et les imports.
"""
            )
        ),
    )

    assert replace_first_match(
        cells,
        "### Lecture — imports et configuration",
        lines(
            """### Après exécution — lecture (imports)

Les bibliothèques **scikit-learn** et le répertoire `ML/models/` sont prêts ; le mode `ML_SQL_ONLY` conditionne la suite (DW strict ou repli local).
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Lecture — jeu de données",
        lines(
            """### Après exécution — lecture (jeu chargé)

Les dimensions `(lignes, colonnes)` et la présence du **statut** conditionnent la faisabilité de la classification ; le repli SQL ou parquet est documenté dans les messages `[2]`.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Lecture — cible et découpage",
        lines(
            """### Après exécution — lecture (cible et découpage)

Le **split stratifié** préserve la proportion des classes sur le train/test ; le nombre de classes et de features numériques fixe la complexité du problème.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Comparaison des modèles, métriques et graphiques de validation",
        lines(
            """---

## Entraînement des modèles et validation *(GridSearch, critère **C**)*

Vue d’ensemble avant les deux pipelines : **matrice de confusion** (champion), **ROC** si binaire, **importances** si RF ; privilégier les métriques sur le **jeu test** et le **F1 pondéré** en cas de classes déséquilibrées.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Modèle 1 — Random Forest (pipeline + GridSearchCV)",
        lines(
            """### Modèle 1 — Random Forest (pipeline + GridSearchCV)

Recherche sur grille (`n_estimators`, `max_depth`) avec **5-fold stratifié** et score **F1 pondéré**.

🎯 **Objectif** : optimiser **Random Forest** par **GridSearchCV** (5-fold **stratifié**, score F1 pondéré) sur le **train** uniquement.

✅ **Résultats attendus** : meilleurs hyperparamètres `gs_rf.best_params_` ; modèle `best_rf` prêt pour le test.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Modèle 2 — Régression logistique multinomiale",
        lines(
            """### Modèle 2 — Régression logistique multinomiale

Grille sur **`C`** ; même procédure de validation croisée sur le **train**.

🎯 **Objectif** : optimiser la **régression logistique** (pipeline `StandardScaler` → multinomial) sur la même grille de validation.

✅ **Résultats attendus** : meilleur estimateur `best_lr` ; comparabilité avec la RF via le même protocole CV.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Comparaison des modèles — jeu test & figures",
        lines(
            """### Comparaison sur le jeu test et figures

Métriques **Accuracy, Precision, Recall, F1, ROC-AUC** ; **champion** = meilleur F1 pondéré sur le test ; matrice de confusion et ROC (si binaire) ; importances si RF champion.

🎯 **Objectif** : **Comparer** RF et LR sur le **jeu test** : métriques globales, **matrice de confusion**, **ROC** (si binaire), **importances** (RF) ; désigner le **champion**.

✅ **Résultats attendus** : Accuracy, Precision / Recall / F1 pondérés, ROC-AUC ; figures ; nom du champion.

📊 **Visualisations** : **Matrice de confusion**, **courbe ROC** (si applicable), barres d’importance des variables (RF).
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Lecture — entraînement, métriques et figures",
        lines(
            """### Après exécution — lecture (métriques et figures)

Les blocs **GridSearch** indiquent les **meilleurs hyperparamètres** par validation croisée. Les **métriques test** comparent RF et LR ; le **champion** retient le meilleur **F1 pondéré**. La **matrice de confusion** localise les confusions entre statuts ; la **ROC** (binaire) ou le texte (multiclasse) complète l’**AUC**. Les **barres d’importance** (RF) hiérarchisent les variables pour l’analyse métier.

---

## Sauvegarde des artefacts

🎯 **Objectif** : persister le **pipeline champion** et les métriques détaillées pour le critère **C**.

✅ **Résultats attendus** : fichiers `.joblib` et `metrics_classification.json` ; message de confirmation.
"""
        ),
    )

    assert replace_first_match(
        cells,
        "### Lecture — artefacts produits",
        lines(
            """### Après exécution — lecture (fichiers produits)

Les fichiers **`.joblib`** et **`metrics_classification.json`** consolident le pipeline retenu et les métriques des deux modèles pour le **rapport** et la synthèse projet.
"""
        ),
    )

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK — notebook réorganisé, cellules :", len(nb["cells"]))


if __name__ == "__main__":
    main()
