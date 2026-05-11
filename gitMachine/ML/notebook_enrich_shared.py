# -*- coding: utf-8 -*-
"""
Textes partagés : ton professionnel, sans impératifs adressés au lecteur.
Référence : grilles de validation A–F (projet intégré ML).
"""

CONN_MD = """## 🔌 Connexion au Data Warehouse (SQL Server / SSMS)

L’accès aux données repose sur le **même environnement** que sous **SSMS** : base **`DW_eventzella`** (ou `EVENTZILLA_SQL_DW`), via **SQLAlchemy** et **pyodbc**, authentification Windows.

La cellule suivante affiche serveur, base, extrait de chaîne de connexion, ainsi qu’un test `SELECT DB_NAME()` / `SERVERPROPERTY`.

> En cas d’anomalie (service SQL, driver ODBC, paramètres `EVENTZILLA_SQL_*`), la configuration est documentée dans `ML/ml_paths.py`.
"""

FIG_MD = """## 🖼️ Rendu des figures (Matplotlib / Seaborn)

Les graphiques apparaissent **sous** la cellule qui appelle `plt.show()` (mode **inline** dans Jupyter / VS Code).

| Comportement observé | Piste de diagnostic |
|---------------------|---------------------|
| Aucun graphique | Cellules non exécutées dans le flux logique du notebook, ou absence d’exécution de la cellule contenant `plt.show()`. |
| Zone d’affichage vide | La directive `%matplotlib inline` figure en tête des notebooks qui produisent des figures. |

Le tableau « Structure » du notebook recense les visualisations ; chaque figure suit la cellule qui la génère.
"""


def replace_connexion_figures_in_markdown(src: str) -> str:
    """Remplace les blocs Connexion / Figures standardisés si présents."""
    import re

    if "## Connexion au Data Warehouse" in src and "🔌" not in src[:120]:
        m = re.match(
            r"## Connexion au Data Warehouse \(SQL Server / SSMS\)\s*\n[\s\S]*",
            src.strip(),
        )
        if m:
            src = CONN_MD.strip() + "\n"
    if "## Où apparaissent les figures" in src and "🖼️" not in src[:120]:
        m = re.match(
            r"## Où apparaissent les figures \(graphiques\) \?\s*\n[\s\S]*",
            src.strip(),
        )
        if m:
            src = FIG_MD.strip() + "\n"
    return src


def soften_imperatives_markdown(src: str) -> str:
    """Atténue les formulations impératives courantes (ton rapport / encadrement)."""
    reps = [
        ("Exécuter la cellule ci-dessous **en premier** :", "**Séquence** — la cellule ci-dessous présente"),
        ("Exécuter les cellules **dans l'ordre** depuis le haut ; vérifier que la cellule avec", "En cas d’absence de figure : reprendre le flux depuis le haut et s’assurer que la cellule avec"),
        ("Exécuter les cellules **dans l’ordre** depuis le haut ; vérifier que", "En cas d’absence de figure : reprendre le flux depuis le haut ; contrôler que"),
        ("1. Vérifier la présence des", "1. Présence attendue des"),
        ("Ce notebook **n’entraîne aucun modèle**", "Ce notebook **n’entraîne pas de modèle**"),
    ]
    for a, b in reps:
        src = src.replace(a, b)
    return src
