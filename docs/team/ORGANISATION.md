# Organisation du dépôt (branche `Naima`)

- **Objectif** : garder la racine lisible sans toucher au cœur applicatif (`ML/api`, pipelines, `n8n` actifs).
- **Changements** :
  - Suppression d’arborescences et scripts **non nécessaires au runtime** (copie `gitMachine/`, scripts ponctuels de traduction / « fix » Python, extractions `.txt` temporaires, doublons de fichiers KPI).
  - Regroupement des guides Markdown épars dans **`docs/project-guides/`** (index : `docs/project-guides/README.md`).
- **`main`** : branche de référence commune ; **`Naima`** : travail de ménage et documentation associée, poussé séparément pour éviter toute confusion avec l’intégration principale.

_Contribution : Naima Sarraj — Esprit._
