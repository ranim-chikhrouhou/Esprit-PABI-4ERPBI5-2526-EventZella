# Dossier `gitMachine` — paquet prêt pour Git (ML + Streamlit)

Ce répertoire est une **copie synchronisée** des éléments à versionner pour le critère **Git versioning and deployment** (focus **ML + Streamlit**). La source de vérité du code est `../ML/` ; après modification des notebooks ou scripts, resynchroniser avec la commande indiquée en fin de fichier.

**Dépôt GitHub du projet :** [Esprit-PABI-4ERPBI5-2526-EventZella](https://github.com/ranim-chikhrouhou/Esprit-PABI-4ERPBI5-2526-EventZella.git)

---

## Arborescence (`gitMachine/`)

| Élément | Rôle |
|---------|------|
| **`README.md`** | Vue d’ensemble projet EventZilla (racine du paquet). |
| **`.gitignore`** | Exclusions (venv, caches, données lourdes, `.joblib`). |
| **`.streamlit/config.toml`** | Thème / options Streamlit pour le déploiement local ou cloud. |
| **`ML/`** | Code Python, notebooks, scripts, assets, `requirements.txt`. |
| **`ML/notebooks/`** | `00_A` … `04_F`, **`06_B`** (critère B), **`05`** (synthèse métriques). |
| **`ML/scripts/`** | `run_00_data_preparation.py` … `run_05_metrics_comparison.py`, `run_all_ml_pipeline.py`, etc. |
| **`ML/streamlit_app.py`** | Application Streamlit (lancer depuis la racine `gitMachine` avec la commande ci-dessous). |
| **`ML/models_artifacts/`** | `metrics_*.json`, JSON clustering fidélité — **pas** de `.joblib` versionné. |
| **`ML/processed/`** | JSON de lignée / résumés (`A_*.json`, `numeric_feature_list.json`) — **pas** de `.npy`, `.parquet`, `.csv`, figures `.png`. |
| **`ML/EventZilla_Dashboards_Improved.pdf`** | Référence KPI (si présent). |

### Fichiers Python utiles au déploiement / clustering

`clustering_deploy.py`, `cluster_labels.py`, `loyalty_artifacts_display.py`, `pca_interpretation_fr.py`, `ml_paths.py`, `schema_eventzilla.py`, `csv_local_fallback.py`, etc.

---

## Exclu volontairement (régénérable ou trop lourd)

- `*.joblib`, `*.npy`, `*.parquet`, `*.csv` générés sous `ML/processed/`.
- Images `ML/processed/*.png`.
- Caches `__pycache__`, checkpoints Jupyter.

Régénération : notebooks **00_A** → **04_F**, ou `ML/scripts/run_*.py`, une fois le DW / l’environnement configurés.

---

## Lancer Streamlit (après `pip install`)

À exécuter depuis la racine **`gitMachine`** (dossier qui contient `ML/` et `.streamlit/`) :

```text
py -3.11 -m pip install -r ML/requirements.txt
streamlit run ML/streamlit_app.py
```

**Variables SQL :** `EVENTZILLA_SQL_*` ou équivalent (voir `ML/ml_paths.py`). Ne pas committer de mots de passe ; utiliser l’environnement ou un `.env` **non versionné**.

---

## Resynchroniser `gitMachine/ML` depuis votre copie de travail

Depuis **cmd** ou **PowerShell** (adapter le chemin si besoin) :

```text
robocopy "c:\Users\ranim\Downloads\PI BI NEW\ML" "c:\Users\ranim\Downloads\PI BI NEW\gitMachine\ML" /E /XD __pycache__ .ipynb_checkpoints /XF *.joblib *.npy *.parquet *.csv /XF _pca_src.txt _md23.txt _md25.txt
```

(`robocopy` code de sortie 0–7 = succès ; 8+ = erreur.)

---

## Commandes Git (cmd sur Windows) — pousser vers GitHub

### Cas 1 : ce dépôt GitHub est vide ou vous remplacez tout par `gitMachine`

```cmd
cd /d "c:\Users\ranim\Downloads\PI BI NEW\gitMachine"
git init
git branch -M main
git remote add origin https://github.com/ranim-chikhrouhou/Esprit-PABI-4ERPBI5-2526-EventZella.git
git add .
git status
git commit -m "feat: ML EventZilla — notebooks 00_A-06_B, Streamlit, scripts, metriques JSON"
git push -u origin main
```

Si `git push` refuse parce que le dépôt distant a déjà des commits :

```cmd
git pull origin main --allow-unrelated-histories
```

Résolvez d’éventuels conflits, puis :

```cmd
git push -u origin main
```

### Cas 2 : vous avez déjà cloné le dépôt ailleurs (avec `DataBase`, `Reports`, etc.)

Copiez le contenu de `gitMachine` **dans la racine du clone** (ou uniquement `ML/` + `.streamlit/` + `.gitignore` si vous fusionnez), puis :

```cmd
cd /d "chemin\vers\votre\clone\Esprit-PABI-4ERPBI5-2526-EventZella"
git add .
git commit -m "feat: ajout couche ML EventZilla (notebooks, Streamlit, scripts)"
git push origin main
```

### Authentification GitHub

- **HTTPS :** Git vous demandera identifiant + **Personal Access Token** (pas le mot de passe du compte).
- **SSH :** `git remote set-url origin git@github.com:ranim-chikhrouhou/Esprit-PABI-4ERPBI5-2526-EventZella.git` après configuration des clés.

---

*Synchroniser `gitMachine` avant chaque remise si vous éditez directement `PI BI NEW\ML`.*
