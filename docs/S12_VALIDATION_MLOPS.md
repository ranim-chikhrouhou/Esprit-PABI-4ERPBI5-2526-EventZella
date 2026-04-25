# Validation S12 — MLOps (EventZilla)

Ce document résume les éléments livrés pour la validation S12.

## 1) Experiment Tracking (MLflow)

- MLflow intégré dans:
  - `ML/scripts/run_01_clustering.py`
  - `ML/scripts/run_02_classification.py`
  - `ML/scripts/run_03_prediction_regression.py`
  - `ML/scripts/run_04_time_series.py`
- Paramètres, métriques et artefacts sont loggés à chaque run.
- Des runs comparables existent pour chaque type (`classification`, `regression`, `clustering`, `time_series`).

## 2) Pipeline automatisé

- Pipeline de bout en bout disponible:
  - `python ML/scripts/run_all_ml_pipeline.py`
- Chaîne: préparation -> entraînement -> évaluation -> sauvegarde.
- Exécutable sans intervention manuelle.

## 3) Model management

- Modèles et JSON métriques sauvegardés dans `ML/models_artifacts/`.
- Versioning local ajouté (historique horodaté) via `ML/scripts/mlops_utils.py`.

## 4) Model serving (API)

- API FastAPI fonctionnelle: `ML/api/main.py`
- Endpoint de validation explicite:
  - `POST /predict`
- Endpoints métiers disponibles:
  - `/predict/classification`
  - `/predict/regression`
  - `/predict/segmentation/{type_entite}`
  - `/predict/timeseries`

## 5) Containerization

- Dockerfile API: `Dockerfile.api`
- Orchestration: `docker-compose.mlops.yml`
- Services:
  - API sur `http://127.0.0.1:8000`
  - MLflow sur `http://127.0.0.1:5000`

## 6) Code quality

- Scripts structurés par étape (`run_00` -> `run_05`).
- Vérifications de syntaxe Python effectuées sur les fichiers modifiés.

## 7) Web App Integration

- `ML/streamlit_app.py` peut appeler l'API (`/predict`) pour le flux UI -> API -> modèle -> résultat.

---

## Commandes de démo (jour J)

```powershell
cd "C:\Users\ranim\Downloads\PI BI NEW"
docker compose -f "docker-compose.mlops.yml" up -d
docker compose -f "docker-compose.mlops.yml" ps
```

Ouvrir:
- API docs: `http://127.0.0.1:8000/docs`
- MLflow: `http://127.0.0.1:5000`

Test API:
- `POST /predict` avec un payload de classification.

MLflow:
- Aller dans `Training runs`.
- Comparer 2 runs du même type.
- Montrer `params`, `metrics`, `artifacts`.
