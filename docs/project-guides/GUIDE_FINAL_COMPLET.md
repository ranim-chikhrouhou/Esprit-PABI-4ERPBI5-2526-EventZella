# 🎯 GUIDE FINAL COMPLET - EventZilla MLOps

> **Note (avril 2026)** : Les scripts `NETTOYER_PROJET.py` et `REPARER_TOUT.py` ne sont plus dans le dépôt (nettoyage déjà effectué). Voir [`docs/team/ORGANISATION.md`](../team/ORGANISATION.md).

## 📋 Table des Matières
1. [Nettoyage du Projet](#1-nettoyage-du-projet)
2. [Réparation et Remplissage](#2-réparation-et-remplissage)
3. [Démarrage des Services](#3-démarrage-des-services)
4. [Vérification et Tests](#4-vérification-et-tests)
5. [Résolution des Problèmes](#5-résolution-des-problèmes)

---

## 1. 🧹 Nettoyage du Projet

### Pourquoi nettoyer?
- Supprimer les fichiers redondants (multiples guides, scripts de test)
- Garder uniquement l'essentiel
- Projet plus clair et organisé

### Comment nettoyer?

```bash
# Dans un terminal:
cd "PI BI NEW (2)/PI BI NEW"
python NETTOYER_PROJET.py
```

**Tapez "oui" quand demandé**

### Résultat:
- ✅ ~50-100 fichiers inutiles supprimés
- ✅ Structure claire et organisée
- ✅ Tous les fichiers essentiels conservés

---

## 2. 🔧 Réparation et Remplissage

### Pourquoi réparer?
- MLflow est vide → besoin de runs
- Metrics protégées → besoin de les rendre publiques
- Services à vérifier

### Comment réparer?

```bash
# Dans le même terminal:
python REPARER_TOUT.py
```

### Ce que fait ce script:
1. ✅ Vérifie que FastAPI, MLflow, Streamlit, n8n fonctionnent
2. ✅ Remplit MLflow avec 14 runs dans 4 expériences:
   - Classification (3 runs)
   - Régression (3 runs)
   - Clustering (3 runs)
   - n8n Finance (5 runs)
3. ✅ Teste FastAPI et les métriques
4. ✅ Affiche un rapport complet

### Résultat:
- ✅ MLflow rempli avec des expériences visibles
- ✅ Rapport de l'état de tous les services

---

## 3. 🚀 Démarrage des Services

### Option A: Tout en un (Recommandé)

```bash
# Double-cliquez sur:
LANCER_PROJET.bat
```

**OU** dans un terminal:
```bash
LANCER_PROJET.bat
```

Cela démarre:
- FastAPI (port 8000)
- MLflow (port 5000)
- Streamlit (port 8502)
- n8n (port 5678)

### Option B: Services individuels

**Terminal 1 - FastAPI:**
```bash
python -m uvicorn ML.api.main:app --reload --port 8000
```

**Terminal 2 - MLflow:**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

**Terminal 3 - Streamlit:**
```bash
python -m streamlit run ML/streamlit_app.py
```

**Terminal 4 - n8n:**
```bash
npx n8n
```

---

## 4. ✅ Vérification et Tests

### 4.1 Vérifier FastAPI

**URL:** http://localhost:8000

**Test 1 - Health Check:**
```bash
curl http://localhost:8000
```
Devrait retourner: `{"status":"ok",...}`

**Test 2 - Metrics (PUBLIC):**
```bash
curl http://localhost:8000/metrics
```
Devrait retourner des métriques Prometheus (pas d'erreur 401)

**Test 3 - API Docs:**
Ouvrez: http://localhost:8000/docs
- Interface interactive
- Testez les endpoints

### 4.2 Vérifier MLflow

**URL:** http://localhost:5000

**Ce que vous devriez voir:**
- ✅ 4 expériences dans la liste
- ✅ Cliquez sur "EventZilla_Classification_Booking_Status" → 3 runs
- ✅ Cliquez sur "EventZilla_Regression_Price_Prediction" → 3 runs
- ✅ Cliquez sur "EventZilla_Clustering_Customer_Segmentation" → 3 runs
- ✅ Cliquez sur "n8n_Finance_Pipeline" → 5 runs

**Dans chaque run:**
- Paramètres (n_estimators, alpha, etc.)
- Métriques (accuracy, r2_score, etc.)
- Modèles sauvegardés

### 4.3 Vérifier Streamlit

**URL:** http://localhost:8502

**Si erreur SQL Server:**
- C'est normal si vous n'avez pas SQL Server configuré
- Utilisez FastAPI Docs à la place (http://localhost:8000/docs)

### 4.4 Vérifier n8n

**URL:** http://localhost:5678

**Test workflow:**
1. Importez: `n8n/workflow_finance_mlflow_ultra_simple.json`
2. Cliquez "Execute Workflow"
3. Vérifiez MLflow pour voir le nouveau run

---

## 5. 🎯 Monitoring (Week S13)

### 5.1 Sans Docker (Simple)

**Métriques disponibles:**
- URL: http://localhost:8000/metrics
- Format: Prometheus (texte brut)
- Rafraîchissez pour voir les changements

**Générer du traffic:**
```bash
python simulate_scenarios.py
```

Pendant que ça tourne, rafraîchissez `/metrics` → les compteurs augmentent!

### 5.2 Avec Docker (Complet)

**Démarrer Prometheus + Grafana:**
```bash
docker-compose -f docker-compose-monitoring.yml up -d
```

**Attendre 2 minutes**, puis:

**Prometheus:**
- URL: http://localhost:9090
- Vérifiez les targets: http://localhost:9090/targets
- Vérifiez les alertes: http://localhost:9090/alerts

**Grafana:**
- URL: http://localhost:3000
- Login: `admin` / `eventzilla2026`
- Dashboard: "EventZilla MLOps - Production Monitoring"

**Lancer les simulations:**
```bash
python simulate_scenarios.py
```

Regardez le dashboard Grafana pendant que ça tourne!

---

## 6. 🔍 Résolution des Problèmes

### Problème 1: FastAPI - 401 Unauthorized sur /metrics

**Cause:** Anciennes modifications pas appliquées

**Solution:**
1. Arrêtez FastAPI (Ctrl+C)
2. Relancez: `python -m uvicorn ML.api.main:app --reload --port 8000`
3. Testez: `curl http://localhost:8000/metrics`

### Problème 2: MLflow vide

**Cause:** Aucun run loggé

**Solution:**
```bash
python REPARER_TOUT.py
```

Cela remplit MLflow avec 14 runs.

### Problème 3: Streamlit - Erreur SQL Server

**Cause:** SQL Server non configuré

**Solutions:**
- **Option A:** Utilisez FastAPI Docs (http://localhost:8000/docs)
- **Option B:** Configurez SQL Server dans `.env`
- **Option C:** Demandez une version Streamlit sans SQL Server

### Problème 4: n8n - Invalid expression

**Cause:** Expressions JSON trop complexes

**Solution:**
Utilisez le workflow simplifié:
- `n8n/workflow_finance_mlflow_ultra_simple.json`

### Problème 5: Grafana - Pas de données

**Cause:** Prometheus ne peut pas scraper

**Solution:**
1. Vérifiez que FastAPI tourne
2. Vérifiez que `/metrics` est public (pas de 401)
3. Redémarrez Prometheus:
   ```bash
   docker-compose -f docker-compose-monitoring.yml restart prometheus
   ```

### Problème 6: Docker - Services ne démarrent pas

**Cause:** Docker Desktop pas démarré

**Solution:**
1. Ouvrez Docker Desktop
2. Attendez qu'il affiche "Engine running"
3. Relancez: `docker-compose -f docker-compose-monitoring.yml up -d`

---

## 7. 📊 Deliverables Week S13

### Checklist Complète

- [x] **1. Prometheus monitoring working**
  - Endpoint `/metrics` fonctionnel
  - Métriques exposées au format Prometheus
  - Scraping configuré (prometheus.yml)

- [x] **2. Grafana dashboard**
  - Dashboard JSON créé (grafana/dashboards/eventzilla_mlops_dashboard.json)
  - Panels: Traffic, Performance, Model Health, Data Quality, System
  - Peut être importé dans n'importe quel Grafana

- [x] **3. Alerting rules configured**
  - 15 règles d'alerte (prometheus_rules.yml)
  - Catégories: Performance, Reliability, Model Health, Data Quality, System, Traffic
  - Niveaux: Info, Warning, Critical

- [x] **4. Drift detection logic**
  - Module monitoring.py avec détection de drift
  - Data drift (distribution shift)
  - Model drift (accuracy drop >5%)
  - Baselines définis

- [x] **5. Simulation scenarios**
  - Script simulate_scenarios.py
  - 3 scénarios: High Traffic, API Errors, Model Drift
  - Automatisé avec ThreadPoolExecutor

- [x] **6. Observability**
  - Métriques (what happens)
  - Logs (why it happens)
  - Alert logs avec timestamp, type, message, severity

- [x] **7. Baseline comparison**
  - Baselines définis pour tous les modèles
  - Thresholds configurables
  - Déviations détectables et alertables

### Screenshots à Prendre

1. **MLflow UI** - Expériences avec runs
2. **FastAPI Docs** - Interface interactive
3. **Metrics Endpoint** - http://localhost:8000/metrics
4. **Grafana Dashboard** - (si Docker fonctionne)
5. **Prometheus Alerts** - http://localhost:9090/alerts

---

## 8. 🎉 Résumé Final

### Ce qui fonctionne:
✅ FastAPI avec API complète
✅ MLflow avec expériences et runs
✅ Métriques Prometheus exposées
✅ Configuration monitoring complète
✅ Simulations fonctionnelles
✅ Documentation complète

### Ce qui nécessite ajustements:
⚠️ Streamlit (nécessite SQL Server)
⚠️ Grafana (nécessite Docker)
⚠️ n8n workflows (à tester)

### Pour la démonstration:
1. Montrez MLflow avec les runs
2. Montrez FastAPI Docs avec les endpoints
3. Montrez les métriques sur /metrics
4. Lancez simulate_scenarios.py
5. Montrez les métriques qui changent
6. Expliquez la configuration Prometheus/Grafana

---

## 9. 🚀 Commandes Rapides

```bash
# Nettoyer le projet
python NETTOYER_PROJET.py

# Réparer et remplir MLflow
python REPARER_TOUT.py

# Démarrer tous les services
LANCER_PROJET.bat

# Lancer les simulations
python simulate_scenarios.py

# Démarrer monitoring avec Docker
docker-compose -f docker-compose-monitoring.yml up -d

# Arrêter monitoring
docker-compose -f docker-compose-monitoring.yml down
```

---

## 10. 📞 Support

**Fichiers importants:**
- `README_FINAL.md` - Documentation complète du projet
- `MONITORING_GUIDE_S13.md` - Guide monitoring détaillé
- `STRUCTURE_FINALE.txt` - Structure du projet après nettoyage

**URLs importantes:**
- FastAPI: http://localhost:8000
- FastAPI Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics
- MLflow: http://localhost:5000
- Streamlit: http://localhost:8502
- n8n: http://localhost:5678
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

**Tout est prêt! Suivez les étapes dans l'ordre et tout fonctionnera! 🚀**
