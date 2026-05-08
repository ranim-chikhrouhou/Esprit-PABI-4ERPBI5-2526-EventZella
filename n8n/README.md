# EventZilla — Workflows n8n (Automatisation ML)

## Architecture Globale

```
[Trigger]
    │
    ▼
[Login API FastAPI]  ──→  JWT Token
    │
    ├──→ [Prédiction Modèle 1]
    │
    ├──→ [Prédiction Modèle 2]
    │
    ▼
[Fusionner Résultats]
    │
    ▼
[Sauvegarder JSON]  +  [Error Handler → Email]
```

---

## Prérequis — Lancer dans cet ordre

```bash
# Terminal 1 — API FastAPI (backend ML)
python -m uvicorn ML.api.main:app --reload --port 8000

# Terminal 2 — n8n (orchestrateur)
n8n start
# → Ouvrir http://localhost:5678

# Terminal 3 — Streamlit (app déploiement)
streamlit run ML/streamlit_app.py
```

---

## Workflows disponibles

| Fichier | Décideur | Trigger | Modèles |
|---------|----------|---------|---------|
| `workflow_marketing.json` | Ranim Chikhrouhou | Cron quotidien 08h00 | K-Means (E) + Random Forest (C) |
| `workflow_finance.json` | Naïma Sarraj | Cron hebdo lundi 07h00 | Ridge (D) + Holt (F) |
| `workflow_crm.json` | Anas Allam | Webhook événementiel | Random Forest (C) + K-Means (E) |
| `workflow_error_handler.json` | Tous | Error Trigger | Notification email |

---

## Comment importer dans n8n

```
1. Ouvrir http://localhost:5678
2. Menu supérieur → "+" → New Workflow
3. Menu "..." (3 points) → Import from File
4. Sélectionner workflow_marketing.json
5. Répéter pour les 3 autres workflows
6. Configurer "EventZilla — Error Handler" dans
   Settings de chaque workflow → Error Workflow
```

---

## Description des Nœuds

### Workflow Marketing (Ranim — marketing_manager)

| Nœud | Type n8n | Rôle |
|------|----------|------|
| Déclencheur Quotidien 08h00 | Schedule Trigger | Déclenche automatiquement chaque matin |
| Login API — Ranim | HTTP Request POST | Authentification SQL Server → JWT token |
| Segmentation Bénéficiaires | HTTP Request POST | Appel `/predict/segmentation/beneficiaire` → K-Means fidélité |
| Classification Statut Réservation | HTTP Request POST | Appel `/predict/classification` → RF champion |
| Fusionner Résultats | Code (JavaScript) | Consolidation des deux prédictions |
| Sauvegarder Prédictions Marketing | Write Binary File | Fichier JSON daté dans `n8n/results/` |
| Vérifier Succès | IF | Déclenchement alerte si prédiction vide |

### Workflow Finance (Naïma — financial_manager)

| Nœud | Type n8n | Rôle |
|------|----------|------|
| Déclencheur Hebdo Lundi 07h00 | Schedule Trigger | Rapport financier hebdomadaire |
| Login API — Naïma | HTTP Request POST | Authentification → rôle financial_manager |
| Prédiction Montant Final (Ridge) | HTTP Request POST | Appel `/predict/regression` → Ridge (R²≈1.0) |
| Prévision CA Mensuel (Holt) | HTTP Request GET | Appel `/predict/timeseries?horizon=3` → Holt |
| Fusionner Résultats Finance | Code (JavaScript) | Rapport financier consolidé |
| Sauvegarder Rapport Finance | Write Binary File | JSON hebdomadaire dans `n8n/results/` |

### Workflow CRM (Anas — crm_manager)

| Nœud | Type n8n | Rôle |
|------|----------|------|
| Webhook — Événement CRM | Webhook POST | Déclenché par événement externe (nouvelle réservation) |
| Login API — Anas | HTTP Request POST | Authentification → rôle crm_manager |
| Anticipation Annulation (RF) | HTTP Request POST | Appel `/predict/classification` → statut prédit |
| Segment Fidélité Client | HTTP Request POST | Appel `/predict/segmentation/beneficiaire` |
| Analyse CRM + Action Recommandée | Code (JavaScript) | Génère alerte si statut = cancelled |
| Sauvegarder Analyse CRM | Write Binary File | JSON horodaté dans `n8n/results/` |

### Workflow Error Handler (Tous les workflows)

| Nœud | Type n8n | Rôle |
|------|----------|------|
| Error Trigger | Error Trigger | Capte toute erreur dans les 3 workflows |
| Formater Message Erreur | Code (JavaScript) | Prépare le message d'alerte |
| Notification Email Erreur | Email Send | Envoie l'alerte à l'équipe |

---

## Endpoints API utilisés

| Endpoint | Méthode | Accès | Modèle |
|----------|---------|-------|--------|
| `/auth/login` | POST | Tous | — |
| `/predict/classification` | POST | Marketing, Finance, CRM | Random Forest (champion C) |
| `/predict/regression` | POST | Finance, Marketing | Ridge (champion D, R²≈1.0) |
| `/predict/segmentation/{type}` | POST | Marketing, CRM | K-Means fidélité (critère E) |
| `/predict/timeseries` | GET | Finance, Marketing | Holt (champion F, MAPE≈6.1%) |
| `/metrics` | GET | Tous | — (métriques JSON) |

---

## Résultats générés

Les fichiers sont sauvegardés dans `n8n/results/` :

```
n8n/results/
├── marketing_predictions_2026-04-16.json
├── finance_predictions_2026-04-16.json
└── crm_predictions_2026-04-16_08-30.json
```

---

## Test manuel d'un workflow

```
n8n → Ouvrir workflow → Bouton "Execute Workflow"
→ Vérifier l'onglet "Output" de chaque nœud
→ Vérifier http://localhost:5678/executions pour l'historique
```

---

## Décisions de conception

| Décision | Justification |
|----------|---------------|
| JWT via FastAPI | Sécurité : token expirant 8h, basé sur logins SQL Server |
| Cron pour Marketing/Finance | Reporting périodique — cohérent avec la fréquence de décision (quotidienne/hebdomadaire) |
| Webhook pour CRM | Réactivité événementielle — le CRM doit agir en temps réel sur les annulations |
| Error Workflow centralisé | Un seul handler pour les 3 workflows — maintenance simplifiée |
| Stockage JSON local | Traçabilité des prédictions — preuves d'automatisation datées |
