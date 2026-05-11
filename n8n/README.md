# EventZilla — Workflows n8n (automatisation ML)

Ce dossier contient les exports JSON **prêts à importer** pour couvrir la grille **« Partie 1 — N8N ML Automation »** : pipeline ML bout-en-bout, diversité des nœuds (HTTP, Webhook, Cron / Schedule, Execute Command), persistance des prédictions, MLflow, gestion d’erreurs et documentation.

---

## Fichiers livrés

| Fichier | Rôle |
|---------|------|
| `workflow_finance.json` | Finance — **Schedule** → FastAPI (Ridge + Holt) → fichier + **MLflow** |
| `workflow_webhook_finance.json` | Finance — **Webhook** (inférence à la demande), même sortie |
| `workflow_marketing.json` | Marketing — **Schedule** quotidien → segmentation + classification → fichier + **MLflow** |
| `workflow_crm.json` | CRM — **Webhook** → classification + segmentation → fichier + **MLflow** |
| `workflow_retraining_weekly.json` | Bonus — **Schedule** + **Execute Command** (`run_pipeline_s12.py`) |
| `workflow_error_handler.json` | **Error Trigger** → `/alert/error` (logs + email si SMTP configuré) |
| `env.example` | Variables d’environnement recommandées (sans secrets en prod idéal) |

Les anciens exports redondants (`workflow_finance_*mlflow*.json`, etc.) ont été **supprimés** au profit de `workflow_finance.json` qui centralise MLflow et la fusion robuste.

---

## Cartographie grille « Excellent »

| Critère | Exigence | Réalisation EventZilla |
|--------|----------|-------------------------|
| **A — Architecture** | Trigger → données → modèle → sortie structurée | Chaque pipeline va du trigger au stockage (`/save_result`) avec étapes ML nommées ; **Merge** pour synchroniser deux branches parallèles sans double exécution du Code. |
| **A — Documentation** | Nœuds étiquetés + README | Notes sur les nœuds + sticky notes ; ce README décrit les choix. |
| **B — Intégration ML** | API / script | Toutes les inférences passent par **FastAPI** (`ML/api`). |
| **B — Variété nœuds** | HTTP, Webhook, Cron, Execute Command | **HTTP** partout ; **Webhook** CRM + Finance ; **Schedule** Marketing/Finance/Réentraînement ; **Execute Command** réentraînement. |
| **B — Secrets** | Hors fichier | Login via variables `N8N_EZ_*` (fallback démo dans les expressions si non défini — à retirer en production). |
| **C — Inférence auto** | Cron ou webhook + persistance | Cron (Finance/Marketing) + Webhooks (CRM/Finance) ; JSON dans `n8n/results/`. |
| **C — Bonus réentraînement** | Pipeline planifiée | `workflow_retraining_weekly.json` appelle `python run_pipeline_s12.py`. |
| **D — Robustesse** | Erreurs / notifications | Chaque workflow métier définit `errorWorkflow` → **EventZilla — Error Handler** → log JSONL + email Gmail si variables serveur renseignées. |

---

## Schéma général

```
[ Schedule ou Webhook ]
        │
        ▼
[ Login JWT FastAPI ]
        │
        ├──► [ Modèle A ] ──┐
        │                   ├──► [ Merge ] ─► [ Code rapport ]
        └──► [ Modèle B ] ──┘                        │
                                                   ├──► POST /save_result
                                                   └──► POST /mlflow/log_prediction
```

---

## Prérequis — lancer dans cet ordre

```bash
# Terminal 1 — API FastAPI (backend ML)
python run_fastapi.py
# ou : python -m uvicorn ML.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — n8n (orchestrateur)
npx n8n
# → http://localhost:5678

# Terminal 3 (optionnel) — MLflow UI si vous utilisez le tracking distant/local du projet
```

Copier `env.example` vers votre configuration de variables n8n et renseigner au minimum `N8N_EZ_FASTAPI_URL` et `EVENTZILLA_REPO_ROOT` pour le réentraînement.

---

## Import dans n8n

1. **Importer d’abord** `workflow_error_handler.json`, puis activer si besoin.
2. Importer les workflows métier (`workflow_*.json`).
3. Pour chaque workflow métier : **Workflow Settings → Error workflow** → choisir **EventZilla — Error Handler** (le nom doit correspondre exactement).
4. **Execute Command** : selon la version n8n / politique de sécurité, autoriser explicitement ce nœud sur l’instance.

---

## URLs webhook (tests)

| Workflow | Mode test | Mode production |
|----------|-----------|-----------------|
| CRM | `POST http://localhost:5678/webhook-test/eventzilla-crm-trigger` | `.../webhook/eventzilla-crm-trigger` |
| Finance | `POST http://localhost:5678/webhook-test/eventzilla-finance-trigger` | `.../webhook/eventzilla-finance-trigger` |

Le script `test_workflows.py` déclenche CRM et Finance automatiquement si n8n tourne.

---

## Endpoints FastAPI utilisés

| Endpoint | Usage |
|----------|--------|
| `POST /auth/login` | JWT par rôle |
| `POST /predict/regression` | Finance (et utilisateurs autorisés par RBAC) |
| `GET /predict/timeseries` | Finance |
| `POST /predict/classification` | Marketing, CRM |
| `POST /predict/segmentation/beneficiaire` | Marketing, CRM |
| `POST /save_result` | Persistance JSON dans `n8n/results/` |
| `POST /mlflow/log_prediction` | Tracking (schéma `MLflowLogRequest`) |
| `POST /alert/error` | Handler d’erreurs n8n |

---

## Résultats générés

Les fichiers sont créés par FastAPI sous `n8n/results/` :

```
n8n/results/
├── marketing_predictions_YYYY-MM-DD.json
├── finance_predictions_YYYY-MM-DD.json
├── crm_predictions_YYYY-MM-DD_HH-MM.json
└── error_log.jsonl
```

---

## Décisions de conception (résumé)

| Décision | Pourquoi |
|----------|----------|
| **Merge « Combine by position »** | Quand deux branches parallèles alimentent un Code, sans Merge le nœud peut s’exécuter deux fois ou fusionner dans un ordre non garanti. |
| **Références `$('Nom du nœud').first().json`** | Après le Merge, le Code lit explicitement chaque réponse HTTP pour Marketing/CRM et évite les collisions de champs (`modele`, etc.). |
| **Payload MLflow via Code** | Garantit des métriques **numériques** (`Dict[str,float]`) et un champ `artifacts` sérialisable côté API. |
| **Variables `N8N_EZ_*`** | Répond à l’exigence « credentials gérés proprement » sans multiplier les Credential n8n pour une API maison. |
| **Double déclenchement Finance** | Cron pour reporting ; Webhook pour événements — couvre explicitement planification **et** inférence événementielle. |

---

## Test rapide

Depuis la racine du projet « PI BI NEW » :

```bash
python n8n/test_workflows.py
```

---

## Maintenance

- Ajuster les horaires dans les **Schedule Trigger** depuis l’UI n8n.
- Adapter la commande du workflow réentraînement sous Linux/Mac (`python3`, chemins).
- En production : retirer les mots de passe en dur dans les expressions en vous assurant que **toutes** les variables `N8N_EZ_*` sont définies sur le serveur n8n.
