# 📊 RÉSUMÉ - État des Workflows n8n EventZilla

## 🎯 STATUT GLOBAL

```
┌─────────────────────────────────────────────┐
│  WORKFLOWS N8N - EVENTZILLA                 │
│                                             │
│  Statut : ⚠️ NON OPÉRATIONNEL              │
│  Raison : npm bloqué par PowerShell         │
│  Solution : Utiliser CMD ou débloquer PS    │
│  Temps : 10-15 minutes                      │
└─────────────────────────────────────────────┘
```

---

## ✅ CE QUI EST PRÊT

| Composant | Statut | Détails |
|-----------|--------|---------|
| **Node.js** | ✅ Installé | v22.22.0 |
| **Workflows JSON** | ✅ Prêts | 4 fichiers complets |
| **FastAPI** | ✅ Code prêt | Tous les endpoints ML |
| **Documentation** | ✅ Complète | 3 guides détaillés |
| **Scripts de test** | ✅ Prêts | Test automatique disponible |

---

## ❌ CE QUI MANQUE

| Composant | Statut | Action requise |
|-----------|--------|----------------|
| **npm** | ❌ Bloqué | Débloquer PowerShell OU utiliser CMD |
| **n8n** | ❓ Inconnu | Installer après déblocage npm |
| **Workflows importés** | ❌ Non | Importer dans interface n8n |
| **FastAPI lancée** | ❌ Non | Lancer dans terminal |
| **n8n lancé** | ❌ Non | Lancer dans terminal |

---

## 🔴 PROBLÈME PRINCIPAL

### PowerShell bloque npm

**Erreur:**
```
npm : Impossible de charger le fichier C:\Program Files\nodejs\npm.ps1, 
car l'exécution de scripts est désactivée sur ce système.
```

**Cause:** Windows bloque les scripts PowerShell par défaut

**Impact:** Impossible d'utiliser npm, npx, donc impossible de lancer n8n

---

## ✅ SOLUTION RAPIDE (2 OPTIONS)

### Option 1: Utiliser CMD (RECOMMANDÉ)

```cmd
# 1. Ouvrir CMD (Win + R → cmd)
# 2. Naviguer vers le projet
cd "C:\Users\ASUS\Desktop\VER1\PI BI NEW (2)\PI BI NEW"

# 3. Lancer l'installation
setup_n8n.bat
```

**Avantage:** Pas besoin de modifier les permissions Windows

---

### Option 2: Débloquer PowerShell

```powershell
# 1. Ouvrir PowerShell en Admin
# 2. Débloquer
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Confirmer avec "O"
# 4. Lancer l'installation
.\setup_n8n.ps1
```

**Avantage:** Résout le problème définitivement

---

## 📋 ÉTAPES APRÈS INSTALLATION

### 1️⃣ Installer n8n
```cmd
npm install -g n8n
```

### 2️⃣ Lancer FastAPI (Terminal 1)
```cmd
python -m uvicorn ML.api.main:app --reload --port 8000
```
→ http://127.0.0.1:8000/docs

### 3️⃣ Lancer n8n (Terminal 2)
```cmd
npx n8n
```
→ http://localhost:5678

### 4️⃣ Créer compte n8n
- Ouvrir http://localhost:5678
- Sign up avec email/password

### 5️⃣ Importer 4 workflows
- workflow_marketing.json
- workflow_finance.json
- workflow_crm.json
- workflow_error_handler.json

### 6️⃣ Configurer Error Handler
- Dans chaque workflow (Marketing, Finance, CRM)
- Settings → Error Workflow → "EventZilla — Error Handler"

### 7️⃣ Tester
```cmd
python n8n/test_workflows.py
```

---

## 📊 WORKFLOWS DISPONIBLES

### 🎯 Workflow Marketing (Ranim)
- **Trigger:** Cron quotidien 08h00
- **Modèles:** K-Means + Random Forest
- **Endpoints:**
  - `/predict/segmentation/beneficiaire`
  - `/predict/classification`
- **Output:** `marketing_predictions_YYYY-MM-DD.json`

### 💰 Workflow Finance (Naïma)
- **Trigger:** Cron hebdo lundi 07h00
- **Modèles:** Ridge + Holt
- **Endpoints:**
  - `/predict/regression`
  - `/predict/timeseries`
- **Output:** `finance_predictions_YYYY-MM-DD.json`

### 👥 Workflow CRM (Anas)
- **Trigger:** Webhook événementiel
- **Modèles:** Random Forest + K-Means
- **Endpoints:**
  - `/predict/classification`
  - `/predict/segmentation/beneficiaire`
- **Output:** `crm_predictions_YYYY-MM-DD_HH-MM.json`

### ⚠️ Workflow Error Handler
- **Trigger:** Erreur dans les 3 workflows
- **Action:** Email d'alerte + log JSON
- **Output:** `error_log.jsonl`

---

## 🎯 CHECKLIST RAPIDE

```
Installation:
□ npm fonctionne
□ n8n installé
□ FastAPI lancée
□ n8n lancé
□ Compte n8n créé

Configuration:
□ 4 workflows importés
□ Error Handler configuré

Tests:
□ Test manuel réussi (nœuds verts)
□ Test automatique réussi (script Python)
□ Fichiers JSON créés dans n8n/results/
```

---

## 📁 FICHIERS CRÉÉS POUR VOUS

| Fichier | Description |
|---------|-------------|
| **SOLUTION_N8N.md** | 📘 Guide complet étape par étape |
| **DIAGNOSTIC_N8N.md** | 🔍 Analyse détaillée des problèmes |
| **setup_n8n.bat** | 🚀 Installation automatique (CMD) |
| **setup_n8n.ps1** | 🚀 Installation automatique (PowerShell) |
| **start_n8n.bat** | ▶️ Démarrage rapide n8n |

---

## 📚 DOCUMENTATION EXISTANTE

| Fichier | Description |
|---------|-------------|
| **GUIDE_INSTALLATION_N8N.md** | Guide d'installation complet |
| **n8n/README.md** | Architecture des workflows |
| **n8n/test_workflows.py** | Script de test automatique |

---

## 🚀 DÉMARRAGE RAPIDE

### Première fois (installation):
```cmd
cd "C:\Users\ASUS\Desktop\VER1\PI BI NEW (2)\PI BI NEW"
setup_n8n.bat
```

### Utilisation quotidienne (3 terminaux):

**Terminal 1:**
```cmd
python -m uvicorn ML.api.main:app --reload --port 8000
```

**Terminal 2:**
```cmd
npx n8n
```

**Terminal 3:**
```cmd
python n8n/test_workflows.py
```

---

## 🌐 INTERFACES WEB

| Interface | URL | Description |
|-----------|-----|-------------|
| **n8n** | http://localhost:5678 | Orchestrateur workflows |
| **FastAPI** | http://127.0.0.1:8000/docs | API ML (Swagger) |
| **Streamlit** | http://localhost:8501 | Application ML |

---

## 🔧 DÉPANNAGE RAPIDE

### npm bloqué?
→ Utiliser CMD au lieu de PowerShell

### n8n pas installé?
→ `npm install -g n8n`

### Port 8000 refusé?
→ Lancer FastAPI dans Terminal 1

### Port 5678 occupé?
→ `netstat -ano | findstr :5678` puis `taskkill /PID <PID> /F`

### 401 Unauthorized?
→ Vérifier SQL Server et table AppUsers

---

## 📞 AIDE

**Lire dans cet ordre:**
1. **SOLUTION_N8N.md** ← Commencez ici
2. **DIAGNOSTIC_N8N.md** ← Si problèmes
3. **GUIDE_INSTALLATION_N8N.md** ← Détails complets

**Tester:**
```cmd
python n8n/test_workflows.py
```

---

## 🎉 PROCHAINE ÉTAPE

**FAITES CECI MAINTENANT:**

1. Ouvrir **CMD** (pas PowerShell)
2. Taper:
   ```cmd
   cd "C:\Users\ASUS\Desktop\VER1\PI BI NEW (2)\PI BI NEW"
   setup_n8n.bat
   ```
3. Suivre les instructions

**Temps estimé:** 10-15 minutes  
**Difficulté:** 🟢 Facile

---

## 📊 ARCHITECTURE VISUELLE

```
┌─────────────────────────────────────────────────────────┐
│                  SYSTÈME EVENTZILLA                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Terminal 1          Terminal 2          Terminal 3    │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐   │
│  │ FastAPI  │◄──────┤   n8n    │       │  Tests   │   │
│  │  :8000   │  JWT  │  :5678   │       │          │   │
│  └────┬─────┘       └────┬─────┘       └──────────┘   │
│       │                  │                             │
│       ▼                  ▼                             │
│  ┌─────────┐       ┌──────────────┐                   │
│  │   SQL   │       │  Workflows   │                   │
│  │ Server  │       │  - Marketing │                   │
│  └─────────┘       │  - Finance   │                   │
│                    │  - CRM       │                   │
│                    │  - Error     │                   │
│                    └──────┬───────┘                   │
│                           │                            │
│                           ▼                            │
│                    ┌──────────────┐                   │
│                    │   Results    │                   │
│                    │  (JSON files)│                   │
│                    └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

**Bonne chance! 🚀**

*Tous les fichiers sont prêts. Il suffit de débloquer npm et suivre les étapes.*
