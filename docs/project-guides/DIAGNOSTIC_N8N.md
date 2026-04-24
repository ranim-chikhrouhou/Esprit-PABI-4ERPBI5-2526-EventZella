# 🔍 DIAGNOSTIC N8N - EventZilla

## ✅ CE QUI FONCTIONNE

1. **Node.js installé** : v22.22.0 ✅
2. **Structure des fichiers** :
   - ✅ 4 workflows JSON bien structurés
   - ✅ Script de test Python complet
   - ✅ Documentation complète (README.md)
   - ✅ FastAPI avec tous les endpoints ML
   - ✅ Script de démarrage (start_n8n.bat)

## ❌ PROBLÈMES IDENTIFIÉS

### 🔴 PROBLÈME 1: PowerShell Execution Policy (CRITIQUE)

**Symptôme:**
```
npm : Impossible de charger le fichier C:\Program Files\nodejs\npm.ps1, car 
l'exécution de scripts est désactivée sur ce système.
```

**Cause:** Windows bloque l'exécution des scripts PowerShell par défaut pour des raisons de sécurité.

**Impact:** Impossible d'utiliser `npm`, `npx`, donc impossible de lancer n8n.

**Solution:**
```powershell
# Option 1: Pour l'utilisateur actuel (RECOMMANDÉ)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Option 2: Pour la session actuelle seulement
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

---

### 🟡 PROBLÈME 2: n8n pas installé (PROBABLE)

**Cause:** n8n n'a jamais été installé sur le système.

**Solution:**
```powershell
# Après avoir résolu le Problème 1, installer n8n
npm install -g n8n
```

---

### 🟡 PROBLÈME 3: Workflows pas importés dans n8n

**Cause:** Les fichiers JSON existent mais ne sont pas chargés dans l'interface n8n.

**Solution:** Après avoir lancé n8n, importer manuellement les 4 workflows.

---

### 🟡 PROBLÈME 4: FastAPI pas lancée

**Cause:** L'API backend doit tourner pour que les workflows fonctionnent.

**Solution:**
```powershell
python -m uvicorn ML.api.main:app --reload --port 8000
```

---

## 📋 PLAN DE RÉSOLUTION (ÉTAPES DANS L'ORDRE)

### ÉTAPE 1: Débloquer PowerShell ⚠️ CRITIQUE

**Action:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Vérification:**
```powershell
Get-ExecutionPolicy -List
# CurrentUser devrait afficher "RemoteSigned"
```

---

### ÉTAPE 2: Vérifier npm fonctionne

**Action:**
```powershell
npm --version
```

**Résultat attendu:** Affiche un numéro de version (ex: 10.x.x)

---

### ÉTAPE 3: Installer n8n

**Action:**
```powershell
npm install -g n8n
```

**Vérification:**
```powershell
n8n --version
```

**Résultat attendu:** Affiche la version de n8n (ex: 1.x.x)

---

### ÉTAPE 4: Lancer FastAPI (Terminal 1)

**Action:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
python -m uvicorn ML.api.main:app --reload --port 8000
```

**Vérification:** Ouvrir http://127.0.0.1:8000/docs dans le navigateur

**Résultat attendu:** Interface Swagger avec tous les endpoints

---

### ÉTAPE 5: Lancer n8n (Terminal 2)

**Action:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
npx n8n
```

**Vérification:** Ouvrir http://localhost:5678 dans le navigateur

**Résultat attendu:** Interface n8n avec écran de connexion/création de compte

---

### ÉTAPE 6: Créer un compte n8n (première fois)

**Action:**
1. Ouvrir http://localhost:5678
2. Créer un compte avec email et mot de passe
3. Se connecter

---

### ÉTAPE 7: Importer les 4 workflows

**Action pour chaque workflow:**
1. Cliquer sur "+" → "New Workflow"
2. Cliquer sur "..." (3 points en haut à droite)
3. Sélectionner "Import from File"
4. Choisir le fichier:
   - `n8n/workflow_marketing.json`
   - `n8n/workflow_finance.json`
   - `n8n/workflow_crm.json`
   - `n8n/workflow_error_handler.json`
5. Cliquer "Save"

---

### ÉTAPE 8: Configurer Error Handler

**Action pour chaque workflow (Marketing, Finance, CRM):**
1. Ouvrir le workflow
2. Cliquer sur "Workflow Settings" (icône engrenage)
3. Dans "Error Workflow", sélectionner "EventZilla — Error Handler"
4. Sauvegarder

---

### ÉTAPE 9: Tester les workflows

**Test manuel:**
1. Ouvrir un workflow (ex: Marketing)
2. Cliquer "Execute Workflow" (bouton en haut)
3. Vérifier que tous les nœuds sont verts ✅

**Test automatique:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
python n8n/test_workflows.py
```

**Résultat attendu:** Tous les tests affichent "[OK]"

---

## 🎯 CHECKLIST FINALE

Avant de considérer n8n comme opérationnel:

- [ ] PowerShell execution policy = RemoteSigned
- [ ] `npm --version` fonctionne
- [ ] `n8n --version` fonctionne
- [ ] FastAPI accessible sur http://127.0.0.1:8000/docs
- [ ] n8n accessible sur http://localhost:5678
- [ ] Compte n8n créé et connecté
- [ ] 4 workflows importés et visibles dans n8n
- [ ] Error Handler configuré dans les 3 workflows
- [ ] Test manuel d'un workflow réussi (tous les nœuds verts)
- [ ] `python n8n/test_workflows.py` affiche "[OK]" partout
- [ ] Fichiers JSON créés dans `n8n/results/`

---

## 🚀 COMMANDES RAPIDES (APRÈS RÉSOLUTION)

### Démarrage quotidien (3 terminaux):

**Terminal 1 - FastAPI:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
python -m uvicorn ML.api.main:app --reload --port 8000
```

**Terminal 2 - n8n:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
npx n8n
```

**Terminal 3 - Tests:**
```powershell
cd "PI BI NEW (2)/PI BI NEW"
python n8n/test_workflows.py
```

---

## 📞 DÉPANNAGE

### Si npm ne fonctionne toujours pas après Set-ExecutionPolicy:

**Solution alternative:** Utiliser CMD au lieu de PowerShell
```cmd
npm --version
npm install -g n8n
n8n
```

### Si n8n ne démarre pas:

**Vérifier le port 5678:**
```powershell
netstat -ano | findstr :5678
```

**Si occupé, tuer le processus:**
```powershell
taskkill /PID <PID> /F
```

### Si FastAPI ne démarre pas:

**Vérifier les dépendances:**
```powershell
pip install fastapi uvicorn pydantic joblib pandas numpy scikit-learn
```

**Vérifier SQL Server:**
```powershell
python ML/scripts/run_test_sql_connection.py
```

---

## 📊 ARCHITECTURE COMPLÈTE

```
┌─────────────────────────────────────────────────────────┐
│                    SYSTÈME N8N                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Terminal 1 │      │   Terminal 2 │               │
│  │              │      │              │               │
│  │   FastAPI    │◄─────┤     n8n      │               │
│  │   :8000      │      │   :5678      │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                       │
│         │                      │                       │
│         ▼                      ▼                       │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  SQL Server  │      │  Workflows   │               │
│  │  (AppUsers)  │      │  - Marketing │               │
│  │              │      │  - Finance   │               │
│  └──────────────┘      │  - CRM       │               │
│                        │  - Error     │               │
│                        └──────────────┘               │
│                               │                        │
│                               ▼                        │
│                        ┌──────────────┐               │
│                        │   Results    │               │
│                        │  (JSON files)│               │
│                        └──────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ STATUT ACTUEL

| Composant | Statut | Action requise |
|-----------|--------|----------------|
| Node.js | ✅ Installé (v22.22.0) | Aucune |
| npm | ❌ Bloqué | Débloquer PowerShell |
| n8n | ❓ Inconnu | Installer après déblocage |
| FastAPI | ✅ Code prêt | Lancer |
| Workflows JSON | ✅ Prêts | Importer dans n8n |
| Documentation | ✅ Complète | Aucune |

---

## 🎉 PROCHAINE ÉTAPE

**COMMENCEZ PAR CECI:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Puis suivez les étapes 2 à 9 dans l'ordre.
