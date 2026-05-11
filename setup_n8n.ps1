# ========================================
#   Script d'installation automatique n8n
#   EventZilla ML System
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation n8n - EventZilla" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Fonction pour afficher les messages
function Write-Step {
    param([string]$Message)
    Write-Host "[ETAPE] $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK]    $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERREUR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO]  $Message" -ForegroundColor Cyan
}

# ========================================
# ETAPE 1: Vérifier Node.js
# ========================================
Write-Step "Verification de Node.js..."
try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Success "Node.js installe : $nodeVersion"
    } else {
        Write-Error "Node.js n'est pas installe"
        Write-Info "Telechargez Node.js depuis : https://nodejs.org/"
        exit 1
    }
} catch {
    Write-Error "Node.js n'est pas installe"
    Write-Info "Telechargez Node.js depuis : https://nodejs.org/"
    exit 1
}

# ========================================
# ETAPE 2: Vérifier/Configurer Execution Policy
# ========================================
Write-Step "Verification de la politique d'execution PowerShell..."
$currentPolicy = Get-ExecutionPolicy -Scope CurrentUser

if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "Undefined") {
    Write-Info "Politique actuelle : $currentPolicy"
    Write-Info "Modification de la politique d'execution..."
    
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Write-Success "Politique d'execution modifiee : RemoteSigned"
    } catch {
        Write-Error "Impossible de modifier la politique d'execution"
        Write-Info "Executez manuellement : Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
        exit 1
    }
} else {
    Write-Success "Politique d'execution OK : $currentPolicy"
}

# ========================================
# ETAPE 3: Vérifier npm
# ========================================
Write-Step "Verification de npm..."
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Success "npm installe : v$npmVersion"
    } else {
        Write-Error "npm n'est pas accessible"
        Write-Info "Redemarrez PowerShell et relancez ce script"
        exit 1
    }
} catch {
    Write-Error "npm n'est pas accessible"
    Write-Info "Redemarrez PowerShell et relancez ce script"
    exit 1
}

# ========================================
# ETAPE 4: Vérifier si n8n est installé
# ========================================
Write-Step "Verification de n8n..."
try {
    $n8nVersion = n8n --version 2>$null
    if ($n8nVersion) {
        Write-Success "n8n deja installe : $n8nVersion"
    } else {
        throw "n8n non installe"
    }
} catch {
    Write-Info "n8n n'est pas installe. Installation en cours..."
    Write-Info "Cela peut prendre quelques minutes..."
    
    try {
        npm install -g n8n
        $n8nVersion = n8n --version 2>$null
        Write-Success "n8n installe avec succes : $n8nVersion"
    } catch {
        Write-Error "Echec de l'installation de n8n"
        Write-Info "Essayez manuellement : npm install -g n8n"
        exit 1
    }
}

# ========================================
# ETAPE 5: Vérifier Python et dépendances
# ========================================
Write-Step "Verification de Python..."
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Success "Python installe : $pythonVersion"
    } else {
        Write-Error "Python n'est pas installe"
        exit 1
    }
} catch {
    Write-Error "Python n'est pas installe"
    exit 1
}

# ========================================
# ETAPE 6: Vérifier les dépendances Python
# ========================================
Write-Step "Verification des dependances Python..."
$requiredPackages = @("fastapi", "uvicorn", "pydantic", "joblib", "pandas", "numpy", "scikit-learn", "requests")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    $installed = python -c "import $package" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Info "Packages manquants : $($missingPackages -join ', ')"
    Write-Info "Installation recommandee : pip install $($missingPackages -join ' ')"
} else {
    Write-Success "Toutes les dependances Python sont installees"
}

# ========================================
# ETAPE 7: Vérifier les fichiers workflows
# ========================================
Write-Step "Verification des fichiers workflows..."
$workflowFiles = @(
    "n8n/workflow_marketing.json",
    "n8n/workflow_finance.json",
    "n8n/workflow_crm.json",
    "n8n/workflow_error_handler.json"
)

$allWorkflowsExist = $true
foreach ($file in $workflowFiles) {
    if (Test-Path $file) {
        Write-Success "Trouve : $file"
    } else {
        Write-Error "Manquant : $file"
        $allWorkflowsExist = $false
    }
}

# ========================================
# ETAPE 8: Créer le dossier results
# ========================================
Write-Step "Creation du dossier results..."
$resultsDir = "n8n/results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
    Write-Success "Dossier cree : $resultsDir"
} else {
    Write-Success "Dossier existe : $resultsDir"
}

# ========================================
# RESUME
# ========================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RESUME DE L'INSTALLATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Composants installes :" -ForegroundColor White
Write-Host "  [OK] Node.js : $nodeVersion" -ForegroundColor Green
Write-Host "  [OK] npm     : v$npmVersion" -ForegroundColor Green
Write-Host "  [OK] n8n     : $n8nVersion" -ForegroundColor Green
Write-Host "  [OK] Python  : $pythonVersion" -ForegroundColor Green
Write-Host ""

Write-Host "Prochaines etapes :" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Lancer FastAPI (Terminal 1) :" -ForegroundColor White
Write-Host "   python -m uvicorn ML.api.main:app --reload --port 8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Lancer n8n (Terminal 2) :" -ForegroundColor White
Write-Host "   npx n8n" -ForegroundColor Cyan
Write-Host "   OU" -ForegroundColor White
Write-Host "   .\start_n8n.bat" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Ouvrir n8n dans le navigateur :" -ForegroundColor White
Write-Host "   http://localhost:5678" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Importer les 4 workflows dans n8n :" -ForegroundColor White
Write-Host "   - workflow_marketing.json" -ForegroundColor Cyan
Write-Host "   - workflow_finance.json" -ForegroundColor Cyan
Write-Host "   - workflow_crm.json" -ForegroundColor Cyan
Write-Host "   - workflow_error_handler.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "5. Tester les workflows :" -ForegroundColor White
Write-Host "   python n8n/test_workflows.py" -ForegroundColor Cyan
Write-Host ""

Write-Host "Documentation complete :" -ForegroundColor Yellow
Write-Host "  - GUIDE_INSTALLATION_N8N.md" -ForegroundColor Cyan
Write-Host "  - DIAGNOSTIC_N8N.md" -ForegroundColor Cyan
Write-Host "  - n8n/README.md" -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation terminee avec succes !" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
