#!/usr/bin/env pwsh
# ═══════════════════════════════════════════════════════════════
#  EventZilla — Lanceur S12 MLOps (PowerShell)
#  Lance séquentiellement :
#    1. Installation des dépendances
#    2. Pipeline d'entraînement (→ .joblib + runs MLflow)
#    3. MLflow UI (port 5000)
#    4. FastAPI (port 8000)
#    5. Streamlit (port 8501)
# ═══════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$PSScriptRoot_local = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot_local

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "  [$n/$total]  $msg" -ForegroundColor Cyan
}

function Write-OK($msg) {
    Write-Host "           OK — $msg" -ForegroundColor Green
}

function Write-Err($msg) {
    Write-Host "  ERREUR : $msg" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  EventZilla — Validation S12 MLOps" -ForegroundColor Magenta
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Magenta

# ── Étape 1 : Dépendances ────────────────────────────────────────────────
Write-Step 1 5 "Installation des dépendances Python..."
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Err "pip install échoué"; exit 1 }
Write-OK "Dépendances OK"

# ── Étape 2 : Pipeline ───────────────────────────────────────────────────
Write-Step 2 5 "Pipeline d'entraînement S12 (modèles + MLflow runs)..."
python run_pipeline_s12.py
if ($LASTEXITCODE -ne 0) { Write-Err "Pipeline échoué — vérifiez les logs ci-dessus"; exit 1 }
Write-OK "Modèles .joblib générés + runs MLflow créés"

# ── Étape 3 : MLflow UI ──────────────────────────────────────────────────
Write-Step 3 5 "Démarrage MLflow UI (port 5000, SQLite — Overview OK)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$PSScriptRoot_local'; python mlflow_ui_sqlite.py" `
    -WindowStyle Normal
Start-Sleep -Seconds 6
Write-OK "MLflow UI → http://localhost:5000"

# ── Étape 4 : FastAPI ────────────────────────────────────────────────────
Write-Step 4 5 "Démarrage FastAPI (port 8000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$PSScriptRoot_local'; python run_fastapi.py" `
    -WindowStyle Normal
Start-Sleep -Seconds 8
Write-OK "FastAPI → http://localhost:8000/docs"

# ── Étape 5 : Streamlit ──────────────────────────────────────────────────
Write-Step 5 5 "Démarrage Streamlit (port 8501)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$PSScriptRoot_local'; streamlit run ML/streamlit_predict.py --server.port 8501" `
    -WindowStyle Normal
Start-Sleep -Seconds 5
Write-OK "Streamlit → http://localhost:8501"

# ── Résumé ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✅  TOUS LES SERVICES SONT LANCÉS !" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  📊 MLflow UI   : http://localhost:5000" -ForegroundColor Yellow
Write-Host "  🔌 API FastAPI : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  🌐 Streamlit   : http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Pour tester l'API :" -ForegroundColor Cyan
Write-Host "    python test_api_s12.py" -ForegroundColor White
Write-Host ""

# Ouvrir les navigateurs
Start-Sleep -Seconds 2
Start-Process "http://localhost:5000"
Start-Process "http://localhost:8000/docs"
Start-Process "http://localhost:8501"

Write-Host "  Appuyez sur Entrée pour quitter..." -ForegroundColor Gray
Read-Host
