@echo off
chcp 65001 >nul
echo ========================================
echo   Installation n8n - EventZilla
echo ========================================
echo.

:: Vérifier Node.js
echo [ETAPE] Verification de Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Node.js n'est pas installe
    echo [INFO]   Telechargez depuis : https://nodejs.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [OK]    Node.js installe : %NODE_VERSION%
echo.

:: Vérifier npm (utiliser CMD au lieu de PowerShell)
echo [ETAPE] Verification de npm...
call npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] npm n'est pas accessible
    echo [INFO]   Redemarrez l'ordinateur et relancez ce script
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('call npm --version') do set NPM_VERSION=%%i
echo [OK]    npm installe : v%NPM_VERSION%
echo.

:: Vérifier n8n
echo [ETAPE] Verification de n8n...
call npx n8n --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO]   n8n n'est pas installe. Installation en cours...
    echo [INFO]   Cela peut prendre quelques minutes...
    call npm install -g n8n
    if %errorlevel% neq 0 (
        echo [ERREUR] Echec de l'installation de n8n
        echo [INFO]   Essayez manuellement : npm install -g n8n
        pause
        exit /b 1
    )
    echo [OK]    n8n installe avec succes
) else (
    for /f "tokens=*" %%i in ('call npx n8n --version') do set N8N_VERSION=%%i
    echo [OK]    n8n deja installe : %N8N_VERSION%
)
echo.

:: Vérifier Python
echo [ETAPE] Verification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK]    Python installe : %PYTHON_VERSION%
echo.

:: Créer le dossier results
echo [ETAPE] Creation du dossier results...
if not exist "n8n\results" (
    mkdir "n8n\results"
    echo [OK]    Dossier cree : n8n\results
) else (
    echo [OK]    Dossier existe : n8n\results
)
echo.

:: Vérifier les workflows
echo [ETAPE] Verification des fichiers workflows...
set ALL_OK=1
if exist "n8n\workflow_marketing.json" (
    echo [OK]    Trouve : n8n\workflow_marketing.json
) else (
    echo [ERREUR] Manquant : n8n\workflow_marketing.json
    set ALL_OK=0
)
if exist "n8n\workflow_finance.json" (
    echo [OK]    Trouve : n8n\workflow_finance.json
) else (
    echo [ERREUR] Manquant : n8n\workflow_finance.json
    set ALL_OK=0
)
if exist "n8n\workflow_crm.json" (
    echo [OK]    Trouve : n8n\workflow_crm.json
) else (
    echo [ERREUR] Manquant : n8n\workflow_crm.json
    set ALL_OK=0
)
if exist "n8n\workflow_error_handler.json" (
    echo [OK]    Trouve : n8n\workflow_error_handler.json
) else (
    echo [ERREUR] Manquant : n8n\workflow_error_handler.json
    set ALL_OK=0
)
echo.

:: Résumé
echo ========================================
echo   RESUME DE L'INSTALLATION
echo ========================================
echo.
echo Composants installes :
echo   [OK] Node.js : %NODE_VERSION%
echo   [OK] npm     : v%NPM_VERSION%
echo   [OK] n8n     : installe
echo   [OK] Python  : %PYTHON_VERSION%
echo.
echo Prochaines etapes :
echo.
echo 1. Lancer FastAPI (Terminal 1) :
echo    python -m uvicorn ML.api.main:app --reload --port 8000
echo.
echo 2. Lancer n8n (Terminal 2) :
echo    npx n8n
echo    OU
echo    start_n8n.bat
echo.
echo 3. Ouvrir n8n dans le navigateur :
echo    http://localhost:5678
echo.
echo 4. Importer les 4 workflows dans n8n :
echo    - workflow_marketing.json
echo    - workflow_finance.json
echo    - workflow_crm.json
echo    - workflow_error_handler.json
echo.
echo 5. Tester les workflows :
echo    python n8n/test_workflows.py
echo.
echo Documentation complete :
echo   - GUIDE_INSTALLATION_N8N.md
echo   - DIAGNOSTIC_N8N.md
echo   - n8n/README.md
echo.
echo ========================================
echo   Installation terminee avec succes !
echo ========================================
echo.
pause
