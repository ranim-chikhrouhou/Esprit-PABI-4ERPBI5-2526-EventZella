@echo off
chcp 65001 >nul
cls
echo ════════════════════════════════════════════════════════
echo   TEST RAPIDE - Composants n8n
echo ════════════════════════════════════════════════════════
echo.

echo [TEST 1/5] Node.js...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('node --version') do echo   [OK] Node.js : %%i
) else (
    echo   [ERREUR] Node.js non installe
)

echo.
echo [TEST 2/5] npm...
call npm --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('call npm --version') do echo   [OK] npm : v%%i
) else (
    echo   [ERREUR] npm non accessible
)

echo.
echo [TEST 3/5] n8n...
call npx n8n --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('call npx n8n --version') do echo   [OK] n8n : %%i
) else (
    echo   [ERREUR] n8n non installe
    echo   [INFO]   Lancez : npm install -g n8n
)

echo.
echo [TEST 4/5] Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version') do echo   [OK] Python : %%i
) else (
    echo   [ERREUR] Python non installe
)

echo.
echo [TEST 5/5] FastAPI...
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] FastAPI et uvicorn installes
) else (
    echo   [ERREUR] FastAPI ou uvicorn manquant
    echo   [INFO]   Lancez : pip install fastapi uvicorn
)

echo.
echo ════════════════════════════════════════════════════════
echo   FICHIERS WORKFLOWS
echo ════════════════════════════════════════════════════════
echo.

if exist "n8n\workflow_marketing.json" (
    echo   [OK] workflow_marketing.json
) else (
    echo   [MANQUANT] workflow_marketing.json
)

if exist "n8n\workflow_finance.json" (
    echo   [OK] workflow_finance.json
) else (
    echo   [MANQUANT] workflow_finance.json
)

if exist "n8n\workflow_crm.json" (
    echo   [OK] workflow_crm.json
) else (
    echo   [MANQUANT] workflow_crm.json
)

if exist "n8n\workflow_error_handler.json" (
    echo   [OK] workflow_error_handler.json
) else (
    echo   [MANQUANT] workflow_error_handler.json
)

echo.
echo ════════════════════════════════════════════════════════
echo   RESULTAT
echo ════════════════════════════════════════════════════════
echo.
echo Si tous les tests sont [OK], vous pouvez lancer :
echo   LANCER_N8N_COMPLET.bat
echo.
echo Si des tests sont [ERREUR], installez les composants manquants.
echo.
pause
