@echo off
chcp 65001 >nul
cls
echo ╔════════════════════════════════════════════════════════╗
echo ║     LANCEMENT COMPLET - EventZilla n8n + FastAPI      ║
echo ╚════════════════════════════════════════════════════════╝
echo.

:: Vérifier que nous sommes dans le bon dossier
if not exist "ML\api\main.py" (
    echo [ERREUR] Fichier ML\api\main.py introuvable
    echo [INFO]   Assurez-vous d'etre dans le dossier "PI BI NEW"
    pause
    exit /b 1
)

echo [INFO] Verification des composants...
echo.

:: Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe
    pause
    exit /b 1
)
echo [OK] Python installe

:: Vérifier npm
call npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] npm n'est pas accessible
    pause
    exit /b 1
)
echo [OK] npm accessible

:: Vérifier n8n
call npx n8n --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] n8n n'est pas installe
    echo [INFO]   Installation de n8n...
    call npm install -g n8n
)
echo [OK] n8n installe

echo.
echo ════════════════════════════════════════════════════════
echo   LANCEMENT DES SERVICES
echo ════════════════════════════════════════════════════════
echo.
echo Ce script va ouvrir 2 fenetres :
echo   1. FastAPI (port 8000)
echo   2. n8n (port 5678)
echo.
echo Appuyez sur une touche pour continuer...
pause >nul

:: Lancer FastAPI dans une nouvelle fenêtre
echo.
echo [ETAPE 1/2] Lancement de FastAPI...
start "EventZilla - FastAPI" cmd /k "python -m uvicorn ML.api.main:app --reload --port 8000"
timeout /t 3 >nul
echo [OK] FastAPI demarre dans une nouvelle fenetre

:: Lancer n8n dans une nouvelle fenêtre
echo.
echo [ETAPE 2/2] Lancement de n8n...
start "EventZilla - n8n" cmd /k "npx n8n"
timeout /t 3 >nul
echo [OK] n8n demarre dans une nouvelle fenetre

echo.
echo ════════════════════════════════════════════════════════
echo   SERVICES LANCES !
echo ════════════════════════════════════════════════════════
echo.
echo Interfaces Web :
echo   • FastAPI Swagger : http://127.0.0.1:8000/docs
echo   • n8n Interface  : http://localhost:5678
echo.
echo Prochaines etapes :
echo   1. Ouvrir http://localhost:5678 dans votre navigateur
echo   2. Creer un compte n8n (premiere fois)
echo   3. Importer les 4 workflows depuis le dossier n8n/
echo   4. Tester avec : python n8n/test_workflows.py
echo.
echo Pour arreter les services :
echo   - Fermez les 2 fenetres qui se sont ouvertes
echo   - OU appuyez sur Ctrl+C dans chaque fenetre
echo.
echo ════════════════════════════════════════════════════════
echo.
pause
