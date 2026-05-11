@echo off
chcp 65001 >nul
title EventZilla S12 — Lanceur MLOps

echo.
echo ═══════════════════════════════════════════════════════════
echo    EventZilla — Lanceur S12 MLOps (Validation)
echo    Tous les services seront demarres dans des fenetres
echo    separees. Attendez chaque etape avant de continuer.
echo ═══════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"

echo [ETAPE 1/5]  Installation des dependances Python...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR: Installation des dependances echouee.
    pause
    exit /b 1
)
echo             OK - Dependances installees.
echo.

echo [ETAPE 2/5]  Execution du Pipeline d'entrainement S12...
echo             (Generation des donnees + modeles .joblib + runs MLflow)
python run_pipeline_s12.py
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR: Pipeline d'entrainement echoue.
    pause
    exit /b 1
)
echo             OK - Modeles et runs MLflow generes.
echo.

echo [ETAPE 3/5]  Demarrage du serveur MLflow (port 5000, SQLite — Overview OK)...
start "MLflow UI - Port 5000" cmd /k "cd /d "%~dp0" && python mlflow_ui_sqlite.py"
echo             Ouverture dans 5 secondes...
timeout /t 5 /nobreak >nul
echo             OK - MLflow UI: http://localhost:5000
echo.

echo [ETAPE 4/5]  Demarrage de l'API FastAPI (port 8000)...
start "FastAPI - Port 8000" cmd /k "cd /d "%~dp0" && python run_fastapi.py"
echo             Ouverture dans 8 secondes...
timeout /t 8 /nobreak >nul
echo             OK - API Docs: http://localhost:8000/docs
echo.

echo [ETAPE 5/5]  Demarrage de l'interface Streamlit (port 8501)...
start "Streamlit - Port 8501" cmd /k "cd /d "%~dp0" && streamlit run ML/streamlit_predict.py --server.port 8501"
echo             Ouverture dans 5 secondes...
timeout /t 5 /nobreak >nul
echo.

echo ═══════════════════════════════════════════════════════════
echo    TOUS LES SERVICES SONT LANCES !
echo ═══════════════════════════════════════════════════════════
echo.
echo    MLflow UI   : http://localhost:5000
echo    API FastAPI : http://localhost:8000/docs
echo    Streamlit   : http://localhost:8501
echo.
echo    Pour tester l'API : python test_api_s12.py
echo.

echo Ouverture automatique des navigateurs...
timeout /t 3 /nobreak >nul
start "" "http://localhost:5000"
start "" "http://localhost:8000/docs"
start "" "http://localhost:8501"

echo.
echo Appuyez sur une touche pour fermer ce lanceur...
pause >nul
