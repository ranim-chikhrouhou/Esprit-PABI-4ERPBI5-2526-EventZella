@echo off
chcp 65001 > nul
echo.
echo ========================================
echo   Apache Airflow - EventZella ETL Audit
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Verification Docker...
docker --version > nul 2>&1
if errorlevel 1 (
    echo ERREUR : Docker n'est pas lance. Ouvrez Docker Desktop d'abord.
    pause
    exit /b 1
)
echo       Docker OK

echo.
echo [2/3] Demarrage Airflow (PostgreSQL + Scheduler + Webserver)...
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler

echo.
echo [3/3] Attente demarrage (30 secondes)...
timeout /t 30 /nobreak > nul

echo.
echo ========================================
echo   Airflow demarre avec succes !
echo ========================================
echo.
echo   Interface web : http://localhost:8088
echo   Login         : admin
echo   Password      : admin
echo.
echo   DAGs disponibles :
echo     - eventzella_master_sa      (chaque jour 02h00)
echo     - eventzella_master_etl     (chaque jour 04h00)
echo     - eventzella_master_global  (chaque dimanche 01h00)
echo.
echo Pour arreter Airflow : docker compose down
echo.
pause
