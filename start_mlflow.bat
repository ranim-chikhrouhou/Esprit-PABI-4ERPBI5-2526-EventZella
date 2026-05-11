@echo off
REM Start MLflow Tracking Server
REM This script launches the MLflow UI for experiment tracking

echo ========================================
echo Starting MLflow Tracking Server
echo ========================================
echo.
echo MLflow UI will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start MLflow server with local file storage
mlflow ui --host 0.0.0.0 --port 5000

pause
