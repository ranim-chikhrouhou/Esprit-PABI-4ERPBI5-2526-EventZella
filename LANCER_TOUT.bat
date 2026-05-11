@echo off
REM EventZilla - Complete Project Launcher (including MLflow)
REM This script launches all services: FastAPI, n8n, Streamlit, and MLflow

title EventZilla - Complete Launcher

echo ========================================
echo EventZilla - Complete Project Launcher
echo ========================================
echo.
echo This will start ALL services:
echo   1. FastAPI (port 8000)
echo   2. n8n (port 5678)
echo   3. Streamlit (port 8502)
echo   4. MLflow (port 5000)
echo.
echo Press Ctrl+C to cancel, or
pause
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo [1/4] Starting FastAPI...
echo ========================================
start "EventZilla - FastAPI" cmd /k "echo FastAPI Server && echo URL: http://localhost:8000 && echo Docs: http://localhost:8000/docs && echo. && python -m uvicorn ML.api.main:app --reload --port 8000"
timeout /t 5 /nobreak >nul
echo FastAPI started on port 8000
echo.

echo ========================================
echo [2/4] Starting n8n...
echo ========================================
start "EventZilla - n8n" cmd /k "echo n8n Workflow Automation && echo URL: http://localhost:5678 && echo. && npx n8n"
timeout /t 5 /nobreak >nul
echo n8n started on port 5678
echo.

echo ========================================
echo [3/4] Starting Streamlit...
echo ========================================
start "EventZilla - Streamlit" cmd /k "echo Streamlit ML Dashboard && echo URL: http://localhost:8502 && echo. && python -m streamlit run ML/streamlit_app.py"
timeout /t 5 /nobreak >nul
echo Streamlit started on port 8502
echo.

echo ========================================
echo [4/4] Starting MLflow...
echo ========================================
start "EventZilla - MLflow" cmd /k "echo MLflow Tracking Server && echo URL: http://localhost:5000 && echo. && mlflow ui --host 0.0.0.0 --port 5000"
timeout /t 3 /nobreak >nul
echo MLflow started on port 5000
echo.

echo ========================================
echo All Services Started Successfully!
echo ========================================
echo.
echo Access your services at:
echo   FastAPI:   http://localhost:8000
echo   n8n:       http://localhost:5678
echo   Streamlit: http://localhost:8502
echo   MLflow:    http://localhost:5000
echo.
echo Press any key to close this window...
echo (Services will continue running in their own windows)
pause >nul
