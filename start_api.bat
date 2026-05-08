@echo off
echo ========================================
echo   Demarrage FastAPI - EventZilla
echo ========================================
echo.
echo API disponible sur : http://127.0.0.1:8000
echo Documentation      : http://127.0.0.1:8000/docs
echo.

python -m uvicorn ML.api.main:app --reload --port 8000
