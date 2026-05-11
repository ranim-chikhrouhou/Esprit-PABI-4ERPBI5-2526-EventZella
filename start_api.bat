@echo off
echo ========================================
echo   Demarrage FastAPI - EventZilla
echo ========================================
echo.
echo API disponible sur : http://127.0.0.1:8000
echo Documentation      : http://127.0.0.1:8000/docs
echo.

REM Evite WinError 10022 sous Windows avec --reload (voir run_fastapi.py).
REM Pour activer le reload : set UVICORN_RELOAD=1 puis relancer ce .bat
python run_fastapi.py
