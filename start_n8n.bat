@echo off
chcp 65001 >nul
echo ========================================
echo   Demarrage n8n - EventZilla
echo ========================================
echo.

:: Charger les variables du fichier .env
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
    )
)

set N8N_RESTRICT_FILE_ACCESS_TO=%~dp0n8n\results

echo [INFO] Interface n8n : http://localhost:5678
echo [INFO] Appuyez sur Ctrl+C pour arreter n8n
echo.

:: Utiliser call pour éviter les problèmes avec npx
call npx n8n
