@echo off
echo ========================================
echo   Demarrage n8n - EventZilla
echo ========================================
echo.

:: Charger les variables du fichier .env
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
)

set N8N_RESTRICT_FILE_ACCESS_TO=C:\Users\ranim\Downloads\PI BI NEW\n8n\results

echo Interface n8n : http://localhost:5678
echo.

npx n8n
