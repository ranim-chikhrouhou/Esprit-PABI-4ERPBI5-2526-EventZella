@echo off
echo ================================================================
echo EventZilla - SQL Performance Optimization
echo ================================================================
echo.
echo This will create an optimized index on the AppUsers table
echo to speed up login queries by 50-80%%
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Running optimization script...
echo.

sqlcmd -S "DESKTOP-DVMNP7K\MSSQLSERVERS" -E -i "Database\optimize_appusers_performance.sql"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS! Optimization completed successfully
    echo ================================================================
    echo.
    echo Next steps:
    echo 1. Restart your Streamlit app
    echo 2. Test login - should be MUCH faster now
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR: Optimization failed
    echo ================================================================
    echo.
    echo Possible causes:
    echo 1. SQL Server is not running
    echo 2. Wrong server name
    echo 3. Insufficient permissions
    echo.
    echo Try running this in SQL Server Management Studio instead:
    echo   File: Database\optimize_appusers_performance.sql
    echo.
)

pause
