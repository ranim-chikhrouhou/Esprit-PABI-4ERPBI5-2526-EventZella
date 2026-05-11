@echo off
echo ========================================
echo Testing MLflow API Endpoint
echo ========================================
echo.

echo Step 1: Login to get JWT token...
curl -X POST "http://localhost:8000/auth/login" ^
  -H "Content-Type: application/json" ^
  -d "{\"login\":\"naima_sarraj\",\"password\":\"Naima@Finance2025!\"}" ^
  -o login_response.json

echo.
echo Step 2: Extract token and log to MLflow...
echo (Manual step - copy token from login_response.json)
echo.

echo ========================================
echo Alternative: Direct MLflow Test
echo ========================================
echo.
echo You can also test by calling the n8n workflow directly in n8n UI
echo Go to: http://localhost:5678
echo Find: "EventZilla — Finance Pipeline with MLflow (Simple)"
echo Click: "Execute Workflow" button
echo.

pause
