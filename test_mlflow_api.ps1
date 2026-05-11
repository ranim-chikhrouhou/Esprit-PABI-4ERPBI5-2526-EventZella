# Test MLflow API Endpoint
# This script will create a test run in MLflow

Write-Host "🔐 Step 1: Login to get JWT token..." -ForegroundColor Cyan

$loginBody = @{
    login = "naima_sarraj"
    password = "Naima@Finance2025!"
} | ConvertTo-Json

try {
    $loginResponse = Invoke-RestMethod -Uri "http://localhost:8000/auth/login" -Method Post -Body $loginBody -ContentType "application/json"
    $token = $loginResponse.access_token
    Write-Host "✅ Login successful! Token received." -ForegroundColor Green
} catch {
    Write-Host "❌ Login failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📊 Step 2: Logging test run to MLflow..." -ForegroundColor Cyan

$headers = @{
    "Authorization" = "Bearer $token"
    "Content-Type" = "application/json"
}

$mlflowBody = @{
    experiment_name = "n8n_Finance_Pipeline"
    run_name = "test_finance_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    params = @{
        workflow = "finance"
        user = "naima_sarraj"
        model_regression = "Ridge"
        model_timeseries = "Holt"
    }
    metrics = @{
        predicted_amount = 1450.75
        timeseries_mape = 6.1
        timeseries_rmse = 245.3
        timeseries_mae = 189.4
    }
    tags = @{
        source = "powershell_test"
        pipeline = "finance"
        automated = "true"
    }
} | ConvertTo-Json -Depth 10

try {
    $mlflowResponse = Invoke-RestMethod -Uri "http://localhost:8000/mlflow/log_prediction" -Method Post -Headers $headers -Body $mlflowBody
    Write-Host "✅ MLflow logging successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Response:" -ForegroundColor Yellow
    Write-Host "   Run ID: $($mlflowResponse.mlflow_run_id)" -ForegroundColor White
    Write-Host "   Experiment ID: $($mlflowResponse.mlflow_experiment_id)" -ForegroundColor White
    Write-Host ""
    Write-Host "🔗 View in MLflow UI:" -ForegroundColor Cyan
    Write-Host "   $($mlflowResponse.mlflow_ui)" -ForegroundColor Blue
    Write-Host ""
    Write-Host "🌐 Or go to: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "   Then click on 'n8n_Finance_Pipeline' experiment" -ForegroundColor White
} catch {
    Write-Host "❌ MLflow logging failed: $_" -ForegroundColor Red
    Write-Host "Response: $($_.Exception.Response)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✨ Test completed successfully!" -ForegroundColor Green
