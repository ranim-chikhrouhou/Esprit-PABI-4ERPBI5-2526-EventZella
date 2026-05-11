# 🔗 n8n + MLflow Integration Guide

## Overview

This guide shows you how to integrate your n8n workflows with MLflow to automatically track all ML predictions.

---

## 🎯 What You'll Get

When integrated, every n8n workflow execution will:
- ✅ Log predictions to MLflow
- ✅ Track model parameters
- ✅ Record metrics (accuracy, MAPE, RMSE, etc.)
- ✅ Store prediction artifacts
- ✅ Enable experiment comparison
- ✅ Provide audit trail

---

## 📦 Files Created

| File | Purpose |
|------|---------|
| `ML/api/mlflow_integration.py` | MLflow logging utilities |
| `ML/api/mlflow_endpoints.py` | FastAPI endpoints for MLflow |
| `n8n/workflow_finance_with_mlflow.json` | Enhanced finance workflow |
| `N8N_MLFLOW_INTEGRATION.md` | This guide |

---

## 🚀 Setup (3 Steps)

### **Step 1: Add MLflow Endpoints to FastAPI**

Open `ML/api/main.py` and add these lines:

```python
# At the top with other imports
from ML.api.mlflow_endpoints import router as mlflow_router

# After creating the app
app.include_router(mlflow_router)
```

**Complete example:**
```python
from fastapi import FastAPI
from ML.api.mlflow_endpoints import router as mlflow_router

app = FastAPI(title="EventZilla ML API")

# Add MLflow routes
app.include_router(mlflow_router)

# ... rest of your code
```

### **Step 2: Restart FastAPI**

Stop and restart your FastAPI server to load the new endpoints:
```bash
# Stop current FastAPI (Ctrl+C in terminal)
# Then restart:
python -m uvicorn ML.api.main:app --reload --port 8000
```

### **Step 3: Import Enhanced n8n Workflow**

1. Open n8n: http://localhost:5678
2. Click "Add workflow" → "Import from File"
3. Select: `n8n/workflow_finance_with_mlflow.json`
4. Activate the workflow

---

## 📊 New API Endpoints

### **1. Generic Logging**
```http
POST /mlflow/log_prediction
Authorization: Bearer {JWT_TOKEN}

{
  "experiment_name": "n8n_Finance_Pipeline",
  "run_name": "finance_2026-05-01",
  "params": {
    "workflow": "finance",
    "user": "naima_sarraj"
  },
  "metrics": {
    "predicted_amount": 1450.75
  },
  "tags": {
    "source": "n8n"
  }
}
```

### **2. Finance Pipeline**
```http
POST /mlflow/log_finance
Authorization: Bearer {JWT_TOKEN}

{
  "regression": {
    "montant_predit": 1450.75,
    "modele": "Ridge"
  },
  "timeseries": {
    "modele_champion": "Holt",
    "metriques_test": {
      "MAPE": 6.1,
      "RMSE": 245.3
    }
  }
}
```

### **3. Marketing Pipeline**
```http
POST /mlflow/log_marketing
Authorization: Bearer {JWT_TOKEN}

{
  "segmentation": {
    "segment_id": 2,
    "segment_label": "Fidèle actif",
    "modele": "KMeans"
  },
  "classification": {
    "statut_predit": "confirmed",
    "probabilites": {
      "confirmed": 0.85
    },
    "modele": "RandomForest"
  }
}
```

### **4. CRM Pipeline**
```http
POST /mlflow/log_crm
Authorization: Bearer {JWT_TOKEN}

{
  "classification": {
    "statut_predit": "confirmed",
    "modele": "RandomForest"
  },
  "segmentation": {
    "segment_id": 3,
    "modele": "KMeans"
  }
}
```

### **5. Check MLflow Status**
```http
GET /mlflow/status

Response:
{
  "status": "connected",
  "tracking_uri": "http://localhost:5000",
  "experiments_count": 3,
  "message": "MLflow is accessible"
}
```

---

## 🔄 Enhanced Finance Workflow

The new workflow (`workflow_finance_with_mlflow.json`) adds MLflow logging:

```
┌─────────────────────────────────────────────────────────┐
│                  Finance Workflow                       │
├─────────────────────────────────────────────────────────┤
│ 1. Trigger (Weekly Monday 7am)                          │
│ 2. Login (Naïma)                                        │
│ 3. Price Prediction (Ridge)                             │
│ 4. Revenue Forecast (Holt)                              │
│ 5. Merge Results                                        │
│ 6. Save to File ────────────┐                           │
│ 7. Log to MLflow (NEW!) ────┘                           │
└─────────────────────────────────────────────────────────┘
```

**What gets logged to MLflow:**
- ✅ Workflow name and user
- ✅ Model names (Ridge, Holt)
- ✅ Predicted amount
- ✅ Time series metrics (MAPE, RMSE, MAE)
- ✅ Full prediction data as artifacts
- ✅ Timestamp and tags

---

## 🎨 View in MLflow UI

After running the workflow:

1. Open MLflow: http://localhost:5000
2. Go to "Experiments"
3. Find "n8n_Finance_Pipeline"
4. Click to see all runs
5. Compare predictions over time!

**You'll see:**
- Run name: `finance_2026-05-01_143022`
- Parameters: workflow, user, models
- Metrics: predicted_amount, mape, rmse
- Artifacts: Full prediction JSON
- Tags: source=n8n, pipeline=finance

---

## 🔧 Customize for Other Workflows

### **Marketing Workflow**

Add this node to your marketing workflow:

```json
{
  "parameters": {
    "method": "POST",
    "url": "http://127.0.0.1:8000/mlflow/log_marketing",
    "sendHeaders": true,
    "headerParameters": {
      "parameters": [{
        "name": "Authorization",
        "value": "=Bearer {{ $('Login API').item.json.access_token }}"
      }]
    },
    "sendBody": true,
    "specifyBody": "json",
    "jsonBody": "={\n  \"segmentation\": {{ JSON.stringify($('Segmentation').item.json) }},\n  \"classification\": {{ JSON.stringify($('Classification').item.json) }}\n}"
  },
  "name": "Log to MLflow",
  "type": "n8n-nodes-base.httpRequest"
}
```

### **CRM Workflow**

Add this node to your CRM workflow:

```json
{
  "parameters": {
    "method": "POST",
    "url": "http://127.0.0.1:8000/mlflow/log_crm",
    "sendHeaders": true,
    "headerParameters": {
      "parameters": [{
        "name": "Authorization",
        "value": "=Bearer {{ $('Login API').item.json.access_token }}"
      }]
    },
    "sendBody": true,
    "specifyBody": "json",
    "jsonBody": "={\n  \"classification\": {{ JSON.stringify($('Classification').item.json) }},\n  \"segmentation\": {{ JSON.stringify($('Segmentation').item.json) }}\n}"
  },
  "name": "Log to MLflow",
  "type": "n8n-nodes-base.httpRequest"
}
```

---

## 📊 MLflow Experiments Structure

After integration, you'll have these experiments:

```
MLflow UI (http://localhost:5000)
├── n8n_Finance_Pipeline
│   ├── finance_2026-05-01_070000
│   ├── finance_2026-05-08_070000
│   └── finance_2026-05-15_070000
│
├── n8n_Marketing_Pipeline
│   ├── marketing_2026-05-01_080000
│   ├── marketing_2026-05-02_080000
│   └── marketing_2026-05-03_080000
│
└── n8n_CRM_Pipeline
    ├── crm_2026-05-01_100000
    ├── crm_2026-05-02_100000
    └── crm_2026-05-03_100000
```

---

## 🎯 Use Cases

### **1. Track Prediction Accuracy Over Time**
- Compare MAPE across weeks
- Identify model drift
- Validate model performance

### **2. Audit Trail**
- Who made predictions?
- When were they made?
- What were the inputs?

### **3. A/B Testing**
- Test different models
- Compare performance
- Choose best model

### **4. Debugging**
- Review failed predictions
- Check input parameters
- Analyze error patterns

### **5. Reporting**
- Generate performance reports
- Show stakeholders results
- Track business metrics

---

## 🔍 Example: View Finance Predictions

1. **Open MLflow**: http://localhost:5000
2. **Click "Experiments"**
3. **Select "n8n_Finance_Pipeline"**
4. **See all runs** with:
   - Date/time of execution
   - Predicted amounts
   - Model metrics
   - User who triggered it

5. **Click a run** to see:
   - Full parameters
   - All metrics
   - Prediction artifacts
   - Comparison charts

6. **Compare runs**:
   - Select multiple runs
   - Click "Compare"
   - View side-by-side metrics
   - Identify trends

---

## 🛠️ Testing the Integration

### **Test 1: Check MLflow Status**
```bash
curl http://localhost:8000/mlflow/status
```

Expected response:
```json
{
  "status": "connected",
  "tracking_uri": "http://localhost:5000",
  "message": "MLflow is accessible"
}
```

### **Test 2: Manual Log**
```bash
curl -X POST http://localhost:8000/mlflow/log_prediction \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "Test_Experiment",
    "run_name": "test_run",
    "params": {"test": "value"},
    "metrics": {"accuracy": 0.95}
  }'
```

### **Test 3: Run n8n Workflow**
1. Open n8n: http://localhost:5678
2. Open finance workflow
3. Click "Execute Workflow"
4. Check MLflow UI for new run

---

## 📚 API Documentation

After adding the endpoints, view interactive docs:

**Swagger UI**: http://localhost:8000/docs

You'll see new endpoints under "MLflow" section:
- POST /mlflow/log_prediction
- POST /mlflow/log_finance
- POST /mlflow/log_marketing
- POST /mlflow/log_crm
- GET /mlflow/status

---

## 🆘 Troubleshooting

### **MLflow logging fails**
**Check:**
1. MLflow server is running: http://localhost:5000
2. FastAPI has MLflow endpoints loaded
3. JWT token is valid

**Solution:**
```bash
# Restart MLflow
start_mlflow.bat

# Restart FastAPI
python -m uvicorn ML.api.main:app --reload --port 8000
```

### **Cannot see experiments in MLflow**
**Check:**
1. Workflow executed successfully
2. No errors in n8n execution log
3. FastAPI logs show MLflow requests

**Solution:**
- Check n8n execution history
- View FastAPI terminal for errors
- Verify MLflow UI is accessible

### **Import errors in Python**
```bash
pip install mlflow
```

---

## ✨ Benefits

### **Before Integration:**
- ❌ No tracking of predictions
- ❌ No comparison between runs
- ❌ No audit trail
- ❌ Manual result checking

### **After Integration:**
- ✅ Automatic prediction tracking
- ✅ Easy comparison in UI
- ✅ Complete audit trail
- ✅ Visual dashboards
- ✅ Experiment management
- ✅ Model versioning

---

## 🎯 Next Steps

1. ✅ Add MLflow endpoints to FastAPI
2. ✅ Restart FastAPI server
3. ✅ Import enhanced workflow to n8n
4. ✅ Test with manual execution
5. ✅ View results in MLflow UI
6. ✅ Enhance other workflows (marketing, CRM)
7. ✅ Set up automated reporting

---

## 📖 Related Documentation

- **MLflow Setup**: `MLFLOW_SETUP_GUIDE.md`
- **MLflow Quick Start**: `MLFLOW_QUICK_START.md`
- **n8n Integration**: `N8N_ML_INTEGRATION_GUIDE.md`
- **Services Overview**: `SERVICES_OVERVIEW.md`

---

**Last Updated**: May 1, 2026  
**Status**: ✅ Ready to Use  
**MLflow Version**: 3.11.1

