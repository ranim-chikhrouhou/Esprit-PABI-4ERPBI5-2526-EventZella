# ⚡ Quick Setup: n8n + MLflow Integration

## 🎯 Goal
Connect your finance n8n workflow to MLflow to automatically track all predictions.

---

## ✅ Prerequisites

- [x] MLflow installed and running (port 5000)
- [x] FastAPI running (port 8000)
- [x] n8n running (port 5678)

---

## 🚀 Setup (3 Simple Steps)

### **Step 1: Add MLflow to FastAPI** (30 seconds)

Run this command:
```bash
cd "PI BI NEW (2)\PI BI NEW"
python add_mlflow_to_api.py
```

This automatically adds MLflow endpoints to your API!

**OR manually add to `ML/api/main.py`:**
```python
# Add this import at the top
from ML.api.mlflow_endpoints import router as mlflow_router

# Add this after app = FastAPI(...)
app.include_router(mlflow_router)
```

### **Step 2: Restart FastAPI** (10 seconds)

Stop FastAPI (Ctrl+C) and restart:
```bash
python -m uvicorn ML.api.main:app --reload --port 8000
```

### **Step 3: Import Enhanced Workflow** (1 minute)

1. Open n8n: http://localhost:5678
2. Click "Add workflow" → "Import from File"
3. Select: `n8n/workflow_finance_with_mlflow.json`
4. Click "Activate" toggle

---

## ✨ That's It!

Your finance workflow now logs to MLflow automatically!

---

## 🧪 Test It

### **Test 1: Check API**
Open: http://localhost:8000/docs

Look for new "MLflow" section with endpoints:
- POST /mlflow/log_prediction
- POST /mlflow/log_finance
- GET /mlflow/status

### **Test 2: Run Workflow**
1. Open n8n: http://localhost:5678
2. Open "EventZilla — Finance Pipeline with MLflow"
3. Click "Execute Workflow"
4. Wait for completion

### **Test 3: View in MLflow**
1. Open MLflow: http://localhost:5000
2. Click "Experiments"
3. Find "n8n_Finance_Pipeline"
4. See your prediction logged! 🎉

---

## 📊 What Gets Logged

Every time your finance workflow runs, MLflow logs:

**Parameters:**
- Workflow name: "finance"
- User: "naima_sarraj"
- Model (regression): "Ridge"
- Model (timeseries): "Holt"

**Metrics:**
- Predicted amount (TND)
- MAPE (%)
- RMSE
- MAE

**Artifacts:**
- Full prediction JSON
- Timestamp
- All input/output data

---

## 🎨 MLflow UI Preview

```
┌─────────────────────────────────────────────────────┐
│ MLflow - Experiments                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📊 n8n_Finance_Pipeline                            │
│    ├─ finance_2026-05-01_070000                    │
│    │   Metrics: amount=1450.75, mape=6.1%          │
│    │                                                │
│    ├─ finance_2026-05-08_070000                    │
│    │   Metrics: amount=1523.40, mape=5.8%          │
│    │                                                │
│    └─ finance_2026-05-15_070000                    │
│        Metrics: amount=1489.20, mape=6.3%          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Flow

```
┌──────────────────────────────────────────────────┐
│         Finance Workflow (Enhanced)              │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. ⏰ Trigger (Weekly Monday 7am)              │
│  2. 🔐 Login (Naïma)                            │
│  3. 💰 Price Prediction (Ridge)                 │
│  4. 📈 Revenue Forecast (Holt)                  │
│  5. 🔀 Merge Results                            │
│  6. 💾 Save to File                             │
│  7. 🔥 Log to MLflow ← NEW!                     │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 🎯 Benefits

### **Before:**
- ❌ No tracking
- ❌ No history
- ❌ Manual checking
- ❌ No comparison

### **After:**
- ✅ Automatic tracking
- ✅ Complete history
- ✅ Visual dashboard
- ✅ Easy comparison
- ✅ Audit trail
- ✅ Performance monitoring

---

## 📚 Full Documentation

For detailed information, see:
- **Complete Guide**: `N8N_MLFLOW_INTEGRATION.md`
- **MLflow Setup**: `MLFLOW_SETUP_GUIDE.md`
- **API Reference**: http://localhost:8000/docs

---

## 🆘 Troubleshooting

### **Problem: MLflow endpoints not showing**
**Solution:**
```bash
# Check if endpoints were added
python add_mlflow_to_api.py

# Restart FastAPI
python -m uvicorn ML.api.main:app --reload --port 8000
```

### **Problem: Workflow fails at MLflow step**
**Solution:**
1. Check MLflow is running: http://localhost:5000
2. Check FastAPI logs for errors
3. Verify JWT token is valid

### **Problem: No experiments in MLflow**
**Solution:**
1. Execute workflow manually in n8n
2. Check n8n execution log for errors
3. Verify MLflow step completed successfully

---

## 🎉 Success Checklist

- [ ] MLflow running on port 5000
- [ ] FastAPI has MLflow endpoints
- [ ] Enhanced workflow imported to n8n
- [ ] Workflow executed successfully
- [ ] Prediction visible in MLflow UI

---

## 🚀 Next Steps

1. ✅ Enhance marketing workflow
2. ✅ Enhance CRM workflow
3. ✅ Set up automated reports
4. ✅ Compare predictions over time
5. ✅ Monitor model performance

---

**Setup Time**: ~2 minutes  
**Difficulty**: Easy  
**Status**: ✅ Ready to Use

