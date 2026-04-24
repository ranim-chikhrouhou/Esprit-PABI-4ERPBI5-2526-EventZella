# 🎯 EventZilla Services Overview

## 📊 All Services at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    EventZilla Platform                          │
│                  Complete ML & Automation Stack                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   FastAPI        │  │   n8n            │  │   Streamlit      │
│   Port: 8000     │  │   Port: 5678     │  │   Port: 8502     │
│                  │  │                  │  │                  │
│   ML API         │  │   Workflow       │  │   ML Dashboard   │
│   Endpoints      │  │   Automation     │  │   UI             │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   MLflow            │
                    │   Port: 5000        │
                    │                     │
                    │   Experiment        │
                    │   Tracking          │
                    └─────────────────────┘
```

---

## 🚀 Service Details

### **1. FastAPI (ML API Backend)**
- **Port**: 8000
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Purpose**: ML prediction endpoints
- **Features**:
  - ✅ Booking status classification
  - ✅ Price estimation
  - ✅ Customer segmentation
  - ✅ Time series forecasting
  - ✅ JWT authentication
  - ✅ Interactive API docs

**Start**: Included in `LANCER_PROJET.bat` or `LANCER_TOUT.bat`

---

### **2. n8n (Workflow Automation)**
- **Port**: 5678
- **URL**: http://localhost:5678
- **Purpose**: Automated ML workflows
- **Features**:
  - ✅ Visual workflow editor
  - ✅ Scheduled predictions
  - ✅ Marketing pipeline
  - ✅ Finance pipeline
  - ✅ CRM pipeline
  - ✅ Error handling

**Start**: Included in `LANCER_PROJET.bat` or `LANCER_TOUT.bat`

**Workflows**:
- `workflow_marketing.json` - Daily at 08:00
- `workflow_finance.json` - Daily at 09:00
- `workflow_crm.json` - Daily at 10:00
- `workflow_error_handler.json` - Error logging

---

### **3. Streamlit (ML Dashboard)**
- **Port**: 8502
- **URL**: http://localhost:8502
- **Purpose**: Interactive ML interface
- **Features**:
  - ✅ Manual predictions
  - ✅ Model testing
  - ✅ Data visualization
  - ✅ User authentication
  - ✅ Real-time results
  - ✅ Clean English UI

**Start**: 
```bash
python -m streamlit run ML/streamlit_app.py
```
Or use `LANCER_TOUT.bat`

**Pages**:
- Home
- Booking Status Prediction
- Price Estimation
- Customer Segmentation
- Trends & Forecast
- Summary

---

### **4. MLflow (Experiment Tracking)** 🆕
- **Port**: 5000
- **URL**: http://localhost:5000
- **Purpose**: ML experiment tracking
- **Features**:
  - ✅ Log experiments
  - ✅ Track metrics
  - ✅ Compare models
  - ✅ Model registry
  - ✅ Version control
  - ✅ Artifact storage

**Start**: 
```bash
start_mlflow.bat
```
Or use `LANCER_TOUT.bat`

**Capabilities**:
- Track model parameters
- Log training metrics
- Store model artifacts
- Compare experiment runs
- Register production models
- Version management

---

## 🎯 Launch Options

### **Option 1: Launch Everything** (Recommended)
```bash
LANCER_TOUT.bat
```
Starts all 4 services automatically!

### **Option 2: Launch Core Services**
```bash
LANCER_PROJET.bat
```
Starts FastAPI + n8n (then manually start Streamlit and MLflow)

### **Option 3: Individual Services**
```bash
# FastAPI
python -m uvicorn ML.api.main:app --reload --port 8000

# n8n
npx n8n

# Streamlit
python -m streamlit run ML/streamlit_app.py

# MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 🔗 Service Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │Streamlit │  │   n8n    │  │  MLflow  │
        │   UI     │  │Workflows │  │Tracking  │
        └──────────┘  └──────────┘  └──────────┘
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                      ┌──────────────┐
                      │   FastAPI    │
                      │   ML API     │
                      └──────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ML Models │  │SQL Server│  │  MLflow  │
        │ .joblib  │  │   DW     │  │ Storage  │
        └──────────┘  └──────────┘  └──────────┘
```

### **Flow Examples**:

**1. Manual Prediction (Streamlit)**
```
User → Streamlit → FastAPI → ML Model → Result → User
                                  ↓
                              MLflow (log)
```

**2. Automated Prediction (n8n)**
```
Schedule → n8n → FastAPI → ML Model → Save Result
                              ↓
                          MLflow (log)
```

**3. Experiment Tracking**
```
Notebook → Train Model → Log to MLflow → View in UI
```

---

## 📊 Port Summary

| Service | Port | Status | Auto-Start |
|---------|------|--------|------------|
| FastAPI | 8000 | ✅ Running | Yes |
| n8n | 5678 | ✅ Running | Yes |
| Streamlit | 8502 | ⚠️ Manual | Optional |
| MLflow | 5000 | 🆕 New | Optional |

---

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `ML/ml_paths.py` | Path configurations |
| `.streamlit/config.toml` | Streamlit theme |
| `n8n/workflow_*.json` | n8n workflows |
| `mlruns/` | MLflow experiments |

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `N8N_ML_INTEGRATION_GUIDE.md` | n8n integration guide |
| `MLFLOW_SETUP_GUIDE.md` | Complete MLflow guide |
| `MLFLOW_QUICK_START.md` | Quick start guide |
| `ALL_UI_FILES.md` | UI files overview |
| `TRANSLATION_COMPLETE.md` | Translation log |
| `SERVICES_OVERVIEW.md` | This file |

---

## 🎯 Common Tasks

### **Start All Services**
```bash
LANCER_TOUT.bat
```

### **Check Service Status**
- FastAPI: http://localhost:8000/docs
- n8n: http://localhost:5678
- Streamlit: http://localhost:8502
- MLflow: http://localhost:5000

### **Log Models to MLflow**
```bash
python log_models_to_mlflow.py
```

### **Test API Endpoints**
```bash
curl http://localhost:8000/health
```

### **View n8n Workflows**
1. Open http://localhost:5678
2. Import workflow JSON files
3. Activate workflows

### **Access Streamlit Dashboard**
1. Open http://localhost:8502
2. Login with credentials
3. Navigate to prediction pages

### **Track Experiments in MLflow**
1. Open http://localhost:5000
2. View experiments
3. Compare runs
4. Register models

---

## 🆘 Troubleshooting

### **Port Already in Use**
```bash
# Check what's using the port
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <process_id> /F
```

### **Service Won't Start**
1. Check if Python is installed
2. Verify dependencies: `pip list`
3. Check firewall settings
4. Try different port

### **Cannot Access UI**
1. Use `127.0.0.1` instead of `localhost`
2. Check if service is running
3. Clear browser cache
4. Try different browser

---

## ✨ Next Steps

1. ✅ Start all services: `LANCER_TOUT.bat`
2. ✅ Access each service in browser
3. ✅ Log models to MLflow
4. ✅ Test predictions in Streamlit
5. ✅ Set up n8n workflows
6. ✅ Monitor experiments in MLflow

---

**Last Updated**: May 1, 2026  
**Services**: 4 (FastAPI, n8n, Streamlit, MLflow)  
**Status**: ✅ All Operational

