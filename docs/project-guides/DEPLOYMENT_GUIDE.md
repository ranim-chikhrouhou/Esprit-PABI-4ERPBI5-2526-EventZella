# 🚀 EventZella MLOps Deployment Guide

## 📋 Complete Checklist Status

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Experiment Tracking (MLflow) | ✅ COMPLETE | Multiple runs with metrics visible |
| 2 | Automated Training Pipeline | ✅ COMPLETE | `automated_training_pipeline.py` |
| 3 | Model Management | ✅ COMPLETE | Models versioned in MLflow Registry |
| 4 | Model Serving (API) | ✅ COMPLETE | FastAPI with /predict endpoints |
| 5 | Containerization | ✅ COMPLETE | Docker + Docker Compose ready |
| 6 | Code Quality | ✅ COMPLETE | Clean, structured, documented |
| 7 | Web App Integration | ✅ COMPLETE | Streamlit + n8n connected to API |

**Overall: 7/7 Requirements Met (100%)** ✅

---

## 🎯 Deliverables

### ✅ 1. Functional API (Local & Dockerized)

**Local:**
- Running on http://localhost:8000
- Status: http://localhost:8000/ returns `{"status":"ok"}`
- Docs: http://localhost:8000/docs

**Docker:**
```bash
docker-compose up fastapi
# Access: http://localhost:8000
```

**Endpoints:**
- POST `/predict/classification` - Booking status
- POST `/predict/regression` - Price estimation
- POST `/predict/segmentation/beneficiaire` - Customer segments
- GET `/predict/timeseries` - Revenue forecast
- POST `/mlflow/log_finance` - Log to MLflow

---

### ✅ 2. MLflow with Visible Runs

**Local:**
- Running on http://localhost:5000
- Experiment: `n8n_Finance_Pipeline`
- Multiple runs logged with metrics

**Docker:**
```bash
docker-compose up mlflow
# Access: http://localhost:5000
```

**What's Tracked:**
- Parameters: workflow, user, model names
- Metrics: predicted_amount, mape, rmse, mae
- Artifacts: Full prediction data
- Tags: source, pipeline, automated

**Evidence:**
```
Experiment ID: 5
Runs: finance_20260501_230516, finance_20260501_230519, etc.
All runs have metrics visible
```

---

### ✅ 3. Automated Pipeline

**File:** `automated_training_pipeline.py`

**What it does:**
1. Loads data from processed files
2. Preprocesses data (scaling, encoding)
3. Trains 3 models (Classification, Regression, Clustering)
4. Evaluates performance
5. Logs everything to MLflow
6. Saves models with versioning
7. Fully reproducible

**Run it:**
```bash
python automated_training_pipeline.py
```

**Output:**
- Models saved in `ML/models_artifacts/`
- Runs logged to MLflow
- Metrics displayed in console
- Registered in MLflow Model Registry

---

### ✅ 4. Model Accessible via API

**Test Classification:**
```bash
curl -X POST http://localhost:8000/predict/classification \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "id_date": 1,
    "id_event": 42,
    "id_servicecategory": 3,
    "final_price": 1500
  }'
```

**Test Regression:**
```bash
curl -X POST http://localhost:8000/predict/regression \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "service_price": 1200,
    "event_budget": 2000,
    "benchmark_avg_price": 1300
  }'
```

**Response:**
```json
{
  "prediction": "confirmed",
  "confidence": 0.85,
  "model": "RandomForest"
}
```

---

### ✅ 5. Web App Connected to API

**Streamlit App:**
- URL: http://localhost:8502
- Pages: Home, Booking Status, Price Estimation, Segmentation, Forecast
- Connected to FastAPI
- Full pipeline: UI → API → Model → Result

**n8n Workflows:**
- URL: http://localhost:5678
- Finance Pipeline: Automated predictions
- Logs to MLflow automatically
- Full pipeline: Trigger → API → Model → MLflow

**End-to-End Flow:**
```
User Input (Streamlit/n8n)
    ↓
FastAPI Endpoint
    ↓
Load Model from artifacts
    ↓
Make Prediction
    ↓
Log to MLflow (optional)
    ↓
Return Result to User
```

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Services

| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000 |
| MLflow | 5000 | http://localhost:5000 |
| Streamlit | 8502 | http://localhost:8502 |

### Individual Services

```bash
# Start only FastAPI
docker-compose up fastapi

# Start only MLflow
docker-compose up mlflow

# Start only Streamlit
docker-compose up streamlit
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  EventZella MLOps Platform              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Streamlit   │  │     n8n      │  │   MLflow     │ │
│  │  Web UI      │  │  Automation  │  │  Tracking    │ │
│  │  :8502       │  │  :5678       │  │  :5000       │ │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘ │
│         │                 │                  │         │
│         └─────────────────┼──────────────────┘         │
│                           │                            │
│                  ┌────────▼────────┐                   │
│                  │    FastAPI      │                   │
│                  │   ML API :8000  │                   │
│                  │                 │                   │
│                  │  Endpoints:     │                   │
│                  │  - /predict/*   │                   │
│                  │  - /mlflow/*    │                   │
│                  │  - /auth/*      │                   │
│                  └────────┬────────┘                   │
│                           │                            │
│                  ┌────────▼────────┐                   │
│                  │   ML Models     │                   │
│                  │   (.joblib)     │                   │
│                  │                 │                   │
│                  │  - Classification                   │
│                  │  - Regression   │                   │
│                  │  - Clustering   │                   │
│                  └─────────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

### 1. Test API Health
```bash
curl http://localhost:8000/
# Expected: {"status":"ok","app":"EventZilla ML API v1.0"}
```

### 2. Test MLflow
```bash
curl http://localhost:5000/health
# Expected: 200 OK
```

### 3. Test Prediction
```bash
# Get auth token first
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"login":"test_user","password":"test_pass"}' \
  | jq -r '.access_token')

# Make prediction
curl -X POST http://localhost:8000/predict/regression \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"service_price":1200,"event_budget":2000}'
```

### 4. Test Automated Pipeline
```bash
python automated_training_pipeline.py
# Check MLflow UI for new runs
```

### 5. Test n8n Workflow
1. Open http://localhost:5678
2. Execute finance workflow
3. Check MLflow for logged run
4. Verify metrics are visible

---

## 📁 Project Structure

```
PI BI NEW (2)/PI BI NEW/
├── ML/
│   ├── api/
│   │   ├── main.py                    # FastAPI app
│   │   ├── auth_sql.py                # Authentication
│   │   ├── mlflow_integration.py      # MLflow utilities
│   │   └── mlflow_endpoints.py        # MLflow API endpoints
│   ├── models_artifacts/              # Saved models
│   ├── data_processed/                # Processed data
│   ├── notebooks/                     # Training notebooks
│   └── streamlit_app.py               # Web UI
├── n8n/
│   ├── workflow_finance_mlflow_simple.json  # n8n workflow
│   └── results/                       # Workflow results
├── mlruns/                            # MLflow experiments
├── mlartifacts/                       # MLflow artifacts
├── automated_training_pipeline.py     # Automated pipeline
├── docker-compose.yml                 # Docker orchestration
├── Dockerfile.fastapi                 # FastAPI container
├── Dockerfile.mlflow                  # MLflow container
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables
└── DEPLOYMENT_GUIDE.md                # This file
```

---

## 🎓 Demonstration Checklist

### For Presentation:

1. **Show MLflow UI**
   - Open http://localhost:5000
   - Navigate to Experiments
   - Show n8n_Finance_Pipeline
   - Click a run to show metrics
   - Compare multiple runs

2. **Show API Documentation**
   - Open http://localhost:8000/docs
   - Show available endpoints
   - Test a prediction endpoint
   - Show response with prediction

3. **Show Automated Pipeline**
   - Run `python automated_training_pipeline.py`
   - Show console output
   - Show new runs in MLflow
   - Show models saved

4. **Show Web App**
   - Open http://localhost:8502
   - Make a prediction
   - Show result displayed
   - Explain full pipeline

5. **Show n8n Workflow**
   - Open http://localhost:5678
   - Show workflow structure
   - Execute workflow
   - Show result in MLflow

6. **Show Docker Deployment**
   - Run `docker-compose up -d`
   - Show all services running
   - Access services via browser
   - Show logs with `docker-compose logs`

---

## ✅ Verification Steps

### Checklist for Evaluator:

- [ ] MLflow UI accessible at http://localhost:5000
- [ ] At least 2 runs visible in MLflow
- [ ] Runs have parameters and metrics
- [ ] Runs are comparable (click Compare button)
- [ ] FastAPI accessible at http://localhost:8000
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] /predict endpoint works (returns prediction)
- [ ] Automated pipeline script exists
- [ ] Pipeline can be run: `python automated_training_pipeline.py`
- [ ] Pipeline logs to MLflow automatically
- [ ] Models are saved and versioned
- [ ] Docker files exist (Dockerfile, docker-compose.yml)
- [ ] Docker deployment works: `docker-compose up`
- [ ] Web app accessible and functional
- [ ] Web app calls API successfully
- [ ] Full pipeline works end-to-end

---

## 📞 Support

**Documentation:**
- API Docs: http://localhost:8000/docs
- MLflow Guide: `MLFLOW_SETUP_GUIDE.md`
- n8n Integration: `N8N_MLFLOW_INTEGRATION.md`
- Project Checklist: `PROJECT_CHECKLIST.md`

**Quick Commands:**
```bash
# Start all services
docker-compose up -d

# Run automated training
python automated_training_pipeline.py

# Check service status
docker-compose ps

# View logs
docker-compose logs -f fastapi
docker-compose logs -f mlflow

# Stop all services
docker-compose down
```

---

**Status: ✅ ALL REQUIREMENTS MET (7/7)**

**Ready for Demonstration and Evaluation!** 🎉

