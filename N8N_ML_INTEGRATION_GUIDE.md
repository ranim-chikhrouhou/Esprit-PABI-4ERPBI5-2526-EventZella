# 🔄 N8N Automation Integration with ML Flow

## Overview
Your n8n workflows are already configured to automate ML predictions through your FastAPI backend. Here's how to monitor and use them.

## 📊 Current Setup

### **Services Running**
1. **FastAPI Backend**: http://localhost:8000 (ML API endpoints)
2. **n8n Workflow Engine**: http://localhost:5678 (Automation platform)
3. **Streamlit Dashboard**: http://localhost:8502 (ML UI)

### **Available Workflows**

#### 1. **Marketing Pipeline** (Ranim)
- **File**: `n8n/workflow_marketing.json`
- **User**: ranim_chikhrouhou
- **Trigger**: Daily at 08:00
- **ML Models Used**:
  - ✅ Customer Segmentation (Beneficiaries)
  - ✅ Booking Status Classification
- **Output**: `n8n/results/marketing_*.json`

#### 2. **Finance Pipeline** (Naima)
- **File**: `n8n/workflow_finance.json`
- **User**: naima_ben_salem
- **Trigger**: Daily at 09:00
- **ML Models Used**:
  - ✅ Price Estimation (Regression)
  - ✅ Time Series Forecasting
- **Output**: `n8n/results/finance_*.json`

#### 3. **CRM Pipeline** (Anas)
- **File**: `n8n/workflow_crm.json`
- **User**: anas_trabelsi
- **Trigger**: Daily at 10:00
- **ML Models Used**:
  - ✅ Booking Status Classification
  - ✅ Customer Segmentation
- **Output**: `n8n/results/crm_*.json`

#### 4. **Error Handler**
- **File**: `n8n/workflow_error_handler.json`
- **Purpose**: Catches and logs errors from all workflows
- **Output**: `n8n/results/errors_*.json`

## 🚀 How to Use N8N with ML Flow

### **Step 1: Access N8N Interface**
1. Open browser to: **http://localhost:5678**
2. You'll see the n8n workflow editor

### **Step 2: Import Workflows**
```bash
# Workflows are already in JSON format
# Import them in n8n:
1. Click "Add workflow" → "Import from File"
2. Select: n8n/workflow_marketing.json
3. Repeat for finance, crm, and error_handler
```

### **Step 3: Activate Workflows**
1. Open each workflow in n8n
2. Click the **"Active"** toggle (top right)
3. Workflows will now run automatically on schedule

### **Step 4: Manual Execution**
To test immediately:
1. Open workflow in n8n
2. Click **"Execute Workflow"** button
3. Watch the nodes execute in real-time

## 📡 ML API Endpoints Used by N8N

### **Authentication**
```http
POST http://localhost:8000/auth/login
Content-Type: application/json

{
  "login": "ranim_chikhrouhou",
  "password": "Ranim@Marketing2025!"
}

Response: { "access_token": "JWT_TOKEN", "token_type": "bearer" }
```

### **Customer Segmentation**
```http
POST http://localhost:8000/predict/segmentation/beneficiaire
Authorization: Bearer {JWT_TOKEN}
Content-Type: application/json

{
  "nb_reservations_loyalty": 12,
  "ca_total_loyalty": 15000,
  "panier_moyen_loyalty": 1250,
  "recency_days_loyalty": 30,
  "avg_nb_visitors_loyalty": 85,
  "volume_reservations_site_loyalty": 5
}

Response: {
  "segment_id": 2,
  "segment_label": "Fidèle actif",
  "modele": "KMeans"
}
```

### **Booking Status Classification**
```http
POST http://localhost:8000/predict/classification
Authorization: Bearer {JWT_TOKEN}
Content-Type: application/json

{
  "id_date": 1,
  "id_event": 42,
  "id_servicecategory": 3,
  "id_benchmark": 2,
  "id_provider": 7,
  "final_price": 1500,
  "service_price": 1200,
  "benchmark_avg_price": 1300,
  "event_budget": 2000,
  "cal_month": 4,
  "cal_year": 2024,
  "quarter": 2
}

Response: {
  "statut_predit": "confirmed",
  "probabilites": {
    "confirmed": 0.85,
    "pending": 0.10,
    "cancelled": 0.05
  },
  "modele": "RandomForest"
}
```

### **Price Estimation**
```http
POST http://localhost:8000/predict/regression
Authorization: Bearer {JWT_TOKEN}
Content-Type: application/json

{
  "service_price": 1200,
  "event_budget": 2000,
  "benchmark_avg_price": 1300,
  "cal_year": 2024,
  "cal_month": 4,
  "quarter": 2
}

Response: {
  "prix_estime": 1450.75,
  "modele": "Ridge"
}
```

## 📊 Monitoring N8N Automation

### **Option 1: N8N Web Interface**
1. Go to http://localhost:5678
2. Click **"Executions"** tab
3. See all workflow runs with:
   - ✅ Success status
   - ❌ Error status
   - ⏱️ Execution time
   - 📊 Data flow between nodes

### **Option 2: Check Result Files**
```bash
# Results are saved in JSON format
cd "PI BI NEW (2)/PI BI NEW/n8n/results"

# View latest marketing results
cat marketing_2026-05-01.json

# View latest finance results
cat finance_2026-05-01.json

# View errors
cat errors_2026-05-01.json
```

### **Option 3: FastAPI Logs**
Check the FastAPI terminal (Terminal 14) for API calls:
```
INFO: 127.0.0.1:xxxxx - "POST /auth/login HTTP/1.1" 200 OK
INFO: 127.0.0.1:xxxxx - "POST /predict/segmentation/beneficiaire HTTP/1.1" 200 OK
INFO: 127.0.0.1:xxxxx - "POST /predict/classification HTTP/1.1" 200 OK
```

## 🔗 Integration Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    N8N WORKFLOW ENGINE                       │
│                   (http://localhost:5678)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Scheduled Trigger
                              │ (Daily 08:00, 09:00, 10:00)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  1. AUTHENTICATION                           │
│  POST /auth/login → Get JWT Token (8h validity)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ JWT Token
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  2. ML PREDICTIONS                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Marketing: Segmentation + Classification             │  │
│  │ Finance:   Regression + Time Series                  │  │
│  │ CRM:       Classification + Segmentation             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ ML Results
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. FASTAPI BACKEND                          │
│              (http://localhost:8000)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • Load ML Models (joblib)                            │  │
│  │ • Run Predictions                                     │  │
│  │ • Return JSON Results                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Save Results
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  4. RESULTS STORAGE                          │
│  n8n/results/marketing_YYYY-MM-DD.json                      │
│  n8n/results/finance_YYYY-MM-DD.json                        │
│  n8n/results/crm_YYYY-MM-DD.json                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ View Results
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  5. STREAMLIT DASHBOARD                      │
│              (http://localhost:8502)                         │
│  • Manual predictions via UI                                 │
│  • View ML model performance                                 │
│  • Interactive testing                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

### **Use Case 1: Daily Automated Predictions**
1. N8N triggers workflows daily
2. Each workflow calls ML APIs
3. Results saved to JSON files
4. Business teams review results

### **Use Case 2: Manual Testing**
1. Open Streamlit (http://localhost:8502)
2. Navigate to prediction pages
3. Enter test data manually
4. See immediate results

### **Use Case 3: API Integration**
1. External systems call FastAPI directly
2. Use JWT authentication
3. Get ML predictions in real-time
4. Integrate into business processes

## 📝 Example: Complete Workflow Execution

### **Marketing Pipeline Example**
```json
// 1. Login
POST /auth/login
{
  "login": "ranim_chikhrouhou",
  "password": "Ranim@Marketing2025!"
}
→ Returns: { "access_token": "eyJ..." }

// 2. Customer Segmentation
POST /predict/segmentation/beneficiaire
Authorization: Bearer eyJ...
{
  "nb_reservations_loyalty": 12,
  "ca_total_loyalty": 15000,
  "panier_moyen_loyalty": 1250,
  "recency_days_loyalty": 30,
  "avg_nb_visitors_loyalty": 85,
  "volume_reservations_site_loyalty": 5
}
→ Returns: { "segment_id": 2, "segment_label": "Fidèle actif" }

// 3. Booking Status
POST /predict/classification
Authorization: Bearer eyJ...
{
  "final_price": 1500,
  "service_price": 1200,
  "event_budget": 2000,
  ...
}
→ Returns: { "statut_predit": "confirmed", "probabilites": {...} }

// 4. Save Results
POST /save_result
Authorization: Bearer eyJ...
{
  "workflow": "marketing",
  "data": { "segmentation": {...}, "classification": {...} }
}
→ Saves to: n8n/results/marketing_2026-05-01.json
```

## 🔧 Troubleshooting

### **Problem: Workflows not executing**
**Solution:**
1. Check if n8n is running: http://localhost:5678
2. Verify workflows are **Active** (toggle on)
3. Check execution history for errors

### **Problem: Authentication fails**
**Solution:**
1. Verify FastAPI is running: http://localhost:8000
2. Check user credentials in workflow
3. Ensure SQL Server database is accessible

### **Problem: ML predictions fail**
**Solution:**
1. Check if ML models exist in `ML/models_artifacts/`
2. Verify model files: `rf_status_kpi_pipeline.joblib`, etc.
3. Check FastAPI logs for error details

### **Problem: Results not saved**
**Solution:**
1. Check `n8n/results/` directory exists
2. Verify FastAPI has write permissions
3. Check `/save_result` endpoint in FastAPI

## 📚 Additional Resources

### **Test N8N Workflows**
```bash
# Run test script
cd "PI BI NEW (2)/PI BI NEW"
python n8n/test_workflows.py
```

### **View API Documentation**
Open: http://localhost:8000/docs
- Interactive API documentation
- Test endpoints directly
- See request/response schemas

### **Check Service Status**
```bash
# FastAPI
curl http://localhost:8000/health

# N8N
curl http://localhost:5678/healthz

# Streamlit
curl http://localhost:8502/healthz
```

## ✅ Summary

**YES, you can follow n8n automation in your ML flow!**

### **How to Monitor:**
1. **N8N Web UI**: http://localhost:5678 → See workflow executions
2. **Result Files**: Check `n8n/results/*.json`
3. **FastAPI Logs**: Watch Terminal 14 for API calls
4. **Streamlit**: Manual testing at http://localhost:8502

### **Integration Points:**
- ✅ N8N calls FastAPI ML endpoints
- ✅ FastAPI loads ML models and predicts
- ✅ Results saved to JSON files
- ✅ Streamlit provides manual UI
- ✅ All services work together seamlessly

---

**Next Steps:**
1. Open http://localhost:5678 to see n8n
2. Import your workflow JSON files
3. Activate workflows
4. Monitor executions in real-time!
