# 🚀 MLflow Quick Start Guide

## ✅ Installation Complete!

MLflow version **3.11.1** has been successfully installed on your system.

---

## 🎯 Quick Start (3 Steps)

### **Step 1: Start MLflow Server**

Double-click: **`start_mlflow.bat`**

Or run in terminal:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### **Step 2: Access MLflow UI**

Open your browser: **http://localhost:5000**

### **Step 3: Log Your Models**

Run the model logging script:
```bash
python log_models_to_mlflow.py
```

This will automatically log all your existing EventZilla models to MLflow!

---

## 🎨 What You'll See in MLflow UI

### **Experiments Tab**
- View all your ML experiments
- Compare different model runs
- Filter and search experiments

### **Models Tab**
- Registered models with versions
- Model staging (Development → Production)
- Model metadata and artifacts

### **Runs Tab**
- Individual training runs
- Parameters, metrics, and artifacts
- Comparison charts

---

## 🔗 Your EventZilla Services

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **FastAPI** | 8000 | http://localhost:8000 | ML API endpoints |
| **n8n** | 5678 | http://localhost:5678 | Workflow automation |
| **Streamlit** | 8502 | http://localhost:8502 | ML Dashboard |
| **MLflow** | 5000 | http://localhost:5000 | Experiment tracking |

---

## 🚀 Launch All Services at Once

Use the new complete launcher:

**`LANCER_TOUT.bat`** - Starts all 4 services automatically!

---

## 📊 Your Models in MLflow

After running `log_models_to_mlflow.py`, you'll have:

1. **Booking Status Classifier** (RandomForest)
   - Predicts: confirmed, pending, cancelled
   - Experiment: EventZilla_Classification

2. **Price Estimation** (Ridge Regression)
   - Predicts: final event price
   - Experiment: EventZilla_Regression

3. **Customer Segmentation - Beneficiary** (KMeans)
   - Segments: loyalty groups
   - Experiment: EventZilla_Segmentation

4. **Customer Segmentation - Visitor** (KMeans)
   - Segments: visitor behavior
   - Experiment: EventZilla_Segmentation

---

## 💡 Common Use Cases

### **1. Compare Model Versions**
1. Open MLflow UI
2. Go to Experiments
3. Select multiple runs
4. Click "Compare"
5. View side-by-side metrics

### **2. Register Best Model**
1. Find your best run
2. Click "Register Model"
3. Give it a name
4. Transition to "Production"

### **3. Load Production Model**
```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/EventZilla_booking_status_prediction/Production"
)
predictions = model.predict(data)
```

### **4. Track New Experiments**
```python
import mlflow

mlflow.set_experiment("EventZilla_NewFeature")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

---

## 📁 Files Created

- ✅ `start_mlflow.bat` - MLflow launcher
- ✅ `log_models_to_mlflow.py` - Model logging script
- ✅ `LANCER_TOUT.bat` - Complete project launcher
- ✅ `MLFLOW_SETUP_GUIDE.md` - Detailed documentation
- ✅ `MLFLOW_QUICK_START.md` - This file

---

## 🔧 Troubleshooting

### **MLflow UI won't open**
- Check if port 5000 is already in use
- Try: `mlflow ui --port 5001`

### **Cannot log models**
- Make sure MLflow server is running
- Check model files exist in `ML/models_artifacts/`

### **Import errors**
```bash
pip install --upgrade mlflow
```

---

## 📚 Learn More

- **Full Guide**: Read `MLFLOW_SETUP_GUIDE.md`
- **Official Docs**: https://mlflow.org/docs/latest/
- **Tutorials**: https://mlflow.org/docs/latest/tutorials-and-examples/

---

## ✨ Next Steps

1. ✅ Start MLflow: `start_mlflow.bat`
2. ✅ Log models: `python log_models_to_mlflow.py`
3. ✅ Open UI: http://localhost:5000
4. ✅ Explore your experiments!
5. ✅ Integrate with your workflows

---

**Happy Tracking! 🎉**

