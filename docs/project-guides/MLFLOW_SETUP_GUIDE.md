# 🚀 MLflow Installation & Setup Guide

## ✅ Installation Complete

MLflow has been successfully installed!

**Version**: 3.11.1

---

## 📊 What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage ML models
- **Model Deployment**: Deploy models to various platforms
- **Project Packaging**: Reproduce ML projects

---

## 🎯 Quick Start

### **1. Start MLflow Tracking Server**

Run the startup script:
```bash
start_mlflow.bat
```

Or manually:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

**Access MLflow UI**: http://localhost:5000

---

## 🔧 Integration with Your EventZilla Project

### **Current Services**
- ✅ **FastAPI**: http://localhost:8000 (ML API)
- ✅ **n8n**: http://localhost:5678 (Workflow automation)
- ✅ **Streamlit**: http://localhost:8502 (ML Dashboard)
- 🆕 **MLflow**: http://localhost:5000 (Experiment tracking)

### **How to Use MLflow with Your ML Models**

#### **Option 1: Track Experiments in Notebooks**

Add this to your Jupyter notebooks:

```python
import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("EventZilla_ML_Experiments")

# Start a run
with mlflow.start_run(run_name="booking_status_classifier"):
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train your model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Model logged with accuracy: {accuracy}")
```

#### **Option 2: Track in FastAPI**

Add to `ML/api/main.py`:

```python
import mlflow

# At the top of your file
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("EventZilla_API_Predictions")

# In your prediction endpoints
@app.post("/predict/classification")
async def predict_classification(data: dict):
    with mlflow.start_run(run_name="classification_prediction"):
        # Log input parameters
        mlflow.log_params(data)
        
        # Make prediction
        prediction = model.predict(...)
        
        # Log prediction result
        mlflow.log_metric("prediction_confidence", confidence)
        
        return {"prediction": prediction}
```

#### **Option 3: Track in Streamlit**

Add to `ML/streamlit_app.py`:

```python
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# In your prediction functions
def make_prediction(features):
    with mlflow.start_run(run_name="streamlit_prediction"):
        mlflow.log_params(features)
        
        prediction = model.predict(features)
        
        mlflow.log_metric("prediction_value", prediction)
        
        return prediction
```

---

## 📁 MLflow Directory Structure

MLflow will create these directories:

```
PI BI NEW (2)/PI BI NEW/
├── mlruns/                    # Experiment tracking data
│   ├── 0/                     # Default experiment
│   ├── 1/                     # Your experiments
│   └── .trash/                # Deleted runs
├── mlartifacts/               # Model artifacts
└── mlflow.db                  # SQLite database (optional)
```

---

## 🎨 MLflow UI Features

### **1. Experiments View**
- Compare multiple runs side-by-side
- Filter and search experiments
- Visualize metrics over time

### **2. Run Details**
- View all logged parameters
- See metrics and charts
- Download model artifacts

### **3. Model Registry**
- Register production models
- Version control for models
- Stage transitions (Staging → Production)

### **4. Compare Runs**
- Select multiple runs
- Compare parameters and metrics
- Identify best performing models

---

## 🔗 Integration with n8n Workflows

You can log n8n workflow predictions to MLflow:

```python
# In your n8n workflow Python scripts
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("n8n_automated_predictions")

with mlflow.start_run(run_name="marketing_pipeline"):
    # Log workflow parameters
    mlflow.log_param("workflow", "marketing")
    mlflow.log_param("user", "ranim_chikhrouhou")
    
    # Make predictions via API
    response = requests.post("http://localhost:8000/predict/segmentation")
    
    # Log results
    mlflow.log_metric("segment_id", response.json()["segment_id"])
    mlflow.log_dict(response.json(), "prediction_result.json")
```

---

## 📊 Example: Track Your Existing Models

### **Track Booking Status Classifier**

```python
import mlflow
import joblib
from pathlib import Path

# Load your existing model
model_path = Path("ML/models_artifacts/rf_status_kpi_pipeline.joblib")
model = joblib.load(model_path)

# Log to MLflow
mlflow.set_experiment("EventZilla_Existing_Models")

with mlflow.start_run(run_name="booking_status_rf"):
    # Log model info
    mlflow.log_param("model_file", "rf_status_kpi_pipeline.joblib")
    mlflow.log_param("model_type", "RandomForest")
    
    # Log the model
    mlflow.sklearn.log_model(model, "booking_status_model")
    
    # Add tags
    mlflow.set_tag("project", "EventZilla")
    mlflow.set_tag("model_purpose", "booking_status_prediction")
    
    print("✅ Model logged to MLflow!")
```

### **Track Price Estimation Model**

```python
with mlflow.start_run(run_name="price_estimation_ridge"):
    model = joblib.load("ML/models_artifacts/ridge_final_price_pipeline.joblib")
    
    mlflow.log_param("model_type", "Ridge Regression")
    mlflow.sklearn.log_model(model, "price_estimation_model")
    
    mlflow.set_tag("project", "EventZilla")
    mlflow.set_tag("model_purpose", "price_estimation")
```

### **Track Customer Segmentation**

```python
with mlflow.start_run(run_name="customer_segmentation_kmeans"):
    model = joblib.load("ML/models_artifacts/kmeans_beneficiaire_pipeline.joblib")
    
    mlflow.log_param("model_type", "KMeans")
    mlflow.log_param("n_clusters", 4)
    mlflow.sklearn.log_model(model, "segmentation_model")
    
    mlflow.set_tag("project", "EventZilla")
    mlflow.set_tag("model_purpose", "customer_segmentation")
```

---

## 🚀 Advanced Features

### **1. Model Registry**

Register your best models:

```python
# After training and logging
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "EventZilla_BookingStatus")
```

### **2. Model Versioning**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to production
client.transition_model_version_stage(
    name="EventZilla_BookingStatus",
    version=1,
    stage="Production"
)
```

### **3. Load Production Models**

```python
import mlflow.pyfunc

# Load latest production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/EventZilla_BookingStatus/Production"
)

# Make predictions
predictions = model.predict(data)
```

---

## 📝 Best Practices

### **1. Naming Conventions**
- **Experiments**: `EventZilla_{Feature}` (e.g., `EventZilla_Classification`)
- **Runs**: `{model_type}_{date}` (e.g., `RandomForest_2026-05-01`)
- **Models**: `EventZilla_{Purpose}` (e.g., `EventZilla_PriceEstimation`)

### **2. What to Log**
- ✅ Model hyperparameters
- ✅ Training/validation metrics
- ✅ Feature importance
- ✅ Confusion matrices
- ✅ Model artifacts (.joblib files)
- ✅ Data preprocessing steps

### **3. Organize Experiments**
- Separate experiments for each ML task
- Use tags to categorize runs
- Add descriptions to experiments

---

## 🔧 Configuration

### **Change MLflow Port**

Edit `start_mlflow.bat`:
```bash
mlflow ui --host 0.0.0.0 --port 5001
```

### **Use Database Backend**

For production, use PostgreSQL or MySQL:
```bash
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow \
  --host 0.0.0.0 \
  --port 5000
```

### **Set Tracking URI**

In your Python code:
```python
import mlflow

# Local file storage (default)
mlflow.set_tracking_uri("file:///path/to/mlruns")

# Remote server
mlflow.set_tracking_uri("http://localhost:5000")

# Database
mlflow.set_tracking_uri("postgresql://user:password@localhost/mlflow")
```

---

## 📚 Useful Commands

```bash
# Start MLflow UI
mlflow ui

# Start with specific port
mlflow ui --port 5001

# Start MLflow server (production)
mlflow server --host 0.0.0.0 --port 5000

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 1

# Delete experiment
mlflow experiments delete --experiment-id 1
```

---

## 🎯 Next Steps

1. **Start MLflow UI**: Run `start_mlflow.bat`
2. **Access Dashboard**: Open http://localhost:5000
3. **Log Your Models**: Add MLflow tracking to your notebooks
4. **Compare Experiments**: Use the UI to compare model performance
5. **Register Best Models**: Move top models to Model Registry
6. **Integrate with API**: Add tracking to FastAPI endpoints

---

## 📖 Resources

- **Official Docs**: https://mlflow.org/docs/latest/index.html
- **Quickstart**: https://mlflow.org/docs/latest/quickstart.html
- **Python API**: https://mlflow.org/docs/latest/python_api/index.html
- **Tracking**: https://mlflow.org/docs/latest/tracking.html
- **Model Registry**: https://mlflow.org/docs/latest/model-registry.html

---

## 🆘 Troubleshooting

### **Port Already in Use**
```bash
# Change port in start_mlflow.bat
mlflow ui --port 5001
```

### **Cannot Access UI**
- Check if MLflow is running
- Verify firewall settings
- Try http://127.0.0.1:5000 instead of localhost

### **Import Errors**
```bash
# Reinstall MLflow
pip install --upgrade mlflow
```

---

**Last Updated**: May 1, 2026  
**MLflow Version**: 3.11.1  
**Status**: ✅ Installed and Ready to Use

