# 🔍 How to Find Models in MLflow

## 📍 Quick Answer

**Open MLflow UI**: http://localhost:5000

Then follow one of these paths:

---

## 🎯 Method 1: View Models in Model Registry (Recommended)

### **Step 1: Open MLflow**
Go to: **http://localhost:5000**

### **Step 2: Click "Models" Tab**
At the top of the page, you'll see tabs:
- Experiments
- **Models** ← Click here!

### **Step 3: See Your Registered Models**
You'll see a list of all registered models:

```
┌─────────────────────────────────────────────────────┐
│ Models                                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📦 EventZilla_booking_status_prediction            │
│    Version 1 - Stage: None                         │
│    Created: 2026-05-01                             │
│                                                     │
│ 📦 EventZilla_price_estimation                     │
│    Version 1 - Stage: Production                   │
│    Created: 2026-05-01                             │
│                                                     │
│ 📦 EventZilla_customer_segmentation                │
│    Version 1 - Stage: Staging                      │
│    Created: 2026-05-01                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### **Step 4: Click a Model**
Click on any model name to see:
- All versions
- Model details
- Artifacts (the actual .joblib file)
- Metadata
- Performance metrics

---

## 🎯 Method 2: View Models in Experiments

### **Step 1: Open MLflow**
Go to: **http://localhost:5000**

### **Step 2: Click "Experiments" Tab**
You'll see experiments like:
- EventZilla_Classification
- EventZilla_Regression
- EventZilla_Segmentation
- n8n_Finance_Pipeline

### **Step 3: Click an Experiment**
Example: Click "EventZilla_Classification"

### **Step 4: See All Runs**
You'll see a table with all training runs:

```
┌──────────────────────────────────────────────────────────┐
│ Run Name                    │ Metrics      │ Parameters  │
├──────────────────────────────────────────────────────────┤
│ Booking Status_20260501     │ accuracy: -  │ model: RF   │
│ Price Estimation_20260501   │ r2: 0.98     │ model: Ridge│
│ Segmentation_20260501       │ silhouette:  │ clusters: 4 │
└──────────────────────────────────────────────────────────┘
```

### **Step 5: Click a Run**
Click any run name to see:
- Parameters
- Metrics
- **Artifacts** ← Your model is here!
- Tags
- Notes

### **Step 6: Download Model**
In the run details:
1. Scroll to "Artifacts" section
2. Click "model" folder
3. See your model files
4. Click to download

---

## 🎯 Method 3: Search for Models

### **In Model Registry:**
1. Go to "Models" tab
2. Use search box at top
3. Type model name (e.g., "booking")
4. See filtered results

### **In Experiments:**
1. Go to "Experiments" tab
2. Use filter options
3. Filter by:
   - Metrics (e.g., accuracy > 0.9)
   - Parameters (e.g., model_type = "Ridge")
   - Tags (e.g., source = "n8n")

---

## 📊 Your Current Models

Based on your files, you have these models:

### **1. Classification Models**
- `classification_status_champion_pipeline.joblib`
- `rf_status_kpi_pipeline.joblib`

**In MLflow:**
- Experiment: "EventZilla_Classification"
- Registered as: "EventZilla_booking_status_prediction"

### **2. Regression Models**
- `ridge_regression_primary.joblib`
- `rf_regression_primary.joblib`
- `rf_panier_kpi_pipeline.joblib`

**To log to MLflow:**
- Experiment: "EventZilla_Regression"
- Register as: "EventZilla_price_estimation"

### **3. Clustering Models**
- `kmeans_loyalty_beneficiary.joblib`
- `kmeans_loyalty_provider.joblib`
- `kmeans_kpi_segments.joblib`
- `kmeans_segment.joblib`

**To log to MLflow:**
- Experiment: "EventZilla_Segmentation"
- Register as: "EventZilla_customer_segmentation"

---

## 🔍 Detailed Navigation Guide

### **MLflow UI Structure:**

```
http://localhost:5000
│
├── 📊 Experiments Tab
│   ├── EventZilla_Classification
│   │   └── Runs
│   │       ├── Booking Status_20260501
│   │       │   ├── Parameters
│   │       │   ├── Metrics
│   │       │   └── Artifacts ← Model here!
│   │       └── ...
│   │
│   ├── EventZilla_Regression
│   ├── EventZilla_Segmentation
│   └── n8n_Finance_Pipeline
│
├── 📦 Models Tab (Model Registry)
│   ├── EventZilla_booking_status_prediction
│   │   ├── Version 1
│   │   │   ├── Details
│   │   │   ├── Artifacts ← Model here!
│   │   │   └── Metadata
│   │   └── Version 2
│   │
│   ├── EventZilla_price_estimation
│   └── EventZilla_customer_segmentation
│
└── 🔍 Search & Filter
    ├── Search by name
    ├── Filter by metrics
    └── Filter by tags
```

---

## 💡 Quick Actions

### **View Latest Model:**
1. Go to "Models" tab
2. Click model name
3. See "Latest Version"

### **Compare Models:**
1. Go to "Experiments" tab
2. Select experiment
3. Check multiple runs
4. Click "Compare" button

### **Download Model:**
1. Go to run details
2. Click "Artifacts"
3. Click "model" folder
4. Click file to download

### **Load Model in Python:**
```python
import mlflow.pyfunc

# Load by run ID
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# Load from Model Registry
model = mlflow.pyfunc.load_model(
    "models:/EventZilla_booking_status_prediction/1"
)

# Load latest production model
model = mlflow.pyfunc.load_model(
    "models:/EventZilla_booking_status_prediction/Production"
)
```

---

## 🎨 Visual Guide

### **Finding Model in Experiments:**

```
1. Open MLflow UI
   http://localhost:5000
   
2. Click "Experiments"
   ┌─────────────────────────┐
   │ [Experiments] [Models]  │
   └─────────────────────────┘
   
3. Click Experiment Name
   ┌─────────────────────────────┐
   │ EventZilla_Classification   │ ← Click
   │ EventZilla_Regression       │
   │ EventZilla_Segmentation     │
   └─────────────────────────────┘
   
4. Click Run Name
   ┌──────────────────────────────┐
   │ Booking Status_20260501      │ ← Click
   └──────────────────────────────┘
   
5. Scroll to Artifacts
   ┌──────────────────────────────┐
   │ Artifacts                    │
   │ └─ model/                    │ ← Your model!
   │    ├─ MLmodel                │
   │    ├─ model.pkl              │
   │    └─ requirements.txt       │
   └──────────────────────────────┘
```

### **Finding Model in Registry:**

```
1. Open MLflow UI
   http://localhost:5000
   
2. Click "Models"
   ┌─────────────────────────┐
   │ [Experiments] [Models]  │ ← Click
   └─────────────────────────┘
   
3. See All Models
   ┌────────────────────────────────────────┐
   │ EventZilla_booking_status_prediction   │ ← Click
   │ EventZilla_price_estimation            │
   │ EventZilla_customer_segmentation       │
   └────────────────────────────────────────┘
   
4. See Versions
   ┌──────────────────────────────┐
   │ Version 1 - Stage: None      │ ← Click
   │ Version 2 - Stage: Production│
   └──────────────────────────────┘
   
5. View Model Details
   ┌──────────────────────────────┐
   │ Source Run: abc123...        │
   │ Created: 2026-05-01          │
   │ [View Artifacts]             │ ← Click
   └──────────────────────────────┘
```

---

## 🧪 Test: Find Your Model

Let's find the model we just logged!

### **Step-by-Step:**

1. **Open MLflow**: http://localhost:5000

2. **Go to Experiments**

3. **Find "EventZilla_Classification"**

4. **Click on it**

5. **See run: "Booking Status Classifier_20260501"**

6. **Click the run**

7. **Scroll down to "Artifacts"**

8. **Click "model" folder**

9. **See your model files!** 🎉

---

## 📱 Direct Links

After logging models, you can access them directly:

### **Experiments:**
- http://localhost:5000/#/experiments/2 (Classification)
- http://localhost:5000/#/experiments/3 (Regression)
- http://localhost:5000/#/experiments/4 (Segmentation)

### **Models:**
- http://localhost:5000/#/models/EventZilla_booking_status_prediction
- http://localhost:5000/#/models/EventZilla_price_estimation
- http://localhost:5000/#/models/EventZilla_customer_segmentation

---

## 🆘 Troubleshooting

### **Problem: No models showing**
**Solution:**
1. Check MLflow is running: http://localhost:5000
2. Log models: `python log_models_to_mlflow.py`
3. Refresh browser

### **Problem: Can't find specific model**
**Solution:**
1. Use search box in "Models" tab
2. Check "Experiments" tab
3. Look in correct experiment

### **Problem: Model shows but can't download**
**Solution:**
1. Click run in experiment
2. Go to "Artifacts" section
3. Click "model" folder
4. Right-click file → Save

---

## ✨ Pro Tips

### **Tip 1: Use Tags**
Tag your models for easy finding:
```python
mlflow.set_tag("team", "finance")
mlflow.set_tag("version", "v2.0")
mlflow.set_tag("production", "true")
```

### **Tip 2: Use Naming Convention**
- Experiments: `EventZilla_{Purpose}`
- Runs: `{model_type}_{date}`
- Models: `EventZilla_{purpose}`

### **Tip 3: Register Important Models**
Only register models you want to track versions:
```python
mlflow.register_model(model_uri, "EventZilla_ModelName")
```

### **Tip 4: Use Model Stages**
- **None**: Development
- **Staging**: Testing
- **Production**: Live
- **Archived**: Old versions

---

## 📚 Summary

**To find models in MLflow:**

1. **Model Registry** (Best for production models)
   - Go to "Models" tab
   - See all registered models
   - Click to view versions

2. **Experiments** (Best for all runs)
   - Go to "Experiments" tab
   - Click experiment
   - Click run
   - View artifacts

3. **Search** (Best for finding specific models)
   - Use search box
   - Filter by metrics/tags
   - Find quickly

---

**Your model is logged! Go check it out:** http://localhost:5000

