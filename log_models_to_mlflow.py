"""
Script to log existing EventZilla ML models to MLflow
Run this after starting MLflow UI to register all your trained models
"""
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from datetime import datetime

# Set MLflow tracking URI (change if using remote server)
mlflow.set_tracking_uri("http://localhost:5000")

# Define models to log
MODELS_DIR = Path("ML/models_artifacts")

MODELS = {
    "Booking Status Classifier": {
        "file": "rf_status_kpi_pipeline.joblib",
        "type": "RandomForest",
        "purpose": "booking_status_prediction",
        "description": "Predicts booking status (confirmed, pending, cancelled)",
        "experiment": "EventZilla_Classification"
    },
    "Price Estimation": {
        "file": "ridge_final_price_pipeline.joblib",
        "type": "Ridge Regression",
        "purpose": "price_estimation",
        "description": "Estimates final price for events",
        "experiment": "EventZilla_Regression"
    },
    "Customer Segmentation (Beneficiary)": {
        "file": "kmeans_beneficiaire_pipeline.joblib",
        "type": "KMeans",
        "purpose": "customer_segmentation",
        "description": "Segments beneficiaries into loyalty groups",
        "experiment": "EventZilla_Segmentation"
    },
    "Customer Segmentation (Visitor)": {
        "file": "kmeans_visiteur_pipeline.joblib",
        "type": "KMeans",
        "purpose": "visitor_segmentation",
        "description": "Segments visitors into behavior groups",
        "experiment": "EventZilla_Segmentation"
    }
}

def log_model_to_mlflow(model_name, model_info):
    """Log a single model to MLflow"""
    model_path = MODELS_DIR / model_info["file"]
    
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Set experiment
        mlflow.set_experiment(model_info["experiment"])
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_info["type"])
            mlflow.log_param("model_file", model_info["file"])
            mlflow.log_param("purpose", model_info["purpose"])
            
            # Log the model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"EventZilla_{model_info['purpose']}"
            )
            
            # Add tags
            mlflow.set_tag("project", "EventZilla")
            mlflow.set_tag("model_purpose", model_info["purpose"])
            mlflow.set_tag("description", model_info["description"])
            mlflow.set_tag("logged_date", datetime.now().strftime("%Y-%m-%d"))
            
            print(f"✅ Successfully logged: {model_name}")
            return True
            
    except Exception as e:
        print(f"❌ Error logging {model_name}: {str(e)}")
        return False

def main():
    """Main function to log all models"""
    print("=" * 60)
    print("🚀 Logging EventZilla ML Models to MLflow")
    print("=" * 60)
    print()
    
    # Check if MLflow is accessible
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        print("✅ MLflow tracking server is accessible")
        print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")
        print()
    except Exception as e:
        print("❌ Cannot connect to MLflow server!")
        print("   Make sure MLflow is running: start_mlflow.bat")
        print(f"   Error: {str(e)}")
        return
    
    # Log each model
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        print(f"📦 Logging: {model_name}")
        if log_model_to_mlflow(model_name, model_info):
            success_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"✅ Successfully logged: {success_count}/{total_count} models")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("1. Open MLflow UI: http://localhost:5000")
    print("2. View your experiments and models")
    print("3. Compare model performance")
    print("4. Register best models to Model Registry")
    print()

if __name__ == "__main__":
    main()
