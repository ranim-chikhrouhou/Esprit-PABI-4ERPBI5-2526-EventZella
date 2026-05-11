"""
Enhanced script to log EventZilla ML models to MLflow WITH metrics
This version includes all performance metrics from your JSON files
"""
import mlflow
import mlflow.sklearn
import joblib
import json
from pathlib import Path
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Define paths
MODELS_DIR = Path("ML/models_artifacts")

def log_classification_model():
    """Log classification model with metrics"""
    print("📦 Logging: Booking Status Classifier (with metrics)")
    
    # Load model
    model_path = MODELS_DIR / "classification_status_champion_pipeline.joblib"
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    # Load metrics
    metrics_path = MODELS_DIR / "metrics_classification.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    
    try:
        model = joblib.load(model_path)
        
        # Set experiment
        mlflow.set_experiment("EventZilla_Classification")
        
        # Start run
        with mlflow.start_run(run_name=f"BookingStatus_RandomForest_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("champion_model", metrics.get("champion_model", "RandomForest"))
            mlflow.log_param("criterion", metrics.get("criterion", "C"))
            mlflow.log_param("task", "classification")
            mlflow.log_param("target", "booking_status")
            
            # Log best params from GridSearch
            best_params = metrics.get("gridsearch_rf_best_params", {})
            for key, value in best_params.items():
                mlflow.log_param(key, value)
            
            # Log test metrics
            test_metrics = metrics.get("test_metrics_champion", {})
            mlflow.log_metric("accuracy", test_metrics.get("accuracy", 0))
            mlflow.log_metric("precision_weighted", test_metrics.get("precision_weighted", 0))
            mlflow.log_metric("recall_weighted", test_metrics.get("recall_weighted", 0))
            mlflow.log_metric("f1_weighted", test_metrics.get("f1_weighted", 0))
            mlflow.log_metric("roc_auc", test_metrics.get("roc_auc", 0))
            
            # Log classes
            classes = metrics.get("classes", [])
            mlflow.log_param("classes", ", ".join(classes))
            mlflow.log_param("n_classes", len(classes))
            
            # Log tags
            mlflow.set_tag("project", "EventZilla")
            mlflow.set_tag("model_purpose", "booking_status_prediction")
            mlflow.set_tag("kpi_alignment", metrics.get("kpi_alignment", ""))
            mlflow.set_tag("logged_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("model_file", "classification_status_champion_pipeline.joblib")
            
            # Log the model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="EventZilla_booking_status_prediction"
            )
            
            # Log metrics JSON as artifact
            mlflow.log_dict(metrics, "metrics_classification.json")
            
            print("✅ Successfully logged: Booking Status Classifier")
            print(f"   - Accuracy: {test_metrics.get('accuracy', 0):.3f}")
            print(f"   - F1 Score: {test_metrics.get('f1_weighted', 0):.3f}")
            print(f"   - ROC AUC: {test_metrics.get('roc_auc', 0):.3f}")
            return True
            
    except Exception as e:
        print(f"❌ Error logging classification model: {str(e)}")
        return False


def log_regression_model():
    """Log regression model with metrics"""
    print("\n📦 Logging: Price Estimation (Ridge) with metrics")
    
    # Load model
    model_path = MODELS_DIR / "ridge_regression_primary.joblib"
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    # Load metrics
    metrics_path = MODELS_DIR / "metrics_regression.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    
    try:
        model = joblib.load(model_path)
        
        # Set experiment
        mlflow.set_experiment("EventZilla_Regression")
        
        # Start run
        with mlflow.start_run(run_name=f"PriceEstimation_Ridge_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("champion_model", metrics.get("champion_model", "Ridge"))
            mlflow.log_param("criterion", metrics.get("criterion", "D"))
            mlflow.log_param("task", "regression")
            mlflow.log_param("target", metrics.get("target", "final_price"))
            
            # Log features
            features = metrics.get("features", [])
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("features", ", ".join(features[:5]) + "...")  # First 5
            
            # Log test metrics
            test_metrics = metrics.get("test_champion", {})
            mlflow.log_metric("r2_score", test_metrics.get("r2", 0))
            mlflow.log_metric("rmse", test_metrics.get("rmse", 0))
            mlflow.log_metric("mae", test_metrics.get("mae", 0))
            mlflow.log_metric("mse", test_metrics.get("mse", 0))
            
            # Log CV metrics
            cv_metrics = metrics.get("cv_ridge", {})
            mlflow.log_metric("cv_r2_mean", cv_metrics.get("cv_r2_mean", 0))
            mlflow.log_metric("cv_rmse_mean", cv_metrics.get("cv_rmse_mean", 0))
            mlflow.log_metric("cv_mae_mean", cv_metrics.get("cv_mae_mean", 0))
            
            # Log tags
            mlflow.set_tag("project", "EventZilla")
            mlflow.set_tag("model_purpose", "price_estimation")
            mlflow.set_tag("kpi_alignment", metrics.get("kpi_alignment", ""))
            mlflow.set_tag("logged_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("model_file", "ridge_regression_primary.joblib")
            
            # Log the model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="EventZilla_price_estimation"
            )
            
            # Log metrics JSON as artifact
            mlflow.log_dict(metrics, "metrics_regression.json")
            
            print("✅ Successfully logged: Price Estimation (Ridge)")
            print(f"   - R² Score: {test_metrics.get('r2', 0):.6f}")
            print(f"   - RMSE: {test_metrics.get('rmse', 0):.2f}")
            print(f"   - MAE: {test_metrics.get('mae', 0):.2f}")
            return True
            
    except Exception as e:
        print(f"❌ Error logging regression model: {str(e)}")
        return False


def log_clustering_beneficiary():
    """Log beneficiary clustering model with metrics"""
    print("\n📦 Logging: Customer Segmentation (Beneficiary) with metrics")
    
    # Load model
    model_path = MODELS_DIR / "kmeans_loyalty_beneficiary.joblib"
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    # Load metrics
    metrics_path = MODELS_DIR / "metrics_clustering.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    
    # Load segment labels
    labels_path = MODELS_DIR / "clustering_segment_labels_loyalty_beneficiary.json"
    segment_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    
    try:
        model = joblib.load(model_path)
        
        # Set experiment
        mlflow.set_experiment("EventZilla_Segmentation")
        
        # Start run
        with mlflow.start_run(run_name=f"Segmentation_Beneficiary_KMeans_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_param("model_type", "KMeans")
            mlflow.log_param("segment_type", "beneficiary")
            mlflow.log_param("n_clusters", metrics.get("k", 3))
            mlflow.log_param("task", "clustering")
            
            # Log metrics
            mlflow.log_metric("silhouette_score", metrics.get("silhouette", 0))
            mlflow.log_metric("silhouette_train", metrics.get("silhouette_train", 0))
            mlflow.log_metric("silhouette_holdout", metrics.get("silhouette_holdout", 0))
            mlflow.log_metric("davies_bouldin", metrics.get("davies_bouldin_kmeans", 0))
            mlflow.log_metric("n_samples", metrics.get("n_samples", 0))
            
            # Log segment labels
            for seg_id, seg_info in segment_labels.items():
                mlflow.log_param(f"segment_{seg_id}_label", seg_info.get("label", ""))
            
            # Log tags
            mlflow.set_tag("project", "EventZilla")
            mlflow.set_tag("model_purpose", "customer_segmentation")
            mlflow.set_tag("segment_type", "beneficiary")
            mlflow.set_tag("kpi_alignment", metrics.get("kpi_alignment", ""))
            mlflow.set_tag("logged_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("model_file", "kmeans_loyalty_beneficiary.joblib")
            
            # Log the model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="EventZilla_customer_segmentation_beneficiary"
            )
            
            # Log artifacts
            mlflow.log_dict(metrics, "metrics_clustering.json")
            mlflow.log_dict(segment_labels, "segment_labels.json")
            
            print("✅ Successfully logged: Customer Segmentation (Beneficiary)")
            print(f"   - Silhouette Score: {metrics.get('silhouette', 0):.3f}")
            print(f"   - Number of Clusters: {metrics.get('k', 3)}")
            print(f"   - Samples: {metrics.get('n_samples', 0)}")
            return True
            
    except Exception as e:
        print(f"❌ Error logging clustering model: {str(e)}")
        return False


def log_clustering_provider():
    """Log provider clustering model with metrics"""
    print("\n📦 Logging: Provider Segmentation with metrics")
    
    # Load model
    model_path = MODELS_DIR / "kmeans_loyalty_provider.joblib"
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return False
    
    # Load segment labels
    labels_path = MODELS_DIR / "clustering_segment_labels_loyalty_provider.json"
    segment_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    
    try:
        model = joblib.load(model_path)
        
        # Set experiment
        mlflow.set_experiment("EventZilla_Segmentation")
        
        # Start run
        with mlflow.start_run(run_name=f"Segmentation_Provider_KMeans_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_param("model_type", "KMeans")
            mlflow.log_param("segment_type", "provider")
            mlflow.log_param("n_clusters", len(segment_labels))
            mlflow.log_param("task", "clustering")
            
            # Log segment labels
            for seg_id, seg_info in segment_labels.items():
                mlflow.log_param(f"segment_{seg_id}_label", seg_info.get("label", ""))
            
            # Log tags
            mlflow.set_tag("project", "EventZilla")
            mlflow.set_tag("model_purpose", "provider_segmentation")
            mlflow.set_tag("segment_type", "provider")
            mlflow.set_tag("logged_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("model_file", "kmeans_loyalty_provider.joblib")
            
            # Log the model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="EventZilla_provider_segmentation"
            )
            
            # Log artifacts
            mlflow.log_dict(segment_labels, "segment_labels.json")
            
            print("✅ Successfully logged: Provider Segmentation")
            print(f"   - Number of Clusters: {len(segment_labels)}")
            return True
            
    except Exception as e:
        print(f"❌ Error logging provider clustering model: {str(e)}")
        return False


def main():
    """Main function to log all models with metrics"""
    print("=" * 60)
    print("🚀 Logging EventZilla ML Models to MLflow (WITH METRICS)")
    print("=" * 60)
    print()
    
    # Check MLflow connection
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
    results = []
    results.append(log_classification_model())
    results.append(log_regression_model())
    results.append(log_clustering_beneficiary())
    results.append(log_clustering_provider())
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    print()
    print("=" * 60)
    print(f"✅ Successfully logged: {success_count}/{total_count} models")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("1. Open MLflow UI: http://localhost:5000")
    print("2. Click 'Experiments' tab")
    print("3. View your experiments with METRICS:")
    print("   - EventZilla_Classification")
    print("   - EventZilla_Regression")
    print("   - EventZilla_Segmentation")
    print()
    print("4. Click 'Models' tab to see registered models")
    print()
    print("📊 Now you can see:")
    print("   ✅ Model parameters")
    print("   ✅ Performance metrics (accuracy, R², silhouette, etc.)")
    print("   ✅ Model artifacts")
    print("   ✅ Comparison charts")
    print()


if __name__ == "__main__":
    main()
