"""
Automated Training Pipeline for EventZilla ML Models
End-to-end pipeline: Data Loading → Preprocessing → Training → Evaluation → MLflow Logging → Model Saving
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
import joblib
import json
import os

from ML.mlflow_tracking_uri import ensure_artifact_dir, get_tracking_uri

# Configuration — SQLite par défaut (Overview MLflow UI)
REPO_ROOT = Path(__file__).resolve().parent
MLRUNS_DIR = REPO_ROOT / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)

ensure_artifact_dir(REPO_ROOT)
MLFLOW_URI = get_tracking_uri(REPO_ROOT)
MODELS_DIR = REPO_ROOT / "ML" / "models_artifacts"
DATA_DIR = REPO_ROOT / "ML" / "data_processed"

mlflow.set_tracking_uri(MLFLOW_URI)

class AutomatedTrainingPipeline:
    """Automated ML training pipeline with MLflow tracking"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load training data from processed files"""
        print("📊 Loading data...")
        
        # Load your processed data
        # Replace with actual data loading logic
        try:
            classification_data = pd.read_parquet(DATA_DIR / "classification_data.parquet")
            regression_data = pd.read_parquet(DATA_DIR / "regression_data.parquet")
            clustering_data = pd.read_parquet(DATA_DIR / "clustering_data.parquet")
            
            print(f"   ✅ Classification data: {classification_data.shape}")
            print(f"   ✅ Regression data: {regression_data.shape}")
            print(f"   ✅ Clustering data: {clustering_data.shape}")
            
            return classification_data, regression_data, clustering_data
        except Exception as e:
            print(f"   ⚠️  Using dummy data (real data not found): {e}")
            # Create dummy data for demonstration
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data matching EventZilla real feature schema"""
        rng = np.random.default_rng(42)
        n_samples = 1200

        event_budget  = rng.uniform(1_000, 20_000, n_samples)
        service_price = event_budget * rng.uniform(0.15, 0.45, n_samples)
        benchmark_avg = service_price * rng.uniform(0.85, 1.20, n_samples)
        commission    = service_price * rng.uniform(0.05, 0.15, n_samples)
        final_price   = service_price + commission

        statuses = np.array(["confirmed", "pending", "cancelled"])
        status = rng.choice(statuses, n_samples)

        clf_data = pd.DataFrame({
            "id_date":             rng.integers(1, 730, n_samples).astype(float),
            "id_event":            rng.integers(1, 150, n_samples).astype(float),
            "id_servicecategory":  rng.integers(1, 12, n_samples).astype(float),
            "id_benchmark":        rng.integers(1, 50, n_samples).astype(float),
            "id_provider":         rng.integers(1, 80, n_samples).astype(float),
            "final_price":         final_price,
            "service_price":       service_price,
            "benchmark_avg_price": benchmark_avg,
            "event_budget":        event_budget,
            "cal_month":           rng.integers(1, 13, n_samples).astype(float),
            "cal_year":            rng.choice([2023, 2024, 2025], n_samples).astype(float),
            "quarter":             rng.integers(1, 5, n_samples).astype(float),
            "status":              status,
        })

        reg_data = pd.DataFrame({
            "id_date":             rng.integers(1, 730, n_samples).astype(float),
            "id_event":            rng.integers(1, 150, n_samples).astype(float),
            "id_servicecategory":  rng.integers(1, 12, n_samples).astype(float),
            "id_benchmark":        rng.integers(1, 50, n_samples).astype(float),
            "id_provider":         rng.integers(1, 80, n_samples).astype(float),
            "service_price":       service_price,
            "benchmark_avg_price": benchmark_avg,
            "event_budget":        event_budget,
            "cal_month":           rng.integers(1, 13, n_samples).astype(float),
            "cal_year":            rng.choice([2023, 2024, 2025], n_samples).astype(float),
            "quarter":             rng.integers(1, 5, n_samples).astype(float),
            "commission_margin":   commission,
            "final_price":         final_price,
        })

        clust_data = pd.DataFrame({
            "nb_reservations_loyalty":          rng.integers(1, 60, n_samples).astype(float),
            "ca_total_loyalty":                 rng.uniform(1000, 50000, n_samples),
            "panier_moyen_loyalty":             rng.uniform(500, 3000, n_samples),
            "recency_days_loyalty":             rng.integers(1, 365, n_samples).astype(float),
            "avg_nb_visitors_loyalty":          rng.uniform(20, 200, n_samples),
            "volume_reservations_site_loyalty": rng.integers(1, 30, n_samples).astype(float),
        })

        return clf_data, reg_data, clust_data
    
    def train_classification_model(self, data):
        """Train classification model with MLflow tracking"""
        print("\n🎯 Training Classification Model...")

        clf_features = [
            "id_date", "id_event", "id_servicecategory", "id_benchmark",
            "id_provider", "final_price", "service_price", "benchmark_avg_price",
            "event_budget", "cal_month", "cal_year", "quarter",
        ]
        mlflow.set_experiment("EventZilla_Automated_Training_Classification")

        with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            X = data[clf_features]
            y = data['status']

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", len(X))

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # Noms EXACTS attendus par ML/api/main.py
            model_path = self.models_dir / "classification_status_champion_pipeline.joblib"
            le_path    = self.models_dir / "label_encoder_status.joblib"
            joblib.dump(model, model_path)
            joblib.dump(le, le_path)

            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name="EventZilla_Classification_Automated"
            )

            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   ✅ F1 Score: {f1:.3f}")
            print(f"   ✅ Model saved: {model_path}")

            return model, accuracy, f1
    
    def train_regression_model(self, data):
        """Train regression model with MLflow tracking"""
        print("\n💰 Training Regression Model...")

        reg_features = [
            "id_date", "id_event", "id_servicecategory", "id_benchmark",
            "id_provider", "service_price", "benchmark_avg_price", "event_budget",
            "cal_month", "cal_year", "quarter", "commission_margin",
        ]
        mlflow.set_experiment("EventZilla_Automated_Training_Regression")

        with mlflow.start_run(run_name=f"Ridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            X = data[reg_features]
            y = data['final_price']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae  = float(np.mean(np.abs(y_test - y_pred)))

            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", 1.0)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", len(X))

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # Nom EXACT attendu par ML/api/main.py
            model_path = self.models_dir / "ridge_regression_primary.joblib"
            joblib.dump(model, model_path)

            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name="EventZilla_Regression_Automated"
            )

            print(f"   ✅ R² Score: {r2:.6f}")
            print(f"   ✅ RMSE: {rmse:.2f}")
            print(f"   ✅ MAE: {mae:.2f}")
            print(f"   ✅ Model saved: {model_path}")

            return model, r2, rmse
    
    def train_clustering_model(self, data):
        """Train clustering model with MLflow tracking (bénéficiaires + prestataires)"""
        print("\n👥 Training Clustering Model...")

        cluster_features = [
            "nb_reservations_loyalty", "ca_total_loyalty", "panier_moyen_loyalty",
            "recency_days_loyalty", "avg_nb_visitors_loyalty", "volume_reservations_site_loyalty",
        ]
        mlflow.set_experiment("EventZilla_Automated_Training_Clustering")

        best_sil = -1
        for suffix in ("beneficiary", "provider"):
            with mlflow.start_run(run_name=f"KMeans_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                X = data[cluster_features].copy()

                imputer = SimpleImputer(strategy="median")
                X_imp   = imputer.fit_transform(X)
                scaler  = StandardScaler()
                X_scaled = scaler.fit_transform(X_imp)

                n_clusters = 4
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                clusters = model.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, clusters)

                mlflow.log_param("model_type", "KMeans")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param("entity", suffix)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_samples", len(X))
                mlflow.log_metric("silhouette_score", silhouette)

                # Noms EXACTS attendus par ML/api/main.py
                joblib.dump(model,   self.models_dir / f"kmeans_loyalty_{suffix}.joblib")
                joblib.dump(scaler,  self.models_dir / f"kmeans_standard_scaler_loyalty_{suffix}.joblib")
                joblib.dump(imputer, self.models_dir / f"kmeans_median_imputer_loyalty_{suffix}.joblib")

                mlflow.sklearn.log_model(model, "model",
                    registered_model_name=f"EventZilla_Clustering_{suffix.capitalize()}_Automated")

                print(f"   ✅ Silhouette Score ({suffix}): {silhouette:.3f}")
                if silhouette > best_sil:
                    best_sil = silhouette

        return model, best_sil
    
    def run_full_pipeline(self):
        """Run complete automated training pipeline"""
        print("=" * 60)
        print("🚀 EventZilla Automated Training Pipeline")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Load data
            clf_data, reg_data, clust_data = self.load_data()
            
            # Train models
            clf_model, clf_acc, clf_f1 = self.train_classification_model(clf_data)
            reg_model, reg_r2, reg_rmse = self.train_regression_model(reg_data)
            clust_model, clust_sil = self.train_clustering_model(clust_data)
            
            # Summary
            print()
            print("=" * 60)
            print("✅ Pipeline Completed Successfully!")
            print("=" * 60)
            print()
            print("📊 Results Summary:")
            print(f"   Classification - Accuracy: {clf_acc:.3f}, F1: {clf_f1:.3f}")
            print(f"   Regression - R²: {reg_r2:.6f}, RMSE: {reg_rmse:.2f}")
            print(f"   Clustering - Silhouette: {clust_sil:.3f}")
            print()
            print("🔗 View results in MLflow: http://localhost:5000")
            print()
            print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print()
            print("=" * 60)
            print("❌ Pipeline Failed!")
            print("=" * 60)
            print(f"Error: {str(e)}")
            return False


def main():
    """Main entry point"""
    pipeline = AutomatedTrainingPipeline()
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n✨ All models trained and logged to MLflow!")
        print("   Next: Check http://localhost:5000 to see your experiments")
    else:
        print("\n⚠️  Pipeline encountered errors. Check logs above.")


if __name__ == "__main__":
    main()
