# -*- coding: utf-8 -*-
"""
MLflow Integration for EventZilla API
Handles logging predictions from n8n workflows to MLflow
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


class MLflowLogRequest(BaseModel):
    """Request model for logging to MLflow"""
    experiment_name: str
    run_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Optional[Dict[str, str]] = None
    artifacts: Optional[Dict[str, Any]] = None


class MLflowLogger:
    """MLflow logging utility for n8n workflows"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """Initialize MLflow logger
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    
    def log_prediction(
        self,
        experiment_name: str,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Log a prediction to MLflow
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of this specific run
            params: Parameters to log (model config, input features, etc.)
            metrics: Metrics to log (predictions, scores, etc.)
            tags: Optional tags for categorization
            artifacts: Optional artifacts to save (JSON data, etc.)
        
        Returns:
            Dictionary with run_id and experiment_id
        """
        try:
            # Set experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run
            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
                
                # Log tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # Add default tags
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                mlflow.set_tag("source", "n8n_workflow")
                
                # Log artifacts as JSON
                if artifacts:
                    import json
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(artifacts, f, indent=2)
                        temp_path = f.name
                    mlflow.log_artifact(temp_path, "prediction_data")
                
                return {
                    "status": "success",
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "tracking_uri": self.tracking_uri
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tracking_uri": self.tracking_uri
            }
    
    def log_finance_pipeline(
        self,
        user: str,
        regression_result: Dict[str, Any],
        timeseries_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Log finance pipeline predictions to MLflow
        
        Args:
            user: Username who triggered the workflow
            regression_result: Price prediction result
            timeseries_result: Time series forecast result
        
        Returns:
            Dictionary with logging status
        """
        run_name = f"finance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        params = {
            "workflow": "finance",
            "user": user,
            "model_regression": regression_result.get("modele", "unknown"),
            "model_timeseries": timeseries_result.get("modele_champion", "unknown"),
            "horizon_months": timeseries_result.get("horizon_mois", 0)
        }
        
        metrics = {
            "predicted_amount": float(regression_result.get("montant_predit", 0)),
            "timeseries_mape": float(timeseries_result.get("metriques_test", {}).get("MAPE", 0)),
            "timeseries_rmse": float(timeseries_result.get("metriques_test", {}).get("RMSE", 0)),
            "timeseries_mae": float(timeseries_result.get("metriques_test", {}).get("MAE", 0))
        }
        
        tags = {
            "pipeline": "finance",
            "automated": "true",
            "source": "n8n"
        }
        
        artifacts = {
            "regression": regression_result,
            "timeseries": timeseries_result,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.log_prediction(
            experiment_name="n8n_Finance_Pipeline",
            run_name=run_name,
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts=artifacts
        )
    
    def log_marketing_pipeline(
        self,
        user: str,
        segmentation_result: Dict[str, Any],
        classification_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Log marketing pipeline predictions to MLflow"""
        run_name = f"marketing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        params = {
            "workflow": "marketing",
            "user": user,
            "model_segmentation": segmentation_result.get("modele", "unknown"),
            "model_classification": classification_result.get("modele", "unknown"),
            "segment_id": segmentation_result.get("segment_id", -1)
        }
        
        metrics = {
            "segment_id": float(segmentation_result.get("segment_id", 0)),
            "classification_confidence": max(classification_result.get("probabilites", {}).values()) if classification_result.get("probabilites") else 0.0
        }
        
        tags = {
            "pipeline": "marketing",
            "automated": "true",
            "source": "n8n",
            "segment_label": segmentation_result.get("segment_label", "unknown")
        }
        
        artifacts = {
            "segmentation": segmentation_result,
            "classification": classification_result,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.log_prediction(
            experiment_name="n8n_Marketing_Pipeline",
            run_name=run_name,
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts=artifacts
        )
    
    def log_crm_pipeline(
        self,
        user: str,
        classification_result: Dict[str, Any],
        segmentation_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Log CRM pipeline predictions to MLflow"""
        run_name = f"crm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        params = {
            "workflow": "crm",
            "user": user,
            "model_classification": classification_result.get("modele", "unknown"),
            "model_segmentation": segmentation_result.get("modele", "unknown"),
            "predicted_status": classification_result.get("statut_predit", "unknown")
        }
        
        metrics = {
            "classification_confidence": max(classification_result.get("probabilites", {}).values()) if classification_result.get("probabilites") else 0.0,
            "segment_id": float(segmentation_result.get("segment_id", 0))
        }
        
        tags = {
            "pipeline": "crm",
            "automated": "true",
            "source": "n8n",
            "predicted_status": classification_result.get("statut_predit", "unknown")
        }
        
        artifacts = {
            "classification": classification_result,
            "segmentation": segmentation_result,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.log_prediction(
            experiment_name="n8n_CRM_Pipeline",
            run_name=run_name,
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts=artifacts
        )


# Global MLflow logger instance
mlflow_logger = MLflowLogger()
