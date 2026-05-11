# -*- coding: utf-8 -*-
"""
MLflow API Endpoints for EventZilla
Add these endpoints to your FastAPI main.py
"""
from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ML.api.mlflow_integration import mlflow_logger, MLflowLogRequest
from ML.api.auth_sql import get_current_user


# Create router
router = APIRouter(prefix="/mlflow", tags=["MLflow"])


class FinancePipelineLog(BaseModel):
    """Finance pipeline logging request"""
    regression: Dict[str, Any]
    timeseries: Dict[str, Any]


class MarketingPipelineLog(BaseModel):
    """Marketing pipeline logging request"""
    segmentation: Dict[str, Any]
    classification: Dict[str, Any]


class CRMPipelineLog(BaseModel):
    """CRM pipeline logging request"""
    classification: Dict[str, Any]
    segmentation: Dict[str, Any]


@router.post("/log_prediction")
async def log_prediction(
    request: MLflowLogRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generic endpoint to log any prediction to MLflow
    
    **Usage from n8n:**
    ```json
    POST /mlflow/log_prediction
    {
      "experiment_name": "n8n_Finance_Pipeline",
      "run_name": "finance_2026-05-01",
      "params": {
        "workflow": "finance",
        "user": "naima_sarraj",
        "model": "Ridge"
      },
      "metrics": {
        "predicted_amount": 1450.75,
        "mape": 6.1
      },
      "tags": {
        "source": "n8n",
        "automated": "true"
      }
    }
    ```
    """
    try:
        result = mlflow_logger.log_prediction(
            experiment_name=request.experiment_name,
            run_name=request.run_name,
            params=request.params,
            metrics=request.metrics,
            tags=request.tags,
            artifacts=request.artifacts
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "message": "Prediction logged to MLflow successfully",
            "mlflow_run_id": result.get("run_id"),
            "mlflow_experiment_id": result.get("experiment_id"),
            "mlflow_ui": f"{result.get('tracking_uri')}/#/experiments/{result.get('experiment_id')}/runs/{result.get('run_id')}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow logging failed: {str(e)}")


@router.post("/log_finance")
async def log_finance_pipeline(
    request: FinancePipelineLog,
    current_user: dict = Depends(get_current_user)
):
    """
    Log finance pipeline predictions to MLflow
    
    **Automatically logs:**
    - Price prediction (regression)
    - Revenue forecast (time series)
    - Model metrics (MAPE, RMSE, MAE)
    """
    try:
        result = mlflow_logger.log_finance_pipeline(
            user=current_user.get("login", "unknown"),
            regression_result=request.regression,
            timeseries_result=request.timeseries
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "message": "Finance pipeline logged to MLflow",
            "mlflow_run_id": result.get("run_id"),
            "experiment": "n8n_Finance_Pipeline",
            "mlflow_ui": f"{result.get('tracking_uri')}/#/experiments/{result.get('experiment_id')}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow logging failed: {str(e)}")


@router.post("/log_marketing")
async def log_marketing_pipeline(
    request: MarketingPipelineLog,
    current_user: dict = Depends(get_current_user)
):
    """
    Log marketing pipeline predictions to MLflow
    
    **Automatically logs:**
    - Customer segmentation
    - Booking status classification
    - Segment labels and confidence scores
    """
    try:
        result = mlflow_logger.log_marketing_pipeline(
            user=current_user.get("login", "unknown"),
            segmentation_result=request.segmentation,
            classification_result=request.classification
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "message": "Marketing pipeline logged to MLflow",
            "mlflow_run_id": result.get("run_id"),
            "experiment": "n8n_Marketing_Pipeline",
            "mlflow_ui": f"{result.get('tracking_uri')}/#/experiments/{result.get('experiment_id')}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow logging failed: {str(e)}")


@router.post("/log_crm")
async def log_crm_pipeline(
    request: CRMPipelineLog,
    current_user: dict = Depends(get_current_user)
):
    """
    Log CRM pipeline predictions to MLflow
    
    **Automatically logs:**
    - Booking status classification
    - Customer segmentation
    - Prediction confidence scores
    """
    try:
        result = mlflow_logger.log_crm_pipeline(
            user=current_user.get("login", "unknown"),
            classification_result=request.classification,
            segmentation_result=request.segmentation
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return {
            "message": "CRM pipeline logged to MLflow",
            "mlflow_run_id": result.get("run_id"),
            "experiment": "n8n_CRM_Pipeline",
            "mlflow_ui": f"{result.get('tracking_uri')}/#/experiments/{result.get('experiment_id')}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow logging failed: {str(e)}")


@router.get("/status")
async def mlflow_status():
    """
    Check MLflow connection status (SQLite ./mlflow.db puis repli ./mlruns).
    """
    import mlflow
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent.parent
    sqlite_uri = f"sqlite:///{(repo / 'mlflow.db').resolve().as_posix()}"
    local_mlruns = repo / "mlruns"
    file_uri = f"file:///{local_mlruns.as_posix()}"

    last_err = ""
    for uri in (sqlite_uri, file_uri):
        try:
            mlflow.set_tracking_uri(uri)
            experiments = mlflow.search_experiments()
            return {
                "status": "connected",
                "tracking_uri": uri,
                "experiments_count": len(experiments),
                "experiments": [e.name for e in experiments[:10]],
                "message": (
                    "MLflow OK — préférez SQLite pour l’onglet Overview"
                    if uri.startswith("sqlite")
                    else "MLflow OK (fichier — Overview UI limité)"
                ),
                "ui_hint": "python mlflow_ui_sqlite.py",
            }
        except Exception as e:
            last_err = str(e)
            continue

    return {
        "status": "error",
        "tracking_uri": sqlite_uri,
        "error": last_err,
        "message": "Impossible de lire MLflow (sqlite ou mlruns).",
    }
