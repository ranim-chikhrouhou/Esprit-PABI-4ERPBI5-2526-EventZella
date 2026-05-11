# -*- coding: utf-8 -*-
"""
EventZilla ML API - Monitoring Module
Week S13: Prometheus Metrics & Drift Detection

This module provides:
1. Prometheus metrics collection
2. Model performance tracking
3. Data drift detection
4. Alerting logic
"""
from __future__ import annotations

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque

from prometheus_client import Counter, Histogram, Gauge, Info
import psutil


# ═══════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════

# Traffic Metrics
prediction_requests_total = Counter(
    'eventzilla_prediction_requests_total',
    'Total number of prediction requests',
    ['model_type', 'endpoint', 'status']
)

# Performance Metrics
prediction_latency = Histogram(
    'eventzilla_prediction_latency_seconds',
    'Prediction request latency in seconds',
    ['model_type', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Error Metrics
prediction_errors_total = Counter(
    'eventzilla_prediction_errors_total',
    'Total number of prediction errors',
    ['model_type', 'endpoint', 'error_type']
)

# Model Health Metrics
model_accuracy = Gauge(
    'eventzilla_model_accuracy',
    'Current model accuracy',
    ['model_type']
)

model_confidence = Gauge(
    'eventzilla_model_confidence',
    'Average prediction confidence',
    ['model_type']
)

model_predictions_count = Counter(
    'eventzilla_model_predictions_count',
    'Number of predictions per model',
    ['model_type']
)

# Data Health Metrics
data_missing_values = Gauge(
    'eventzilla_data_missing_values_ratio',
    'Ratio of missing values in input data',
    ['feature']
)

data_freshness = Gauge(
    'eventzilla_data_freshness_seconds',
    'Time since last data update in seconds'
)

# Drift Detection Metrics
data_drift_detected = Gauge(
    'eventzilla_data_drift_detected',
    'Data drift detection flag (1=drift, 0=no drift)',
    ['feature']
)

model_drift_detected = Gauge(
    'eventzella_model_drift_detected',
    'Model performance drift flag (1=drift, 0=no drift)',
    ['model_type']
)

# System Metrics
system_cpu_usage = Gauge(
    'eventzilla_system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage = Gauge(
    'eventzilla_system_memory_usage_percent',
    'System memory usage percentage'
)

system_disk_usage = Gauge(
    'eventzilla_system_disk_usage_percent',
    'System disk usage percentage'
)

# API Info
api_info = Info(
    'eventzilla_api',
    'EventZilla ML API information'
)


# ═══════════════════════════════════════════════════════════════════
# BASELINE VALUES (Production Reference)
# ═══════════════════════════════════════════════════════════════════

BASELINE_METRICS = {
    "classification": {
        "accuracy": 0.85,  # 85% baseline accuracy
        "confidence": 0.75,  # 75% average confidence
        "latency": 0.1,  # 100ms baseline latency
    },
    "regression": {
        "r2_score": 0.95,  # R² baseline
        "mae": 50.0,  # Mean Absolute Error baseline
        "latency": 0.08,  # 80ms baseline latency
    },
    "clustering": {
        "silhouette": 0.45,  # Silhouette score baseline
        "latency": 0.05,  # 50ms baseline latency
    },
    "timeseries": {
        "mape": 8.0,  # 8% MAPE baseline
        "rmse": 300.0,  # RMSE baseline
        "latency": 0.15,  # 150ms baseline latency
    }
}

# Drift thresholds
DRIFT_THRESHOLDS = {
    "accuracy_drop": 0.05,  # 5% accuracy drop triggers alert
    "confidence_drop": 0.10,  # 10% confidence drop
    "latency_increase": 2.0,  # 2x latency increase
    "missing_values": 0.15,  # 15% missing values
}


# ═══════════════════════════════════════════════════════════════════
# MONITORING STATE (In-Memory Storage)
# ═══════════════════════════════════════════════════════════════════

class MonitoringState:
    """In-memory storage for monitoring data"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Prediction history (for drift detection)
        self.predictions_history = {
            "classification": deque(maxlen=max_history),
            "regression": deque(maxlen=max_history),
            "clustering": deque(maxlen=max_history),
        }
        
        # Confidence scores history
        self.confidence_history = {
            "classification": deque(maxlen=max_history),
            "regression": deque(maxlen=max_history),
        }
        
        # Latency history
        self.latency_history = {
            "classification": deque(maxlen=max_history),
            "regression": deque(maxlen=max_history),
            "clustering": deque(maxlen=max_history),
            "timeseries": deque(maxlen=max_history),
        }
        
        # Feature distributions (for drift detection)
        self.feature_distributions = {}
        
        # Alerts log
        self.alerts = deque(maxlen=100)
        
        # Last update timestamp
        self.last_update = datetime.now()
    
    def add_prediction(self, model_type: str, prediction: Any, confidence: Optional[float] = None):
        """Record a prediction"""
        if model_type in self.predictions_history:
            self.predictions_history[model_type].append({
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "confidence": confidence
            })
        
        if confidence is not None and model_type in self.confidence_history:
            self.confidence_history[model_type].append(confidence)
    
    def add_latency(self, model_type: str, latency: float):
        """Record request latency"""
        if model_type in self.latency_history:
            self.latency_history[model_type].append(latency)
    
    def add_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Record an alert"""
        self.alerts.append({
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity
        })
    
    def get_average_confidence(self, model_type: str) -> float:
        """Calculate average confidence for a model"""
        if model_type not in self.confidence_history:
            return 0.0
        
        history = list(self.confidence_history[model_type])
        if not history:
            return 0.0
        
        return float(np.mean(history))
    
    def get_average_latency(self, model_type: str) -> float:
        """Calculate average latency for a model"""
        if model_type not in self.latency_history:
            return 0.0
        
        history = list(self.latency_history[model_type])
        if not history:
            return 0.0
        
        return float(np.mean(history))
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]


# Global monitoring state
monitoring_state = MonitoringState()


# ═══════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_accuracy_drift(model_type: str, current_accuracy: float) -> bool:
    """Detect if model accuracy has drifted below baseline"""
    baseline = BASELINE_METRICS.get(model_type, {}).get("accuracy", 0.0)
    threshold = DRIFT_THRESHOLDS["accuracy_drop"]
    
    if baseline == 0.0:
        return False
    
    drift = (baseline - current_accuracy) > threshold
    
    if drift:
        model_drift_detected.labels(model_type=model_type).set(1)
        monitoring_state.add_alert(
            "accuracy_drift",
            f"{model_type} accuracy dropped from {baseline:.2%} to {current_accuracy:.2%}",
            severity="critical"
        )
    else:
        model_drift_detected.labels(model_type=model_type).set(0)
    
    return drift


def detect_confidence_drift(model_type: str) -> bool:
    """Detect if average confidence has dropped significantly"""
    current_confidence = monitoring_state.get_average_confidence(model_type)
    baseline = BASELINE_METRICS.get(model_type, {}).get("confidence", 0.0)
    threshold = DRIFT_THRESHOLDS["confidence_drop"]
    
    if baseline == 0.0 or current_confidence == 0.0:
        return False
    
    drift = (baseline - current_confidence) > threshold
    
    if drift:
        monitoring_state.add_alert(
            "confidence_drift",
            f"{model_type} confidence dropped from {baseline:.2%} to {current_confidence:.2%}",
            severity="warning"
        )
    
    return drift


def detect_latency_drift(model_type: str) -> bool:
    """Detect if latency has increased significantly"""
    current_latency = monitoring_state.get_average_latency(model_type)
    baseline = BASELINE_METRICS.get(model_type, {}).get("latency", 0.0)
    threshold = DRIFT_THRESHOLDS["latency_increase"]
    
    if baseline == 0.0 or current_latency == 0.0:
        return False
    
    drift = current_latency > (baseline * threshold)
    
    if drift:
        monitoring_state.add_alert(
            "latency_drift",
            f"{model_type} latency increased from {baseline*1000:.0f}ms to {current_latency*1000:.0f}ms",
            severity="warning"
        )
    
    return drift


def detect_data_drift(feature_name: str, values: list) -> bool:
    """Detect data distribution drift using simple statistical test"""
    if len(values) < 10:
        return False
    
    # Store baseline distribution if not exists
    if feature_name not in monitoring_state.feature_distributions:
        monitoring_state.feature_distributions[feature_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }
        return False
    
    # Compare current distribution with baseline
    baseline = monitoring_state.feature_distributions[feature_name]
    current_mean = float(np.mean(values))
    current_std = float(np.std(values))
    
    # Simple drift detection: mean shifted by more than 2 standard deviations
    mean_shift = abs(current_mean - baseline["mean"]) / (baseline["std"] + 1e-6)
    drift = mean_shift > 2.0
    
    if drift:
        data_drift_detected.labels(feature=feature_name).set(1)
        monitoring_state.add_alert(
            "data_drift",
            f"Feature '{feature_name}' distribution shifted: mean {baseline['mean']:.2f} → {current_mean:.2f}",
            severity="warning"
        )
    else:
        data_drift_detected.labels(feature=feature_name).set(0)
    
    return drift


# ═══════════════════════════════════════════════════════════════════
# MONITORING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def update_system_metrics():
    """Update system resource metrics"""
    try:
        system_cpu_usage.set(psutil.cpu_percent(interval=0.1))
        system_memory_usage.set(psutil.virtual_memory().percent)
        system_disk_usage.set(psutil.disk_usage('/').percent)
    except Exception:
        pass  # Ignore errors in system metrics


def track_prediction(
    model_type: str,
    endpoint: str,
    latency: float,
    status: str = "success",
    error_type: Optional[str] = None,
    confidence: Optional[float] = None,
    prediction: Any = None
):
    """Track a prediction request with all metrics"""
    
    # Update counters
    prediction_requests_total.labels(
        model_type=model_type,
        endpoint=endpoint,
        status=status
    ).inc()
    
    # Update latency
    prediction_latency.labels(
        model_type=model_type,
        endpoint=endpoint
    ).observe(latency)
    
    # Track errors
    if status == "error" and error_type:
        prediction_errors_total.labels(
            model_type=model_type,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
    
    # Update model-specific metrics
    if status == "success":
        model_predictions_count.labels(model_type=model_type).inc()
        
        if confidence is not None:
            model_confidence.labels(model_type=model_type).set(confidence)
        
        # Store in monitoring state
        monitoring_state.add_prediction(model_type, prediction, confidence)
        monitoring_state.add_latency(model_type, latency)
        
        # Check for drift
        detect_latency_drift(model_type)
        if confidence is not None:
            detect_confidence_drift(model_type)
    
    # Update system metrics
    update_system_metrics()


def check_missing_values(data: dict) -> float:
    """Calculate ratio of missing values in input data"""
    total_fields = len(data)
    if total_fields == 0:
        return 0.0
    
    missing = sum(1 for v in data.values() if v is None or v == "")
    ratio = missing / total_fields
    
    # Update metric for each feature
    for key, value in data.items():
        if value is None or value == "":
            data_missing_values.labels(feature=key).set(1.0)
        else:
            data_missing_values.labels(feature=key).set(0.0)
    
    # Alert if too many missing values
    if ratio > DRIFT_THRESHOLDS["missing_values"]:
        monitoring_state.add_alert(
            "data_quality",
            f"High missing values ratio: {ratio:.1%}",
            severity="warning"
        )
    
    return ratio


def update_data_freshness():
    """Update data freshness metric"""
    seconds_since_update = (datetime.now() - monitoring_state.last_update).total_seconds()
    data_freshness.set(seconds_since_update)
    monitoring_state.last_update = datetime.now()


def get_monitoring_summary() -> dict:
    """Get current monitoring summary"""
    return {
        "timestamp": datetime.now().isoformat(),
        "predictions_count": {
            model: len(monitoring_state.predictions_history[model])
            for model in monitoring_state.predictions_history
        },
        "average_confidence": {
            model: monitoring_state.get_average_confidence(model)
            for model in monitoring_state.confidence_history
        },
        "average_latency": {
            model: monitoring_state.get_average_latency(model)
            for model in monitoring_state.latency_history
        },
        "recent_alerts": monitoring_state.get_recent_alerts(10),
        "baseline_metrics": BASELINE_METRICS,
        "drift_thresholds": DRIFT_THRESHOLDS
    }


# ═══════════════════════════════════════════════════════════════════
# INITIALIZE API INFO
# ═══════════════════════════════════════════════════════════════════

api_info.info({
    'version': '1.0.0',
    'name': 'EventZilla ML API',
    'environment': 'production',
    'monitoring': 'enabled'
})
