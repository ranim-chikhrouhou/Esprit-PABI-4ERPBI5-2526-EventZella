from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ML.ml_paths import REPO_ROOT


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _artifact_version_path(target_path: Path) -> Path:
    versions_root = target_path.parent / "versions" / target_path.stem
    versions_root.mkdir(parents=True, exist_ok=True)
    return versions_root / f"{_now_tag()}__{target_path.name}"


def dump_joblib_versioned(obj: Any, target_path: Path) -> dict[str, str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target_path)
    versioned_path = _artifact_version_path(target_path)
    shutil.copy2(target_path, versioned_path)
    return {"latest": str(target_path), "versioned": str(versioned_path)}


def write_json_versioned(payload: dict[str, Any], target_path: Path) -> dict[str, str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    versioned_path = _artifact_version_path(target_path)
    shutil.copy2(target_path, versioned_path)
    return {"latest": str(target_path), "versioned": str(versioned_path)}


def _mlflow_tracking_uri() -> str:
    configured = os.environ.get("EVENTZILLA_MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return (REPO_ROOT / "mlruns").resolve().as_uri()


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None):
    try:
        import mlflow
    except Exception:
        yield None
        return

    mlflow.set_tracking_uri(_mlflow_tracking_uri())
    mlflow.set_experiment(os.environ.get("EVENTZILLA_MLFLOW_EXPERIMENT", "EventZilla-S12"))
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "EventZilla",
                "phase": "S12",
                "pipeline": "automated_training",
                **(tags or {}),
            }
        )
        yield mlflow
