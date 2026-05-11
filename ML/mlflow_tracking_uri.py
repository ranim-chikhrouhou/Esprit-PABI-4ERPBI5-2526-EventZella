# -*- coding: utf-8 -*-
"""
URI de tracking MLflow — SQLite par défaut (Overview / charts UI).

Le backend « fichier » (file:///.../mlruns) ne permet pas l’onglet Overview
complet dans MLflow 2.11+ ; utiliser sqlite:///.../mlflow.db + artefact local.

Variables d'environnement :
  MLFLOW_TRACKING_URI — surcharge (ex. http://localhost:5000 si serveur distant)
  MLFLOW_UI_HOST / MLFLOW_UI_PORT / MLFLOW_UI_WORKERS — lancement UI (voir mlflow_ui_sqlite.py)
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

WIN = platform.system() == "Windows"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_sqlite_tracking_uri(root: Path | None = None) -> str:
    r = root or repo_root()
    db = (r / "mlflow.db").resolve()
    return f"sqlite:///{db.as_posix()}"


def ensure_artifact_dir(root: Path | None = None) -> Path:
    art = (root or repo_root()) / "mlartifacts"
    art.mkdir(parents=True, exist_ok=True)
    return art


def get_tracking_uri(root: Path | None = None) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", default_sqlite_tracking_uri(root))


def mlflow_ui_argv(root: Path | None = None) -> list[str]:
    """Arguments pour `python -m mlflow ui` avec backend SQL + artefacts.

    Sous Windows : --workers 1 et host 127.0.0.1 par défaut pour éviter
    OSError WinError 10022 (plusieurs workers uvicorn + multiprocessing).

    Variables optionnelles :
      MLFLOW_UI_HOST — ex. 0.0.0.0 pour accès LAN
      MLFLOW_UI_PORT — défaut 5000
      MLFLOW_UI_WORKERS — défaut 1 (ne pas augmenter sous Windows)
    """
    import sys

    r = root or repo_root()
    db_uri = default_sqlite_tracking_uri(r)
    art = ensure_artifact_dir(r)
    art_uri = art.resolve().as_uri()

    host = os.environ.get(
        "MLFLOW_UI_HOST",
        "127.0.0.1" if WIN else "0.0.0.0",
    )
    port = os.environ.get("MLFLOW_UI_PORT", "5000")
    workers = os.environ.get("MLFLOW_UI_WORKERS", "1")

    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        db_uri,
        "--default-artifact-root",
        art_uri,
        "--host",
        host,
        "--port",
        port,
        "--workers",
        workers,
    ]
