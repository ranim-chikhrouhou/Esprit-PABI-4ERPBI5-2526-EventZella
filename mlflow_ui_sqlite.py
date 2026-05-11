#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lance MLflow UI avec SQLite (Overview + graphiques fonctionnels).

Usage (à la racine du projet) :
    python mlflow_ui_sqlite.py

Sous Windows, un seul worker uvicorn est forcé par défaut (évite WinError 10022).

Pour écouter sur tout le réseau :
    set MLFLOW_UI_HOST=0.0.0.0
    python mlflow_ui_sqlite.py
"""
from __future__ import annotations

import subprocess

from ML.mlflow_tracking_uri import mlflow_ui_argv


def main() -> None:
    argv = mlflow_ui_argv()
    print("MLflow UI — backend SQLite + artefacts ./mlartifacts (workers=1 Windows-safe)")
    print(" ".join(argv[2:]))
    subprocess.run(argv, check=False)


if __name__ == "__main__":
    main()
