# -*- coding: utf-8 -*-
"""Exécute tout le pipeline ML dans l’ordre (terminal Cursor / VS Code).

    cd "chemin\\vers\\PI BI NEW"
    python ML/scripts/run_all_ml_pipeline.py

Optionnel : arrêter au premier échec (défaut). Pour continuer : --continue-on-error
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_STEPS = [
    "run_test_sql_connection.py",
    "run_00_data_preparation.py",
    "run_01_clustering.py",
    "run_02_classification.py",
    "run_03_prediction_regression.py",
    "run_04_time_series.py",
    "run_05_metrics_comparison.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--continue-on-error", action="store_true", help="Poursuivre si une étape échoue")
    ap.add_argument("--skip-connection-test", action="store_true", help="Ne pas exécuter le test SQL")
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent.parent
    for name in _STEPS:
        if args.skip_connection_test and name == "run_test_sql_connection.py":
            continue
        script = scripts_dir / name
        print("\n===", name, "===\n")
        r = subprocess.run([sys.executable, str(script)], cwd=repo_root)
        if r.returncode != 0:
            print(f"Échec: {name} (code {r.returncode})")
            if not args.continue_on_error:
                sys.exit(r.returncode)
    print("\nPipeline ML terminé.")


if __name__ == "__main__":
    main()
