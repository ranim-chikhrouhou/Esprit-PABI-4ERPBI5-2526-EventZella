# -*- coding: utf-8 -*-
"""Compile les métriques des étapes 01–04.

    python ML/scripts/run_05_metrics_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

from ML.ml_paths import ML_MODELS, ML_PROCESSED


def main() -> None:
    files = sorted(ML_MODELS.glob("metrics_*.json"))
    if not files:
        raise SystemExit("Aucun metrics_*.json — exécuter run_01 … run_04 d'abord.")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_md = ML_PROCESSED.parent / "ML_METRICS_SUMMARY.md"
    text = (
        "# EventZilla — Résumé métriques ML\n\n"
        "Alignement KPI : voir `kpi_alignment` dans chaque JSON "
        "et `docs/eventzilla/EventZilla_Dashboards_KPIs_Objectifs.md`.\n\n"
        + df.to_csv(index=False)
    )
    out_md.write_text(text, encoding="utf-8")
    print("Ecrit", out_md)


if __name__ == "__main__":
    main()
