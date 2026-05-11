# -*- coding: utf-8 -*-
import json
from pathlib import Path

ml_dir = Path(__file__).resolve().parent
base = ml_dir / "notebooks"
out_lines = []
for name in [
    "01_E_clustering_segmentation.ipynb",
    "02_C_classification_statut_reservation.ipynb",
    "03_D_regression_montants_KPI.ipynb",
    "04_F_series_temporelles.ipynb",
]:
    p = base / name
    nb = json.loads(p.read_text(encoding="utf-8"))
    out_lines.append(f"=== {name} cells={len(nb['cells'])} ===")
    for i, c in enumerate(nb["cells"]):
        t = c["cell_type"]
        src = "".join(c.get("source", []))
        out_lines.append(f"--- {i} {t} ---")
        out_lines.append(src[:3500])
    out_lines.append("")

(ml_dir / "_nb01_04_outline.txt").write_text("\n".join(out_lines), encoding="utf-8")
print("OK")
