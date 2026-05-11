# -*- coding: utf-8 -*-
"""
Supprime la numérotation des titres markdown (1., 1.1, 1.2.3, etc.) dans les notebooks.
Conserve le **grand titre** `# …`, les **emojis** et le texte — sans préfixes numériques.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"


def strip_outline_prefix(rest: str) -> str:
    rest = rest.lstrip()
    changed = True
    while changed:
        changed = False
        for rx in (
            r"^\d+(?:\.\d+)*\s+",
            r"^\d+\.\s+",
        ):
            m = re.match(rx, rest)
            if m:
                rest = rest[m.end() :]
                changed = True
                break
    return rest.lstrip()


def demote_heading_line(line: str) -> str:
    s = line.rstrip("\n")
    if not s.strip():
        return line
    if s.startswith("####"):
        return line

    hashes: str | None = None
    rest_part: str = ""

    if s.startswith("### ") or s == "###":
        hashes, rest_part = "###", s[4:] if s.startswith("### ") else ""
    elif s.startswith("## ") or s == "##":
        hashes, rest_part = "##", s[3:] if s.startswith("## ") else ""
    elif s.startswith("# ") or s == "#":
        hashes, rest_part = "#", s[2:] if s.startswith("# ") else ""
    else:
        return line

    rest = strip_outline_prefix(rest_part)
    new_s = f"{hashes} {rest}".strip()
    if line.endswith("\n"):
        new_s += "\n"
    return new_s


def process_markdown_source(source: list[str]) -> list[str]:
    return [demote_heading_line(l) for l in source]


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = [src]
        cell["source"] = process_markdown_source(list(src))
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    for name in [
        "00_A_preparation_donnees_feature_engineering.ipynb",
        "01_E_clustering_segmentation.ipynb",
        "02_C_classification_statut_reservation.ipynb",
        "03_D_regression_montants_KPI.ipynb",
        "04_F_series_temporelles.ipynb",
        "05_synthese_metriques_validation.ipynb",
    ]:
        p = ROOT / name
        if not p.is_file():
            continue
        patch_notebook(p)
        print("OK", name)


if __name__ == "__main__":
    main()
