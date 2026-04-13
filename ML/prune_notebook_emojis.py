# -*- coding: utf-8 -*-
"""
Réduit les emojis dans les cellules markdown : ne conserve que
  - 🎯 / ✅ / 📑 en tête de titres pertinents (objectifs, résultats, sommaire)
  - 📊 pour les titres contenant « Visualisation(s) »
  - les emojis des lignes du sommaire numéroté (1. 🔌 …)
  - les lignes 🎯 **Objectif** / ✅ **Résultats attendus** / 📊 **Visualisations**
Retire les emojis décoratifs en fin de phrase (conclusions, etc.).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "notebooks"

# Premier caractère pictographique en tête de titre (approximation)
LEADING_SYMBOL = re.compile(
    "^([\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF])"
)


def sommaire_list_line(line: str) -> bool:
    """Ligne type « 1. 🔌 Connexion » dans le corps du sommaire."""
    return bool(re.match(r"^\s*\d+\.\s+\S", line))


def keep_heading_leading_emoji(emoji: str, rest: str) -> bool:
    r = rest
    if emoji == "🎯" and "Objectif" in r:
        return True
    if emoji == "✅":
        return True
    if emoji == "📑" and "Sommaire" in r:
        return True
    if emoji == "📊" and re.search(r"Visualisation", r, re.I):
        return True
    return False


def strip_leading_emojis_from_heading(rest: str) -> str:
    """Supprime les emojis en tête jusqu’à obtenir du texte ou une règle de conservation."""
    rest = rest.lstrip()
    while rest:
        m = LEADING_SYMBOL.match(rest)
        if not m:
            break
        em = m.group(1)
        if keep_heading_leading_emoji(em, rest):
            break
        rest = rest[len(em) :].lstrip()
    return rest


def clean_heading_line(line: str) -> str:
    m = re.match(r"^(#{1,3})\s+(.*)$", line.rstrip("\n"))
    if not m:
        return line
    hashes, rest = m.group(1), m.group(2)
    new_rest = strip_leading_emojis_from_heading(rest)
    # Une seule tête 📊 pour les titres de section « Visualisations » (sans autre emoji devant)
    if re.match(r"^Visualisations\b", new_rest) and not new_rest.startswith("📊"):
        new_rest = "📊 " + new_rest
    nl = "\n" if line.endswith("\n") else ""
    return f"{hashes} {new_rest}{nl}"


def is_objectif_template_line(line: str) -> bool:
    return "🎯 **Objectif**" in line or "✅ **Résultats attendus**" in line or "📊 **Visualisations**" in line


def strip_trailing_decorative_emoji(line: str) -> str:
    """Retire les suites d’emojis décoratifs en fin de ligne (pas les templates objectifs)."""
    if is_objectif_template_line(line):
        return line
    # Fin de ligne : espaces + emojis « décor » (répétés)
    return re.sub(
        r"[\s\u00a0]*[🎯✅📊🔥📁🧰✨💬🖼️]+[\s\u00a0]*$",
        "",
        line,
    )


def clean_body_line(line: str) -> str:
    if is_objectif_template_line(line):
        return line
    if sommaire_list_line(line):
        return line
    return strip_trailing_decorative_emoji(line)


def process_markdown_source(source: list[str]) -> list[str]:
    out: list[str] = []
    for line in source:
        if isinstance(line, str) and re.match(r"^#{1,3}\s", line):
            out.append(clean_heading_line(line))
        else:
            out.append(clean_body_line(line))
    return out


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
