# -*- coding: utf-8 -*-
"""Affichage des segments fidélité RFM depuis les JSON produits par ``run_01_clustering.py``."""

from __future__ import annotations

import json
from pathlib import Path

from ML.ml_paths import ML_MODELS


def _strip_md_bold(s: str) -> str:
    return str(s).replace("**", "")


def print_loyalty_segments_table(models_dir: Path | None = None) -> bool:
    """
    Affiche les segments bénéficiaires et prestataires (libellés métier + parts).
    Retourne True si au moins un fichier JSON a été lu.
    """
    base = models_dir or ML_MODELS
    specs = (
        ("Bénéficiaires (fidélité RFM)", base / "clustering_segment_labels_loyalty_beneficiary.json"),
        ("Prestataires (fidélité RFM)", base / "clustering_segment_labels_loyalty_provider.json"),
    )
    any_ok = False
    for title, fp in specs:
        if not fp.is_file():
            print(f"\n[{title}] — fichier absent : {fp.name}")
            print(f"  → Générez-le avec : python ML/scripts/run_01_clustering.py")
            continue
        any_ok = True
        data = json.loads(fp.read_text(encoding="utf-8"))
        segs = data.get("segments") or []
        k = data.get("k", len(segs))
        print(f"\n=== {title} — k={k} (une ligne = un acteur agrégé, colonnes *_loyalty) ===")
        for s in sorted(segs, key=lambda x: int(x.get("cluster_id", 0))):
            cid = int(s.get("cluster_id", 0))
            metier = _strip_md_bold(s.get("label_metier_fr", ""))
            short = _strip_md_bold(s.get("label_short", ""))
            sh = float(s.get("share_train_sample", 0.0))
            print(f"  Cluster {cid} — ~{sh * 100:.1f} % de l’échantillon d’entraînement")
            print(f"    Lecture métier : {metier}")
            print(f"    Libellé technique (centres) : {short}")
    return any_ok
