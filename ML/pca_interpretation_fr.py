# -*- coding: utf-8 -*-
"""Interprétations PCA dynamiques en français (notebook clustering).

Import optionnel : from ML.pca_interpretation_fr import print_pca_dynamic_interpretation
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PcaScope = Literal["wide", "loyalty_rfm"]


def _fmt_feat(name: str, max_len: int = 28) -> str:
    s = str(name).replace("_", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _semantic_tag(name: str) -> str:
    n = str(name).lower().replace(" ", "_")
    if n.startswith("id_") or n.endswith("_id"):
        return "id_dw"
    if any(x in n for x in ("cal_month", "cal_year", "holiday", "full_date", "quarter")):
        return "temps"
    if any(x in n for x in ("final_price", "service_price", "event_budget", "benchmark", "avg_price", "price")):
        return "montant"
    if any(x in n for x in ("nb_visitor", "nb_reservation", "reservation")):
        return "volume"
    return "autre"


def _theme_sentence(tags: list[str]) -> str:
    if not tags:
        return "profils mixtes"
    tset = list(dict.fromkeys(tags))
    if "temps" in tset and "id_dw" in tset and "montant" not in tset:
        return (
            "un mélange calendrier (mois, année) et identifiants DW (visiteur, etc.) — "
            "axe interprétable comme « période × référentiel technique »"
        )
    prio = ["montant", "temps", "volume", "id_dw", "autre"]
    for p in prio:
        if p in tags:
            if p == "montant":
                return "une dimension économique (prix, budget, tarifs)"
            if p == "temps":
                return "une dimension temporelle / calendaire (mois, année, saison)"
            if p == "volume":
                return "une dimension d’activité (visiteurs, réservations)"
            if p == "id_dw":
                return "les clés de dimensions du DW (surrogate keys)"
    return "d’autres variables du périmètre"


def print_pca_dynamic_interpretation(
    pca,
    X2: np.ndarray,
    feat_names_arr: np.ndarray,
    labels_km: np.ndarray,
    labels_agg: np.ndarray,
    km_centers_pca: np.ndarray,
    k_best: int,
    cluster_label_short: list[str],
    km_cluster_centers: np.ndarray | None = None,
    feat_names_original: list[str] | None = None,
    *,
    scope: PcaScope = "wide",
) -> None:
    """
    Affiche des interprétations **concrètes** basées sur loadings, positions des centres
    et cohérence éventuelle avec les centres K-Means (heatmap).

    ``scope`` : ``wide`` = notebook 01_E (lignes de fait performance) ;
    ``loyalty_rfm`` = agrégats bénéficiaire/prestataire (script ``run_01_clustering.py``).
    """
    _fn = feat_names_arr
    n_feat = pca.components_.shape[1]
    _evr = pca.explained_variance_ratio_
    v1, v2 = float(_evr[0]), float(_evr[1])

    print()
    print("=" * 66)
    print(" INTERPRÉTATIONS DYNAMIQUES (basées sur cette exécution)")
    print("=" * 66)
    _scope_line = (
        "Périmètre : vue LARGE (une ligne ≈ fait / réservation dans X_work) — pas l’agrégat RFM par acteur."
        if scope == "wide"
        else "Périmètre : fidélité RFM (une ligne ≈ bénéficiaire ou prestataire agrégé, colonnes *_loyalty)."
    )
    print(f" ({_scope_line})")
    print(" (Repère graphique : abscisse = PC1, ordonnée = PC2 — aligné sur les figures ci-dessus.)")

    # --- Rôle de chaque axe ---
    for pc_idx, pc_lab, v_ex in ((0, "PC1", v1), (1, "PC2", v2)):
        comp = pca.components_[pc_idx]
        top = np.argsort(np.abs(comp))[::-1][:min(8, n_feat)]
        pos = [(int(i), float(comp[i])) for i in top if comp[i] >= 0.06]
        neg = [(int(i), float(comp[i])) for i in top if comp[i] <= -0.06]
        tags = [_semantic_tag(_fn[i]) for i, _ in pos + neg]
        theme = _theme_sentence(list(dict.fromkeys(tags)))

        print()
        print(f"▸ {pc_lab} ({100 * v_ex:.1f} % de la variance totale)")
        print(
            f"   Lecture globale : cet axe exprime surtout {theme}, "
            f"d’après les variables aux loadings les plus marquants."
        )
        if pos:
            print(
                f"   Côté scores {pc_lab} positifs (à droite si PC1, en haut si PC2) : "
                + ", ".join(f"«{_fmt_feat(_fn[i])}» ({c:+.2f})" for i, c in pos[:4])
                + (" …" if len(pos) > 4 else "")
            )
        if neg:
            print(
                f"   Côté scores {pc_lab} négatifs (gauche si PC1, bas si PC2) : "
                + ", ".join(f"«{_fmt_feat(_fn[i])}» ({c:+.2f})" for i, c in neg[:4])
                + (" …" if len(neg) > 4 else "")
            )

    # --- Position des segments (K-Means) dans le plan PCA ---
    print()
    print("▸ Position des segments K-Means dans le plan PC1–PC2 (centres ★)")
    order_pc2 = np.argsort(-km_centers_pca[:, 1])
    order_pc1 = np.argsort(-km_centers_pca[:, 0])
    for k in range(k_best):
        cx, cy = float(km_centers_pca[k, 0]), float(km_centers_pca[k, 1])
        rk2 = int(np.where(order_pc2 == k)[0][0]) + 1
        rk1 = int(np.where(order_pc1 == k)[0][0]) + 1
        short = cluster_label_short[k] if k < len(cluster_label_short) else f"segment {k}"
        print(
            f"   • Cluster {k} ({short}) : centre ≈ (PC1={cx:+.2f}, PC2={cy:+.2f}) "
            f"— classement des centres : rang PC1 = {rk1}/{k_best}, rang PC2 = {rk2}/{k_best}."
        )

    hi_pc2 = X2[:, 1] >= np.percentile(X2[:, 1], 90)
    hi_pc1 = X2[:, 0] >= np.percentile(X2[:, 0], 90)
    lo_pc1 = X2[:, 0] <= np.percentile(X2[:, 0], 10)
    print()
    print("▸ Sous-populations extrêmes dans ce plan (percentiles sur l’échantillon courant)")
    for mask, nom in (
        (hi_pc2, "PC2 haut (top 10 %)"),
        (hi_pc1, "PC1 haut (top 10 %)"),
        (lo_pc1, "PC1 bas (bottom 10 %)"),
    ):
        if mask.sum() < 2:
            continue
        print(f"   Zone « {nom} » (n={int(mask.sum())}) :")
        for algo_name, lab in (("K-Means", labels_km), ("Ward", labels_agg)):
            sub = lab[mask]
            uniq, cnt = np.unique(sub, return_counts=True)
            dom = uniq[np.argmax(cnt)]
            pct = 100.0 * float(np.max(cnt)) / float(len(sub))
            print(f"      • {algo_name} : surtout cluster {int(dom)} (~{pct:.0f} %).")

    # --- Cohérence loadings PC2 ↔ colonne prix service (si présente) ---
    print()
    print("▸ Cohérence PCA ↔ profils des centres (à relire avec la heatmap)")
    if km_cluster_centers is not None and feat_names_original is not None:
        names_lower = [str(x).lower() for x in feat_names_original]
        j_price = None
        for key in ("service_price", "final_price"):
            if key in names_lower:
                j_price = names_lower.index(key)
                break
        if j_price is not None and j_price < km_cluster_centers.shape[1]:
            z_price = km_cluster_centers[:, j_price]
            k_max_p = int(np.argmax(z_price))
            k_max_pc2 = int(np.argmax(km_centers_pca[:, 1]))
            coln = _fmt_feat(feat_names_original[j_price])
            print(
                f"   Variable « {coln} » : le segment au centre le plus élevé sur cette variable est "
                f"le cluster {k_max_p} (z≈{z_price[k_max_p]:+.2f}) ; "
                f"le segment le plus haut sur PC2 est le cluster {k_max_pc2}."
            )
            if k_max_p == k_max_pc2:
                print(
                    "   → Alignement : le même segment domine à la fois le prix (centre K-Means) "
                    "et l’axe PC2 — lecture « valeur / tarif » sur PC2 cohérente avec la heatmap."
                )
            else:
                print(
                    "   → Les deux classements diffèrent : PC2 résume plusieurs variables (voir loadings) ; "
                    "le prix se lit de façon plus directe sur la heatmap que sur un seul axe."
                )

    # --- K-Means vs Ward ---
    agree = float(np.mean(labels_km == labels_agg))
    _unit = "lignes (fait wide)" if scope == "wide" else "profils agrégés (entités)"
    print()
    print(f"▸ K-Means vs agglomératif Ward (même k={k_best}, même projection)")
    print(
        f"   Accord ponctuel sur les étiquettes : ~{100 * agree:.1f} % des {_unit} "
        "(les deux algorithmes peuvent découper différemment le nuage malgré le même plan PCA)."
    )
    print(
        "   En général : Ward peut isoler un groupe très serré à une extrémité ; "
        "K-Means cherche des partitions type Voronoï autour de centres — d’où des frontières différentes sur PC1."
    )

    print()
    print("=" * 66)
