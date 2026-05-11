"""Libellés de segments à partir des centres K-Means (espace standardisé).

Les facettes (montants, volumes, calendrier, ids, colonnes ``*_loyalty``) servent **en interne** à bâtir le
libellé court ; le texte long d’export reprend ce libellé sans afficher de coefficients ni z-scores.

**Deux contextes** : (1) **vue large** — notebook ``01_E`` : une ligne de matrice = fait performance ;
(2) **fidélité RFM** — ``run_01_clustering.py`` : une ligne = bénéficiaire ou prestataire agrégé. La fonction
est la même ; les noms de colonnes diffèrent.

Utilisé par le notebook 01_E, ``run_01_clustering.py`` et ``streamlit_app``.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _facet_for_column(col: str) -> str:
    """Regroupe une colonne du périmètre performance (wide) dans une facette interprétable."""
    n = _norm_col(col)
    # Clés techniques DW en premier (évite id_visitors → « visitor » → volumes)
    if n.startswith("id_") or n.endswith("_id") or re.search(r"\bid\b", n):
        return "ids"
    # Montants / prix (prioritaires pour le sens métier)
    if any(
        k in n
        for k in (
            "final_price",
            "service_price",
            "event_budget",
            "benchmark",
            "avg_price",
            "price",
            "montant",
            "budget",
            "revenue",
            "ca_",
        )
    ):
        return "montants"
    # Fidélité / RFM (agrégats bénéficiaire ou prestataire)
    if any(
        k in n
        for k in (
            "_loyalty",
            "recency_days",
            "ca_total_loyalty",
            "panier_moyen_loyalty",
            "volume_reservations_site",
        )
    ):
        return "fidelite"
    # Volumes d'activité
    if any(
        k in n
        for k in (
            "nb_visitor",
            "nb_reservation",
            "visitor",
            "reservation_site",
            "count_",
            "volume",
        )
    ):
        return "volumes"
    # Temps / saisonnalité (mois, année, jour férié…)
    if any(
        k in n
        for k in (
            "cal_month",
            "cal_year",
            "is_holiday",
            "holiday",
            "full_date",
            "month",
            "year",
            "quarter",
            "week",
        )
    ):
        return "calendrier"
    return "autre"


def _collect_facets(
    feat_names: list[str], row: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Pour une ligne de centre : facette -> (force moyenne |z|, signe moyen z)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for j, name in enumerate(feat_names):
        buckets[_facet_for_column(name)].append(float(row[j]))
    out: dict[str, tuple[float, float]] = {}
    for facet, vals in buckets.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[facet] = (float(np.mean(np.abs(arr))), float(np.mean(arr)))
    return out


def _phrase_facet(facet: str, strength: float, sign: float) -> str | None:
    """Phrase courte en français si la facette est suffisamment marquée."""
    if strength < 0.12:
        return None
    # Seuils de signe (échelle standardisée)
    hi, lo = 0.18, -0.18
    if facet == "fidelite":
        if sign > hi:
            return "Forte fidélité (fréquence / CA / récence favorable)"
        if sign < lo:
            return "Fidélité modérée ou récence élevée"
        return "Profil fidélité intermédiaire"
    if facet == "montants":
        if sign > hi:
            return "Montants & prix au-dessus de la moyenne"
        if sign < lo:
            return "Montants & prix sous la moyenne"
        return "Profil prix / budget intermédiaire"
    if facet == "volumes":
        if sign > hi:
            return "Forte volumétrie (visiteurs / réservations)"
        if sign < lo:
            return "Faible volumétrie"
        return "Volumétrie intermédiaire"
    if facet == "calendrier":
        return "Saisonnalité marquée (mois / période)"
    if facet == "ids":
        if sign > hi:
            return "Mix d’univers événements très étendu"
        if sign < lo:
            return "Univers d’événements plus restreint"
        return "Profil catalogue différenciant"
    if facet == "autre":
        return f"Autres variables notables (écart |z| ~ {strength:.2f})"
    return None


def _strength_adjusted(facet: str, strength: float, facets: dict[str, tuple[float, float]]) -> float:
    """Réduit le poids des IDs et « autre » lorsque le signal prix / volume / temps est présent."""
    adj = strength
    if facet == "ids":
        other_max = max(
            (facets[f][0] for f in ("montants", "volumes", "calendrier") if f in facets),
            default=0.0,
        )
        if other_max > 0.12:
            adj *= 0.45
        else:
            adj *= 0.72
    if facet == "autre":
        adj *= 0.88
    return adj


def _short_from_facets(facets: dict[str, tuple[float, float]], k_fallback: str) -> str:
    """Libellé court : priorité montants → volumes → calendrier ; IDs / autre en complément seulement."""
    core_keys = ("fidelite", "montants", "volumes", "calendrier")

    def _rank(keys: tuple[str, ...]):
        return sorted(
            ((f, facets[f][0], facets[f][1]) for f in keys if f in facets),
            key=lambda x: -_strength_adjusted(x[0], x[1], facets),
        )

    phrases: list[str] = []
    for facet, strength, sign in _rank(core_keys):
        ph = _phrase_facet(facet, strength, sign)
        if ph:
            phrases.append(ph)
        if len(phrases) >= 2:
            break

    if len(phrases) < 2:
        rest = tuple(f for f in facets if f not in core_keys)
        for facet, strength, sign in _rank(rest):
            ph = _phrase_facet(facet, strength, sign)
            if ph and ph not in phrases:
                phrases.append(ph)
            if len(phrases) >= 2:
                break
    if not phrases:
        mx = max((facets[f][0] for f in facets), default=0.0)
        if mx < 0.12:
            return "Segment proche de la moyenne (profil équilibré)"
        return k_fallback
    if len(phrases) == 1:
        return phrases[0]
    return f"{phrases[0]} — {phrases[1]}"


def _long_from_facets(cluster_id: int, short: str) -> str:
    """Texte long pour exports / JSON : aligné sur le libellé court (pas de z-scores ni facettes brutes)."""
    return f"**Segment {cluster_id}** — {short}"


def cluster_labels_from_centers(
    centers: np.ndarray, feat_names: list[str] | None
) -> tuple[list[str], list[str]]:
    """
    Libellés à partir des centres standardisés.

    Retourne (libellés courts pour légendes, textes longs type rapport).
    Si les noms de colonnes sont génériques (dim_0…), repli sur l'heuristique 2 axes |z| max.
    """
    centers = np.asarray(centers)
    k, p = centers.shape
    names = feat_names if feat_names and len(feat_names) == p else [f"Dim.{i + 1}" for i in range(p)]

    def _short_raw(name: str, maxlen: int = 22) -> str:
        s = str(name).replace("_", " ")
        return s if len(s) <= maxlen else s[: max(1, maxlen - 1)] + "…"

    short: list[str] = []
    long_: list[str] = []

    generic = all(re.match(r"^dim_\d+$", _norm_col(n)) for n in names)

    for i in range(k):
        row = centers[i]
        if generic:
            # Repli : 2 axes |z| max (comportement historique)
            order = np.argsort(-np.abs(row))
            j0 = int(order[0])
            j1 = int(order[1]) if p > 1 else j0
            n0, n1 = _short_raw(names[j0]), _short_raw(names[j1])
            s0, s1 = float(np.sign(row[j0])), float(np.sign(row[j1]))
            a0, a1 = ("↑" if s0 >= 0 else "↓"), ("↑" if s1 >= 0 else "↓")
            short.append(f"{n0} {a0} · {n1} {a1}")
            long_.append(
                f"**Segment {i}** — axes dominants (noms génériques) : « {names[j0]} », « {names[j1]} »."
            )
            continue

        facets = _collect_facets(names, row)
        order = np.argsort(-np.abs(row))
        j0 = int(order[0])
        j1 = int(order[1]) if p > 1 else j0
        fb = (
            f"{_short_raw(names[j0])} ({'↑' if row[j0] >= 0 else '↓'}) · "
            f"{_short_raw(names[j1])} ({'↑' if row[j1] >= 0 else '↓'})"
        )
        sc = _short_from_facets(facets, fb)
        short.append(sc)
        long_.append(_long_from_facets(i, sc))

    return short, long_
