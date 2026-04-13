# -*- coding: utf-8 -*-
"""Utilitaires déploiement / test du clustering K-Means (labels, sanity checks, scoring prototype).

Aligné sur les artefacts : ``kmeans_kpi_segments.joblib``, ``clustering_segment_labels.json``,
``clustering_feature_names.json`` (notebook 01_E).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ML.cluster_labels import cluster_labels_from_centers

CLUSTER_IMPUTER_FNAME = "kmeans_median_imputer.joblib"
CLUSTER_SCALER_FNAME = "kmeans_standard_scaler.joblib"


def load_median_imputer(models_dir: Path):
    """Imputer médian **fitté** sur l’entraînement (même ordre de colonnes que le scaler)."""
    p = models_dir / CLUSTER_IMPUTER_FNAME
    return joblib.load(p) if p.is_file() else None


def load_standard_scaler(models_dir: Path):
    p = models_dir / CLUSTER_SCALER_FNAME
    return joblib.load(p) if p.is_file() else None


def split_business_vs_id_feature_indices(feat_names: list[str]) -> tuple[list[int], list[int]]:
    """
    Sépare les indices pour l’UI : **métier** (montants, volumes, temps) vs **clés DW** (id_*).
    Les IDs restent nécessaires au vecteur complet pour coller au modèle, mais ne sont pas
    le cœur d’un « test » métier — ils sont pré-remplis aux médianes par défaut.
    """
    business: list[int] = []
    technical: list[int] = []
    for i, raw in enumerate(feat_names):
        n = str(raw).lower().replace(" ", "_")
        if n.startswith("id_") or n.endswith("_id") or n == "id":
            technical.append(i)
        else:
            business.append(i)
    return business, technical


def number_input_format_for_feature(col_name: str) -> str:
    """
    Format d’affichage Streamlit (`st.number_input`) : évite les « 200,000000 » inutiles.
    Entiers (compteurs, mois, années, clés) → pas de décimales ; montants → 2 décimales.
    """
    n = str(col_name).lower().replace(" ", "_")
    if any(
        x in n
        for x in (
            "nb_visitor",
            "nb_reservation",
            "visitor",
            "reservation",
            "cal_month",
            "cal_year",
            "is_holiday",
            "count_",
        )
    ):
        return "%.0f"
    if n.startswith("id_") or n.endswith("_id") or n == "id":
        return "%.0f"
    if any(
        x in n
        for x in (
            "price",
            "budget",
            "montant",
            "revenue",
            "ca_",
            "avg_price",
            "benchmark",
        )
    ):
        return "%.2f"
    return "%.2f"


def friendly_feature_label(name: str) -> str:
    """Libellé court FR pour les champs courants du périmètre performance."""
    n = str(name).lower().replace(" ", "_")
    mapping = {
        "final_price": "Prix final",
        "service_price": "Prix service",
        "event_budget": "Budget événement",
        "nb_visitors": "Nb visiteurs",
        "nb_reservations_site": "Nb réservations (site)",
        "cal_month": "Mois calendaire",
        "cal_year": "Année",
        "is_holiday": "Jour férié (0/1)",
        "id_event": "Clé événement (DW)",
        "id_beneficiary": "Clé bénéficiaire (DW)",
        "id_servicecategory": "Clé catégorie service (DW)",
        "id_provider": "Clé prestataire (DW)",
        "id_visitors": "Clé visiteurs (DW)",
        "id_date": "Clé date (DW)",
        "id_reservation": "Clé réservation (DW)",
        "nb_reservations_loyalty": "Nombre de réservations (fréquence d’activité)",
        "ca_total_loyalty": "Chiffre d’affaires cumulé",
        "panier_moyen_loyalty": "Panier moyen",
        "recency_days_loyalty": "Récence — jours depuis la dernière réservation",
        "avg_nb_visitors_loyalty": "Taille moyenne des événements (visiteurs)",
        "volume_reservations_site_loyalty": "Volume cumulé (réservations site)",
    }
    if n in mapping:
        return mapping[n]
    return str(name).replace("_", " ").title()


# --- Déploiement Streamlit : objet métier du formulaire & lecture du segment ---

FORM_OBJECT_METIER_MARKDOWN = """
### À quoi sert ce formulaire ?

**En une phrase :** il sert à **décrire une situation d’activité EventZilla** (offre, fréquentation, montants, période) **telle qu’elle est représentée dans notre data warehouse**, puis à voir **à quel segment** le modèle de clustering rattache cette situation.

Ce n’est pas un formulaire « métier complet » ni une fiche client : c’est **le même genre de ligne** que celles utilisées pour l’entraînement, accessible ici pour que **les enseignants testent le modèle** sur des scénarios plausibles **sans passer par du SQL** sur le DW.

**Ce que vous obtenez :** le **nom du segment** le plus proche et une **comparaison visuelle** simple avec le profil-type de ce segment — pour valider le pipeline ML sur nos données entrepôt, pas pour interpréter chaque champ isolément.
"""

FORM_CLUSTERING_LOYALTY_MARKDOWN = """
### Segmentation fidélité (RFM simplifié)

**Bénéficiaires** : chaque ligne d’entraînement agrège l’**historique** d’un bénéficiaire (nombre de réservations, **CA** total, **panier moyen**, **récence** = jours depuis la dernière réservation, volumes associés).

**Prestataires** : même logique **par prestataire** (charge d’activité, CA, récence d’activité).

**Champs du formulaire** : ils reprennent les **mêmes colonnes numériques** que le modèle — **fréquence** (nombre de réservations), **montants** (CA, panier), **récence**, volumes agrégés. Ce ne sont pas les clés de dimension d’une ligne « réservation » isolée.

**Note :** une **note moyenne** (satisfaction) n’est pas incluse dans l’export SQL actuel du clustering fidélité ; pour l’ajouter il faudrait étendre la requête DW et régénérer le modèle.

Le modèle **K-Means** groupe ces profils en **segments interprétables** (proches de *VIP / fidèle / occasionnel / à relancer* côté bénéficiaires, équivalent prestataires). Les libellés courts peuvent rester **techniques** ; les phrases **label_metier_fr** dans les JSON portent la **lecture fidélité**.
"""


def _segment_title_reader_explain(label_short: str) -> str:
    """Une phrase d’appui pour la démo enseignants (sans jargon technique)."""
    _ = label_short
    return (
        "Dans ce contexte de test, ce libellé indique **quel type de situation** du DW votre scénario ressemble le plus ; "
        "le graphique en dessous **compare** votre saisie au profil-type du segment."
    )


def format_segment_deployment_explanation(
    label_short: str,
    label_metier_fr: str | None,
    *,
    metier_already_shown_above: bool = False,
) -> str:
    """
    Bloc markdown affiché après prédiction : ce que signifie le segment pour l’utilisateur EventZilla.
    ``label_metier_fr`` provient du JSON (optionnel). Si déjà affiché dans l’encadré HTML, passer
    ``metier_already_shown_above=True`` pour éviter la répétition.
    """
    parts: list[str] = [
        "#### Que signifie ce résultat ?",
        "",
        "Le modèle place votre scénario **dans le segment le plus proche** par rapport à l’apprentissage sur les données du DW "
        "(il ne retrouve pas une ligne précise en base).",
        "",
    ]
    if label_metier_fr and str(label_metier_fr).strip() and not metier_already_shown_above:
        parts.append(str(label_metier_fr).strip())
        parts.append("")
    elif metier_already_shown_above and label_metier_fr and str(label_metier_fr).strip():
        parts.append("*La synthèse affichée ci-dessus reprend le libellé du segment.*")
        parts.append("")
    if not metier_already_shown_above:
        parts.append(f"**Libellé du segment :** {label_short.strip() if label_short else '—'}")
        parts.append("")
    parts.append(_segment_title_reader_explain(label_short))
    return "\n".join(parts)


def indices_for_radar_storytelling(
    business_idx: list[int], total_dim: int, min_axes: int = 3
) -> list[int]:
    """Radar lisible : privilégier les axes métier ; sinon tout le vecteur."""
    if len(business_idx) >= min_axes:
        return business_idx
    if len(business_idx) >= 2:
        return business_idx
    return list(range(total_dim))


def predict_cluster_from_raw_features(
    values: Sequence[float],
    imp,
    scaler,
    km,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Chaîne **identique au notebook** : imputation médiane → ``StandardScaler`` → ``predict``.

    Retourne ``(id_cluster, vecteur_standardisé (p,), vecteur_imputé_avant_scale (p,))``.
    """
    X = np.asarray(values, dtype=np.float64).reshape(1, -1)
    Xi = imp.transform(X)
    Xs = scaler.transform(Xi)
    pred = int(km.predict(Xs)[0])
    return pred, Xs.ravel(), Xi.ravel()


def load_clustering_segment_labels_json(models_dir: Path, filename: str = "clustering_segment_labels.json") -> dict | None:
    """Charge ``clustering_segment_labels*.json`` si présent et valide."""
    path = models_dir / filename
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "segments" not in data:
        return None
    return data


def resolve_segment_labels(
    km,
    feat_names: list[str] | None,
    models_dir: Path,
    *,
    labels_json: str = "clustering_segment_labels.json",
) -> tuple[list[str], list[str], str, list[str]]:
    """
    Retourne (libellés courts, libellés longs, source, textes métier optionnels par segment).

    ``label_metier_fr`` dans le JSON : courte explication business (EventZilla) par cluster.
    """
    payload = load_clustering_segment_labels_json(models_dir, filename=labels_json)
    k_model = int(getattr(km, "n_clusters", 0) or 0)
    if (
        payload
        and int(payload.get("k", -1)) == k_model
        and isinstance(payload.get("segments"), list)
        and len(payload["segments"]) == k_model
    ):
        segs = sorted(payload["segments"], key=lambda x: int(x.get("cluster_id", 0)))
        short = [str(s.get("label_short", "")) for s in segs]
        long_ = [str(s.get("label_long_plain", s.get("label_long", ""))) for s in segs]
        metier = [str(s.get("label_metier_fr", "") or "").strip() for s in segs]
        return short, long_, "clustering_segment_labels.json (export notebook)", metier
    if hasattr(km, "cluster_centers_") and km.cluster_centers_ is not None:
        cc = np.asarray(km.cluster_centers_)
        sh, lo = cluster_labels_from_centers(cc, feat_names)
        return sh, lo, "recalcul depuis centres K-Means", [""] * len(sh)
    return [], [], "indisponible", []


def segment_reference_table(k: int, short: list[str], long_: list[str]) -> pd.DataFrame:
    """Table de référence pour l'UI (un segment par ligne)."""
    rows = []
    for i in range(k):
        rows.append(
            {
                "Id": i,
                "Libellé (court)": short[i] if i < len(short) else "—",
                "Description (détail)": long_[i] if i < len(long_) else "—",
            }
        )
    return pd.DataFrame(rows)


def sanity_check_centroid_predictions(km) -> pd.DataFrame:
    """
    Chaque centre projeté dans ``predict`` doit retourner son propre indice de cluster
    (cohérence sklearn / fichier joblib).
    """
    if not hasattr(km, "cluster_centers_") or km.cluster_centers_ is None:
        return pd.DataFrame()
    cc = np.asarray(km.cluster_centers_)
    pred = km.predict(cc)
    rows = []
    for i in range(cc.shape[0]):
        ok = int(pred[i]) == i
        rows.append(
            {
                "Segment": i,
                "Prédiction": int(pred[i]),
                "OK": "✓" if ok else "✗",
            }
        )
    return pd.DataFrame(rows)


def batch_predict_around_centroid(
    km,
    segment_id: int,
    n_trials: int,
    noise_std: float,
    seed: int,
) -> tuple[pd.Series, np.ndarray]:
    """
    Tirages gaussiens autour du centre ``segment_id`` (espace **standardisé**, comme l'entraînement).
    Retourne (effectifs par cluster prédit, dernier vecteur utilisé).
    """
    if not hasattr(km, "cluster_centers_") or km.cluster_centers_ is None:
        return pd.Series(dtype=int), np.array([])
    cc = np.asarray(km.cluster_centers_)
    k, p = cc.shape
    if not (0 <= segment_id < k):
        return pd.Series(dtype=int), np.array([])
    rng = np.random.default_rng(seed)
    center = cc[segment_id]
    preds = []
    last = center.copy()
    for _ in range(max(1, n_trials)):
        z = center + rng.normal(0.0, noise_std, size=p)
        last = z
        preds.append(int(km.predict(z.reshape(1, -1))[0]))
    return pd.Series(preds).value_counts().sort_index(), last


def contrast_midpoint_prediction(km, seg_a: int, seg_b: int) -> tuple[int, np.ndarray]:
    """Milieu des deux centres — utile pour illustrer une zone « frontière »."""
    cc = np.asarray(km.cluster_centers_)
    mid = 0.5 * (cc[seg_a] + cc[seg_b])
    pred = int(km.predict(mid.reshape(1, -1))[0])
    return pred, mid


def distances_to_centroids(point: np.ndarray, km) -> pd.DataFrame:
    """Distances euclidiennes au centre de chaque cluster (espace standardisé)."""
    if point.size == 0 or not hasattr(km, "cluster_centers_"):
        return pd.DataFrame()
    cc = np.asarray(km.cluster_centers_)
    d = np.linalg.norm(cc - point.reshape(1, -1), axis=1)
    return pd.DataFrame({"Segment": np.arange(len(d)), "Distance au centre": d})


# Ordre d’affichage des champs fidélité dans Streamlit (logique métier : activité → montants → récence)
LOYALTY_FEATURE_DISPLAY_ORDER: tuple[str, ...] = (
    "nb_reservations_loyalty",
    "volume_reservations_site_loyalty",
    "avg_nb_visitors_loyalty",
    "ca_total_loyalty",
    "panier_moyen_loyalty",
    "recency_days_loyalty",
)


def loyalty_form_group_key(col_name: str) -> str:
    n = str(col_name).lower().replace(" ", "_")
    if "recency" in n:
        return "récence"
    if "panier" in n or "ca_total" in n:
        return "montants"
    return "activité"


def loyalty_form_group_title(key: str) -> str:
    return {
        "activité": "Fréquence & volumes",
        "montants": "Chiffre d’affaires & panier",
        "récence": "Récence d’activité",
    }.get(key, "Autres")


def ordered_feature_indices_for_form(feat_names: list[str], *, loyalty: bool) -> list[int]:
    """Indices colonnes dans un ordre lisible pour le formulaire (fidélité RFM ou wide)."""
    if not loyalty:
        return list(range(len(feat_names)))
    names = [str(x) for x in feat_names]
    name_to_i = {n: j for j, n in enumerate(names)}
    out: list[int] = []
    for name in LOYALTY_FEATURE_DISPLAY_ORDER:
        if name in name_to_i:
            out.append(name_to_i[name])
    for j in range(len(names)):
        if j not in out:
            out.append(j)
    return out


def loyalty_artifacts_complete(models_dir: Path, prefix: str) -> bool:
    """Fichiers requis pour entraîner / scorer une voie fidélité (comme ``run_01_clustering.py``)."""
    names = (
        f"kmeans_{prefix}.joblib",
        f"kmeans_standard_scaler_{prefix}.joblib",
        f"kmeans_median_imputer_{prefix}.joblib",
        f"clustering_segment_labels_{prefix}.json",
        f"clustering_feature_names_{prefix}.json",
    )
    return all((models_dir / n).is_file() for n in names)


def build_loyalty_modes_from_disk(models_dir: Path, legacy_m: dict | None) -> dict[str, dict] | None:
    """
    Reconstruit le bloc ``modes`` attendu par Streamlit lorsque ``metrics_clustering.json``
    est encore au format notebook (legacy) alors que les artefacts fidélité sont présents.
    """
    modes: dict[str, dict] = {}
    for entity, prefix in (("beneficiary", "loyalty_beneficiary"), ("provider", "loyalty_provider")):
        if not loyalty_artifacts_complete(models_dir, prefix):
            continue
        labels_fp = models_dir / f"clustering_segment_labels_{prefix}.json"
        try:
            data = json.loads(labels_fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segs = data.get("segments") or []
        if not segs:
            continue
        k = int(data.get("k", len(segs)))
        shares = {str(int(s["cluster_id"])): float(s.get("share_train_sample", 0.0)) for s in segs}
        sil = None
        n_samp = n_tr = n_ho = 0
        if legacy_m and isinstance(legacy_m.get("modes"), dict):
            om = legacy_m["modes"].get(entity)
            if isinstance(om, dict):
                sil = om.get("silhouette_holdout") or om.get("silhouette")
                n_samp = int(om.get("n_samples") or 0)
                n_tr = int(om.get("n_train") or 0)
                n_ho = int(om.get("n_holdout") or 0)
        modes[entity] = {
            "entity": entity,
            "artifact_prefix": prefix,
            "k": k,
            "silhouette": sil,
            "silhouette_train": sil,
            "silhouette_holdout": sil,
            "davies_bouldin_kmeans": legacy_m.get("davies_bouldin_kmeans") if legacy_m else None,
            "n_samples": n_samp,
            "n_train": n_tr,
            "n_holdout": n_ho,
            "kpi_alignment": f"fidelite_{entity}s_rfm",
            "features_file": f"clustering_feature_names_{prefix}.json",
            "segment_labels_file": f"clustering_segment_labels_{prefix}.json",
            "model_file": f"kmeans_{prefix}.joblib",
            "scaler_file": f"kmeans_standard_scaler_{prefix}.joblib",
            "imputer_file": f"kmeans_median_imputer_{prefix}.joblib",
            "cluster_share_train_sample": shares,
        }
    return modes if modes else None


def merge_metrics_for_loyalty_ui(models_dir: Path, legacy_m: dict | None) -> dict:
    """Enrichit les métriques chargées depuis JSON si les artefacts fidélité sont complets sur disque."""
    if legacy_m is None:
        legacy_m = {}
    modes = build_loyalty_modes_from_disk(models_dir, legacy_m)
    if not modes:
        return legacy_m
    primary = "beneficiary" if "beneficiary" in modes else "provider"
    pb = modes[primary]
    return {
        **legacy_m,
        "task": "clustering_loyalty_rfm",
        "modes": modes,
        "default_mode": primary,
        "model_primary": "KMeans",
        "k": pb.get("k"),
        "silhouette": pb.get("silhouette_holdout"),
        "silhouette_holdout": pb.get("silhouette_holdout"),
        "n_samples": pb.get("n_samples"),
        "n_train": pb.get("n_train"),
        "n_holdout": pb.get("n_holdout"),
        "kpi_alignment": "fidelite_beneficiaires_et_prestataires",
    }


def filter_clustering_metrics_if_models_missing(models_dir: Path, m: dict) -> dict:
    """
    Ne garde que les entrées ``modes`` dont les fichiers ``model_file`` existent.
    Si plus aucun modèle fidélité n’est utilisable, repasse ``task`` en ``clustering`` (vue large).
    """
    modes = m.get("modes")
    if not isinstance(modes, dict) or not modes:
        return m
    filt: dict[str, dict] = {}
    for k, v in modes.items():
        if isinstance(v, dict):
            mf = v.get("model_file")
            if mf and (models_dir / str(mf)).is_file():
                filt[str(k)] = v
    if len(filt) == len(modes):
        return m
    if filt:
        out = {**m, "modes": filt}
        dm = str(m.get("default_mode") or "")
        if dm and dm not in filt:
            out["default_mode"] = next(iter(filt.keys()))
        return out
    out = {k: v for k, v in m.items() if k not in ("modes", "default_mode")}
    if out.get("task") == "clustering_loyalty_rfm":
        out["task"] = "clustering"
    return out


def loyalty_json_hint_run_script(models_dir: Path) -> bool:
    """True si un export JSON fidélité est présent sans la pile complète (``.joblib`` + features)."""
    for prefix in ("loyalty_beneficiary", "loyalty_provider"):
        if (models_dir / f"clustering_segment_labels_{prefix}.json").is_file():
            if not loyalty_artifacts_complete(models_dir, prefix):
                return True
    return False


def segment_card_title_loyalty(metier_fr: str | None, label_short: str) -> str:
    """Titre lisible : privilégie la phrase métier (VIP, fidèle…) plutôt que le libellé technique des centres."""
    if metier_fr and str(metier_fr).strip():
        t = str(metier_fr).replace("**", "").strip()
        for sep in ("—", ":", "\n"):
            if sep in t:
                t = t.split(sep)[0].strip()
                break
        return t[:200] if t else label_short
    return label_short
