# -*- coding: utf-8 -*-
"""Chargement tabulaire local : ``.csv``, ``.xlsx``, ``.xls`` sous ``ML/`` (+ FilesMachine)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Dossier package ``ML/`` (évite import circulaire avec ``ml_paths`` au chargement).
ML_CSV_ROOT = Path(__file__).resolve().parent
_ML_DATA_EXCLUDED_DIR_NAMES = frozenset({"processed", "models_artifacts", "__pycache__"})
_TABULAR_SUFFIXES = frozenset({".csv", ".xlsx", ".xls", ".xlsm"})
_RESERVATION_STEMS = frozenset({"reservation"})


def _legacy_filesmachine_csv_dirs() -> list[Path]:
    try:
        from ML.ml_paths import DATA_ORIGINAL, DATA_SCRAPPED

        return [p for p in (DATA_ORIGINAL, DATA_SCRAPPED) if p.is_dir()]
    except Exception:
        return []


def _is_tabular_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _TABULAR_SUFFIXES


def iter_project_tabular_paths() -> list[Path]:
    """
    Fichiers tableur : ``.csv``, ``.xlsx``, ``.xls``, ``.xlsm`` dans ``ML/`` (hors dossiers
    techniques) et dans ``FilesMachine/data_original`` / ``datascrapped``.
    """
    out: list[Path] = []
    root = ML_CSV_ROOT
    if root.is_dir():
        for p in root.rglob("*"):
            if not _is_tabular_file(p):
                continue
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if any(seg.lower() in _ML_DATA_EXCLUDED_DIR_NAMES for seg in rel.parts[:-1]):
                continue
            out.append(p)

    r_ml = root.resolve()
    for extra in _legacy_filesmachine_csv_dirs():
        try:
            if extra.resolve() == r_ml:
                continue
        except OSError:
            pass
        if not extra.is_dir():
            continue
        for p in extra.rglob("*"):
            if _is_tabular_file(p):
                out.append(p)

    dedup: dict[str, Path] = {}
    for p in out:
        try:
            key = str(p.resolve()).lower()
        except OSError:
            key = str(p).lower()
        dedup[key] = p
    return sorted(dedup.values(), key=lambda x: str(x).lower())


def iter_project_csv_paths() -> list[Path]:
    """Alias : uniquement les ``.csv`` (compat ancien code)."""
    return [p for p in iter_project_tabular_paths() if p.suffix.lower() == ".csv"]


def csv_search_roots_hint() -> str:
    parts = [str(ML_CSV_ROOT)]
    for d in _legacy_filesmachine_csv_dirs():
        parts.append(str(d))
    return " ; ".join(parts)


def read_csv_safe(
    path: Path,
    sep: str = ";",
    encodings: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if encodings is None:
        encodings = ("utf-8", "latin1", "iso-8859-1", "cp1252")
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, sep=sep, encoding="utf-8", errors="replace", low_memory=False)


def read_tabular_safe(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Lit CSV ou Excel (première feuille par défaut). Nécessite ``openpyxl`` (xlsx) / ``xlrd`` (xls)."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return read_csv_safe(path)
    if suf in (".xlsx", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
    if suf == ".xls":
        return pd.read_excel(path, engine="xlrd", sheet_name=sheet_name)
    raise ValueError(f"Format non supporté : {path}")


def reservation_source_path() -> Path:
    """Fichier réservations : ``Reservation.xlsx`` / ``RESERVATION.csv`` / etc."""
    for p in iter_project_tabular_paths():
        stem = p.stem.lower().strip()
        if stem in _RESERVATION_STEMS:
            return p
    raise FileNotFoundError(
        "Aucun fichier Reservation (reservation.xlsx, reservation.csv, …). "
        f"Vérifiez {csv_search_roots_hint()}"
    )


def load_reservation_dataframe(sheet_name: str | int = 0) -> pd.DataFrame:
    """Charge le jeu réservations (Excel ou CSV) avec colonnes normalisées en minuscules."""
    df = read_tabular_safe(reservation_source_path(), sheet_name=sheet_name)
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def _series_looks_like_class_target(s: pd.Series) -> bool:
    """Heuristique : pas une date, au moins 2 modalités, pas trop de valeurs uniques (évite IDs texte)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    nu = int(s.nunique(dropna=True))
    return 2 <= nu <= 200


def financial_wide_has_status_column(df: pd.DataFrame) -> bool:
    """True si ``df`` contient déjà une colonne exploitable comme cible statut (nom connu ou alias DW)."""
    if df is None or len(df.columns) == 0:
        return False
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for k in (
        "reservation_status",
        "status",
        "statut",
        "booking_status",
        "state",
        "res_status",
        "libelle_statut",
        "statut_reservation",
    ):
        if k in lower_map:
            return True
    return False


def enrich_financial_wide_with_performance_reservation_status(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Ajoute ``reservation_status`` au jeu issu du **fait rentabilité** (sans FK réservation) en fusionnant
    avec une requête **Fact_Performance + DimReservation** sur ``(id_date, id_event)``.
    """
    if df is None or len(df) == 0:
        return df
    if financial_wide_has_status_column(df):
        return df
    lm = {str(c).lower().strip(): c for c in df.columns}
    if "id_date" not in lm or "id_event" not in lm:
        return df
    kd, ke = lm["id_date"], lm["id_event"]
    try:
        from ML.schema_eventzilla import build_sql_ml_performance_status_bridge
        from ML.ml_paths import read_dw_sql

        q = build_sql_ml_performance_status_bridge(engine)
        bridge = read_dw_sql(q, engine)
    except Exception:
        return df
    if bridge is None or len(bridge) == 0:
        return df
    b = bridge.copy()
    b.columns = [str(c).lower() for c in b.columns]
    if "reservation_status" not in b.columns:
        return df
    sub = b[["id_date", "id_event", "reservation_status"]].drop_duplicates(
        subset=["id_date", "id_event"],
        keep="first",
    )
    ren = sub.rename(columns={"id_date": kd, "id_event": ke})
    try:
        return df.merge(ren, on=[kd, ke], how="left")
    except Exception:
        dfc = df.copy()
        ren2 = ren.copy()
        for c in (kd, ke):
            dfc[c] = dfc[c].astype(str)
            ren2[c] = ren2[c].astype(str)
        return dfc.merge(ren2, on=[kd, ke], how="left")


def resolve_classification_status_column(df: pd.DataFrame) -> str:
    """
    Colonne cible **statut de réservation** pour le critère C (classification).

    1. ``EVENTZILLA_CLASS_TARGET_COL`` : nom exact ou insensible à la casse.
    2. Noms connus (alias DW / exports) : ``reservation_status``, ``status``, ``statut``, etc.
    3. Première colonne dont le **nom** évoque un statut (regex) et passe l’heuristique ci-dessus.

    Si le parquet ``dw_financial_wide`` vient du SQL **sans** jointure ``DimReservation``,
    aucune colonne statut n’existe : il faut régénérer les données (notebook 00) ou définir la variable
    d’environnement si la cible existe sous un autre nom.
    """
    if df is None or len(df.columns) == 0:
        raise ValueError("DataFrame vide — impossible de résoudre la colonne cible (classification).")

    lower_map = {str(c).lower().strip(): c for c in df.columns}
    env = os.environ.get("EVENTZILLA_CLASS_TARGET_COL", "").strip()
    if env:
        if env in df.columns:
            return env
        if env.lower() in lower_map:
            return lower_map[env.lower()]
        raise ValueError(
            f"EVENTZILLA_CLASS_TARGET_COL={env!r} introuvable parmi les colonnes : "
            + ", ".join(map(str, list(df.columns)[:30]))
            + (" …" if len(df.columns) > 30 else "")
        )

    priority = (
        "reservation_status",
        "status",
        "statut",
        "reservationstatus",
        "booking_status",
        "reservation_state",
        "state",
        "res_status",
        "statut_reservation",
        "statutresa",
        "libelle_statut",
        "lib_statut",
        "statut_resa",
    )
    for p in priority:
        if p in lower_map:
            return lower_map[p]

    name_pat = re.compile(
        r"(status|statut|état|etat|booking_state|reservation_state|res_status|libelle_statut)",
        re.I,
    )
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        if not name_pat.search(str(col)):
            continue
        s = df[col]
        if not _series_looks_like_class_target(s):
            continue
        scored.append((int(s.nunique(dropna=True)), str(col)))
    if scored:
        scored.sort(key=lambda t: (abs(t[0] - 6), t[0]))
        return scored[0][1]

    preview = ", ".join(map(str, list(df.columns)[:45]))
    if len(df.columns) > 45:
        preview += ", …"
    raise ValueError(
        "Colonne de statut introuvable. Colonnes : "
        + preview
        + "\n\n"
        "• Définissez EVENTZILLA_CLASS_TARGET_COL avec le nom exact de la colonne cible.\n"
        "• Régénérez ``dw_financial_wide.parquet`` via le notebook **00** ou laissez le notebook **02** "
        "tenter le **pont Fact_Performance** (``id_reservation`` → ``status``) fusionné sur "
        "``(id_date, id_event)``.\n"
        "• En local sans DW : **Reservation.xlsx** / **reservation.csv** avec ``status`` / ``statut`` "
        "(ML_SQL_ONLY=0)."
    )


def monthly_series_from_reservation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrégat mensuel (nb de réservations par mois) pour fallback série temporelle sans DW.
    Colonnes : ``cal_year``, ``cal_month``, ``nb_fact_rows`` (compatible requête SQL existante).
    """
    d = df.copy()
    if "reservation_date" not in d.columns:
        raise ValueError("Colonne reservation_date absente dans les données réservation")
    d["reservation_date"] = pd.to_datetime(d["reservation_date"], errors="coerce")
    d = d.dropna(subset=["reservation_date"])
    d["cal_year"] = d["reservation_date"].dt.year.astype(int)
    d["cal_month"] = d["reservation_date"].dt.month.astype(int)
    g = d.groupby(["cal_year", "cal_month"], as_index=False).size()
    return g.rename(columns={"size": "nb_fact_rows"})


def _collect_data_paths() -> list[Path]:
    paths = list(iter_project_tabular_paths())

    def is_reservation(p: Path) -> bool:
        return p.stem.lower().strip() in _RESERVATION_STEMS

    pref = sorted((p for p in paths if is_reservation(p)), key=lambda x: x.name.lower())
    rest = sorted(
        (p for p in paths if not is_reservation(p)),
        key=lambda x: (x.parent.as_posix().lower(), x.name.lower()),
    )
    return pref + rest


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = pd.to_numeric(
                out[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    num = out.select_dtypes(include=[np.number]).columns.tolist()
    return out[num]


def load_numeric_from_local_csvs(max_cols: int = 40) -> pd.DataFrame:
    """
    Matrice numérique pour l'étape 00 : priorité au fichier **Reservation** (xlsx/csv),
    puis autres tableurs même nombre de lignes (concat), sinon plus grand fichier.
    """
    paths = _collect_data_paths()
    if not paths:
        raise FileNotFoundError(
            "Aucun fichier .csv / .xlsx / .xls trouvé. Cherché dans : "
            f"{csv_search_roots_hint()}. "
            "Placez vos exports EventZilla (ex. Reservation.xlsx) dans ``ML/`` ou corrigez le DW."
        )

    loaded: list[tuple[Path, pd.DataFrame]] = []
    for p in paths:
        try:
            raw = read_tabular_safe(p)
            n = _numeric_only(raw)
            if n.empty:
                continue
            loaded.append((p, n))
        except Exception:
            continue

    if not loaded:
        raise FileNotFoundError(
            "Fichiers présents mais illisibles (installez openpyxl pour .xlsx, xlrd pour .xls) "
            "ou aucune colonne numérique exploitable."
        )

    def is_res_path(path: Path) -> bool:
        return path.stem.lower().strip() in _RESERVATION_STEMS

    if len(loaded) == 1:
        X = loaded[0][1]
    else:
        lengths = [len(df) for _, df in loaded]
        if len(set(lengths)) == 1:
            X = pd.concat([df.add_prefix(f"{p.stem.lower()}_") for p, df in loaded], axis=1)
        else:
            for p, n in loaded:
                if is_res_path(p):
                    print(f"Fichiers aux tailles différentes — utilisation de {p.name} ({len(n)} lignes).")
                    X = n
                    break
            else:
                i = int(np.argmax(lengths))
                p, n = loaded[i]
                print(f"Fichiers aux tailles différentes — utilisation de {p.name} ({len(n)} lignes).")
                X = n

    ncols = min(max_cols, X.shape[1])
    return X.iloc[:, :ncols].copy()
