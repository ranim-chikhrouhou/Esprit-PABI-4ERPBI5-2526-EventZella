# -*- coding: utf-8 -*-
"""
Chemins et connexion SQL — projet EventZilla (PI BI NEW).

Mode par défaut configuré pour votre environnement :
- Serveur SQL : `ASUSRANIM`
- Authentification : Windows (`Trusted_Connection=yes`)
- Base DW : `DW_eventzella`

Vous pouvez surcharger via variables d'environnement :
- `EVENTZILLA_SQL_URI` : URI SQLAlchemy complète
- `EVENTZILLA_SQL_SERVER` : nom serveur SQL
- `EVENTZILLA_SQL_PORT` : port (optionnel, ex. 1433)
- `EVENTZILLA_SQL_DW` / `EVENTZILLA_SQL_SA` : noms de bases
- `EVENTZILLA_SQL_SCHEMA` : schéma des tables DW (défaut `dbo`) — utilisé par les requêtes dans `schema_eventzilla.py`
- `EVENTZILLA_FINANCIAL_FACT_RES_COL` / `EVENTZILLA_FINANCIAL_DIM_RES_COL` : paire explicite pour joindre ``Fact_RentabiliteFinanciere`` à ``DimReservation`` si les noms diffèrent (ex. pas de ``id_reservation`` sur le fait)
- `EVENTZILLA_FACT_RENTABILITE_RESERVATION_FK` : nom de la colonne **sur le fait rentabilité** seul (côté dimension = ``EVENTZILLA_DIM_RESERVATION_PK``)
- `EVENTZILLA_DIM_DATE_PK` (défaut ``id_date_SK``), `EVENTZILLA_DIM_RESERVATION_PK` (défaut ``id_reservation_SK``), `EVENTZILLA_DIM_EVENT_PK` (défaut ``id_event_SK``) : colonnes PK côté dimensions (script SSMS EventZilla)
- `EVENTZILLA_ML_SQL_ONLY` : ``1`` (défaut) = pas de repli Excel/CSV dans les notebooks ML ; mettre ``0`` pour autoriser les fichiers locaux
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FILES_MACHINE = REPO_ROOT / "FilesMachine"
DB_BACKUP_DIR = FILES_MACHINE / "DB"
# Fichiers CSV du flux ML : tout sous ce dossier (sauf sous-répertoires exclus ci-dessous).
ML_DIR = REPO_ROOT / "ML"
ML_CSV_ROOT = ML_DIR
ML_PROCESSED = ML_DIR / "processed"
ML_MODELS = ML_DIR / "models_artifacts"
# Archives / ETL hors ML (optionnel, non utilisés par le fallback CSV des scripts ML)
DATA_ORIGINAL = FILES_MACHINE / "data_original"
DATA_SCRAPPED = FILES_MACHINE / "datascrapped"

def iter_project_csv_paths() -> list[Path]:
    """Liste les fichiers ``.csv`` sous ``ML/`` (délégation à ``ML.csv_local_fallback``)."""
    from ML.csv_local_fallback import iter_project_csv_paths as _impl

    return _impl()


def reservation_csv_path() -> Path:
    """Fichier réservations : ``Reservation.xlsx``, ``RESERVATION.csv``, etc. (voir ``csv_local_fallback``)."""
    from ML.csv_local_fallback import reservation_source_path

    return reservation_source_path()

# Fichiers de sauvegarde physiques (sans extension .bak dans ce dépôt — noms tels que sur disque)
BACKUP_DW_FILE = DB_BACKUP_DIR / "DW_Eventzilla"
BACKUP_SA_FILE = DB_BACKUP_DIR / "SA_eventzilla"

# Noms SQL des bases **après** RESTORE (adapter si vous restaurez sous d'autres noms)
DATABASE_DW = os.environ.get("EVENTZILLA_SQL_DW", "DW_eventzella")
DATABASE_SA = os.environ.get("EVENTZILLA_SQL_SA", "SA_eventzilla")

SQL_SERVER = os.environ.get("EVENTZILLA_SQL_SERVER", "ASUSRANIM")
SQL_PORT = os.environ.get("EVENTZILLA_SQL_PORT", "")
SQL_DRIVER = os.environ.get("EVENTZILLA_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
SQL_CONNECTION_URI = os.environ.get("EVENTZILLA_SQL_URI", "")

# Dernière erreur si ``get_sql_engine()`` retourne ``None`` (diagnostic Jupyter).
_SQL_ENGINE_INIT_ERROR: str | None = None

KPI_DOC_EN = FILES_MACHINE / "Liste_Des_Kpis_Updated_English.pdf"
KPI_MD_EN = REPO_ROOT / "docs" / "Liste_Des_Kpis_Updated_English_DAX.md"


def ml_sql_only() -> bool:
    """True = pipelines ML utilisent uniquement le DW (pas de Reservation.xlsx / CSV)."""
    v = os.environ.get("EVENTZILLA_ML_SQL_ONLY", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "allow_local")


def backup_paths_status() -> dict[str, bool]:
    """Indique la présence des fichiers de backup dans FilesMachine/DB/."""
    return {
        "DW_Eventzilla": BACKUP_DW_FILE.is_file(),
        "SA_eventzilla": BACKUP_SA_FILE.is_file(),
    }


def ensure_processed_dirs() -> None:
    ML_PROCESSED.mkdir(parents=True, exist_ok=True)
    ML_MODELS.mkdir(parents=True, exist_ok=True)


def build_windows_auth_uri() -> str:
    """Construit une URI SQLAlchemy pyodbc avec authentification Windows."""
    server = SQL_SERVER if not SQL_PORT else f"{SQL_SERVER}:{SQL_PORT}"
    driver_enc = SQL_DRIVER.replace(" ", "+")
    # trusted_connection fonctionne avec pyodbc/mssql+pyodbc
    return (
        f"mssql+pyodbc://@{server}/{DATABASE_DW}"
        f"?driver={driver_enc}&trusted_connection=yes&TrustServerCertificate=yes"
        f"&Connection+Timeout=15"
    )


def sql_engine_init_error() -> str | None:
    """Message d’erreur si la dernière tentative ``get_sql_engine()`` a échoué (imports, URI)."""
    return _SQL_ENGINE_INIT_ERROR


def get_sql_engine():
    """Engine SQLAlchemy DW; URI explicite sinon fallback Windows Auth par défaut."""
    global _SQL_ENGINE_INIT_ERROR
    _SQL_ENGINE_INIT_ERROR = None
    try:
        from sqlalchemy import create_engine

        uri = SQL_CONNECTION_URI or build_windows_auth_uri()
        # ``timeout`` (secondes) : évite un blocage très long si le serveur SQL est arrêté ou injoignable.
        return create_engine(uri, pool_pre_ping=True, connect_args={"timeout": 15})
    except Exception as e:
        _SQL_ENGINE_INIT_ERROR = f"{type(e).__name__}: {e}"
        return None


def read_dw_sql(query: str, engine=None, params=None):
    """Exécute une requête SELECT sur le DW et retourne un DataFrame pandas."""
    import pandas as pd
    from sqlalchemy import text

    eng = engine or get_sql_engine()
    if eng is None:
        raise RuntimeError("Connexion SQL indisponible (vérifier serveur/driver/accès Windows).")
    with eng.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
