# -*- coding: utf-8 -*-
"""Test connexion DW (Windows Auth par défaut). Exécuter depuis la racine du repo :

    python ML/scripts/run_test_sql_connection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ML.ml_paths import DATABASE_DW, SQL_SERVER, build_windows_auth_uri, read_dw_sql


def main() -> None:
    print("Serveur :", SQL_SERVER)
    print("Base DW  :", DATABASE_DW)
    print("URI (sans mot de passe) :", build_windows_auth_uri())
    try:
        df = read_dw_sql("SELECT DB_NAME() AS db, @@SERVERNAME AS server, GETDATE() AS dt")
        print(df.to_string(index=False))
        print("OK — connexion SQL réussie.")
    except Exception as e:
        print("ÉCHEC :", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
