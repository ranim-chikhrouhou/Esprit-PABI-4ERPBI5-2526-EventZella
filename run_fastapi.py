#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lancement FastAPI / uvicorn — évite OSError [WinError 10022] sous Windows
quand « --reload » respawn un worker avec un socket invalide (Python 3.11+).

Usage (racine projet) :
    python run_fastapi.py

Reload activé explicitement (toutes plateformes) :
    set UVICORN_RELOAD=1
    python run_fastapi.py

Écoute sur toutes les interfaces (Docker / LAN) :
    set UVICORN_HOST=0.0.0.0
    python run_fastapi.py
"""
from __future__ import annotations

import os
import platform

WIN = platform.system() == "Windows"


def main() -> None:
    import uvicorn

    host = os.environ.get("UVICORN_HOST", "127.0.0.1")
    port = int(os.environ.get("UVICORN_PORT", "8000"))
    env_reload = os.environ.get("UVICORN_RELOAD")
    if env_reload is None:
        reload = not WIN
    else:
        reload = env_reload.strip().lower() in ("1", "true", "yes", "on")

    kw: dict = {"host": host, "port": port}
    if reload:
        kw["reload_delay"] = float(os.environ.get("UVICORN_RELOAD_DELAY", "1.5"))

    print(f"uvicorn ML.api.main:app host={host} port={port} reload={reload}")
    uvicorn.run("ML.api.main:app", reload=reload, **kw)


if __name__ == "__main__":
    main()
