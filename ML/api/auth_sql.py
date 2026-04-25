# -*- coding: utf-8 -*-
"""
EventZilla — Authentification FastAPI via SQL Server.

Stratégie optimisée (rapide < 3 s) :
  Niveau 1 (immédiat) : validation locale du mot de passe (< 1 ms)
  Niveau 2 : Windows Auth pour lire dbo.AppUsers (1-2 s)
  Niveau 3 : SQL Server Auth si Mixed Mode activé (optionnel, futur)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# ── Configuration ────────────────────────────────────────────────
SECRET_KEY  = os.environ.get("EVENTZILLA_JWT_SECRET", "eventzilla-jwt-secret-2025")
ALGORITHM   = "HS256"
TOKEN_HOURS = 8

_SQL_SERVER = os.environ.get("EVENTZILLA_SQL_SERVER", "ASUSRANIM")
_SQL_DB     = os.environ.get("EVENTZILLA_SQL_DW",     "DW_eventzella")
_SQL_DRIVER = os.environ.get("EVENTZILLA_SQL_DRIVER", "ODBC Driver 17 for SQL Server")

# Serveurs candidats testés en ordre (Windows Auth uniquement)
_WIN_CANDIDATES: tuple[str, ...] = (
    _SQL_SERVER,
    "localhost",
    "127.0.0.1",
)

# Mots de passe de référence — doit correspondre à setup_roles_logins.sql
_FALLBACK_PASSWORDS: dict[str, str] = {
    "ranim_chikhrouhou": "Ranim@Marketing2025!",
    "naima_sarraj":      "Naima@Finance2025!",
    "anas_allam":        "Anas@CRM2025!",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── URI helpers ──────────────────────────────────────────────────
def _win_auth_uri(server: str) -> str:
    drv = _SQL_DRIVER.replace(" ", "+")
    return (
        f"mssql+pyodbc://@{server}/{_SQL_DB}"
        f"?driver={drv}&trusted_connection=yes&TrustServerCertificate=yes"
        f"&Connection+Timeout=3"
    )


# ── Windows Auth + AppUsers ──────────────────────────────────────
def _get_user_from_db(login: str) -> dict | None:
    """Connexion Windows Auth → lire dbo.AppUsers. Retourne dict ou None."""
    from sqlalchemy import create_engine, text
    for server in _WIN_CANDIDATES:
        try:
            eng = create_engine(
                _win_auth_uri(server),
                pool_pre_ping=False, pool_size=1, max_overflow=0,
            )
            with eng.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT login_name, role_name, full_name, email
                        FROM   dbo.AppUsers
                        WHERE  login_name = :login AND is_active = 1
                    """),
                    {"login": login},
                ).fetchone()
            eng.dispose()
            if row is None:
                return None
            return {
                "login":     row.login_name,
                "role":      row.role_name,
                "full_name": row.full_name,
                "email":     row.email,
            }
        except Exception:
            continue
    return None


# ── Authentification principale ──────────────────────────────────
def authenticate_sql_user(login: str, password: str) -> dict:
    """
    Authentifie l'utilisateur en < 3 secondes :
      1. Validation immédiate du mot de passe (dict local)
      2. Windows Auth pour récupérer le profil depuis dbo.AppUsers

    Retourne dict utilisateur ou lève HTTPException.
    """
    login    = (login or "").strip()
    password = (password or "").strip()

    if not login:
        raise HTTPException(status_code=400, detail="Login requis.")

    # ── Étape 1 : validation locale du mot de passe ──────────────
    expected = _FALLBACK_PASSWORDS.get(login)
    if expected is None:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Login '{login}' inconnu. "
                "Noms valides : ranim_chikhrouhou, naima_sarraj, anas_allam."
            ),
        )
    if password != expected:
        raise HTTPException(status_code=401, detail="Mot de passe incorrect.")

    # ── Étape 2 : lecture profil dans AppUsers ───────────────────
    user = _get_user_from_db(login)
    if user is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Impossible de lire dbo.AppUsers. "
                "Vérifiez : 1) SQL Server démarré, "
                "2) setup_roles_logins.sql exécuté dans SSMS."
            ),
        )
    return user


# ── JWT ──────────────────────────────────────────────────────────
def create_jwt_token(user: dict) -> str:
    payload = {
        "sub":       user["login"],
        "role":      user["role"],
        "full_name": user["full_name"],
        "exp":       datetime.utcnow() + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "login":     data["sub"],
            "role":      data["role"],
            "full_name": data["full_name"],
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Token JWT invalide ou expiré.")


def require_role(*allowed_roles: str):
    def _guard(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Rôle '{user['role']}' non autorisé. Requis : {list(allowed_roles)}"
            )
        return user
    return _guard
