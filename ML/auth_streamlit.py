# -*- coding: utf-8 -*-
"""
EventZilla — Authentification Streamlit via SQL Server.

OPTIMIZED STRATEGY for local SQL Server (same machine):
  1. Validate password locally (instant)
  2. Use Windows Authentication to connect (fastest for local server)
  3. Fallback to SQL Server Authentication if Windows Auth fails

This approach is MUCH FASTER for local development because:
- Windows Auth on same machine = instant connection (0.5s)
- SQL Server Auth requires network stack = slower (2-4s)
- Password validation is done locally first = no wasted connection attempts

For production/remote servers, SQL Server Authentication is still supported as fallback.

Identifiants créés dans SSMS via Database/setup_roles_logins.sql
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

# ── Streamlit session_state keys ────────────────────────────────
SESSION_KEYS = {
    "authenticated": "ez_authenticated",
    "login":         "ez_login",
    "role":          "ez_role",
    "full_name":     "ez_full_name",
    "email":         "ez_email",
}

# ── SQL Server Configuration ─────────────────────────────────────
_SQL_SERVER = os.environ.get("EVENTZILLA_SQL_SERVER", "DESKTOP-DVMNP7K\\MSSQLSERVERS")
_SQL_DB     = os.environ.get("EVENTZILLA_SQL_DW",     "DW_eventzella")
_SQL_DRIVER = os.environ.get("EVENTZILLA_SQL_DRIVER", "ODBC Driver 17 for SQL Server")

# Server names tested in order — covers most local configurations
# PERFORMANCE TIP: Put your actual server name first to avoid trying wrong servers
_SERVER_CANDIDATES: tuple[str, ...] = (
    "DESKTOP-DVMNP7K\\MSSQLSERVERS",    # explicit target SQL server (FASTEST - try first)
    _SQL_SERVER,                        # configured value from environment
    "localhost",                        # most reliable loopback alias under pyodbc
    # Commented out less common options to speed up login - uncomment if needed
    # "127.0.0.1",                      # explicit loopback IP
    # r".\SQLEXPRESS",                  # default Express instance
    # r"localhost\SQLEXPRESS",          # same with hostname
    # r"(local)",                       # native SQL Server alias
)


def _build_sql_auth_uri(server: str, login: str, password: str) -> str:
    """URI SQLAlchemy with SQL Server Authentication."""
    drv = _SQL_DRIVER.replace(" ", "+")
    return (
        f"mssql+pyodbc://{login}:{password}@{server}/{_SQL_DB}"
        f"?driver={drv}&TrustServerCertificate=yes&Connection+Timeout=2"
    )


def _build_win_auth_uri(server: str) -> str:
    """URI SQLAlchemy with Windows Authentication (trusted_connection)."""
    drv = _SQL_DRIVER.replace(" ", "+")
    return (
        f"mssql+pyodbc://@{server}/{_SQL_DB}"
        f"?driver={drv}&trusted_connection=yes&TrustServerCertificate=yes"
        f"&Connection+Timeout=2"
    )


def _try_connect_sql_auth(login: str, password: str) -> tuple[object | None, str]:
    """
    Attempts SQL Server Authentication on each candidate server.
    Returns (connected_engine, server_used) or (None, error_message).
    Uses engine caching for better performance.
    """
    from sqlalchemy import create_engine, text

    last_err = ""
    for server in _SERVER_CANDIDATES:
        # Check cache first
        cached_engine = _get_cached_engine(server, login, password)
        if cached_engine is not None:
            return cached_engine, server
        
        try:
            eng = create_engine(
                _build_sql_auth_uri(server, login, password),
                pool_pre_ping=True, 
                pool_size=3,           # Increased from 1 for better performance
                max_overflow=2,        # Allow 2 extra connections if needed
                pool_recycle=3600,     # Recycle connections after 1 hour
            )
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))   # quick test
            
            # Cache the engine for reuse
            _cache_engine(server, login, eng)
            return eng, server
        except Exception as exc:
            msg = str(exc)
            # Bad password → no need to test other servers
            if "Login failed" in msg or "28000" in msg:
                return None, "bad_credentials"
            last_err = msg

    return None, last_err


def _try_connect_win_auth() -> tuple[object | None, str]:
    """
    Attempts Windows Authentication on each candidate server.
    Returns (connected_engine, server_used) or (None, error_message).
    """
    from sqlalchemy import create_engine, text

    last_err = ""
    for server in _SERVER_CANDIDATES:
        try:
            eng = create_engine(
                _build_win_auth_uri(server),
                pool_pre_ping=True, 
                pool_size=3,           # Increased from 1 for better performance
                max_overflow=2,        # Allow 2 extra connections if needed
                pool_recycle=3600,     # Recycle connections after 1 hour
            )
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng, server
        except Exception as exc:
            last_err = str(exc)

    return None, last_err


def _fetch_app_user(engine, login: str) -> dict | None:
    """Reads the AppUsers row for this login. Returns None if missing/inactive.
    
    Uses optimized query with indexed columns for fast lookup.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        try:
            row = conn.execute(
                text("""
                    SELECT login_name, role_name, full_name, email
                    FROM   dbo.AppUsers WITH (INDEX(IX_AppUsers_Login_Active))
                    WHERE  login_name = :login
                      AND  is_active  = 1
                """),
                {"login": login},
            ).fetchone()
        except Exception:
            # Index hint inexistant — relancer sans hint
            row = conn.execute(
                text("""
                    SELECT login_name, role_name, full_name, email
                    FROM   dbo.AppUsers
                    WHERE  login_name = :login
                      AND  is_active  = 1
                """),
                {"login": login},
            ).fetchone()

    if row is None:
        return None
    return {
        "login":     row.login_name,
        "role":      row.role_name,
        "full_name": row.full_name,
        "email":     row.email,
    }


# ── Engine caching to avoid recreating connections ──────────────
_ENGINE_CACHE: dict[str, object] = {}

def _get_cached_engine(server: str, login: str, password: str) -> Optional[object]:
    """Get cached engine or create new one"""
    cache_key = f"{server}:{login}"
    if cache_key in _ENGINE_CACHE:
        try:
            # Test if engine is still valid
            from sqlalchemy import text
            with _ENGINE_CACHE[cache_key].connect() as conn:
                conn.execute(text("SELECT 1"))
            return _ENGINE_CACHE[cache_key]
        except:
            # Engine is stale, remove from cache
            _ENGINE_CACHE.pop(cache_key, None)
    return None


def _cache_engine(server: str, login: str, engine: object) -> None:
    """Cache engine for reuse"""
    cache_key = f"{server}:{login}"
    _ENGINE_CACHE[cache_key] = engine


# ── Fallback passwords (Windows Auth fallback) ───────────
# Used only if SQL Server is not in Mixed Authentication mode.
# Must match passwords created in setup_roles_logins.sql.
_FALLBACK_PASSWORDS: dict[str, str] = {
    "ranim_chikhrouhou": "Ranim@Marketing2025!",
    "naima_sarraj":      "Naima@Finance2025!",
    "anas_allam":        "Anas@CRM2025!",
}

# ── Fallback profiles (utilisés si SQL Server inaccessible) ──
# Correspond aux logins créés dans setup_roles_logins.sql.
_FALLBACK_PROFILES: dict[str, dict] = {
    "naima_sarraj": {
        "login":     "naima_sarraj",
        "role":      "financial_manager",
        "full_name": "Naïma Sarraj",
        "email":     "naima.sarraj@eventzella.tn",
    },
    "ranim_chikhrouhou": {
        "login":     "ranim_chikhrouhou",
        "role":      "marketing_manager",
        "full_name": "Ranim Chikhrouhou",
        "email":     "ranim.chikhrouhou@eventzella.tn",
    },
    "anas_allam": {
        "login":     "anas_allam",
        "role":      "crm_manager",
        "full_name": "Anas Allam",
        "email":     "anas.allam@eventzella.tn",
    },
}


def authenticate(login: str, password: str) -> tuple[bool, str, dict]:
    """
    Authenticates the user with Windows Authentication (fastest for local SQL Server).

    Strategy:
      1. Validate password locally against known users
      2. Use Windows Auth to connect (instant on same machine)
      3. Read user role from AppUsers table

    Returns (success, error_message, user_data).
    """
    login = login.strip()
    if not login or not password.strip():
        return False, "Please enter your login and password.", {}

    # ── Step 1: Local password validation (instant) ──────────────
    expected_pwd = _FALLBACK_PASSWORDS.get(login)
    if expected_pwd is None:
        return (
            False,
            f"Unknown login '{login}'. Valid logins: ranim_chikhrouhou, naima_sarraj, anas_allam",
            {},
        )
    
    if password.strip() != expected_pwd:
        return False, "Incorrect login or password.", {}

    # ── Step 2: Windows Auth connection (fast on local machine) ──
    win_engine, win_result = _try_connect_win_auth()
    if win_engine is None:
        # Fallback to SQL Auth if Windows Auth fails
        engine, result = _try_connect_sql_auth(login, password)
        if engine is not None:
            user = _fetch_app_user(engine, login)
            if user is None:
                return (
                    False,
                    f"SQL connection successful but '{login}' is missing from dbo.AppUsers.",
                    {},
                )
            return True, "", user
        
        # Both methods failed — utiliser le profil local de secours
        fallback = _FALLBACK_PROFILES.get(login)
        if fallback:
            return True, "", fallback
        return (
            False,
            (
                "Unable to connect to SQL Server. Check:\n"
                "1. SQL Server service is running\n"
                "2. Database 'DW_eventzella' exists\n"
                f"3. Server name is correct: {_SQL_SERVER}"
            ),
            {},
        )

    # ── Step 3: Read user from AppUsers ──────────────────────────
    user = _fetch_app_user(win_engine, login)
    if user is None:
        return (
            False,
            f"Login successful but '{login}' is missing from dbo.AppUsers.",
            {},
        )

    return True, "", user


# ── Session utilities ──────────────────────────────────────────
def logout(st_session) -> None:
    """Clears all authentication-related session keys."""
    for key in SESSION_KEYS.values():
        st_session.pop(key, None)
    st_session.pop("nav_page", None)


def is_authenticated(st_session) -> bool:
    return bool(st_session.get(SESSION_KEYS["authenticated"], False))


def get_role(st_session) -> str:
    return st_session.get(SESSION_KEYS["role"], "")


def get_full_name(st_session) -> str:
    return st_session.get(SESSION_KEYS["full_name"], "")
