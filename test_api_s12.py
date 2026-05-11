#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventZilla — Test API End-to-End (Validation S12)
==================================================
Vérifie tous les critères de la grille S12 :
  ✅ API opérationnelle
  ✅ Authentification JWT
  ✅ /predict/regression → prédiction retournée
  ✅ /predict/classification → prédiction retournée
  ✅ /predict/segmentation/beneficiaire → segment retourné
  ✅ /predict/timeseries → métriques retournées
  ✅ /metrics → métriques globales
  ✅ /mlflow/status → MLflow connecté

Usage :
    python test_api_s12.py
    python test_api_s12.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://localhost:8000"
TIMEOUT     = 15

USERS = {
    "naima_sarraj":      {"password": "Naima@Finance2025!",   "role": "financial_manager"},
    "ranim_chikhrouhou": {"password": "Ranim@Marketing2025!", "role": "marketing_manager"},
    "anas_allam":        {"password": "Anas@CRM2025!",        "role": "crm_manager"},
}

# Couleurs terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

RESULTS: list[tuple[str, bool, str]] = []


def ok(name: str, detail: str = "") -> None:
    RESULTS.append((name, True, detail))
    print(f"  {GREEN}✅ PASS{RESET} {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str = "") -> None:
    RESULTS.append((name, False, detail))
    print(f"  {RED}❌ FAIL{RESET} {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'─'*55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'─'*55}{RESET}")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_health(base: str) -> bool:
    section("1. Santé de l'API (Health Check)")
    try:
        r = requests.get(f"{base}/", timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            modeles = data.get("modeles_charges", {})
            ok("GET /  → 200", f"status={data.get('status')}")
            loaded = [k for k, v in modeles.items() if v]
            missing = [k for k, v in modeles.items() if not v]
            if loaded:
                ok("Modèles chargés", ", ".join(loaded))
            if missing:
                fail("Modèles manquants", ", ".join(missing))
            return True
        fail(f"GET /  → {r.status_code}")
    except requests.exceptions.ConnectionError:
        fail("GET /", f"Impossible de contacter {base}")
    return False


def test_auth(base: str) -> dict[str, str]:
    section("2. Authentification JWT")
    tokens: dict[str, str] = {}
    for login, info in USERS.items():
        r = requests.post(
            f"{base}/auth/login",
            json={"login": login, "password": info["password"]},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            tokens[login] = data["access_token"]
            ok(f"POST /auth/login [{login}]", f"rôle={data.get('role')}")
        else:
            fail(f"POST /auth/login [{login}]", f"HTTP {r.status_code}: {r.text[:100]}")
    return tokens


def test_regression(base: str, tokens: dict) -> None:
    section("3. Prédiction — Régression (final_price)")
    token = tokens.get("naima_sarraj") or tokens.get("ranim_chikhrouhou")
    if not token:
        fail("POST /predict/regression", "Pas de token disponible")
        return

    body = {
        "id_date": 1.0, "id_event": 42.0, "id_servicecategory": 3.0,
        "id_benchmark": 2.0, "id_provider": 7.0,
        "service_price": 1200.0, "benchmark_avg_price": 1300.0,
        "event_budget": 5000.0, "cal_month": 4.0, "cal_year": 2024.0,
        "quarter": 2.0, "commission_margin": 150.0,
    }
    r = requests.post(f"{base}/predict/regression",
                      headers={"Authorization": f"Bearer {token}"},
                      json=body, timeout=TIMEOUT)
    if r.status_code == 200:
        data = r.json()
        prix = data.get("montant_predit")
        ok("POST /predict/regression → 200",
           f"montant_predit={prix} TND, modèle={data.get('modele','?')}")
    else:
        fail("POST /predict/regression", f"HTTP {r.status_code}: {r.text[:200]}")


def test_classification(base: str, tokens: dict) -> None:
    section("4. Prédiction — Classification (statut réservation)")
    token = tokens.get("ranim_chikhrouhou") or tokens.get("naima_sarraj")
    if not token:
        fail("POST /predict/classification", "Pas de token disponible")
        return

    body = {
        "id_date": 1.0, "id_event": 42.0, "id_servicecategory": 3.0,
        "id_benchmark": 2.0, "id_provider": 7.0,
        "final_price": 1500.0, "service_price": 1200.0,
        "benchmark_avg_price": 1300.0, "event_budget": 5000.0,
        "cal_month": 4.0, "cal_year": 2024.0, "quarter": 2.0,
    }
    r = requests.post(f"{base}/predict/classification",
                      headers={"Authorization": f"Bearer {token}"},
                      json=body, timeout=TIMEOUT)
    if r.status_code == 200:
        data = r.json()
        statut = data.get("statut_predit")
        ok("POST /predict/classification → 200",
           f"statut_predit={statut}, modèle={data.get('modele','?')}")
    else:
        fail("POST /predict/classification", f"HTTP {r.status_code}: {r.text[:200]}")


def test_segmentation(base: str, tokens: dict) -> None:
    section("5. Prédiction — Segmentation RFM (K-Means)")
    token = tokens.get("ranim_chikhrouhou") or tokens.get("anas_allam")
    if not token:
        fail("POST /predict/segmentation", "Pas de token disponible")
        return

    body = {
        "nb_reservations_loyalty": 12.0,
        "ca_total_loyalty": 15000.0,
        "panier_moyen_loyalty": 1250.0,
        "recency_days_loyalty": 30.0,
        "avg_nb_visitors_loyalty": 85.0,
        "volume_reservations_site_loyalty": 5.0,
    }
    for entity in ("beneficiaire", "prestataire"):
        r = requests.post(f"{base}/predict/segmentation/{entity}",
                          headers={"Authorization": f"Bearer {token}"},
                          json=body, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            ok(f"POST /predict/segmentation/{entity} → 200",
               f"segment_id={data.get('segment_id')}, label={data.get('segment_label','?')}")
        else:
            fail(f"POST /predict/segmentation/{entity}", f"HTTP {r.status_code}: {r.text[:200]}")


def test_timeseries(base: str, tokens: dict) -> None:
    section("6. Prédiction — Séries Temporelles (Holt)")
    token = tokens.get("naima_sarraj") or tokens.get("ranim_chikhrouhou")
    if not token:
        fail("GET /predict/timeseries", "Pas de token disponible")
        return

    r = requests.get(f"{base}/predict/timeseries",
                     headers={"Authorization": f"Bearer {token}"},
                     params={"horizon": 3}, timeout=TIMEOUT)
    if r.status_code == 200:
        data = r.json()
        champ = data.get("modele_champion", "?")
        metriques = data.get("metriques_test", {})
        ok("GET /predict/timeseries → 200",
           f"champion={champ}, RMSE={metriques.get('rmse','?')}, MAPE={metriques.get('mape','?')}%")
    else:
        fail("GET /predict/timeseries", f"HTTP {r.status_code}: {r.text[:200]}")


def test_global_metrics(base: str, tokens: dict) -> None:
    section("7. Métriques globales /models/metrics")
    token = next(iter(tokens.values()), None)
    if not token:
        fail("GET /models/metrics", "Pas de token disponible")
        return

    r = requests.get(f"{base}/models/metrics",
                     headers={"Authorization": f"Bearer {token}"},
                     timeout=TIMEOUT)
    if r.status_code == 200:
        try:
            data = r.json()
            present = [k for k in ["classification","regression","clustering","timeseries"]
                       if data.get(k)]
            ok("GET /models/metrics → 200", f"sections présentes: {present}")
        except Exception as e:
            fail("GET /models/metrics", f"Réponse non-JSON: {r.text[:100]}")
    else:
        fail("GET /models/metrics", f"HTTP {r.status_code}: {r.text[:200]}")


def test_mlflow_status(base: str, tokens: dict) -> None:
    section("8. MLflow Status")
    token = next(iter(tokens.values()), None)
    if not token:
        fail("GET /mlflow/status", "Pas de token disponible")
        return

    try:
        r = requests.get(f"{base}/mlflow/status",
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=20)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status")
            exp_count = data.get("experiments_count", "?")
            if status in ("connected", "degraded"):
                ok("GET /mlflow/status → " + status,
                   f"URI={data.get('tracking_uri','?')[:60]}, experiments={exp_count}")
            else:
                fail("GET /mlflow/status", f"status={status}, msg={data.get('message','?')}")
        else:
            fail("GET /mlflow/status", f"HTTP {r.status_code}: {r.text[:200]}")
    except requests.exceptions.Timeout:
        fail("GET /mlflow/status", "Timeout 20s — MLflow server non joignable")


def test_docs(base: str) -> None:
    section("9. Documentation Interactive")
    for path in ["/docs", "/openapi.json"]:
        try:
            r = requests.get(f"{base}{path}", timeout=10)
            if r.status_code == 200:
                ok(f"GET {path} → 200")
            else:
                fail(f"GET {path}", f"HTTP {r.status_code}")
        except Exception as e:
            fail(f"GET {path}", str(e))


# ── Rapport final ──────────────────────────────────────────────────────────

def print_summary() -> int:
    total  = len(RESULTS)
    passed = sum(1 for _, ok_, _ in RESULTS if ok_)
    failed = total - passed
    pct    = int(100 * passed / total) if total else 0

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  RAPPORT S12 — VALIDATION MLOPS{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  Tests réussis : {GREEN}{passed}/{total}{RESET}  ({pct}%)")

    if failed:
        print(f"\n  {RED}Tests échoués :{RESET}")
        for name, ok_, detail in RESULTS:
            if not ok_:
                print(f"    ✗ {name}" + (f" : {detail}" if detail else ""))

    print(f"\n  {BOLD}Grille S12 :{RESET}")
    categories = [
        ("Experiment Tracking (MLflow)",   "GET /mlflow/status"),
        ("Automated Pipeline",             "run_pipeline_s12.py"),
        ("Model Management",               "Modèles chargés"),
        ("Model Serving (/predict)",       "POST /predict/regression"),
        ("Containerization (Docker)",      "docker-compose.yml"),
        ("Code Quality",                   "GET /  → 200"),
        ("Web App → API",                  "POST /predict/regression"),
    ]
    for cat, test_name in categories:
        related = [ok_ for name, ok_, _ in RESULTS if test_name in name]
        status = (GREEN + "✅") if (related and all(related)) else \
                 (YELLOW + "⚠️ ") if not related else (RED + "❌")
        print(f"    {status} {cat}{RESET}")

    print(f"\n{'═'*55}")
    return 0 if failed == 0 else 1


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test API EventZilla S12")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"URL de base de l'API (défaut: {DEFAULT_URL})")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  EventZilla — Test End-to-End API (S12 MLOps){RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  API cible : {CYAN}{base}{RESET}")
    print(f"  Heure     : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    alive = test_health(base)
    if not alive:
        print(f"\n{RED}⛔ L'API n'est pas joignable. Arrêt des tests.{RESET}")
        print("   Lancez d'abord : python run_fastapi.py")
        sys.exit(1)

    tokens = test_auth(base)
    if not tokens:
        print(f"\n{RED}⛔ Aucun token obtenu. Vérifiez l'authentification.{RESET}")
        sys.exit(1)

    test_regression(base, tokens)
    test_classification(base, tokens)
    test_segmentation(base, tokens)
    test_timeseries(base, tokens)
    test_global_metrics(base, tokens)
    test_mlflow_status(base, tokens)
    test_docs(base)

    sys.exit(print_summary())


if __name__ == "__main__":
    main()
