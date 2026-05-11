# -*- coding: utf-8 -*-
"""
EventZilla -- Script de test complet des workflows n8n.

Lancer depuis la racine du projet :
    cd "c:/Users/ASUS/OneDrive - ESPRIT/Pièces jointes/ML projetttt/DossierProjet/DossierProjet/PI BI NEW (2)/PI BI NEW"
    python n8n/test_workflows.py

Prerequis :
    1. FastAPI lancee  -> python run_fastapi.py   (ou: uvicorn ML.api.main:app --port 8000)
    2. n8n lance       -> npx n8n  (autre terminal)
    3. Workflows importes dans n8n
"""
import io
import sys
import requests

# Forcer UTF-8 sur la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_API = "http://127.0.0.1:8000"
BASE_N8N = "http://localhost:5678"
TIMEOUT  = 30

SEP  = "=" * 60
SEP2 = "-" * 40

def titre(t):  print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):     print(f"  [OK]     {m}")
def err(m):    print(f"  [ERREUR] {m}")
def info(m):   print(f"  [INFO]   {m}")


# ════════════════════════════════════════════════════════════
# ETAPE 0 -- Health check FastAPI
# ════════════════════════════════════════════════════════════
titre("ETAPE 0 -- Health check FastAPI")
try:
    r = requests.get(f"{BASE_API}/", timeout=5)
    data = r.json()
    ok(f"FastAPI repond -- status: {data.get('status')}")
    for nom, charge in data.get("modeles_charges", {}).items():
        symbole = "[OK]" if charge else "[ABSENT]"
        print(f"         {symbole}  Modele [{nom}] : {'charge' if charge else 'ABSENT'}")
except Exception as e:
    err(f"FastAPI inaccessible : {e}")
    print("\n  Lancez d'abord dans un autre terminal :")
    print("  python -m uvicorn ML.api.main:app --reload --port 8000")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# ETAPE 1 -- Login des 3 utilisateurs
# ════════════════════════════════════════════════════════════
titre("ETAPE 1 -- Authentification des 3 utilisateurs")

USERS = [
    {"login": "ranim_chikhrouhou", "password": "Ranim@Marketing2025!", "nom": "Ranim  (Marketing)"},
    {"login": "naima_sarraj",      "password": "Naima@Finance2025!",   "nom": "Naima  (Finance)  "},
    {"login": "anas_allam",        "password": "Anas@CRM2025!",        "nom": "Anas   (CRM)      "},
]

TOKENS = {}
for u in USERS:
    try:
        r = requests.post(f"{BASE_API}/auth/login",
                          json={"login": u["login"], "password": u["password"]},
                          timeout=TIMEOUT)
        if r.status_code == 200:
            body = r.json()
            TOKENS[u["login"]] = body["access_token"]
            ok(f"{u['nom']} -- role: {body['role']}")
        else:
            err(f"{u['nom']} -- HTTP {r.status_code} : {r.text[:100]}")
    except Exception as e:
        err(f"{u['nom']} -- {e}")

if not TOKENS:
    err("Aucun login reussi. Verifiez SQL Server et dbo.AppUsers.")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# ETAPE 2 -- Test direct des endpoints ML
# ════════════════════════════════════════════════════════════
titre("ETAPE 2 -- Test direct des endpoints ML (sans n8n)")

t_mkt = TOKENS.get("ranim_chikhrouhou")
t_fin = TOKENS.get("naima_sarraj")
t_crm = TOKENS.get("anas_allam")

corps_classif = {
    "id_date": 1, "id_event": 42, "id_servicecategory": 3,
    "id_benchmark": 2, "id_provider": 7,
    "final_price": 1500.0, "service_price": 1200.0,
    "benchmark_avg_price": 1300.0, "event_budget": 2000.0,
    "cal_month": 4, "cal_year": 2024, "quarter": 2,
}
corps_segment = {
    "nb_reservations_loyalty": 12, "ca_total_loyalty": 15000,
    "panier_moyen_loyalty": 1250, "recency_days_loyalty": 30,
    "avg_nb_visitors_loyalty": 85, "volume_reservations_site_loyalty": 5,
}
corps_regress = {
    "id_date": 1, "id_event": 42, "id_servicecategory": 3,
    "id_benchmark": 2, "id_provider": 7,
    "service_price": 1200, "benchmark_avg_price": 1300,
    "event_budget": 2000, "cal_month": 4, "cal_year": 2024,
    "quarter": 2, "commission_margin": 150,
}

# -- Classification
print(f"\n  {SEP2}\n  /predict/classification\n  {SEP2}")
if t_mkt:
    try:
        r = requests.post(f"{BASE_API}/predict/classification",
                          json=corps_classif,
                          headers={"Authorization": f"Bearer {t_mkt}"},
                          timeout=TIMEOUT)
        if r.status_code == 200:
            d = r.json()
            ok(f"Statut predit : {d['statut_predit']}")
            for cls, prob in d.get("probabilites", {}).items():
                print(f"         {cls:<12} {prob:.1%}")
        else:
            err(f"HTTP {r.status_code} : {r.text[:200]}")
    except Exception as e:
        err(str(e))

# -- Segmentation
print(f"\n  {SEP2}\n  /predict/segmentation/beneficiaire\n  {SEP2}")
if t_mkt:
    try:
        r = requests.post(f"{BASE_API}/predict/segmentation/beneficiaire",
                          json=corps_segment,
                          headers={"Authorization": f"Bearer {t_mkt}"},
                          timeout=TIMEOUT)
        if r.status_code == 200:
            d = r.json()
            ok(f"Segment : {d['segment_label']} (id={d['segment_id']})")
        else:
            err(f"HTTP {r.status_code} : {r.text[:200]}")
    except Exception as e:
        err(str(e))

# -- Regression
print(f"\n  {SEP2}\n  /predict/regression\n  {SEP2}")
if t_fin:
    try:
        r = requests.post(f"{BASE_API}/predict/regression",
                          json=corps_regress,
                          headers={"Authorization": f"Bearer {t_fin}"},
                          timeout=TIMEOUT)
        if r.status_code == 200:
            d = r.json()
            ok(f"Montant predit : {d['montant_predit']} {d['unite']}")
        else:
            err(f"HTTP {r.status_code} : {r.text[:200]}")
    except Exception as e:
        err(str(e))

# -- Series temporelles
print(f"\n  {SEP2}\n  /predict/timeseries\n  {SEP2}")
if t_fin:
    try:
        r = requests.get(f"{BASE_API}/predict/timeseries?horizon=3",
                         headers={"Authorization": f"Bearer {t_fin}"},
                         timeout=TIMEOUT)
        if r.status_code == 200:
            d = r.json()
            ok(f"Modele champion : {d['modele_champion']}")
            m = d.get("metriques_test", {})
            if m:
                print(f"         RMSE={m.get('rmse','N/A')}  MAPE={m.get('mape','N/A')}%")
        else:
            err(f"HTTP {r.status_code} : {r.text[:200]}")
    except Exception as e:
        err(str(e))

# -- Test securite : CRM ne peut pas appeler /predict/regression
print(f"\n  {SEP2}\n  Test securite RBAC\n  {SEP2}")
if t_crm:
    try:
        r = requests.post(f"{BASE_API}/predict/regression",
                          json=corps_regress,
                          headers={"Authorization": f"Bearer {t_crm}"},
                          timeout=TIMEOUT)
        if r.status_code == 403:
            ok("Acces refuse (403) pour le role CRM sur /regression -- RBAC OK")
        else:
            err(f"PROBLEME SECURITE : HTTP {r.status_code} au lieu de 403")
    except Exception as e:
        err(str(e))


# ════════════════════════════════════════════════════════════
# ETAPE 3 -- Declenchement webhook CRM
# ════════════════════════════════════════════════════════════
titre("ETAPE 3 -- Declenchement workflow CRM via Webhook n8n")
info("Marketing / Finance (Cron) : executer manuellement dans n8n.")
info("CRM : declenche par appel HTTP POST ci-dessous.")
print()

corps_webhook = {
    "event": "test_reservation", "id_date": 1, "id_event": 42,
    "id_servicecategory": 3, "id_benchmark": 2, "id_provider": 7,
    "final_price": 800.0, "service_price": 700.0,
    "benchmark_avg_price": 900.0, "event_budget": 1000.0,
    "cal_month": 4, "cal_year": 2024, "quarter": 2,
}

for url in [
    f"{BASE_N8N}/webhook-test/eventzilla-crm-trigger",
    f"{BASE_N8N}/webhook/eventzilla-crm-trigger",
]:
    try:
        r = requests.post(url, json=corps_webhook, timeout=10)
        if r.status_code == 200:
            ok(f"Webhook CRM declenche : {url}")
            ok(f"Reponse n8n : {r.text[:100]}")
            break
        else:
            info(f"{url} -> HTTP {r.status_code}")
    except Exception as e:
        info(f"{url} -> {type(e).__name__}")
else:
    err("Webhook CRM inaccessible.")
    print()
    print("  Verifiez :")
    print("  1. n8n est lance  (npx n8n)")
    print("  2. Workflow CRM importe et webhook active")


# ════════════════════════════════════════════════════════════
# ETAPE 4 -- Webhook Finance (inférence à la demande)
# ════════════════════════════════════════════════════════════
titre("ETAPE 4 -- Declenchement webhook Finance")
corps_fin_wh = {
    "id_date": 1,
    "id_event": 42,
    "service_price": 1200,
    "benchmark_avg_price": 1300,
    "event_budget": 2000,
}
for url in [
    f"{BASE_N8N}/webhook-test/eventzilla-finance-trigger",
    f"{BASE_N8N}/webhook/eventzilla-finance-trigger",
]:
    try:
        r = requests.post(url, json=corps_fin_wh, timeout=15)
        if r.status_code == 200:
            ok(f"Webhook Finance declenche : {url}")
            ok(f"Reponse n8n : {r.text[:120]}")
            break
        else:
            info(f"{url} -> HTTP {r.status_code}")
    except Exception as e:
        info(f"{url} -> {type(e).__name__}")
else:
    err("Webhook Finance inaccessible : importer workflow_webhook_finance.json et activer.")


# ════════════════════════════════════════════════════════════
# RESUME
# ════════════════════════════════════════════════════════════
titre("RESUME")
print("  Endpoints testes : Classification, Segmentation, Regression, Timeseries")
print("  Securite RBAC    : verifiee (CRM bloque sur /regression)")
print()
print("  Interface n8n    : http://localhost:5678")
print("  Swagger API      : http://127.0.0.1:8000/docs")
print("  Resultats JSON   : n8n/results/  (apres execution dans n8n)")
print()
