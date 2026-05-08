"""
EventZilla -- Script de sauvegarde des resultats n8n.
Appele par le noeud Execute Command dans n8n.

Usage:
    python save_result.py <workflow_name> <json_data>
"""
import sys
import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if len(sys.argv) < 3:
    print("Usage: python save_result.py <workflow> <json_data>")
    sys.exit(1)

workflow = sys.argv[1]
json_data = sys.argv[2]

try:
    data = json.loads(json_data)
except Exception as e:
    print(f"Erreur parsing JSON: {e}")
    sys.exit(1)

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%Y-%m-%d_%H-%M")

name_map = {
    "marketing": f"marketing_predictions_{date_str}.json",
    "finance":   f"finance_predictions_{date_str}.json",
    "crm":       f"crm_predictions_{time_str}.json",
}

filename = name_map.get(workflow.lower(), f"{workflow}_{time_str}.json")
filepath = RESULTS_DIR / filename

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"OK: {filepath}")
