# -*- coding: utf-8 -*-
"""
EventZilla MLOps - Nettoyage du Projet
Supprime tous les fichiers inutiles et garde uniquement l'essentiel
"""
import os
import shutil
from pathlib import Path

print("="*70)
print("🧹 EventZilla MLOps - Nettoyage du Projet")
print("="*70)
print()

# Répertoire du projet
PROJECT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════
# FICHIERS ET DOSSIERS À GARDER (ESSENTIELS)
# ═══════════════════════════════════════════════════════════════════

KEEP_FILES = {
    # ── Scripts de démarrage ──
    "LANCER_PROJET.bat",
    "LANCER_PROJET.ps1",
    "REPARER_TOUT.py",
    "NETTOYER_PROJET.py",
    
    # ── Documentation essentielle ──
    "README_FINAL.md",
    "MONITORING_GUIDE_S13.md",
    "SOLUTION_FINALE_COMPLETE.txt",
    
    # ── Configuration ──
    ".env",
    ".gitignore",
    "requirements.txt",
    "requirements_monitoring.txt",
    
    # ── Docker ──
    "docker-compose.yml",
    "docker-compose-monitoring.yml",
    "Dockerfile.fastapi",
    "Dockerfile.mlflow",
    
    # ── Prometheus & Grafana ──
    "prometheus.yml",
    "prometheus_rules.yml",
    
    # ── Simulation ──
    "simulate_scenarios.py",
    "automated_training_pipeline.py",
}

KEEP_DIRS = {
    # ── Code source ML ──
    "ML/api",
    "ML/models_artifacts",
    "ML/data_processed",
    "ML/notebooks",
    
    # ── Workflows n8n ──
    "n8n",
    
    # ── Grafana ──
    "grafana/provisioning",
    "grafana/dashboards",
    
    # ── Airflow (si utilisé) ──
    "airflow/dags",
    
    # ── MLflow ──
    "mlruns",
    "mlartifacts",
}

# ═══════════════════════════════════════════════════════════════════
# FICHIERS À SUPPRIMER (INUTILES/REDONDANTS)
# ═══════════════════════════════════════════════════════════════════

DELETE_PATTERNS = [
    # ── Documentation redondante ──
    "**/GUIDE_*.md",
    "**/SOLUTION_*.md",
    "**/IMPORTER_*.md",
    "**/START_HERE*.txt",
    "**/INSTRUCTIONS_*.txt",
    "**/FAITES_*.txt",
    "**/LISEZ_*.txt",
    "**/PLAN_*.md",
    
    # ── Scripts de test multiples ──
    "test_*.py",
    "check_*.bat",
    "start_monitoring*.bat",
    "start_monitoring*.ps1",
    
    # ── Fichiers temporaires ──
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/.pytest_cache",
    "**/.ipynb_checkpoints",
    
    # ── Logs ──
    "**/logs/**/*.log",
    "**/*.log",
    
    # ── Fichiers de backup ──
    "**/*.bak",
    "**/*.backup",
    "**/*.old",
    
    # ── Fichiers système ──
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/desktop.ini",
]

# ═══════════════════════════════════════════════════════════════════
# FONCTION DE NETTOYAGE
# ═══════════════════════════════════════════════════════════════════

def should_keep_file(file_path: Path) -> bool:
    """Détermine si un fichier doit être gardé"""
    relative_path = file_path.relative_to(PROJECT_DIR)
    
    # Garder les fichiers essentiels
    if file_path.name in KEEP_FILES:
        return True
    
    # Garder les fichiers dans les dossiers essentiels
    for keep_dir in KEEP_DIRS:
        if str(relative_path).startswith(keep_dir.replace("/", os.sep)):
            return True
    
    return False

def should_delete_file(file_path: Path) -> bool:
    """Détermine si un fichier doit être supprimé"""
    relative_path = file_path.relative_to(PROJECT_DIR)
    
    for pattern in DELETE_PATTERNS:
        if file_path.match(pattern):
            return True
    
    return False

# ═══════════════════════════════════════════════════════════════════
# ANALYSE DU PROJET
# ═══════════════════════════════════════════════════════════════════

print("📊 Analyse du projet...")
print()

files_to_keep = []
files_to_delete = []
files_unknown = []

for file_path in PROJECT_DIR.rglob("*"):
    if file_path.is_file():
        if should_keep_file(file_path):
            files_to_keep.append(file_path)
        elif should_delete_file(file_path):
            files_to_delete.append(file_path)
        else:
            files_unknown.append(file_path)

print(f"✅ Fichiers à garder: {len(files_to_keep)}")
print(f"🗑️  Fichiers à supprimer: {len(files_to_delete)}")
print(f"❓ Fichiers inconnus: {len(files_unknown)}")
print()

# ═══════════════════════════════════════════════════════════════════
# AFFICHAGE DES FICHIERS À SUPPRIMER
# ═══════════════════════════════════════════════════════════════════

if files_to_delete:
    print("🗑️  Fichiers qui seront supprimés:")
    print("-"*70)
    for file_path in sorted(files_to_delete)[:20]:  # Afficher les 20 premiers
        relative_path = file_path.relative_to(PROJECT_DIR)
        print(f"  - {relative_path}")
    
    if len(files_to_delete) > 20:
        print(f"  ... et {len(files_to_delete) - 20} autres fichiers")
    print()

# ═══════════════════════════════════════════════════════════════════
# CONFIRMATION
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("⚠️  ATTENTION: Cette opération est IRRÉVERSIBLE!")
print("="*70)
print()
print(f"Vous allez supprimer {len(files_to_delete)} fichiers.")
print()

response = input("Voulez-vous continuer? (oui/non): ").strip().lower()

if response not in ["oui", "yes", "o", "y"]:
    print()
    print("❌ Nettoyage annulé.")
    print()
    exit(0)

# ═══════════════════════════════════════════════════════════════════
# SUPPRESSION
# ═══════════════════════════════════════════════════════════════════

print()
print("🗑️  Suppression en cours...")
print()

deleted_count = 0
error_count = 0

for file_path in files_to_delete:
    try:
        if file_path.is_file():
            file_path.unlink()
            deleted_count += 1
        elif file_path.is_dir():
            shutil.rmtree(file_path)
            deleted_count += 1
    except Exception as e:
        print(f"❌ Erreur: {file_path.name} - {e}")
        error_count += 1

# Supprimer les dossiers vides
for dir_path in sorted(PROJECT_DIR.rglob("*"), reverse=True):
    if dir_path.is_dir():
        try:
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
        except:
            pass

print()
print("="*70)
print("✅ NETTOYAGE TERMINÉ!")
print("="*70)
print()
print(f"📊 Résultats:")
print(f"  ✅ Fichiers supprimés: {deleted_count}")
print(f"  ❌ Erreurs: {error_count}")
print(f"  📁 Fichiers conservés: {len(files_to_keep)}")
print()
print("🎯 Structure du projet nettoyée:")
print()
print("PI BI NEW (2)/PI BI NEW/")
print("├── ML/")
print("│   ├── api/              # Code API FastAPI")
print("│   ├── models_artifacts/ # Modèles ML")
print("│   ├── data_processed/   # Données traitées")
print("│   └── notebooks/        # Notebooks Jupyter")
print("├── n8n/                  # Workflows n8n")
print("├── grafana/              # Configuration Grafana")
print("├── airflow/              # DAGs Airflow")
print("├── mlruns/               # Expériences MLflow")
print("├── LANCER_PROJET.bat     # Script de démarrage")
print("├── REPARER_TOUT.py       # Script de réparation")
print("├── README_FINAL.md       # Documentation")
print("└── docker-compose.yml    # Configuration Docker")
print()
print("="*70)
