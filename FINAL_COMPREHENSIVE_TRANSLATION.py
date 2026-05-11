#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final comprehensive translation - find and replace ALL French text
"""

from pathlib import Path

def final_comprehensive_translation():
    """Translate absolutely all remaining French text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Comprehensive list of ALL French text found
    translations = {
        # Headers and titles
        '"Ce que teste cet écran"': '"What this screen tests"',
        '"Metrics clés calculés depuis le database"': '"Key metrics from the database"',
        
        # Metrics labels
        '"Exactitude (réf.)"': '"Accuracy (ref.)"',
        '"Balance Score pondéré (réf.)"': '"Weighted F1-Score (ref.)"',
        '"Quality Score (réf.)"': '"Quality Score (ref.)"',
        '"Statuts possibles (Y) :"': '"Possible statuses (Y):"',
        
        # Section titles with Unicode
        '"Régression (D)"': '"Regression (D)"',
        '"Séries temp. (F)"': '"Time Series (F)"',
        
        # Info messages
        '"Customer Grouping : aucune métrique disponible."': '"Customer Segmentation: no metrics available."',
        '"Régression : aucune métrique disponible."': '"Regression: no metrics available."',
        '"Status Prediction : aucune métrique disponible."': '"Booking Status: no metrics available."',
        '"Séries temporelles : aucune métrique disponible."': '"Time Series: no metrics available."',
        
        # Warning messages
        '"Aucune métrique trouvée dans ML/models_artifacts/ — exécutez les scripts run_01 … run_04."': '"No metrics found in ML/models_artifacts/ — run scripts run_01 … run_04."',
        
        # Table columns
        '"Critère"': '"Criterion"',
        '"Domaine"': '"Domain"',
        '"Qualité"': '"Quality"',
        '"Règle de choix"': '"Selection Rule"',
        
        # Expander titles
        '"Export texte détaillé (ML_METRICS_SUMMARY.md)"': '"Detailed text export (ML_METRICS_SUMMARY.md)"',
        '"Aperçu tronqué — fichier complet dans le dossier ML/."': '"Truncated preview — full file in ML/ folder."',
        
        # Navigation
        '"Accès rapide"': '"Quick Access"',
        '"Accéder aux pages de test"': '"Access test pages"',
        
        # Login caption
        '"Identifiants créés dans SSMS — DW_eventzella"': '"Credentials from SSMS — DW_eventzella"',
        
        # Comments (not visible but good to translate)
        '# --- Navigation rapide ---': '# --- Quick navigation ---',
        '# --- Cartes détaillées par famille ---': '# --- Detailed cards by family ---',
        '# --- Tableau synthétique compact ---': '# --- Compact summary table ---',
        '# --- Export optional ---': '# --- Optional export ---',
        
        # Additional French words
        'clés': 'key',
        'calculés': 'calculated',
        'depuis': 'from',
        'aucune': 'no',
        'métrique': 'metric',
        'métriques': 'metrics',
        'disponible': 'available',
        'trouvée': 'found',
        'exécutez': 'run',
        'détaillé': 'detailed',
        'tronqué': 'truncated',
        'fichier complet': 'full file',
        'dossier': 'folder',
        'rapide': 'quick',
        'pages de test': 'test pages',
        'Identifiants': 'Credentials',
        'créés': 'from',
        'Statuts possibles': 'Possible statuses',
        'Exactitude': 'Accuracy',
        'pondéré': 'weighted',
        'réf.': 'ref.',
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    print("✅ Final comprehensive translation complete!")
    print("📄 All remaining French text translated")
    print("📊 Translated: headers, metrics, messages, table columns, expanders")

if __name__ == "__main__":
    final_comprehensive_translation()
