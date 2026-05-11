#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix French text with Unicode escapes
"""

from pathlib import Path
import codecs

def fix_unicode_french():
    """Fix French text that appears as Unicode escapes."""
    
    file_path = Path("ML/streamlit_app.py")
    
    # Read with proper encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Decode any Unicode escapes
    try:
        # Try to decode Unicode escapes if they exist
        content = codecs.decode(content, 'unicode_escape')
    except:
        pass  # If it fails, content is already decoded
    
    # Now translate
    translations = {
        # Section headers
        'Modèles ML déployés': 'Deployed ML Models',
        'Quatre familles de modèles entraînés sur le database': 'Four model families trained on the database',
        
        # Model cards
        'Régression': 'Regression',
        'segments clients': 'customer segments',
        'Séries temporelles': 'Time Series',
        'Prévision': 'Forecast',
        ' mois': '-month forecast',
        
        # Expander
        'En savoir plus — intérêt du ML pour EventZilla': 'Learn more — ML benefits for EventZilla',
        
        # Summary page
        'Récapitulatif des modèles': 'Models Summary',
        "Vue d'ensemble des **quatre familles ML** déployées : performance, modèle Best System et indicateur métier.": "Overview of **four deployed ML families**: performance, best model, and business indicator.",
        'Synthèse': 'Summary',
        
        # Additional
        'entraînés': 'trained',
        'déployés': 'deployed',
        'métier': 'business',
        'modèles': 'models',
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Unicode French text fixed!")
    print("📄 All section headers and model cards now in English")

if __name__ == "__main__":
    fix_unicode_french()
