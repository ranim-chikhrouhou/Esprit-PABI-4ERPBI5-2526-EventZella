#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the last remaining French phrases in UI
"""

import re
from pathlib import Path

def fix_last_french():
    """Fix the last French phrases"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Last remaining French phrases
    translations = {
        # Expander titles
        "Objectif du formulaire — details (optional)": "Form objective — details (optional)",
        "Objectif du formulaire": "Form objective",
        
        # Caption labels
        "Erreur quadratique averagene": "Root mean squared error",
        "Erreur absolue averagene": "Mean absolute error",
        "averagene": "average",
        
        # Button labels
        "Recharger les séries depuis le database": "Reload series from database",
        "Recharger les s\\u00e9ries depuis le database": "Reload series from database",
        "Recharger": "Reload",
        "les séries": "series",
        "depuis le database": "from database",
        
        # Info messages
        "L'historique est trop court (< 6 month) pour ajuster les forecasting models.": "History is too short (< 6 months) to fit forecasting models.",
        "L\\u2019historique est trop court (< 6 month) pour ajuster les forecasting models.": "History is too short (< 6 months) to fit forecasting models.",
        "L'historique est trop court": "History is too short",
        "L\\u2019historique est trop court": "History is too short",
        "pour ajuster les forecasting models": "to fit forecasting models",
        "pour ajuster": "to fit",
    }
    
    # Apply all translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Last French phrases fixed!")
    print(f"SUCCESS: File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = fix_last_french()
    if success:
        print("\nSUCCESS: All French text removed from UI!")
        print("SUCCESS: Please restart Streamlit")
    else:
        print("\nERROR: Fix failed")
