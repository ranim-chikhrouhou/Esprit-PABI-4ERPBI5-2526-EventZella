#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix all remaining French text found in the UI
"""

import re
from pathlib import Path

def fix_remaining_french():
    """Fix all remaining French text"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # All remaining French to English translations
    translations = {
        # Form header - Price Estimation
        "Formulaire — prédire le prix final": "Form — predict final price",
        "Formulaire \\u2014 pr\\u00e9dire le prix final": "Form — predict final price",
        
        # Subtitle text
        "Priority : variables outside Booking Status screen ; puis complété jusqu'à": "Priority: variables outside Booking Status screen; then completed up to",
        "puis complété jusqu'à": "then completed up to",
        "y compris des variables already presentes en Status Prediction si besoin": "including variables already present in Booking Status if needed",
        "y compris des variables already presentes": "including variables already present",
        "en Status Prediction si besoin": "in Booking Status if needed",
        "Reference : dropdown si présent": "Reference: dropdown if present",
        "dropdown si présent": "dropdown if present",
        "Autres X → médiane database": "Other X → database median",
        "médiane database": "database median",
        
        # Model description
        "Random forest + mise à l'échelle": "Random forest + scaling",
        "mise à l'échelle": "scaling",
        "Prediction System entraîné pour minimiser l'erreur on test set": "Prediction System trained to minimize error on test set",
        "entraîné pour minimiser l'erreur": "trained to minimize error",
        
        # Figure note
        "Histogramme de `final_price`, importances des variables, valeur estimée vs médiane du database": "Histogram of `final_price`, variable importances, estimated value vs database median",
        "importances des variables": "variable importances",
        "valeur estimée vs médiane du database": "estimated value vs database median",
        "valeur estimée": "estimated value",
        "vs médiane du database": "vs database median",
        
        # Additional French phrases that might be present
        "En priorité": "Priority",
        "variables hors écran": "variables outside screen",
        "hors écran": "outside screen",
        "au moins six champs": "at least six fields",
        "au moins": "at least",
        "champs": "fields",
        "importance du Prediction System": "Prediction System importance",
        "importance du modèle": "model importance",
        "déjà présentes": "already present",
        "si besoin": "if needed",
        "liste déroulante": "dropdown list",
        "déroulante": "dropdown",
        "si présent": "if present",
        "présent": "present",
        "médiane DW": "DW median",
        
        # Context card labels
        "pourquoi": "rationale",
        "figure_note": "figure_note",
        
        # Any remaining "modèle" references
        "modèle": "model",
        "mod\\u00e8le": "model",
        
        # Any remaining "erreur" references  
        "erreur": "error",
        
        # Any remaining "entraîné" references
        "entraîné": "trained",
        "entra\\u00een\\u00e9": "trained",
    }
    
    # Apply all translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: All remaining French text fixed!")
    print(f"SUCCESS: File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = fix_remaining_french()
    if success:
        print("\nSUCCESS: Translation complete!")
        print("SUCCESS: Please restart Streamlit and refresh browser with Ctrl+F5")
    else:
        print("\nERROR: Fix failed")
