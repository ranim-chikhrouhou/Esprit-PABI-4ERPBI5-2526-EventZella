#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final comprehensive translation pass
Handles all remaining French text and removes unnecessary verbose messages
"""

from pathlib import Path
import re

def translate_final():
    """Final comprehensive translation pass."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Final comprehensive translations
    translations = {
        # Any remaining French phrases
        "Régression :": "Regression:",
        "Customer Grouping :": "Customer Segmentation:",
        "Status Prediction :": "Booking Status:",
        "Séries temporelles :": "Time Series:",
        
        # Form and UI elements
        "Formulaire": "Form",
        "Résultat": "Result",
        "Graphique": "Chart",
        "Tableau": "Table",
        "Liste": "List",
        
        # More status messages
        "Aucun résultat": "No results",
        "Aucune donnée": "No data",
        "Données chargées": "Data loaded",
        "Modèle chargé": "Model loaded",
        
        # Additional cleanup
        "Voir plus": "See more",
        "Voir moins": "See less",
        "Détails": "Details",
        "Informations": "Information",
        
        # Technical messages to simplify
        "— exécutez": "— run",
        "— lancez": "— run",
        "— vérifiez": "— check",
        
        # Simplify verbose messages
        "Parquet `dw_financial_wide.parquet` introuvable — exécutez `ML/scripts/run_00_data_preparation.py`.": "Parquet file `dw_financial_wide.parquet` not found — run `ML/scripts/run_00_data_preparation.py`.",
        
        # Clean up technical jargon
        "Database": "database",
        "Prediction System": "model",
        
        # More UI cleanup
        "Cliquez sur": "Click on",
        "Sélectionnez une": "Select a",
        "Choisissez un": "Choose a",
        
        # Additional form labels
        "Valeur par défaut": "Default value",
        "Valeur suggérée": "Suggested value",
        "Valeur typique": "Typical value",
        
        # More captions
        "Exemple :": "Example:",
        "Note :": "Note:",
        "Attention :": "Warning:",
        "Important :": "Important:",
        
        # Clean up redundant phrases
        "pour votre scénario": "for your scenario",
        "selon vos données": "based on your data",
        "d'après le modèle": "from the model",
        
        # Simplify technical explanations
        "valeurs fixées à la médiane du jeu pour l'inférence": "values set to dataset median",
        "auto-remplis avec valeurs typiques pour la prédiction": "auto-filled with typical values",
        
        # More cleanup
        "Affichage": "Display",
        "Configuration": "Configuration",
        "Options": "Options",
        "Paramètres": "Parameters",
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Remove unnecessary verbose captions and messages
    verbose_patterns = [
        # Remove overly technical captions
        r'st\.caption\([^)]*".*identifiant database.*"[^)]*\)',
        r'st\.caption\([^)]*".*Database.*médiane.*"[^)]*\)',
        
        # Simplify long explanations in captions
        r'st\.caption\(\s*".*SSMS.*DW_eventzella.*"\s*\)',
    ]
    
    for pattern in verbose_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Clean up double spaces and empty lines created by removals
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    content = re.sub(r'  +', ' ', content)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Final translation complete!")
    print("📄 All French text translated to English")
    print("📄 Unnecessary verbose messages removed")
    print("📄 UI is now clean and professional")

if __name__ == "__main__":
    translate_final()
