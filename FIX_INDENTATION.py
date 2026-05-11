#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix indentation issues caused by translation
"""

from pathlib import Path
import re

def fix_indentation():
    """Fix indentation issues in the streamlit app."""
    
    # Restore from backup and re-translate properly
    backup_file = Path("ML/streamlit_app.py.french_backup")
    target_file = Path("ML/streamlit_app.py")
    
    if not backup_file.exists():
        print("❌ Backup file not found!")
        return
    
    print("📦 Restoring from backup...")
    content = backup_file.read_text(encoding="utf-8")
    
    # Apply all translations without breaking indentation
    translations = {
        # Page docstrings
        "Accueil : dashboard KPI, modèles déployés, navigation": "Home: KPI dashboard, deployed models, navigation",
        "Dernière page : tableau synthétique lisible avec cartes par famille": "Summary page: comprehensive overview with model cards",
        
        # Hero sections
        "Plateforme d'**intelligence artificielle** appliquée au **business** EventZilla.": "**AI-powered** platform for EventZilla **business intelligence**.",
        
        # Section headers
        "Indicateurs clés calculés depuis le Database": "Key metrics from the data warehouse",
        "Modèles ML déployés": "Deployed ML Models",
        "Quatre familles de modèles entraînés sur le Database": "Four model families trained on the data warehouse",
        "Accéder aux écrans de test interactif": "Access interactive prediction tools",
        "Accéder aux pages de test": "Access prediction pages",
        
        # KPI labels
        "Total Réservations": "Total Bookings",
        "Valeur Commande Moy.": "Avg Order Value",
        "Taux Annulation": "Cancellation Rate",
        "Segments Clients": "Customer Segments",
        "Balance Score Status Prediction": "Classification F1-Score",
        "R² Régression": "Regression R²",
        "Quality Score Score": "Clustering Quality",
        "Prediction Error Séries Temp.": "Time Series RMSE",
        "Horizon Prévision": "Forecast Horizon",
        
        # Model cards
        "Risque d'annulation": "Cancellation risk prediction",
        "Estimation prix": "Price prediction",
        "segments clients": "customer segments",
        
        # Navigation
        "Voir le récapitulatif": "View Summary",
        "En savoir plus — intérêt du ML pour EventZilla": "Learn more — ML benefits for EventZilla",
        
        # Form instructions
        "Comment utiliser ce formulaire": "How to use this form",
        
        # Results
        "Résultat & Visualisations": "Result & Visualizations",
        "Statut prédit": "Predicted Status",
        "Probabilités par classe": "Class Probabilities",
        "Montant prédit": "Predicted Amount",
        
        # Messages
        "aucune métrique disponible": "no metrics available",
        "Modèles Status Prediction absents": "Booking Status models missing",
        "Fichier introuvable": "File not found",
        
        # Sidebar
        "Connecté en tant que": "Logged in as",
        
        # Common words
        " mois": " months",
        "Prévision": "Forecast",
    }
    
    # Apply translations carefully
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    target_file.write_text(content, encoding="utf-8")
    print("✅ Indentation fixed!")
    print("📄 File restored and translated properly")

if __name__ == "__main__":
    fix_indentation()
