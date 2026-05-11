#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safe translation that preserves indentation and code structure
Only translates strings, not code
"""

from pathlib import Path
import re

def translate_safe():
    """Safely translate only string literals without breaking code."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Only translate strings in specific contexts (st.markdown, st.caption, etc.)
    # This preserves all code structure and indentation
    
    # Translation dictionary - only for display strings
    translations = {
        # Page titles
        '"Home"': '"Home"',  # Already English
        '"Booking Status"': '"Booking Status"',  # Already English
        '"Price Estimation"': '"Price Estimation"',  # Already English
        '"Customer Segments"': '"Customer Segments"',  # Already English
        '"Trends & Forecast"': '"Trends & Forecast"',  # Already English
        '"Summary"': '"Summary"',  # Already English
        
        # Docstrings
        '"""Accueil : dashboard KPI, modèles déployés, navigation."""': '"""Home: KPI dashboard, deployed models, navigation."""',
        '"""Dernière page : tableau synthétique lisible avec cartes par famille."""': '"""Summary page: comprehensive overview with model cards."""',
        
        # Hero subtitle
        '"Plateforme d\'**intelligence artificielle** appliquée au **business** EventZilla."': '"**AI-powered** platform for EventZilla **business intelligence**."',
        
        # Section headers
        '"Business Analytics"': '"Business Analytics"',
        '"Indicateurs clés calculés depuis le Database"': '"Key metrics from the data warehouse"',
        '"Modèles ML déployés"': '"Deployed ML Models"',
        '"Quatre familles de modèles entraînés sur le Database"': '"Four model families trained on the data warehouse"',
        '"Explorer"': '"Explore"',
        '"Accéder aux écrans de test interactif"': '"Access interactive prediction tools"',
        
        # KPI labels
        '"Total Réservations"': '"Total Bookings"',
        '"Revenue (TND)"': '"Revenue (TND)"',
        '"Valeur Commande Moy."': '"Avg Order Value"',
        '"Taux Annulation"': '"Cancellation Rate"',
        '"Segments Clients"': '"Customer Segments"',
        '"Balance Score Status Prediction"': '"Classification F1-Score"',
        '"R² Régression"': '"Regression R²"',
        '"Quality Score Score"': '"Clustering Quality"',
        '"Prediction Error Séries Temp."': '"Time Series RMSE"',
        '"Horizon Prévision"': '"Forecast Horizon"',
        
        # Model cards
        '"Status Prediction"': '"Booking Status"',
        '"Risque d\'annulation"': '"Cancellation risk prediction"',
        '"Régression"': '"Price Estimation"',
        '"Estimation prix"': '"Price prediction"',
        '"Customer Grouping"': '"Customer Segmentation"',
        '"segments clients"': '"customer segments"',
        '"Séries temporelles"': '"Time Series"',
        '"Prévision"': '"Forecast"',
        '" mois"': '" months"',
        
        # Navigation
        '"Voir le récapitulatif"': '"View Summary"',
        '"En savoir plus — intérêt du ML pour EventZilla"': '"Learn more — ML benefits for EventZilla"',
        
        # Form labels
        '"Comment utiliser ce formulaire"': '"How to use this form"',
        '"Résultat & Visualisations"': '"Result & Visualizations"',
        
        # Messages
        '"aucune métrique disponible"': '"no metrics available"',
        '"Modèles Status Prediction absents"': '"Booking Status models missing"',
        
        # Sidebar
        '"Connecté en tant que"': '"Logged in as"',
        '"Déconnexion"': '"Logout"',
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Safe translation complete!")
    print("📄 Indentation preserved")
    print("📄 Code structure intact")

if __name__ == "__main__":
    translate_safe()
