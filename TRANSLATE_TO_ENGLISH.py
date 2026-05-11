#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive translation script for Streamlit UI
Translates French to English and removes unnecessary messages
"""

from pathlib import Path
import re

def translate_streamlit_app():
    """Translate the Streamlit app from French to English."""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Create backup
    backup_path = file_path.with_suffix(".py.french_backup")
    if not backup_path.exists():
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"📦 Backup created: {backup_path}")
    
    # Read content
    content = file_path.read_text(encoding="utf-8")
    
    # Comprehensive translation dictionary
    translations = {
        # Docstrings
        "Accueil : dashboard KPI, modèles déployés, navigation": "Home: KPI dashboard, deployed models, navigation",
        "Dernière page : tableau synthétique lisible avec cartes par famille": "Summary page: comprehensive overview with model cards",
        
        # Hero sections
        "Plateforme d'**intelligence artificielle** appliquée au **business** EventZilla.": "**AI-powered** platform for EventZilla **business intelligence**.",
        
        # Section headers
        "Business Analytics": "Business Analytics",
        "Indicateurs clés calculés depuis le Database": "Key metrics from the data warehouse",
        "Modèles ML déployés": "Deployed ML Models",
        "Quatre familles de modèles entraînés sur le Database": "Four model families trained on the data warehouse",
        "Explorer": "Explore",
        "Accéder aux écrans de test interactif": "Access interactive prediction tools",
        "Accès rapide": "Quick Access",
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
        "Prévision": "Forecast",
        " mois": " months",
        
        # Navigation
        "Voir le récapitulatif": "View Summary",
        "En savoir plus — intérêt du ML pour EventZilla": "Learn more — ML benefits for EventZilla",
        
        # Form instructions
        "Comment utiliser ce formulaire": "How to use this form",
        "Chaque champ représente une **variable numérique** du jeu d'entraînement.": "Each field represents a **numeric variable** from the training data.",
        "Choisissez une valeur parmi les **suggestions** (valeurs typiques du Database), puis lancez la prédiction.": "Choose a value from the **suggestions** (typical values from database), then run the prediction.",
        
        # Results
        "Résultat & Visualisations": "Result & Visualizations",
        "Résultat de la prédiction": "Prediction Result",
        "Statut prédit": "Predicted Status",
        "Probabilités par classe": "Class Probabilities",
        "Montant prédit": "Predicted Amount",
        "Segment identifié": "Identified Segment",
        
        # Metrics
        "Métriques du modèle": "Model Metrics",
        "Précision": "Accuracy",
        "Rappel": "Recall",
        "Score F1": "F1-Score",
        "Erreur moyenne": "Mean Error",
        "Erreur quadratique": "RMSE",
        
        # Messages
        "Aucune métrique disponible": "No metrics available",
        "Modèles absents": "Models not found",
        "Fichier introuvable": "File not found",
        "Exécutez les scripts": "Run the scripts",
        "aucune métrique disponible": "no metrics available",
        
        # Login
        "Connexion": "Login",
        "Se connecter": "Sign In",
        "Identifiants créés dans SSMS — DW_eventzella": "Credentials from SSMS — DW_eventzella",
        
        # Sidebar
        "Déconnexion": "Logout",
        "Connecté en tant que": "Logged in as",
        
        # Expanders
        "Rappel pédagogique — détails (optionnel)": "Educational reminder — details (optional)",
        "Export texte détaillé": "Detailed text export",
        "Aperçu tronqué — fichier complet dans le dossier ML/.": "Truncated preview — full file in ML/ folder.",
        
        # Field groups
        "Identifiants": "Identifiers",
        "Montants": "Amounts",
        "Temporel": "Temporal",
        "Contexte": "Context",
        
        # Warnings
        "⛔ La page": "⛔ The page",
        "n'est pas accessible avec votre rôle": "is not accessible with your role",
        "Parquet": "Parquet file",
        "introuvable — exécutez": "not found — run",
        
        # Captions
        "Champs saisis": "Fields entered",
        "Champs masqués": "Hidden fields",
        "identifiants système — auto-remplis avec valeurs typiques pour la prédiction.": "system identifiers — auto-filled with typical values for prediction.",
        "colonne(s) identifiant Database non affichées — valeurs fixées à la médiane du jeu pour l'inférence.": "database identifier column(s) not displayed — values set to dataset median for inference.",
        
        # Time series
        "Historique": "Historical",
        "Validation": "Validation",
        "train / validation / prévision": "train / validation / forecast",
        "Fin historique": "End of history",
        "Début validation": "Start of validation",
        "Valeur": "Value",
        "Mois": "Month",
        
        # Clustering
        "Bénéficiaires": "Beneficiaries",
        "Prestataires": "Providers",
        "Type d'entité": "Entity Type",
        "Métriques RFM": "RFM Metrics",
        
        # Summary
        "Vue d'ensemble": "Overview",
        "Critère": "Criterion",
        "Domaine": "Domain",
        "Best System": "Best Model",
        "Qualité": "Quality",
        "Règle de choix": "Selection Rule",
        
        # Misc
        "Ce que teste cet écran": "What this screen tests",
        "Illustration": "Illustration",
        "exemple de probabilités égales": "equal probability example",
        "Après prédiction, les barres et la jauge montrent les **probabilités réelles** du Prediction System pour votre scénario.": "After prediction, bars and gauge show the **actual probabilities** from the model for your scenario.",
        
        # Common words
        "lancez": "run",
        "exécutez": "execute",
        "cliquez": "click",
        "sélectionnez": "select",
        "choisissez": "choose",
        "fichier": "file",
        "dossier": "folder",
        "données": "data",
        "modèle": "model",
        "entraîné": "trained",
        "prédit": "predicted",
        "prédiction": "prediction",
        "disponible": "available",
        "absent": "missing",
        "trouvé": "found",
        
        # Sidebar caption (remove)
        "EventZilla ML Studio — données du Database, modèles dans ML/models_artifacts/.": "",
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Additional regex-based replacements
    # Remove unnecessary technical messages
    content = re.sub(r'st\.caption\(\s*"Identifiants créés dans SSMS[^"]*"\s*\)', '', content)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Translation complete!")
    print(f"📄 Translated file: {file_path}")
    print(f"📄 Backup: {backup_path}")

if __name__ == "__main__":
    translate_streamlit_app()
