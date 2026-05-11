#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to translate Streamlit UI from French to English
and remove unnecessary messages.
"""

import re
from pathlib import Path

# Translation dictionary: French -> English
TRANSLATIONS = {
    # Page titles and headers
    "Accueil : dashboard KPI, modèles déployés, navigation": "Home: KPI dashboard, deployed models, navigation",
    "Dernière page : tableau synthétique lisible avec cartes par famille": "Summary page: comprehensive overview with model cards",
    "Plateforme d'**intelligence artificielle** appliquée au **business** EventZilla": "**AI-powered** platform for EventZilla **business intelligence**",
    
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
    "Revenue (TND)": "Revenue (TND)",
    "Valeur Commande Moy.": "Avg Order Value",
    "Taux Annulation": "Cancellation Rate",
    "Segments Clients": "Customer Segments",
    "Balance Score Status Prediction": "Classification F1-Score",
    "R² Régression": "Regression R²",
    "Quality Score Score": "Clustering Quality",
    "Prediction Error Séries Temp.": "Time Series RMSE",
    "Horizon Prévision": "Forecast Horizon",
    
    # Model cards
    "Status Prediction": "Booking Status",
    "Risque d'annulation": "Cancellation risk",
    "Régression": "Price Estimation",
    "Estimation prix": "Price prediction",
    "Customer Grouping": "Customer Segmentation",
    "segments clients": "customer segments",
    "Séries temporelles": "Time Series",
    "Prévision": "Forecast",
    "mois": "months",
    
    # Navigation
    "Voir le récapitulatif": "View Summary",
    "En savoir plus — intérêt du ML pour EventZilla": "Learn more — ML benefits for EventZilla",
    
    # Form labels
    "Comment utiliser ce formulaire": "How to use this form",
    "Chaque champ représente une **variable numérique** du jeu d'entraînement": "Each field represents a **numeric variable** from the training data",
    "Choisissez une valeur parmi les **suggestions** (valeurs typiques du Database)": "Choose a value from the **suggestions** (typical values from database)",
    "puis lancez la prédiction": "then run the prediction",
    
    # Buttons
    "Prédire le statut": "Predict Status",
    "Estimer le prix": "Estimate Price",
    "Segmenter": "Segment",
    "Prévoir": "Forecast",
    "Predict Booking Status": "Predict Booking Status",
    "Estimate Final Price": "Estimate Final Price",
    
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
    
    # Login
    "Connexion": "Login",
    "Email": "Email",
    "Mot de passe": "Password",
    "Se connecter": "Sign In",
    "Identifiants créés dans SSMS — DW_eventzella": "Credentials created in SSMS — DW_eventzella",
    
    # Sidebar
    "Déconnexion": "Logout",
    "Connecté en tant que": "Logged in as",
    
    # Expanders
    "Rappel pédagogique — détails (optionnel)": "Educational reminder — details (optional)",
    "Export texte détaillé": "Detailed text export",
    "Aperçu tronqué — fichier complet dans le dossier ML/": "Truncated preview — full file in ML/ folder",
    
    # Field groups
    "Identifiants": "Identifiers",
    "Montants": "Amounts",
    "Temporel": "Temporal",
    "Contexte": "Context",
    
    # Warnings and errors
    "⛔ La page": "⛔ The page",
    "n'est pas accessible avec votre rôle": "is not accessible with your role",
    "Parquet": "Parquet file",
    "introuvable": "not found",
    "exécutez": "run",
    
    # Captions
    "Champs saisis": "Fields entered",
    "Champs masqués": "Hidden fields",
    "identifiants système — auto-remplis avec valeurs typiques": "system identifiers — auto-filled with typical values",
    "colonne(s) identifiant Database non affichées": "database identifier column(s) not displayed",
    "valeurs fixées à la médiane du jeu pour l'inférence": "values set to dataset median for inference",
    
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
    
    # Summary page
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
    "Après prédiction": "After prediction",
    "les barres et la jauge montrent": "bars and gauge show",
    "les probabilités réelles": "the actual probabilities",
    "du Prediction System pour votre scénario": "from the model for your scenario",
}

# Patterns to remove (unnecessary messages)
REMOVE_PATTERNS = [
    r"EventZilla ML Studio — données du Database, modèles dans ML/models_artifacts/\.",
    r"Identifiants créés dans SSMS — DW_eventzella",
    r"Aperçu tronqué — fichier complet dans le dossier ML/\.",
]

def translate_file(file_path: Path) -> None:
    """Translate French text to English in the given file."""
    print(f"Translating {file_path.name}...")
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # Apply translations
    for french, english in TRANSLATIONS.items():
        content = content.replace(french, english)
    
    # Remove unnecessary patterns
    for pattern in REMOVE_PATTERNS:
        content = re.sub(pattern, "", content)
    
    # Additional replacements for common French words
    replacements = {
        # Common UI terms
        " lancez ": " run ",
        " exécutez ": " execute ",
        " cliquez ": " click ",
        " sélectionnez ": " select ",
        " choisissez ": " choose ",
        
        # File/data terms
        " fichier ": " file ",
        " dossier ": " folder ",
        " données ": " data ",
        " jeu ": " dataset ",
        
        # Model terms
        " modèle ": " model ",
        " entraîné ": " trained ",
        " prédit ": " predicted ",
        " prédiction ": " prediction ",
        
        # Common phrases
        " disponible": " available",
        " absent": " missing",
        " trouvé": " found",
    }
    
    for fr, en in replacements.items():
        content = content.replace(fr, en)
    
    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"✅ Translated {file_path.name}")
    else:
        print(f"ℹ️  No changes needed for {file_path.name}")

def main():
    """Main translation function."""
    streamlit_file = Path("ML/streamlit_app.py")
    
    if not streamlit_file.exists():
        print(f"❌ File not found: {streamlit_file}")
        return
    
    # Backup original file
    backup_file = streamlit_file.with_suffix(".py.backup")
    if not backup_file.exists():
        import shutil
        shutil.copy2(streamlit_file, backup_file)
        print(f"📦 Backup created: {backup_file.name}")
    
    # Translate
    translate_file(streamlit_file)
    
    print("\n✅ Translation complete!")
    print(f"📄 Original backup: {backup_file}")
    print(f"📄 Translated file: {streamlit_file}")

if __name__ == "__main__":
    main()
