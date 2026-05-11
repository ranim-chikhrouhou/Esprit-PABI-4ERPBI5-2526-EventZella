#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Second pass translation - catch all remaining French text
"""

from pathlib import Path
import re

def translate_pass2():
    """Second pass to catch all remaining French text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Comprehensive translations - second pass
    translations = {
        # Error/Info messages
        "Modèles Status Prediction missings": "Booking Status models missing",
        "Métriques Price Estimation incomplètes (pas de Goal documentée).": "Price Estimation metrics incomplete (no target documented).",
        "Colonnes missinges du parquet — valeurs par défaut :": "Missing columns from parquet — default values:",
        "Histogramme inavailable — colonne `final_price` missinge du jeu préparé.": "Histogram unavailable — `final_price` column missing from prepared dataset.",
        "Métriques manquantes pour ce périmètre.": "Metrics missing for this scope.",
        "Prediction System Customer Grouping missing — métriques JSON uniquement.": "Customer Segmentation model missing — JSON metrics only.",
        "Prédiction impossible (vérifiez le nombre de variables et les artefacts) :": "Prediction failed (check number of variables and artifacts):",
        
        # Form labels
        "Indicateurs à renseigner": "Metrics to fill in",
        "Les valeurs proposées correspondent aux **médianes** du jeu d'apprentissage — modifiez-les pour simuler un cas.": "Suggested values are **medians** from the training dataset — modify them to simulate a scenario.",
        "Identifiers techniques (optionnel)": "Technical identifiers (optional)",
        "Utile seulement si vous reproduisez une ligne complète du Database ; sinon laissez les défauts.": "Useful only if reproducing a complete database row; otherwise leave defaults.",
        "Répartition illustrative des segments": "Illustrative segment distribution",
        
        # More common phrases
        "Aucun": "None",
        "Tous": "All",
        "Sélectionner": "Select",
        "Choisir": "Choose",
        "Valider": "Validate",
        "Annuler": "Cancel",
        "Fermer": "Close",
        "Ouvrir": "Open",
        "Afficher": "Show",
        "Masquer": "Hide",
        
        # Technical terms
        "jeu d'apprentissage": "training dataset",
        "jeu de test": "test dataset",
        "jeu de validation": "validation dataset",
        "jeu préparé": "prepared dataset",
        "artefacts": "artifacts",
        "périmètre": "scope",
        
        # More UI elements
        "Chargement": "Loading",
        "En cours": "In progress",
        "Terminé": "Completed",
        "Échec": "Failed",
        "Succès": "Success",
        
        # Additional captions
        "Statuts possibles (Y) :": "Possible statuses (Y):",
        "Fields entered": "Fields entered",
        "Hidden fields": "Hidden fields",
        
        # More section headers that might have been missed
        "Résultat de la validation interactive": "Interactive validation result",
        "Erreurs calculées sur les": "Errors calculated on the",
        "mois de holdout sélectionnés": "selected holdout months",
        
        # Clustering specific
        "Segment identifié": "Identified segment",
        "Caractéristiques": "Characteristics",
        "Profil": "Profile",
        
        # Time series specific
        "Historique (Database)": "Historical (Database)",
        "L'historique est trop court": "History is too short",
        "pour ajuster les modèles de prévision": "to fit forecasting models",
        "Colonnes attendues absentes du résultat SQL": "Expected columns missing from SQL result",
        "vérifiez": "check",
        
        # Summary page
        "Aucune métrique trouvée dans ML/models_artifacts/": "No metrics found in ML/models_artifacts/",
        "exécutez les scripts run_01 … run_04": "run scripts run_01 … run_04",
        
        # Login page
        "Email ou mot de passe incorrect": "Incorrect email or password",
        "Veuillez remplir tous les champs": "Please fill in all fields",
        
        # Sidebar
        "Navigation": "Navigation",
        "Paramètres": "Settings",
        "Aide": "Help",
        
        # Buttons
        "Précédent": "Previous",
        "Suivant": "Next",
        "Retour": "Back",
        "Continuer": "Continue",
        
        # More form fields
        "Entrez": "Enter",
        "Saisissez": "Input",
        "Renseignez": "Fill in",
        
        # Status messages
        "Chargement en cours...": "Loading...",
        "Traitement en cours...": "Processing...",
        "Calcul en cours...": "Calculating...",
        
        # More technical terms
        "entraînement": "training",
        "apprentissage": "learning",
        "inférence": "inference",
        "déploiement": "deployment",
        
        # Additional UI cleanup
        "Ce que teste cet écran": "What this screen tests",
        "Rappel pédagogique": "Educational reminder",
        "détails (optionnel)": "details (optional)",
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Additional cleanup - remove verbose technical captions
    patterns_to_remove = [
        r'st\.caption\(\s*f?"Identifiants créés dans SSMS[^"]*"\s*\)',
        r'st\.caption\(\s*"EventZilla ML Studio[^"]*"\s*\)',
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Second pass translation complete!")

if __name__ == "__main__":
    translate_pass2()
