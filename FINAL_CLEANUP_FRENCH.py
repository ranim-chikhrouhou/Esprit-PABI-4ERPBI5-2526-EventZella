#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final cleanup of remaining French words in streamlit_app.py
Focuses on comments, docstrings, and remaining UI strings
"""

import re
from pathlib import Path

def final_cleanup():
    """Remove all remaining French text"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Additional translations for remaining French text
    replacements = {
        # Comments and docstrings
        "alignée sur le critère D": "aligned with criterion D",
        "panier / prix final": "basket / final price",
        "Statut réservation": "Booking status",
        "statut de réservation": "booking status",
        "statuts et montants": "statuses and amounts",
        "ordre de grandeur des montants": "order of magnitude of amounts",
        "montants utiles à la lecture business": "amounts useful for business reading",
        "montants & budget": "amounts & budget",
        "montants / saisonnalité / catalogue": "amounts / seasonality / catalog",
        "indicateurs EventZilla": "EventZilla indicators",
        "indicateurs database": "database indicators",
        "des indicateurs": "indicators",
        "saisies à l'écran": "entered on screen",
        "Affiche l'écran de connexion": "Display login screen",
        "hors écran Status Prediction": "outside Booking Status screen",
        "aperçu du graphique": "chart preview",
        "aperçu (répartition fictive": "preview (fictitious distribution",
        "Probabilités par statut — aperçu": "Probabilities by status — preview",
        "Confiance (probabilité max.) — aperçu": "Confidence (max. probability) — preview",
        "répartition fictive": "fictitious distribution",
        "multi-classes": "multi-class",
        
        # More UI strings
        "Ordre par défaut": "Default order",
        "chargez le Prediction System": "load the Prediction System",
        "vue large": "broad view",
        "saisonnalité": "seasonality",
        "Exécutez": "Execute",
        "récence": "recency",
        "activity": "activity",
        
        # Additional context
        "permettent d'anticipate": "allow to anticipate",
        "de segment l'offre": "to segment the offering",
        "de track les monthly": "to track monthly",
        "Associer un profil": "Associate a profile",
        "cohérent avec le database": "consistent with the database",
        "au statut le plus plausible": "to the most plausible status",
        "Lecture réservation": "Booking reading",
        "file active": "active queue",
        "Forêt aléatoire": "Random forest",
        "mise à l'échelle": "scaling",
        "Bon compromis précision": "Good accuracy compromise",
        "sur le test set": "on test set",
        "probabilité max.": "max. probability",
        "classe dominante": "dominant class",
        
        # Form and field labels
        "En priorité": "Priority",
        "variables hors": "variables outside",
        "puis complété jusqu'à": "then completed up to",
        "au moins": "at least",
        "champs": "fields",
        "même si présents": "even if present",
        
        # Segment and clustering
        "profil numérique": "numeric profile",
        "profil-type": "typical profile",
        "groupe-type": "typical group",
        "comportements proches": "similar behaviors",
        "Partitions lisibles": "Readable partitions",
        "pour regrouper": "to group",
        "Segments stables": "Stable segments",
        "après normalisation": "after normalization",
        
        # Additional technical terms
        "échelle normalisée": "normalized scale",
        "espace standardisé": "standardized space",
        "espace d'entrée": "input space",
        "Gaussienne multivariée": "multivariate Gaussian",
        "taille relative": "relative size",
        "volumes business bruts": "raw business volumes",
        "répartition simulée": "simulated distribution",
        "nombreuses attributions": "many assignments",
        
        # More captions and hints
        "Valeurs typiques": "Typical values",
        "jeu database": "database set",
        "prepared dataset": "prepared dataset",
        "Champs saisis": "Fields entered",
        "Champs masqués": "Hidden fields",
        "identifiants système": "system identifiers",
        "auto-remplis": "auto-filled",
        "valeurs typiques": "typical values",
        
        # Business context
        "niveau d'activity": "activity level",
        "cas réel": "real case",
        "déjà présent": "already present",
        "même univers": "same universe",
        "situation cohérente": "consistent situation",
        "alignée sur": "aligned with",
        
        # Loyalty and RFM
        "combien de bookings": "how many bookings",
        "quel volume d'affaires": "what business volume",
        "à quelle récence": "how recent",
        "remonte la dernière activity": "was the last activity",
        "très fidèle": "very loyal",
        "occasionnel": "occasional",
        "à relancer": "to reactivate",
        "fidélité / RFM": "loyalty / RFM",
        
        # Additional UI elements
        "Aide au ciblage": "Targeting assistance",
        "offres, relances, priorisation": "offers, reactivation, prioritization",
        "Lecture par segment": "Reading by segment",
        "Repères sur": "Benchmarks on",
        "Qualité globale": "Overall quality",
        "avant de remplir": "before filling",
        "Quelques indicateurs": "Some indicators",
        "avant simulation": "before simulation",
        
        # Chart and visualization
        "Comparaison visuelle": "Visual comparison",
        "profil saisi": "entered profile",
        "vs profil-type": "vs typical profile",
        "Chaque axe correspond": "Each axis corresponds",
        "à un indicateur": "to an indicator",
        "comme à l'Learning": "as in training",
        "Centres des clusters": "Cluster centers",
        "variables RFM": "RFM variables",
        "dimensions = variables num.": "dimensions = numeric variables",
        "du scope perf.": "from perf. scope",
        "not available": "not available",
        
        # Distribution section
        "Distribution illustrative": "Illustrative distribution",
        "pour visualiser": "to visualize",
        "Les parts par segment": "Segment shares",
        "affichées plus haut": "displayed above",
        "proviennent de": "come from",
        "échantillon d'apprentissage": "training sample",
        "simulation aléatoire": "random simulation",
        "n'est pas calibrée": "is not calibrated",
        "désactivée en mode": "disabled in mode",
        "Simulation de": "Simulation of",
        "dans l'espace": "in space",
        "donne une idée": "gives an idea",
        "pas les volumes": "not the volumes",
        
        # More specific phrases
        "l'évolution monthlyle": "monthly evolution",
        "de vos indicateurs": "of your indicators",
        "comparez Trend Analysis": "compare Trend Analysis",
        "vs Advanced Forecast": "vs Advanced Forecast",
    }
    
    # Apply all replacements
    for french, english in replacements.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Final cleanup complete!")
    print(f"✓ File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = final_cleanup()
    if success:
        print("\n✓ All remaining French text has been cleaned up")
        print("✓ Please restart the Streamlit app to see the changes")
    else:
        print("\n✗ Cleanup failed")
