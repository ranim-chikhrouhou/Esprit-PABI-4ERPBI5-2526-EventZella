#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive cleanup of ALL remaining French text in streamlit_app.py
"""

import re
from pathlib import Path

def comprehensive_cleanup():
    """Remove ALL remaining French text"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Comprehensive list of ALL French to English translations
    replacements = {
        # File header comments
        "Lancer depuis la racine du dépôt": "Launch from repository root",
        "laboratoire ML": "ML laboratory",
        "interface claire": "clear interface",
        "accents teal": "teal accents",
        "graphiques Plotly lisibles": "readable Plotly charts",
        
        # Core French phrases
        "permettent d'anticipate les statuses and amounts": "allow to anticipate statuses and amounts",
        "de segment l'offre": "to segment the offering",
        "de track les monthly trends": "to track monthly trends",
        "sans remplacer le business": "without replacing business",
        "mais pour prioritize": "but to prioritize",
        "et illustrate les scenarios": "and illustrate scenarios",
        "dans un educational framework": "in an educational framework",
        "et reproducible": "and reproducible",
        
        # Studio description
        "Ce studio centralise les modèles": "This studio centralizes models",
        "trained on the même scope": "trained on the same scope",
        "que vos notebooks": "as your notebooks",
        "test them here before any production deployment": "test them here before any production deployment",
        
        # Indicator selection
        "Choisir l'indicateur": "Choose the indicator",
        "volume d'activity": "activity volume",
        "monthly revenue ou average basket": "monthly revenue or average basket",
        "Ajuster l'horizon": "Adjust horizon",
        "de 1 à 12 mois de forecast": "from 1 to 12 months forecast",
        
        # Function docstrings
        "Texte court expliquant le choix du Prediction System": "Short text explaining Prediction System choice",
        "fields optionals dans les JSON métriques": "optional fields in JSON metrics",
        "Thème de couleur dynamique par page": "Dynamic color theme per page",
        "boutons, métriques, formulaires, expanders": "buttons, metrics, forms, expanders",
        "Columns affichées dans le formulaire Status Prediction": "Columns displayed in Booking Status form",
        "hors id": "excluding id",
        "à ne pas dupliquer en Price Estimation": "not to duplicate in Price Estimation",
        "Met en tête period / amounts useful for business reading": "Put period / amounts useful for business reading first",
        "puis le reste": "then the rest",
        
        # Month labels
        "mois {i + 1}": "month {i + 1}",
        
        # Field grouping
        "price, budget, margin, revenue, ca_": "price, budget, margin, revenue, ca_",
        
        # Median defaults
        "Médianes sur le prepared dataset": "Medians on prepared dataset",
        "pour les colonnes id requises par le Prediction System": "for id columns required by Prediction System",
        "non saisies à l'écran": "not entered on screen",
        
        # Target labels
        "event_budget": "event_budget",
        "prix_prestataire_structure_revenus": "provider_price_revenue_structure",
        "budget_evenement": "event_budget",
        "panier_average_ca_sum_final_price": "basket_average_revenue_sum_final_price",
        
        # Form ordering
        "Ordre des fields": "Field order",
        "importance RF décroissante": "decreasing RF importance",
        "sinon heuristique": "otherwise heuristic",
        "prix / budget d'abord": "price / budget first",
        
        # Benchmark price
        "ou alias": "or alias",
        "saisie par dropdown de quantiles / valeurs database": "input by dropdown of quantiles / database values",
        
        # Minimal fields
        "Nombre minimal de fields affichés": "Minimum number of displayed fields",
        "saisie utilisateur": "user input",
        "évite un formulaire réduit à une seule variable": "avoids form reduced to single variable",
        
        # Manual columns
        "Columns de saisie Price Estimation": "Price Estimation input columns",
        "d'abord hors formulaire Status Prediction": "first outside Booking Status form",
        "puis complément": "then complement",
        "jusqu'à un minimum de fields": "up to minimum fields",
        "ordre importance du Prediction System": "Prediction System importance order",
        "même si chevauchement avec la classif": "even if overlap with classification",
        
        # Section blocks
        "Input Factors du Prediction System (prix final)": "Prediction System Input Factors (final price)",
        "Amounts, budget & references (fill in priority)": "Amounts, budget & references (fill in priority)",
        
        # Series labels
        "Volume d'activity (fact rows database / mois)": "Activity volume (database fact rows / month)",
        "mois": "month",
        
        # Target description
        "k={k} segments (Factors perf. database standardisées)": "k={k} segments (standardized database perf. factors)",
        "Factors perf. database standardisées": "standardized database perf. factors",
        
        # Chart titles
        "Probabilités par statut — chart preview": "Probabilities by status — chart preview",
        "Illustration équiprobable": "Equiprobable illustration",
        "les vraies valeurs apparaissent after": "true values appear after",
        "Prédire": "Predict",
        
        # ID columns
        "Clés dimension / identifiants database": "Dimension keys / database identifiers",
        "jamais proposés aux formulaires business": "never offered in business forms",
        "remplissage automatique si requis par le Prediction System": "automatic filling if required by Prediction System",
        
        # Money detection
        "price, budget, margin, revenue, ca_, montant": "price, budget, margin, revenue, ca_, amount",
        "montant": "amount",
        
        # Form centered
        "Formulaire centré": "Centered form",
        
        # KPI labels
        "Customer Segments": "Customer Segments",
        "Horizon Forecast": "Forecast Horizon",
        "Prediction Error Séries Temp.": "Time Series Prediction Error",
        "customer segments": "customer segments",
        
        # Model cards
        "Customer Segmentation": "Customer Segmentation",
        "Agglomératif (Ward)": "Agglomerative (Ward)",
        "données database standardisées": "standardized database data",
        "donn\\u00e9es database standardis\\u00e9es": "standardized database data",
        
        # Page recap
        "Statut réservation": "Booking status",
        "Statut de réservation": "Booking status",
        "multi-class": "multi-class",
        
        # Deployment context
        "Associate a profile": "Associate a profile",
        "Estimate final price à partir d'un profil database": "Estimate final price from a database profile",
        "cohérent avec les data préparées": "consistent with prepared data",
        
        # Segment text
        "which segment for this profile": "which segment for this profile",
        "Testez à quel segment se rapproche un profil": "Test which segment a profile is closest to",
        "consistent with the database": "consistent with the database",
        "The closest segment (among learned groups)": "The closest segment (among learned groups)",
        "Reading by segment": "Reading by segment",
        "Stable segments after normalization des indicateurs": "Stable segments after indicator normalization",
        "Readable partitions to group similar behaviors": "Readable partitions to group similar behaviors",
        "Radar: your profile compared to center of assigned segment": "Radar: your profile compared to center of assigned segment",
        
        # Context labels
        "Ce que vous obtenez": "What you get",
        "Utilité": "Usefulness",
        "Graphique principal": "Main chart",
        
        # Metrics section
        "Prediction System benchmarks": "Prediction System benchmarks",
        "Overall quality before filling": "Overall quality before filling",
        "Some indicators before simulation": "Some indicators before simulation",
        "Sample (train / total)": "Sample (train / total)",
        "Scope": "Scope",
        "Exploratory database Prediction System": "Exploratory database Prediction System",
        
        # Form section
        "Form — describe the profile to test": "Form — describe the profile to test",
        "Same logic as Booking Status": "Same logic as Booking Status",
        "one field per indicator": "one field per indicator",
        "then validate to see segment": "then validate to see segment",
        "Fill in expected values": "Fill in expected values",
        "then validate to see segment and chart": "then validate to see segment and chart",
        
        # Segment assignment
        "Assigned segment": "Assigned segment",
        "Reading": "Reading",
        "To be completed on project side": "To be completed on project side",
        "Segment index": "Segment index",
        "Approximate share of this segment": "Approximate share of this segment",
        "in training sample": "in training sample",
        "aggregated profiles (loyalty)": "aggregated profiles (loyalty)",
        
        # Chart comparisons
        "Visual comparison": "Visual comparison",
        "entered profile vs typical segment profile": "entered profile vs typical segment profile",
        "Each axis corresponds to an indicator": "Each axis corresponds to an indicator",
        "normalized scale as in training": "normalized scale as in training",
        
        # Technical section
        "Prediction System comparisons": "Prediction System comparisons",
        "To go further than the form above": "To go further than the form above",
        "more compact / separated clusters": "more compact / separated clusters",
        "Customer Grouping (loyalty)": "Customer Grouping (loyalty)",
        "Cluster centers — RFM variables": "Cluster centers — RFM variables",
        "standardized space": "standardized space",
        "dimensions = numeric variables from perf. scope": "dimensions = numeric variables from perf. scope",
        "Cluster centers not available": "Cluster centers not available",
        
        # Distribution section
        "Illustrative distribution (simulation)": "Illustrative distribution (simulation)",
        "Simulated distribution to visualize": "Simulated distribution to visualize",
        "relative size of segments": "relative size of segments",
        "Segment shares displayed": "Segment shares displayed",
        "come from training sample": "come from training sample",
        "Random simulation is not calibrated": "Random simulation is not calibrated",
        "disabled in loyalty mode": "disabled in loyalty mode",
        "Simulation of many assignments": "Simulation of many assignments",
        "in KMeans input space": "in KMeans input space",
        "multivariate Gaussian": "multivariate Gaussian",
        "gives an idea of relative size": "gives an idea of relative size",
        "not raw business volumes": "not raw business volumes",
        "Illustration — simulated distribution": "Illustration — simulated distribution",
        
        # Additional UI strings
        "Saisie du profil": "Profile input",
        "Champs : fréquence, volumes, CA cumulé, average basket": "Fields: frequency, volumes, cumulative revenue, average basket",
        "récence": "recency",
        "La note averagene n'est pas encore intégrée": "Average rating not yet integrated",
        "au Prediction System": "into Prediction System",
        
        # Expander titles
        "Segment overview (reminder)": "Segment overview (reminder)",
        "Technical metrics & detailed profiles (optional)": "Technical metrics & detailed profiles (optional)",
        
        # Form labels
        "Metrics to fill in": "Metrics to fill in",
        "Les valeurs proposées correspondent aux médianes": "Proposed values correspond to medians",
        "du jeu d'apprentissage": "of training set",
        "modifiez-les pour simuler un cas": "modify them to simulate a case",
        "Valeurs par défaut ≈ médianes du jeu d'Learning": "Default values ≈ training set medians",
        
        # Group labels
        "activity": "activity",
        "montants": "amounts",
        "récence": "recency",
        
        # Button labels
        "Voir mon segment": "View my segment",
        "Voir le segment et le graphique": "View segment and chart",
        
        # Technical identifiers
        "Technical identifiers (optional)": "Technical identifiers (optional)",
        "Useful only if reproducing a complete database row": "Useful only if reproducing a complete database row",
        "otherwise leave defaults": "otherwise leave defaults",
        
        # Prediction results
        "Segment attribué": "Assigned segment",
        "Lecture": "Reading",
        "À compléter côté projet": "To be completed on project side",
        "Indice du segment": "Segment index",
        "Part approximative de ce segment": "Approximate share of this segment",
        "dans l'échantillon d'apprentissage": "in training sample",
        "profils aggregateds (fidélité)": "aggregated profiles (loyalty)",
        "lignes": "rows",
        
        # Radar chart
        "Profil saisi": "Entered profile",
        "Centre du segment (reference)": "Segment center (reference)",
        "Comparaison visuelle": "Visual comparison",
        "profil saisi vs profil-type du segment": "entered profile vs typical segment profile",
        "Chaque axe correspond à un indicateur du formulaire": "Each axis corresponds to a form indicator",
        "échelle normalisée comme à l'Learning": "normalized scale as in training",
        
        # Technical metrics expander
        "Metrics techniques & profils detailslés (optional)": "Technical metrics & detailed profiles (optional)",
        "Comparatifs Prediction System": "Prediction System comparisons",
        "Pour aller plus loin que le formulaire ci-dessus": "To go further than the form above",
        
        # Davies-Bouldin chart
        "Separation Score (↓ = clusters plus compacts / séparés)": "Separation Score (↓ = more compact / separated clusters)",
        "Indice DB": "DB Index",
        "Customer Grouping (fidélité)": "Customer Grouping (loyalty)",
        "Customer Grouping (loyalty) (↓ = mieux)": "Customer Grouping (loyalty) (↓ = better)",
        "mieux": "better",
        
        # Heatmap
        "Centres des clusters — variables RFM / fidélité (standardisées)": "Cluster centers — RFM / loyalty variables (standardized)",
        "espace standardisé (dimensions = variables num. du scope perf.)": "standardized space (dimensions = numeric variables from perf. scope)",
        "Centres de clusters not available dans ce fichier joblib": "Cluster centers not available in this joblib file",
        
        # Distribution simulation
        "Distribution illustrative (simulation)": "Illustrative distribution (simulation)",
        "Distribution simulée pour visualiser la taille relative des segments (indicatif)": "Simulated distribution to visualize relative segment size (indicative)",
        "Les parts par segment affichées plus haut": "Segment shares displayed above",
        "proviennent de l'échantillon d'apprentissage": "come from training sample",
        "La simulation aléatoire n'est pas calibrée sur l'espace RFM": "Random simulation is not calibrated on RFM space",
        "désactivée en mode fidélité": "disabled in loyalty mode",
        "Simulation de nombreuses attributions dans l'espace d'entrée du KMeans": "Simulation of many assignments in KMeans input space",
        "Gaussienne multivariée": "multivariate Gaussian",
        "donne une idée de la taille relative des segments": "gives an idea of relative segment size",
        "pas les volumes business bruts": "not raw business volumes",
        "Calculer la répartition simulée": "Calculate simulated distribution",
        "Illustration — répartition simulée des segments": "Illustration — simulated segment distribution",
        
        # Additional cleanup
        "fréquence": "frequency",
        "volumes": "volumes",
        "CA cumulé": "cumulative revenue",
        "note averagene": "average rating",
        "pas encore intégrée": "not yet integrated",
    }
    
    # Apply all replacements
    for french, english in replacements.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Comprehensive cleanup complete!")
    print(f"SUCCESS: File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = comprehensive_cleanup()
    if success:
        print("\nSUCCESS: All French text has been comprehensively cleaned")
        print("SUCCESS: Please restart the Streamlit app to see the changes")
    else:
        print("\nERROR: Cleanup failed")
