#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete translation of all remaining French text in streamlit_app.py
This script handles all French strings including those with Unicode escapes
"""

import re
from pathlib import Path

def translate_streamlit_app():
    """Translate all remaining French text to English"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary of all French to English translations
    translations = {
        # Section headers and titles
        "Metrics clés calculés depuis le database": "Key metrics calculated from the database",
        "Metrics cl\\u00e9s calcul\\u00e9s depuis le database": "Key metrics calculated from the database",
        "Régression": "Regression",
        "R\\u00e9gression": "Regression",
        "Séries temporelles": "Time Series",
        "S\\u00e9ries temporelles": "Time Series",
        "Prévision": "Forecast",
        "Pr\\u00e9vision": "Forecast",
        
        # Learn more section
        "En savoir plus — intérêt du ML pour EventZilla": "Learn more — ML benefits for EventZilla",
        "En savoir plus \\u2014 int\\u00e9r\\u00eat du ML pour EventZilla": "Learn more — ML benefits for EventZilla",
        
        # Table column headers
        "Critère": "Criterion",
        "Crit\\u00e8re": "Criterion",
        "Domaine": "Domain",
        "Règle de choix": "Selection Rule",
        "R\\u00e8gle de choix": "Selection Rule",
        "Qualité": "Quality",
        "Qualit\\u00e9": "Quality",
        
        # Export and file messages
        "Export texte détaillé": "Detailed text export",
        "Export texte d\\u00e9taill\\u00e9": "Detailed text export",
        "Aperçu tronqué — fichier complet dans le dossier ML/.": "Truncated preview — full file in ML/ folder.",
        "Aper\\u00e7u tronqu\\u00e9 \\u2014 fichier complet dans le dossier ML/.": "Truncated preview — full file in ML/ folder.",
        
        # Navigation
        "Accès rapide": "Quick access",
        "Acc\\u00e8s rapide": "Quick access",
        "Accéder aux pages de test": "Access test pages",
        "Acc\\u00e9der aux pages de test": "Access test pages",
        "Accéder aux écrans de test interactif": "Access interactive test screens",
        "Acc\\u00e9der aux \\u00e9crans de test interactif": "Access interactive test screens",
        
        # Metric labels
        "Exactitude (réf.)": "Accuracy (ref.)",
        "Exactitude \\(r\\u00e9f\\.\\)": "Accuracy (ref.)",
        "Balance Score pondéré (réf.)": "Weighted F1-Score (ref.)",
        "Balance Score pond\\u00e9r\\u00e9 \\(r\\u00e9f\\.\\)": "Weighted F1-Score (ref.)",
        "Quality Score (réf.)": "ROC-AUC Score (ref.)",
        "Quality Score \\(r\\u00e9f\\.\\)": "ROC-AUC Score (ref.)",
        "Statuts possibles (Y)": "Possible statuses (Y)",
        "Statuts possibles \\(Y\\)": "Possible statuses (Y)",
        
        # Error messages
        "aucune métrique disponible": "no metrics available",
        "aucune m\\u00e9trique disponible": "no metrics available",
        "Aucune métrique trouvée": "No metrics found",
        "Aucune m\\u00e9trique trouv\\u00e9e": "No metrics found",
        "Aucune métrique — run ML/scripts/run_01": "No metrics — run ML/scripts/run_01",
        "Aucune m\\u00e9trique \\u2014 run ML/scripts/run_01": "No metrics — run ML/scripts/run_01",
        
        # Button labels
        "Se déconnecter": "Log out",
        "Se d\\u00e9connecter": "Log out",
        "Voir le récapitulatif": "View summary",
        "Voir le r\\u00e9capitulatif": "View summary",
        
        # Page titles and descriptions
        "Status Prediction — statut de réservation": "Booking Status — Reservation Status",
        "Status Prediction \\u2014 statut de r\\u00e9servation": "Booking Status — Reservation Status",
        "Price Estimation — montants & indicateurs": "Price Estimation — Amounts & Indicators",
        "Price Estimation \\u2014 montants & indicateurs": "Price Estimation — Amounts & Indicators",
        
        # Badges
        "Critère C": "Criterion C",
        "Crit\\u00e8re C": "Criterion C",
        "Critère D": "Criterion D",
        "Crit\\u00e8re D": "Criterion D",
        "Critère E": "Criterion E",
        "Crit\\u00e8re E": "Criterion E",
        "Critère F": "Criterion F",
        "Crit\\u00e8re F": "Criterion F",
        "Test interactif": "Interactive test",
        
        # Form labels
        "Ce que teste cet écran": "What this screen tests",
        "Ce que teste cet \\u00e9cran": "What this screen tests",
        
        # Specific metric names
        "Classif. Balance Score": "Classification F1-Score",
        "Régr. Correct Predictions Score": "Regression R² Score",
        "R\\u00e9gr\\. Correct Predictions Score": "Regression R² Score",
        
        # Page sections
        "Détail par famille": "Details by family",
        "D\\u00e9tail par famille": "Details by family",
        "Modèle Best System": "Best model",
        "Mod\\u00e8le Best System": "Best model",
        
        # Additional French phrases
        "Plateforme d'intelligence artificielle": "Artificial intelligence platform",
        "Plateforme d'\\*\\*intelligence artificielle\\*\\*": "**Artificial intelligence** platform",
        "appliquée au business EventZilla": "applied to EventZilla business",
        "appliqu\\u00e9e au \\*\\*business\\*\\* EventZilla": "applied to **EventZilla business**",
        
        # Deployment context
        "Lecture réservation / file active": "Booking reading / active queue",
        "Lecture r\\u00e9servation / file active": "Booking reading / active queue",
        "Forêt aléatoire + mise à l'échelle": "Random forest + scaling",
        "For\\u00eat al\\u00e9atoire \\+ mise \\u00e0 l'\\u00e9chelle": "Random forest + scaling",
        "Bon compromis précision / Balance Score sur le test set": "Good balance between accuracy and F1-Score on test set",
        "Bon compromis pr\\u00e9cision / Balance Score sur le test set": "Good balance between accuracy and F1-Score on test set",
        
        # Chart labels
        "Barres = probabilités par statut": "Bars = probabilities by status",
        "Barres = probabilit\\u00e9s par statut": "Bars = probabilities by status",
        "jauge = confiance sur la classe dominante": "gauge = confidence on dominant class",
        
        # Additional sections
        "Comparatifs Prediction System": "Prediction System comparisons",
        "Pour aller plus loin que le formulaire ci-dessus": "To go further than the form above",
        "Centres des clusters": "Cluster centers",
        "Distribution simulée": "Simulated distribution",
        "Distribution simul\\u00e9e": "Simulated distribution",
        "Calculer la répartition simulée": "Calculate simulated distribution",
        "Calculer la r\\u00e9partition simul\\u00e9e": "Calculate simulated distribution",
        
        # Expander titles
        "Aperçu des segments (rappel)": "Segment overview (reminder)",
        "Aper\\u00e7u des segments \\(rappel\\)": "Segment overview (reminder)",
        "Metrics techniques & profils detailslés (optional)": "Technical metrics & detailed profiles (optional)",
        "Metrics techniques & profils detailsl\\u00e9s \\(optional\\)": "Technical metrics & detailed profiles (optional)",
        
        # Additional UI text
        "Renseignez les champs numériques": "Fill in the numeric fields",
        "Estimer le prix final": "Estimate final price",
        "Voir mon segment": "View my segment",
        "Voir le segment et le graphique": "View segment and chart",
        
        # Captions and hints
        "Valeurs typiques pour la colonne": "Typical values for column",
        "prepared dataset": "prepared dataset",
        "Fields entered": "Fields entered",
        "Hidden fields": "Hidden fields",
        "system identifiers": "system identifiers",
        "auto-filled with typical values": "auto-filled with typical values",
        
        # Additional French text
        "Indiquez à quel stade": "Indicate at which stage",
        "Indiquez \\*\\*\\u00e0 quel stade\\*\\*": "Indicate **at which stage**",
        "se situe une réservation": "a booking is located",
        "confirmée, en attente, annulée": "confirmed, pending, cancelled",
        "confirm\\u00e9e, en attente, annul\\u00e9e": "confirmed, pending, cancelled",
        "du même univers que le database": "from the same universe as the database",
        "du m\\u00eame univers que le database": "from the same universe as the database",
        
        # More deployment text
        "Obtenez une estimation numérique": "Get a numeric estimate",
        "Obtenez une \\*\\*estimation num\\u00e9rique\\*\\*": "Get a **numeric estimate**",
        "panier, budget, etc.": "basket, budget, etc.",
        "alignée sur le database": "aligned with the database",
        "align\\u00e9e sur le database": "aligned with the database",
        
        # Segment text
        "quel segment pour ce profil": "which segment for this profile",
        "profils d'activity": "activity profiles",
        "profils d'activit\\u00e9": "activity profiles",
        "combien de bookings": "how many bookings",
        "quel volume d'affaires": "what business volume",
        "à quelle récence": "how recent",
        "\\u00e0 quelle r\\u00e9cence": "how recent",
        "remonte la dernière activity": "was the last activity",
        "groupe-type": "typical group",
        "très fidèle, occasionnel, à relancer": "very loyal, occasional, to reactivate",
        "tr\\u00e8s fid\\u00e8le, occasionnel, \\u00e0 relancer": "very loyal, occasional, to reactivate",
        
        # More UI elements
        "Comment utiliser cet écran": "How to use this screen",
        "Comment utiliser cet \\u00e9cran": "How to use this screen",
        "Profil à simuler": "Profile to simulate",
        "Profil \\u00e0 simuler": "Profile to simulate",
        "Bénéficiaires": "Beneficiaries",
        "B\\u00e9n\\u00e9ficiaires": "Beneficiaries",
        "Prestataires": "Providers",
        "charge, CA, récence": "load, revenue, recency",
        "charge, CA, r\\u00e9cence": "load, revenue, recency",
        
        # Context cards
        "Segmentation fidélité": "Loyalty segmentation",
        "Segmentation fid\\u00e9lit\\u00e9": "Loyalty segmentation",
        "Le segment le plus proche": "The closest segment",
        "parmi les groupes appris": "among learned groups",
        "L'un des": "One of the",
        "profils-type du Prediction System": "typical profiles of the Prediction System",
        "Comparer votre saisie aux profils": "Compare your input to profiles",
        "fidélité / RFM": "loyalty / RFM",
        "fid\\u00e9lit\\u00e9 / RFM": "loyalty / RFM",
        "nommer le groupe le plus proche": "name the closest group",
        "Rapprocher un cas du profil-type le plus proche": "Match a case to the closest typical profile",
        "Aide au ciblage": "Targeting assistance",
        "offres, relances, priorisation": "offers, reactivation, prioritization",
        "Lecture par segment": "Reading by segment",
        "Segments stables after normalisation": "Stable segments after normalization",
        "Segments stables apr\\u00e8s normalisation": "Stable segments after normalization",
        "Partitions lisibles pour regrouper": "Readable partitions to group",
        "comportements proches": "similar behaviors",
        "Radar : votre profil comparé au centre": "Radar: your profile compared to center",
        "Radar : votre profil compar\\u00e9 au centre": "Radar: your profile compared to center",
        "du segment attribué": "of assigned segment",
        "du segment attribu\\u00e9": "of assigned segment",
        
        # Metrics section
        "Repères sur le Prediction System": "Prediction System benchmarks",
        "Rep\\u00e8res sur le Prediction System": "Prediction System benchmarks",
        "Qualité globale avant de remplir": "Overall quality before filling",
        "Qualit\\u00e9 globale avant de remplir": "Overall quality before filling",
        "Quelques indicateurs avant simulation": "Some indicators before simulation",
        "Échantillon (train / total)": "Sample (train / total)",
        "\\u00c9chantillon \\(train / total\\)": "Sample (train / total)",
        "Périmètre": "Scope",
        "P\\u00e9rim\\u00e8tre": "Scope",
        "Prediction System exploratoire database": "Exploratory database Prediction System",
        
        # Form section
        "Formulaire — décrire le profil à tester": "Form — describe the profile to test",
        "Formulaire \\u2014 d\\u00e9crire le profil \\u00e0 tester": "Form — describe the profile to test",
        "Même logique que la Status Prediction": "Same logic as Booking Status",
        "M\\u00eame logique que la Status Prediction": "Same logic as Booking Status",
        "un champ par indicateur": "one field per indicator",
        "puis validation pour voir le segment": "then validate to see segment",
        "Renseignez les valeurs attendues": "Fill in expected values",
        "puis validez pour voir le segment et le graphique": "then validate to see segment and chart",
        
        # Segment assignment
        "Segment attribué": "Assigned segment",
        "Segment attribu\\u00e9": "Assigned segment",
        "Lecture": "Reading",
        "À compléter côté projet": "To be completed on project side",
        "\\u00c0 compl\\u00e9ter c\\u00f4t\\u00e9 projet": "To be completed on project side",
        "Indice du segment": "Segment index",
        "Part approximative de ce segment": "Approximate share of this segment",
        "dans l'échantillon d'apprentissage": "in training sample",
        "dans l'\\u00e9chantillon d'apprentissage": "in training sample",
        "profils aggregateds (fidélité)": "aggregated profiles (loyalty)",
        "profils aggregateds \\(fid\\u00e9lit\\u00e9\\)": "aggregated profiles (loyalty)",
        
        # Chart titles
        "Comparaison visuelle": "Visual comparison",
        "profil saisi vs profil-type du segment": "entered profile vs typical segment profile",
        "Chaque axe correspond à un indicateur": "Each axis corresponds to an indicator",
        "Chaque axe correspond \\u00e0 un indicateur": "Each axis corresponds to an indicator",
        "échelle normalisée comme à l'Learning": "normalized scale as in training",
        "\\u00e9chelle normalis\\u00e9e comme \\u00e0 l'Learning": "normalized scale as in training",
        
        # Technical section
        "Comparatifs Prediction System": "Prediction System comparisons",
        "Pour aller plus loin que le formulaire ci-dessus": "To go further than the form above",
        "Separation Score": "Separation Score",
        "clusters plus compacts / séparés": "more compact / separated clusters",
        "clusters plus compacts / s\\u00e9par\\u00e9s": "more compact / separated clusters",
        "Customer Grouping (fidélité)": "Customer Grouping (loyalty)",
        "Customer Grouping \\(fid\\u00e9lit\\u00e9\\)": "Customer Grouping (loyalty)",
        "Centres des clusters — variables RFM": "Cluster centers — RFM variables",
        "espace standardisé": "standardized space",
        "espace standardis\\u00e9": "standardized space",
        "dimensions = variables num. du scope perf.": "dimensions = numeric variables from perf. scope",
        "Centres de clusters not available": "Cluster centers not available",
        
        # Distribution section
        "Distribution illustrative (simulation)": "Illustrative distribution (simulation)",
        "Distribution simulée pour visualiser": "Simulated distribution to visualize",
        "Distribution simul\\u00e9e pour visualiser": "Simulated distribution to visualize",
        "taille relative des segments": "relative size of segments",
        "Les parts par segment affichées": "The segment shares displayed",
        "proviennent de l'échantillon d'apprentissage": "come from training sample",
        "La simulation aléatoire n'est pas calibrée": "Random simulation is not calibrated",
        "La simulation al\\u00e9atoire n'est pas calibr\\u00e9e": "Random simulation is not calibrated",
        "désactivée en mode fidélité": "disabled in loyalty mode",
        "d\\u00e9sactiv\\u00e9e en mode fid\\u00e9lit\\u00e9": "disabled in loyalty mode",
        "Simulation de nombreuses attributions": "Simulation of many assignments",
        "dans l'espace d'entrée du KMeans": "in KMeans input space",
        "dans l'espace d'entr\\u00e9e du KMeans": "in KMeans input space",
        "Gaussienne multivariée": "multivariate Gaussian",
        "Gaussienne multivari\\u00e9e": "multivariate Gaussian",
        "donne une idée de la taille relative": "gives an idea of relative size",
        "pas les volumes business bruts": "not raw business volumes",
        "Illustration — répartition simulée": "Illustration — simulated distribution",
        "Illustration \\u2014 r\\u00e9partition simul\\u00e9e": "Illustration — simulated distribution",
    }
    
    # Apply all translations
    for french, english in translations.items():
        # Try both literal and regex patterns
        content = content.replace(french, english)
        try:
            content = re.sub(french, english, content)
        except:
            pass
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Translation complete!")
    print(f"✓ File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = translate_streamlit_app()
    if success:
        print("\n✓ All French text has been translated to English")
        print("✓ Please restart the Streamlit app to see the changes")
    else:
        print("\n✗ Translation failed")
