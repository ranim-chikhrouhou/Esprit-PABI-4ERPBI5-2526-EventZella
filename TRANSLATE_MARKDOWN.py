#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translate remaining French text in markdown strings and comments
"""

from pathlib import Path

def translate_markdown():
    """Translate markdown content and remaining French text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Translate markdown content
    markdown_translations = {
        # ML_INTEREST_MARKDOWN
        "**Pourquoi le machine learning pour EventZilla ?**": "**Why machine learning for EventZilla?**",
        "Les data du **database** (réservations, finances agrégées, volumes) permettent d'**anticiper** les statuts et montants, de **segmenter** l'offre et de **suivre** les tendances mensuelles — sans remplacer le métier, mais pour **prioriser** et **illustrer** les scénarios dans un cadre pédagogique et reproductible.": "The **database** data (bookings, aggregated finances, volumes) allows us to **anticipate** statuses and amounts, **segment** offerings, and **track** monthly trends — not to replace business expertise, but to **prioritize** and **illustrate** scenarios in an educational and reproducible framework.",
        "**Ce studio** centralise les models traineds sur le même scope que vos notebooks (00→05) : testez-les ici avant toute mise en production.": "**This studio** centralizes models trained on the same scope as your notebooks (00→05): test them here before any production deployment.",
        
        # DEPLOY_TS_MARKDOWN
        "**Ce que vous pouvez tester :**": "**What you can test:**",
        "**Choose l'indicateur**": "**Choose the indicator**",
        "volume d'activité, CA mensuel ou panier moyen": "activity volume, monthly revenue, or average basket",
        "**Ajuster l'horizon**": "**Adjust the horizon**",
        "de 1 à 12 months de prévision": "from 1 to 12 months forecast",
        "**Comparer visuellement**": "**Visually compare**",
        "le train, la zone de validation et la prévision": "training, validation zone, and forecast",
        "**Lire les métriques**": "**Read the metrics**",
        "Prediction Error, Average Error, MAPE sur la fenêtre de test": "RMSE, MAE, MAPE on the test window",
        "**Modèles comparés :**": "**Models compared:**",
        "lissage exponentiel avec tendance": "exponential smoothing with trend",
        "autorégression + différenciation + moyenne mobile": "autoregression + differencing + moving average",
        "Le **Best Model** est celui qui a le **Prediction Error le plus bas** sur la validation": "The **best model** is the one with the **lowest RMSE** on validation",
        
        # Function docstrings
        "Texte court expliquant le choix du model (champs optionnels dans les JSON métriques).": "Short text explaining model choice (optional fields in JSON metrics).",
        "Thème de couleur dynamique par page — boutons, métriques, formulaires, expanders.": "Dynamic color theme per page — buttons, metrics, forms, expanders.",
        
        # Form labels and captions
        "Estimer le prix final à partir d'un profil database cohérent avec les data préparées.": "Estimate final price from a database profile consistent with prepared data.",
        "Indicateurs finance / performance (cf. métriques)": "Finance / performance indicators (see metrics)",
        "Forêt aléatoire + mise à l'échelle": "Random forest + scaling",
        "model trained pour minimiser l'erreur sur le test dataset.": "model trained to minimize error on test dataset.",
        "Fidélité — quel segment pour ce profil ?": "Loyalty — which segment for this profile?",
        "Segmentation — profils d'activité": "Segmentation — activity profiles",
        "Indiquez **combien de réservations**, **quel volume d'affaires** et **à quelle récence** remonte la dernière activité :": "Indicate **how many bookings**, **what business volume**, and **how recent** the last activity was:",
        "nous rapprochons ce comportement d'un **groupe-type** (ex. très fidèle, occasionnel, à relancer).": "we match this behavior to a **typical group** (e.g., very loyal, occasional, to re-engage).",
        "Beneficiaries (réservations, CA, récence…)": "Beneficiaries (bookings, revenue, recency…)",
        "Providers (charge, CA, récence…)": "Providers (load, revenue, recency…)",
        "Lecture": "Reading",
        "Fidélité RFM": "RFM Loyalty",
        "Beneficiaries & prestataires": "Beneficiaries & providers",
        
        # Additional cleanup
        "réservations": "bookings",
        "prévision": "forecast",
        "months": "months",
        "data": "data",
        "models traineds": "trained models",
        "model trained": "trained model",
        "database": "database",
        "métriques": "metrics",
        "prestataires": "providers",
    }
    
    # Apply translations
    for french, english in markdown_translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Markdown and remaining text translated!")
    print("📄 All French content now in English")

if __name__ == "__main__":
    translate_markdown()
