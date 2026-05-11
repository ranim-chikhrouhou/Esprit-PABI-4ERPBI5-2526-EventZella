#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translate all markdown blocks and visible text
"""

from pathlib import Path

def translate_all_markdown():
    """Translate all markdown blocks and user-visible text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Translate large markdown blocks
    markdown_translations = {
        # ML_INTEREST_MARKDOWN
        '**Pourquoi le machine learning pour EventZilla ?**': '**Why machine learning for EventZilla?**',
        'Les données du **Database** (réservations, finances agrégées, volumes) permettent d\'**anticiper** les statuts et montants, de **segmenter** l\'offre et de **suivre** les tendances mensuelles — sans remplacer le métier, mais pour **prioriser** et **illustrer** les scénarios dans un cadre pédagogique et reproductible.': 'The **database** data (bookings, aggregated finances, volumes) allows us to **anticipate** statuses and amounts, **segment** offerings, and **track** monthly trends — not to replace business expertise, but to **prioritize** and **illustrate** scenarios in an educational and reproducible framework.',
        '**Ce studio** centralise les Prediction Systems traineds sur le même scope que vos notebooks (00→05) : testez-les ici avant toute mise en production.': '**This studio** centralizes models trained on the same scope as your notebooks (00→05): test them here before any production deployment.',
        
        # DEPLOY_CLASSIF_MARKDOWN
        '### À quoi sert cet écran ?': '### What is this screen for?',
        '**Objectif :** tester le **Prediction System de Status Prediction** (cancelled / confirmed / pending) sur un **profil de réservation** que vous composez.': '**Objective:** test the **booking status prediction model** (cancelled / confirmed / pending) on a **booking profile** that you compose.',
        
        # DEPLOY_REGR_MARKDOWN  
        '**Objectif :** estimer le **montant final** (`final_price`) d\'une réservation à partir de ses caractéristiques (prix service, budget événement, période, etc.).': '**Objective:** estimate the **final amount** (`final_price`) of a booking from its characteristics (service price, event budget, period, etc.).',
        
        # DEPLOY_TS_MARKDOWN
        '**Objectif :** visualiser **l\'évolution mensuelle** d\'indicateurs agrégés (volume d\'activité, chiffre d\'affaires, panier moyen) **calculés depuis le Database**, puis **projeter quelques mois** pour illustrer la dynamique observée.': '**Objective:** visualize **monthly evolution** of aggregated indicators (activity volume, revenue, average basket) **calculated from the database**, then **project a few months** to illustrate the observed dynamics.',
        '**Ce que vous pouvez tester :**': '**What you can test:**',
        '1. **Choose l\'indicateur** — volume d\'activité, CA mensuel ou panier moyen.': '1. **Choose the indicator** — activity volume, monthly revenue, or average basket.',
        '2. **Ajuster l\'horizon** — de 1 à 12 months de prévision.': '2. **Adjust the horizon** — from 1 to 12 months forecast.',
        '3. **Comparer visuellement** le train, la zone de validation et la prévision.': '3. **Visually compare** training, validation zone, and forecast.',
        '4. **Lire les métriques** — Prediction Error, Average Error, MAPE sur la fenêtre de test.': '4. **Read the metrics** — RMSE, MAE, MAPE on the test window.',
        '**Prediction Systems comparés :**': '**Models compared:**',
        '**Trend Analysis** (lissage exponentiel avec tendance) vs **Advanced Forecast** (autorégression + différenciation + moyenne mobile).': '**Trend Analysis** (exponential smoothing with trend) vs **Advanced Forecast** (autoregression + differencing + moving average).',
        'Le **Best System** est celui qui a le **Prediction Error le plus bas** sur la validation.': 'The **best model** is the one with the **lowest RMSE** on validation.',
        
        # Common French phrases in UI
        'données': 'data',
        'Database': 'database',
        'réservations': 'bookings',
        'finances agrégées': 'aggregated finances',
        'volumes': 'volumes',
        'anticiper': 'anticipate',
        'segmenter': 'segment',
        'suivre': 'track',
        'tendances mensuelles': 'monthly trends',
        'métier': 'business',
        'prioriser': 'prioritize',
        'illustrer': 'illustrate',
        'scénarios': 'scenarios',
        'cadre pédagogique': 'educational framework',
        'reproductible': 'reproducible',
        'Prediction Systems traineds': 'trained models',
        'mise en production': 'production deployment',
        'profil de réservation': 'booking profile',
        'montant final': 'final amount',
        'prix service': 'service price',
        'budget événement': 'event budget',
        'période': 'period',
        'évolution mensuelle': 'monthly evolution',
        'indicateurs agrégés': 'aggregated indicators',
        'volume d\'activité': 'activity volume',
        'chiffre d\'affaires': 'revenue',
        'panier moyen': 'average basket',
        'calculés depuis': 'calculated from',
        'projeter quelques mois': 'project a few months',
        'dynamique observée': 'observed dynamics',
        'l\'indicateur': 'the indicator',
        'CA mensuel': 'monthly revenue',
        'l\'horizon': 'the horizon',
        'prévision': 'forecast',
        'zone de validation': 'validation zone',
        'fenêtre de test': 'test window',
        'lissage exponentiel': 'exponential smoothing',
        'tendance': 'trend',
        'autorégression': 'autoregression',
        'différenciation': 'differencing',
        'moyenne mobile': 'moving average',
        'le plus bas': 'the lowest',
        'validation': 'validation',
    }
    
    # Apply translations
    for french, english in markdown_translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ All markdown and visible text translated!")
    print("📄 UI should now be fully in English")

if __name__ == "__main__":
    translate_all_markdown()
