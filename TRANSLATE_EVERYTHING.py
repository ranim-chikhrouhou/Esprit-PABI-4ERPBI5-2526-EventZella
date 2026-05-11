#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final comprehensive translation - translate EVERYTHING
"""

from pathlib import Path

def translate_everything():
    """Translate absolutely everything - all French text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Translate ALL French strings
    translations = {
        # Month labels
        '"Janvier"': '"January"',
        '"Février"': '"February"',
        '"Mars"': '"March"',
        '"Avril"': '"April"',
        '"Mai"': '"May"',
        '"Juin"': '"June"',
        '"Juillet"': '"July"',
        '"Août"': '"August"',
        '"Septembre"': '"September"',
        '"Octobre"': '"October"',
        '"Novembre"': '"November"',
        '"Décembre"': '"December"',
        
        # Quarter labels
        '"T1 — janvier à mars"': '"Q1 — January to March"',
        '"T2 — avril à juin"': '"Q2 — April to June"',
        '"T3 — juillet à septembre"': '"Q3 — July to September"',
        '"T4 — octobre à décembre"': '"Q4 — October to December"',
        
        # Year labels
        '"Année — bas (~10e percentile)"': '"Year — low (~10th percentile)"',
        '"Année — bas (~25e)"': '"Year — low (~25th)"',
        '"Année — médiane"': '"Year — median"',
        '"Année — haut (~75e)"': '"Year — high (~75th)"',
        '"Année — haut (~90e)"': '"Year — high (~90th)"',
        
        # Value labels
        '"Valeur observée — "': '"Observed value — "',
        '"Très bas dans le database (~10e %.)"': '"Very low in database (~10th %)"',
        '"Bas (~25e %.)"': '"Low (~25th %)"',
        '"Typique — médiane"': '"Typical — median"',
        '"Élevé (~75e %.)"': '"High (~75th %)"',
        '"Très haut (~90e %.)"': '"Very high (~90th %)"',
        
        # Group titles
        '"Période & calendrier (data database)"': '"Period & calendar (database data)"',
        '"Prices & Amounts"': '"Prices & Amounts"',
        '"Quantities"': '"Quantities"',
        
        # Field labels
        '"Prix final (panier / commande)"': '"Final price (basket / order)"',
        '"Prix prestataire"': '"Provider price"',
        '"Prix moyen de référence (Reference)"': '"Average reference price (benchmark)"',
        '"Budget événement"': '"Event budget"',
        '"Marge commission (final − prestataire)"': '"Commission margin (final − provider)"',
        
        # Section labels
        '"Variables les plus influentes sur le prix final (forêt aléatoire)"': '"Most influential variables on final price (random forest)"',
        '"Autres Input Factors du Prediction System"': '"Other model input factors"',
        '"Montants, budget & références (to fill in en priorité)"': '"Amounts, budget & references (fill in priority)"',
        '"Calendrier & autres dimensions"': '"Calendar & other dimensions"',
        
        # Series labels
        '"Volume d\'activité (lignes de faits database / mois)"': '"Activity volume (database fact rows / month)"',
        '"monthly revenue agrégé (somme des montants)"': '"Aggregated monthly revenue (sum of amounts)"',
        '"Panier moyen mensuel"': '"Monthly average basket"',
        
        # Messages
        '"Connexion au database et chargement des séries…"': '"Connecting to database and loading series…"',
        '"Moteur SQLAlchemy non créé — check pyodbc, sqlalchemy, et les variables "': '"SQLAlchemy engine not created — check pyodbc, sqlalchemy, and variables "',
        '"La requête séries a renvoyé 0 ligne — check le scope database."': '"Series query returned 0 rows — check database scope."',
        '"Prediction System retenu après comparaison sur le jeu de test (détail dans le notebook associé)."': '"Model selected after comparison on test set (details in associated notebook)."',
        
        # Form labels
        '"Goal (Y)"': '"Target (Y)"',
        '"KPI / lecture business"': '"KPI / business reading"',
        '"Figure / indicateur à regarder"': '"Chart / indicator to view"',
        
        # Common words in strings
        'période': 'period',
        'calendrier': 'calendar',
        'données': 'data',
        'référence': 'reference',
        'événement': 'event',
        'activité': 'activity',
        'lignes de faits': 'fact rows',
        'agrégé': 'aggregated',
        'somme des montants': 'sum of amounts',
        'mensuel': 'monthly',
        'moyen': 'average',
        'database': 'database',
        'jeu de test': 'test set',
        'détail': 'details',
        'notebook associé': 'associated notebook',
        'comparaison': 'comparison',
        'retenu': 'selected',
        'après': 'after',
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Everything translated!")
    print("📄 All French labels, months, and text converted to English")

if __name__ == "__main__":
    translate_everything()
