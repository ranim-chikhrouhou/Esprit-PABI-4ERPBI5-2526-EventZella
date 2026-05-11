#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix remaining French strings in UI elements (st.markdown, st.caption, etc.)
"""

import re
from pathlib import Path

def fix_ui_strings():
    """Fix French strings in Streamlit UI elements"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # UI-specific translations
    ui_translations = {
        # Caption and info strings
        'st.caption("Statuts possibles (Y) : **"': 'st.caption("Possible statuses (Y): **"',
        "Statuts possibles (Y) : **": "Possible statuses (Y): **",
        
        # Expander titles
        'with st.expander("Rappel pédagogique — détail (optionnel)"': 'with st.expander("Educational reminder — details (optional)"',
        "Rappel pédagogique — détail (optionnel)": "Educational reminder — details (optional)",
        "Rappel p\\u00e9dagogique \\u2014 d\\u00e9tail \\(optionnel\\)": "Educational reminder — details (optional)",
        
        # Form section headers
        'st.markdown("##### Saisie")': 'st.markdown("##### Input")',
        "##### Saisie": "##### Input",
        
        # Error messages
        'st.error("Métriques régression incomplètes (pas de cible documentée).")': 'st.error("Incomplete regression metrics (no documented target).")',
        "Métriques régression incomplètes (pas de cible documentée).": "Incomplete regression metrics (no documented target).",
        "M\\u00e9triques r\\u00e9gression incompl\\u00e8tes \\(pas de cible document\\u00e9e\\).": "Incomplete regression metrics (no documented target).",
        
        # Predictor labels
        'st.markdown("##### Prédicteurs (X) — saisie numérique ; benchmark en liste déroulante")': 'st.markdown("##### Predictors (X) — numeric input; benchmark as dropdown")',
        "##### Prédicteurs (X) — saisie numérique ; benchmark en liste déroulante": "##### Predictors (X) — numeric input; benchmark as dropdown",
        "##### Pr\\u00e9dicteurs \\(X\\) \\u2014 saisie num\\u00e9rique ; benchmark en liste d\\u00e9roulante": "##### Predictors (X) — numeric input; benchmark as dropdown",
        
        # Scope errors
        'st.error("Métriques manquantes pour ce périmètre.")': 'st.error("Missing metrics for this scope.")',
        "Métriques manquantes pour ce périmètre.": "Missing metrics for this scope.",
        "M\\u00e9triques manquantes pour ce p\\u00e9rim\\u00e8tre.": "Missing metrics for this scope.",
        
        # Model info
        'st.info("Modèle K-Means absent — métriques JSON uniquement.")': 'st.info("K-Means model missing — JSON metrics only.")',
        "Modèle K-Means absent — métriques JSON uniquement.": "K-Means model missing — JSON metrics only.",
        "Mod\\u00e8le K-Means absent \\u2014 m\\u00e9triques JSON uniquement.": "K-Means model missing — JSON metrics only.",
        
        # Segment headers
        'st.markdown(f"**Segment {i} — {_head}**")': 'st.markdown(f"**Segment {i} — {_head}**")',
        
        # Indicator labels
        'st.markdown("##### Indicateurs à renseigner")': 'st.markdown("##### Indicators to fill in")',
        "##### Indicateurs à renseigner": "##### Indicators to fill in",
        "##### Indicateurs \\u00e0 renseigner": "##### Indicators to fill in",
        
        # Caption text
        'st.caption(\n                "Les valeurs proposées correspondent aux **médianes** du jeu d\'apprentissage — modifiez-les pour simuler un cas."': 'st.caption(\n                "Proposed values correspond to **medians** of training set — modify them to simulate a case."',
        "Les valeurs proposées correspondent aux **médianes** du jeu d'apprentissage — modifiez-les pour simuler un cas.": "Proposed values correspond to **medians** of training set — modify them to simulate a case.",
        "Les valeurs propos\\u00e9es correspondent aux \\*\\*m\\u00e9dianes\\*\\* du jeu d'apprentissage \\u2014 modifiez-les pour simuler un cas.": "Proposed values correspond to **medians** of training set — modify them to simulate a case.",
        
        # Distribution headers
        'st.markdown("##### Répartition illustrative des segments")': 'st.markdown("##### Illustrative segment distribution")',
        "##### Répartition illustrative des segments": "##### Illustrative segment distribution",
        "##### R\\u00e9partition illustrative des segments": "##### Illustrative segment distribution",
        
        # Horizon caption
        'st.caption(f"Horizon : {m.get(\'horizon\', \'?\')} mois")': 'st.caption(f"Horizon: {m.get(\'horizon\', \'?\')} months")',
        "Horizon : {m.get('horizon', '?')} mois": "Horizon: {m.get('horizon', '?')} months",
        "Horizon : ": "Horizon: ",
        " mois\"": " months\"",
        
        # Success messages
        'st.success(f"**Données DW chargées** — {len(df_ts)} mois d\'historique disponibles.")': 'st.success(f"**DW data loaded** — {len(df_ts)} months of history available.")',
        "**Données DW chargées** — {len(df_ts)} mois d'historique disponibles.": "**DW data loaded** — {len(df_ts)} months of history available.",
        "Données DW chargées": "DW data loaded",
        "mois d'historique disponibles": "months of history available",
        "mois d\\'historique disponibles": "months of history available",
        
        # Info messages
        'st.info("L\'historique est trop court (< 6 mois) pour ajuster les modèles de prévision.")': 'st.info("History is too short (< 6 months) to fit forecasting models.")',
        "L'historique est trop court (< 6 mois) pour ajuster les modèles de prévision.": "History is too short (< 6 months) to fit forecasting models.",
        "L'historique est trop court": "History is too short",
        "pour ajuster les modèles de prévision": "to fit forecasting models",
        
        # Additional patterns
        "jeu d'apprentissage": "training set",
        "jeu d\\'apprentissage": "training set",
        "modèles de prévision": "forecasting models",
        "mod\\u00e8les de pr\\u00e9vision": "forecasting models",
        "d'historique disponibles": "of history available",
        "d\\u2019historique disponibles": "of history available",
    }
    
    # Apply all translations
    for french, english in ui_translations.items():
        content = content.replace(french, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: UI strings fixed!")
    print(f"SUCCESS: File updated: {file_path}")
    return True

if __name__ == "__main__":
    success = fix_ui_strings()
    if success:
        print("\nSUCCESS: All UI French strings have been translated")
        print("SUCCESS: Please restart the Streamlit app")
    else:
        print("\nERROR: Fix failed")
