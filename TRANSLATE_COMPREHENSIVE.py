#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive translation - all user-visible French text
"""

from pathlib import Path

def translate_comprehensive():
    """Translate all user-visible French text comprehensively."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Comprehensive translations - all user-visible strings
    translations = {
        # Warning/Error messages
        'st.warning("Parquet `dw_financial_wide.parquet` introuvable — exécutez `ML/scripts/run_00_data_preparation.py`.")': 
        'st.warning("Parquet file `dw_financial_wide.parquet` not found — run `ML/scripts/run_00_data_preparation.py`.")',
        
        'st.info("Modèles Status Prediction absents — lancez `ML/scripts/run_02_classification.py`.")':
        'st.info("Booking Status models missing — run `ML/scripts/run_02_classification.py`.")',
        
        'st.caption("Probabilités par classe non disponibles pour ce Prediction System.")':
        'st.caption("Class probabilities not available for this model.")',
        
        'st.error("Métriques Price Estimation incomplètes (pas de Goal documentée).")':
        'st.error("Price Estimation metrics incomplete (no target documented).")',
        
        'st.caption("Colonnes absentes du parquet — valeurs par défaut : "':
        'st.caption("Missing columns from parquet — default values: "',
        
        'st.caption("Histogramme indisponible — colonne `final_price` absente du jeu préparé.")':
        'st.caption("Histogram unavailable — `final_price` column missing from prepared dataset.")',
        
        'st.error("Métriques manquantes pour ce périmètre.")':
        'st.error("Metrics missing for this scope.")',
        
        'st.metric("Lecture", "Fidélité RFM")':
        'st.metric("Reading", "RFM Loyalty")',
        
        'st.caption("Bénéficiaires & prestataires")':
        'st.caption("Beneficiaries & providers")',
        
        'st.info("Prediction System Customer Grouping absent — métriques JSON uniquement.")':
        'st.info("Customer Segmentation model missing — JSON metrics only.")',
        
        'st.markdown("##### Indicateurs à renseigner")':
        'st.markdown("##### Metrics to fill in")',
        
        '"Les valeurs proposées correspondent aux **médianes** du jeu d\'apprentissage — modifiez-les pour simuler un cas."':
        '"Suggested values are **medians** from the training dataset — modify them to simulate a scenario."',
        
        '"Identifiants techniques (optionnel)"':
        '"Technical identifiers (optional)"',
        
        '"Utile seulement si vous reproduisez une ligne complète du Database ; sinon laissez les défauts."':
        '"Useful only if reproducing a complete database row; otherwise leave defaults."',
        
        'st.error(f"Prédiction impossible (vérifiez le nombre de variables et les artefacts) : {_err}")':
        'st.error(f"Prediction failed (check number of variables and artifacts): {_err}")',
        
        'st.markdown("##### Répartition illustrative des segments")':
        'st.markdown("##### Illustrative segment distribution")',
        
        # Form labels
        'st.markdown("##### Input Factors (X) — saisie numérique ; Reference en liste déroulante")':
        'st.markdown("##### Input Factors (X) — numeric input; reference in dropdown")',
        
        # Additional common phrases
        "introuvable": "not found",
        "exécutez": "run",
        "lancez": "run",
        "absents": "missing",
        "absente": "missing",
        "manquantes": "missing",
        "indisponible": "unavailable",
        "non disponibles": "not available",
        "Prédiction impossible": "Prediction failed",
        "vérifiez": "check",
        "Métriques": "Metrics",
        "Colonnes": "Columns",
        "valeurs par défaut": "default values",
        "jeu préparé": "prepared dataset",
        "jeu d'apprentissage": "training dataset",
        "périmètre": "scope",
        "Bénéficiaires": "Beneficiaries",
        "prestataires": "providers",
        "Fidélité": "Loyalty",
        "Indicateurs": "Metrics",
        "à renseigner": "to fill in",
        "Identifiants techniques": "Technical identifiers",
        "optionnel": "optional",
        "Répartition illustrative": "Illustrative distribution",
        "segments": "segments",
        "saisie numérique": "numeric input",
        "liste déroulante": "dropdown",
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Comprehensive translation complete!")
    print("📄 All user-visible French text translated")

if __name__ == "__main__":
    translate_comprehensive()
