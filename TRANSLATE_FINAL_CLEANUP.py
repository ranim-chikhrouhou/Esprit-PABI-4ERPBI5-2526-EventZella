#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final cleanup - catch any remaining French text
"""

from pathlib import Path

def final_cleanup():
    """Final cleanup of any remaining French text."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Final translations
    final_translations = {
        "Simulation de **nombreuses** attributions dans l'espace d'entrée du KMeans (Gaussienne multivariée)": "Simulation of **many** assignments in the KMeans input space (multivariate Gaussian)",
        "donne une idée de la **taille relative** des segments, pas les volumes métier bruts.": "gives an idea of the **relative size** of segments, not raw business volumes.",
        "Calculer la répartition simulée": "Calculate simulated distribution",
        "nombreuses": "many",
        "métier": "business",
        "bruts": "raw",
        "taille relative": "relative size",
        "répartition": "distribution",
        "simulée": "simulated",
    }
    
    # Apply translations
    for french, english in final_translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ Final cleanup complete!")
    print("📄 All remaining French text translated")

if __name__ == "__main__":
    final_cleanup()
