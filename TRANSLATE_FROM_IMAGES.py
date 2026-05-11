#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translate French text visible in the screenshots
"""

from pathlib import Path

def translate_from_images():
    """Translate all French text visible in the user's screenshots."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding="utf-8")
    
    # Translations based on screenshots
    translations = {
        # Section headers
        '"Modèles ML déployés"': '"Deployed ML Models"',
        '"Quatre familles de modèles entraînés sur le database"': '"Four model families trained on the database"',
        
        # Model cards
        '"3 segments clients"': '"3 customer segments"',
        '"Prévision 3 mois"': '"3-month forecast"',
        
        # Section titles
        '"SÉRIES TEMPORELLES"': '"TIME SERIES"',
        
        # Navigation section
        '"Accéder aux écrans de test interactif"': '"Access interactive test screens"',
        
        # Buttons
        '"Régression"': '"Regression"',
        '"Séries temporelles"': '"Time Series"',
        '"Voir le récapitulatif"': '"View Summary"',
        
        # Expander
        '"En savoir plus — intérêt du ML pour EventZilla"': '"Learn more — ML benefits for EventZilla"',
        
        # Inside expander
        '"Les data du database"': '"The database data"',
        '"permettent d\'anticipate les statuts et montants, de segment l\'offre et de track les monthly trends — sans remplacer le business, mais pour prioritize et illustrate les scenarios dans un educational framework et reproducible."': '"allows us to anticipate statuses and amounts, segment offerings, and track monthly trends — not to replace business expertise, but to prioritize and illustrate scenarios in an educational and reproducible framework."',
        '"Ce studio centralise les modèles entraînés sur le même scope que vos notebooks (00→05) : testez-les ici avant toute production deployment."': '"This studio centralizes models trained on the same scope as your notebooks (00→05): test them here before any production deployment."',
        
        # More text from expander
        'Les data du ': 'The ',
        ' (bookings, aggregated finances, volumes) permettent d\'': ' (bookings, aggregated finances, volumes) allows us to ',
        'anticipate les statuts et montants, de ': 'anticipate statuses and amounts, ',
        'segment l\'offre et de ': 'segment offerings and ',
        'track les monthly trends — sans remplacer le business, mais pour ': 'track monthly trends — not to replace business expertise, but to ',
        'prioritize et ': 'prioritize and ',
        'illustrate les scenarios dans un educational framework et reproducible.': 'illustrate scenarios in an educational and reproducible framework.',
        
        'Ce studio centralise les modèles entraînés sur le même scope que vos notebooks (00→05) : testez-les ici avant toute production deployment.': 'This studio centralizes models trained on the same scope as your notebooks (00→05): test them here before any production deployment.',
        
        # Additional common phrases
        'entraînés sur le': 'trained on the',
        'testez-les ici avant toute': 'test them here before any',
        'production deployment': 'production deployment',
        'écrans de test interactif': 'interactive test screens',
        'récapitulatif': 'summary',
    }
    
    # Apply translations
    for french, english in translations.items():
        content = content.replace(french, english)
    
    # Write back
    file_path.write_text(content, encoding="utf-8")
    print("✅ All text from screenshots translated!")
    print("📄 Section headers, buttons, and expander text now in English")

if __name__ == "__main__":
    translate_from_images()
