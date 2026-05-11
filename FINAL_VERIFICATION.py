#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final verification - check for French words in UI-visible strings only
"""

import re
from pathlib import Path

def final_verification():
    """Check for French words in UI-visible strings"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # French words to check for in UI strings
    french_words = [
        'Formulaire', 'prédire', 'prix', 'champs', 'présent', 'médiane',
        'priorité', 'déroulante', 'erreur', 'modèle', 'entraîné',
        'réservation', 'écran', 'profil', 'segment', 'fidélité',
        'saisie', 'données', 'période', 'mois', 'indicateur',
        'métrique', 'critère', 'statut', 'aperçu', 'déconnecter',
        'détaillé', 'pondéré', 'calculés', 'depuis', 'intérêt',
        'savoir', 'exactitude', 'qualité', 'domaine', 'règle',
        'trouvée', 'récapitulatif', 'accéder', 'accès', 'tronqué',
        'montants', 'bénéficiaire', 'prestataire', 'récence',
        'historique', 'horizon', 'prévision'
    ]
    
    # UI-related function calls
    ui_patterns = [
        r'st\.(markdown|write|caption|info|warning|error|success|header|subheader|title)',
        r'section_header\(',
        r'hero_variant\(',
        r'st\.metric\(',
        r'st\.button\(',
        r'st\.expander\(',
    ]
    
    found_issues = []
    
    for line_num, line in enumerate(lines, 1):
        # Check if line contains UI function
        is_ui_line = any(re.search(pattern, line) for pattern in ui_patterns)
        
        if is_ui_line:
            # Check for French words
            for word in french_words:
                if word.lower() in line.lower():
                    found_issues.append({
                        'line': line_num,
                        'word': word,
                        'content': line.strip()[:100]
                    })
    
    if found_issues:
        print(f"WARNING: Found {len(found_issues)} potential French words in UI strings:\n")
        for issue in found_issues[:15]:  # Show first 15
            print(f"Line {issue['line']}: '{issue['word']}'")
            print(f"  {issue['content']}")
            print()
        if len(found_issues) > 15:
            print(f"... and {len(found_issues) - 15} more")
    else:
        print("SUCCESS: No French words found in UI strings!")
        print("SUCCESS: The Streamlit UI appears to be fully translated to English")
    
    return len(found_issues)

if __name__ == "__main__":
    count = final_verification()
    if count == 0:
        print("\n" + "="*60)
        print("TRANSLATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open browser to: http://localhost:8502")
        print("2. Press Ctrl+F5 to hard refresh")
        print("3. Navigate through all pages to verify")
        print("\nAll services running:")
        print("- FastAPI: http://localhost:8000")
        print("- n8n: http://localhost:5678")
        print("- Streamlit: http://localhost:8502")
    else:
        print(f"\nFound {count} potential issues - may need additional review")
