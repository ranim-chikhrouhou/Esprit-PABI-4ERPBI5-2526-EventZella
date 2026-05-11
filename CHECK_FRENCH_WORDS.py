#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check for remaining French words in streamlit_app.py
"""

import re
from pathlib import Path

def check_french_words():
    """Check for common French words in the file"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Common French words to check for
    french_patterns = [
        r'\bRégression\b',
        r'\bSéries temporelles\b',
        r'\bPrévision\b',
        r'\bCritère\b',
        r'\bDomaine\b',
        r'\bQualité\b',
        r'\bRègle\b',
        r'\bExactitude\b',
        r'\bStatuts\b',
        r'\bAucune métrique\b',
        r'\baucune métrique\b',
        r'\bAccès\b',
        r'\bAccéder\b',
        r'\bAperçu\b',
        r'\bdéconnecter\b',
        r'\brécapitulatif\b',
        r'\bréservation\b',
        r'\bmontants\b',
        r'\bindicateurs\b',
        r'\bécran\b',
        r'\bchamps\b',
        r'\bprofil\b',
        r'\bsegment\b',
        r'\bfidélité\b',
        r'\bBénéficiaires\b',
        r'\bPrestataires\b',
        r'\bÉchantillon\b',
        r'\bPérimètre\b',
        r'\bRepères\b',
        r'\bdétaillé\b',
        r'\btronqué\b',
        r'\bpondéré\b',
        r'\bpossibles\b',
        r'\btrouvée\b',
        r'\bcalculés\b',
        r'\bdepuis\b',
        r'\bintérêt\b',
        r'\bsavoir plus\b',
        # Unicode escape patterns
        r'\\u00e9',  # é
        r'\\u00e8',  # è
        r'\\u00ea',  # ê
        r'\\u00e0',  # à
        r'\\u00f4',  # ô
        r'\\u00fb',  # û
        r'\\u00e7',  # ç
        r'\\u00ee',  # î
    ]
    
    found_french = []
    
    for pattern in french_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # Get line number
            line_num = content[:match.start()].count('\n') + 1
            # Get context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            found_french.append({
                'pattern': pattern,
                'line': line_num,
                'match': match.group(),
                'context': context
            })
    
    if found_french:
        print(f"WARNING: Found {len(found_french)} potential French words/patterns:\n")
        for item in found_french[:20]:  # Show first 20
            print(f"Line {item['line']}: {item['match']}")
            print(f"  Pattern: {item['pattern']}")
            print(f"  Context: ...{item['context']}...")
            print()
    else:
        print("SUCCESS: No common French words found!")
        print("SUCCESS: The UI appears to be fully translated to English")
    
    return len(found_french)

if __name__ == "__main__":
    count = check_french_words()
    if count == 0:
        print("\nSUCCESS: Translation verification complete - UI is in English")
    else:
        print(f"\nWARNING: Found {count} potential French words - may need additional translation")
