#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final fix - replace text with Unicode escapes
"""

from pathlib import Path

def fix_final():
    """Fix French text that has Unicode escape sequences."""
    
    file_path = Path("ML/streamlit_app.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Replace with exact Unicode escape sequences as they appear in file
    replacements = {
        # Section header with Unicode escapes
        'section_header("Mod\\u00e8les ML d\\u00e9ploy\\u00e9s", "Quatre familles de mod\\u00e8les entra\\u00een\\u00e9s sur le database")': 
        'section_header("Deployed ML Models", "Four model families trained on the database")',
        
        # Model cards with Unicode escapes
        '("R\\u00e9gression", "Price prediction",': 
        '("Regression", "Price prediction",',
        
        'f"{k_seg} segments clients"': 
        'f"{k_seg} customer segments"',
        
        '("S\\u00e9ries temporelles", f"Pr\\u00e9vision {ts_horizon} mois",': 
        '("Time Series", f"{ts_horizon}-month forecast",',
        
        # Expander
        '"En savoir plus \\u2014 int\\u00e9r\\u00eat du ML pour EventZilla"': 
        '"Learn more — ML benefits for EventZilla"',
        
        # Summary page
        '"R\\u00e9capitulatif des mod\\u00e8les"': 
        '"Models Summary"',
        
        '"Vue d\'ensemble des **quatre familles ML** d\\u00e9ploy\\u00e9es : performance, mod\\u00e8le Best System et indicateur m\\u00e9tier."': 
        '"Overview of **four deployed ML families**: performance, best model, and business indicator."',
        
        '("Synth\\u00e8se",)': 
        '("Summary",)',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    print("✅ Final fix applied!")
    print("📄 Unicode escape sequences replaced")

if __name__ == "__main__":
    fix_final()
