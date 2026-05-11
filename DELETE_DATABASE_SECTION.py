#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delete the entire "Données du database" section from page_timeseries
"""

from pathlib import Path

def delete_database_section():
    """Delete the database connection section"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the start and end of the section to delete
    start_marker = "    # --- Connexion database"
    end_marker = "def main():"
    
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if start_marker in line and start_idx is None:
            start_idx = i
        if end_marker in line and start_idx is not None:
            end_idx = i
            break
    
    if start_idx is not None and end_idx is not None:
        # Keep everything before the section and after (including def main())
        new_lines = lines[:start_idx] + ['\n\n'] + lines[end_idx:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"SUCCESS: Deleted database section (lines {start_idx+1} to {end_idx})")
        print(f"SUCCESS: Removed {end_idx - start_idx} lines")
        return True
    else:
        print("ERROR: Could not find section markers")
        return False

if __name__ == "__main__":
    success = delete_database_section()
    if success:
        print("\nSUCCESS: Database section deleted!")
        print("SUCCESS: Please restart Streamlit")
    else:
        print("\nERROR: Deletion failed")
