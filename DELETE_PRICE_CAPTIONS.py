#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delete the caption texts from Price Estimation page
"""

from pathlib import Path

def delete_price_captions():
    """Delete caption texts after the form submit button"""
    
    file_path = Path("ML/streamlit_app.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and remove the caption block
    # Start marker: after form_submit_button
    # End marker: before "if submitted:"
    
    start_marker = 'submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)'
    end_marker = '    if submitted:'
    
    start_pos = content.find(start_marker)
    if start_pos == -1:
        print("ERROR: Could not find start marker")
        return False
    
    # Find the end of the line with start_marker
    start_line_end = content.find('\n', start_pos)
    
    # Find the "if submitted:" line after the captions
    end_pos = content.find(end_marker, start_line_end)
    if end_pos == -1:
        print("ERROR: Could not find end marker")
        return False
    
    # Replace the section between with just a newline
    new_content = content[:start_line_end + 1] + '\n' + content[end_pos:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("SUCCESS: Deleted all caption texts from Price Estimation form")
    return True

if __name__ == "__main__":
    success = delete_price_captions()
    if success:
        print("\nSUCCESS: Captions deleted!")
        print("SUCCESS: Please restart Streamlit")
    else:
        print("\nERROR: Deletion failed")
