#!/usr/bin/env python3
"""
Script to resolve merge conflicts in Jupyter notebook files.
This script will:
1. Keep HEAD version for execution counts
2. Keep HEAD version for learning rate values
3. Keep HEAD version for outputs (cleaner)
"""

import json
import re
import sys

def resolve_notebook_conflicts(file_path):
    """Resolve merge conflicts in a Jupyter notebook file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match merge conflicts
    conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [a-f0-9]+'
    
    def resolve_conflict(match):
        head_content = match.group(1)
        other_content = match.group(2)
        
        # For execution_count conflicts, keep HEAD version
        if '"execution_count"' in head_content and '"execution_count"' in other_content:
            return head_content
        
        # For learning_rate conflicts, keep HEAD version (1e-3 vs 1e-2)
        if 'learning_rate_layers_3_4' in head_content and 'learning_rate_layers_3_4' in other_content:
            return head_content
        
        # For outputs conflicts, keep HEAD version (cleaner)
        if '"outputs"' in head_content and '"outputs"' in other_content:
            return head_content
        
        # Default: keep HEAD version
        return head_content
    
    # Replace all conflicts
    resolved_content = re.sub(conflict_pattern, resolve_conflict, content, flags=re.DOTALL)
    
    # Write back the resolved content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(resolved_content)
    
    print(f"Resolved conflicts in {file_path}")

if __name__ == "__main__":
    notebook_file = "conjunctive_small_lr_tiexp/conjuntive_small_lr copy.ipynb"
    resolve_notebook_conflicts(notebook_file)
