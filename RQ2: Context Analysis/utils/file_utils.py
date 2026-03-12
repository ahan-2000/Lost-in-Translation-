"""
File and directory utilities
"""

import os
from datetime import datetime
from typing import Optional


def create_results_folder(base_name: str = "results", 
                         base_path: Optional[str] = None,
                         subdirs: Optional[list] = None) -> str:
    """
    Create dedicated folder for analysis results
    
    Args:
        base_name: Base name for the results folder
        base_path: Base path for results (default: current directory)
        subdirs: List of subdirectories to create (e.g., ['plots', 'data'])
        
    Returns:
        Path to created results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if base_path is None:
        results_dir = f"{base_name}_{timestamp}"
    else:
        results_dir = os.path.join(base_path, f"{base_name}_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    return results_dir

