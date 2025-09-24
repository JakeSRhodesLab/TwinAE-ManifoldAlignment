"""
Path utilities for the Graph Manifold Alignment package.

This module provides utilities to get proper relative paths to data directories
from anywhere in the package structure.
"""

import os
from pathlib import Path

def get_project_root():
    """
    Get the project root directory path.
    
    Returns:
        Path: Path to the project root directory
    """
    # Get the directory containing this file
    current_file = Path(__file__)
    
    # Navigate up the directory structure to find the project root
    # From: src/graph_manifold_alignment/helpers/path_utils.py
    # To:   project_root
    project_root = current_file.parent.parent.parent.parent
    
    return project_root.resolve()

def get_resources_path():
    """
    Get the path to the Resources directory.
    
    Returns:
        Path: Path to the Resources directory
    """
    return get_project_root() / "Resources"

def get_data_path():
    """
    Get the path to the data directory.
    
    Returns:
        Path: Path to the data directory
    """
    return get_project_root() / "data"

def get_outputs_path():
    """
    Get the path to the outputs directory.
    
    Returns:
        Path: Path to the outputs directory
    """
    return get_project_root() / "outputs"

def get_classification_csv_path():
    """
    Get the path to the Classification CSV directory.
    
    Returns:
        Path: Path to the Resources/Classification_CSV directory
    """
    return get_resources_path() / "Classification_CSV"

def get_regression_csv_path():
    """
    Get the path to the Regression CSV directory.
    
    Returns:
        Path: Path to the Resources/Regression_CSV directory
    """
    return get_resources_path() / "Regression_CSV"

def get_results_path():
    """
    Get the path to the Results directory (create outputs/results if doesn't exist).
    
    Returns:
        Path: Path to the outputs/results directory
    """
    results_path = get_outputs_path() / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

def get_splits_data_path():
    """
    Get the path to the Splits_Data directory.
    
    Returns:
        Path: Path to the outputs/results/Splits_Data directory
    """
    splits_path = get_results_path() / "Splits_Data"
    splits_path.mkdir(parents=True, exist_ok=True)
    return splits_path

def get_mantel_path():
    """
    Get the path to the Mantel results directory.
    
    Returns:
        Path: Path to the outputs/results/Mantel directory
    """
    mantel_path = get_results_path() / "Mantel"
    mantel_path.mkdir(parents=True, exist_ok=True)
    return mantel_path

def get_mantel_lam_path():
    """
    Get the path to the Mantel_lam results directory.
    
    Returns:
        Path: Path to the outputs/results/Mantel_lam directory
    """
    mantel_lam_path = get_results_path() / "Mantel_lam"
    mantel_lam_path.mkdir(parents=True, exist_ok=True)
    return mantel_lam_path

def get_grae_builds_path():
    """
    Get the path to the Grae_Builds results directory.
    
    Returns:
        Path: Path to the outputs/results/Grae_Builds directory
    """
    grae_path = get_results_path() / "Grae_Builds"
    grae_path.mkdir(parents=True, exist_ok=True)
    return grae_path
