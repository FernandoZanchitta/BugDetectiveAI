"""
Data loader module for BugDetectiveAI.

This module provides functions to load and process the buggy dataset
from pickle files.
"""

from .loader import load_buggy_dataset, load_stable_dataset, get_dataset_paths

__all__ = ["load_buggy_dataset", "load_stable_dataset", "get_dataset_paths"]
