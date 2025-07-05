"""
Simple pickle utilities for saving and loading DataFrames in BugDetectiveAI.
"""

import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
import os


def save_data(df: pd.DataFrame, file_name: str | None = None, data_path: str | None = None) -> str:
    """
    Save a DataFrame to pickle file.
    
    Args:
        df: DataFrame to save
        file_name: Name of the file (without .pkl extension). 
                  If None, uses 'data_{current_timestamp}'
        data_path: Directory to save the file. 
                  If None, uses '/Users/zanchitta/Developer/BugDetectiveAI/data/checkpoints'
    
    Returns:
        Full path to the saved file
    """
    # Set default path
    if data_path is None:
        data_path = '/Users/zanchitta/Developer/BugDetectiveAI/Bugdetectiveai/data/checkpoints'
    
    # Set default filename
    if file_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'data_{timestamp}'
    
    # Ensure .pkl extension
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    
    # Create directory if it doesn't exist
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    # Full file path
    file_path = os.path.join(data_path, file_name)
    
    # Save DataFrame
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Data saved to: {file_path}")
    return file_path


def load_data(file_name: str, data_path: str | None = None) -> pd.DataFrame:
    """
    Load a DataFrame from pickle file.
    
    Args:
        file_name: Name of the file (with or without .pkl extension)
        data_path: Directory where the file is located. 
                  If None, uses '/Users/zanchitta/Developer/BugDetectiveAI/data/checkpoints'
    
    Returns:
        Loaded DataFrame
    """
    # Set default path
    if data_path is None:
        data_path = '/Users/zanchitta/Developer/BugDetectiveAI/data/checkpoints'
    
    # Ensure .pkl extension
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    
    # Full file path
    file_path = os.path.join(data_path, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load DataFrame
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Data loaded from: {file_path}")
    return df 