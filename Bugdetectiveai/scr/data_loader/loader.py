"""
Data loader functions for BugDetectiveAI dataset.

This module provides functions to load the buggy and stable datasets
from pickle files and return pandas DataFrames.
"""

import os
import pickle
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime


def get_dataset_paths(base_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get the paths to all dataset files.

    Args:
        base_path: Base path to the dataset. If None, uses default path.

    Returns:
        Dictionary with paths to all dataset files.
    """
    if base_path is None:
        # Default path based on the notebook structure
        base_path = "/Users/zanchitta/Developer/BugDetectiveAI/Bugdetectiveai/data/pytracebugs_dataset_v1"

    buggy_datapath = os.path.join(base_path, "buggy_dataset")
    stable_datapath = os.path.join(base_path, "stable_dataset")

    paths = {
        # Buggy dataset paths
        "bugfixes_test": os.path.join(buggy_datapath, "bugfixes_test.pickle"),
        "bugfixes_train": os.path.join(buggy_datapath, "bugfixes_train.pickle"),
        "bugfixes_valid": os.path.join(buggy_datapath, "bugfixes_valid.pickle"),
        # Stable dataset paths
        "stable_code_test": os.path.join(stable_datapath, "stable_code_test.pickle"),
        "stable_code_train": os.path.join(stable_datapath, "stable_code_train.pickle"),
        "stable_code_valid": os.path.join(stable_datapath, "stable_code_valid.pickle"),
    }

    return paths


def load_pickle_file(file_path: str) -> pd.DataFrame:
    """
    Load a pickle file and return a pandas DataFrame.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Pandas DataFrame loaded from the pickle file.

    Raises:
        FileNotFoundError: If the pickle file doesn't exist.
        ValueError: If the file cannot be loaded as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Ensure it's a DataFrame
        if isinstance(data, pd.DataFrame):
            return data
        else:
            # Try to convert to DataFrame if it's not already
            return pd.DataFrame(data)

    except Exception as e:
        raise ValueError(f"Error loading pickle file {file_path}: {str(e)}")


def save_pickle_file(df: pd.DataFrame, file_path: str) -> str:
    """
    Save a DataFrame to a pickle file.

    Args:
        df: DataFrame to save
        file_path: Path where to save the file

    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    with open(file_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Data saved to: {file_path}")
    return file_path


def save_data(
    df: pd.DataFrame, file_name: Optional[str] = None, data_path: Optional[str] = None
) -> str:
    """
    Save a DataFrame to pickle file with automatic naming.

    Args:
        df: DataFrame to save
        file_name: Name of the file (without .pkl extension).
                  If None, uses a descriptive name like 'sample_data_{num_rows}_rows_{response_names}.pkl'
        data_path: Directory to save the file.
                  If None, uses default checkpoints directory

    Returns:
        Full path to the saved file
    """
    # Set default path
    if data_path is None:
        data_path = (
            "/Users/zanchitta/Developer/BugDetectiveAI/Bugdetectiveai/data/checkpoints"
        )

    # Set default filename with descriptive info if not provided
    if file_name is None:
        num_rows = len(df)
        response_columns = [col for col in df.columns if "response_" in col]
        if response_columns:
            response_names = "_".join([col.replace("response_", "") for col in response_columns])
        else:
            response_names = "no_responses"
        file_name = f"sample_data_{num_rows}_rows_{response_names}.pkl"
    else:
        # Ensure .pkl extension
        if not file_name.endswith(".pkl"):
            file_name += ".pkl"

    # Full file path
    file_path = os.path.join(data_path, file_name)

    return save_pickle_file(df, file_path)

    # Full file path
    file_path = os.path.join(data_path, file_name)

    return save_pickle_file(df, file_path)


def load_data(file_name: str, data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from pickle file.

    Args:
        file_name: Name of the file (with or without .pkl extension)
        data_path: Directory where the file is located.
                  If None, uses default checkpoints directory

    Returns:
        Loaded DataFrame
    """
    # Set default path
    if data_path is None:
        data_path = (
            "/Users/zanchitta/Developer/BugDetectiveAI/Bugdetectiveai/data/checkpoints"
        )

    # Ensure .pkl extension
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"

    # Full file path
    file_path = os.path.join(data_path, file_name)

    return load_pickle_file(file_path)


def load_buggy_dataset(
    split: str = "test", base_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load buggy dataset for a specific split.

    Args:
        split: Dataset split ('train', 'validation', 'valid', or 'test').
        base_path: Base path to the dataset. If None, uses default path.

    Returns:
        Pandas DataFrame with buggy dataset.

    Raises:
        ValueError: If split is not valid.
        FileNotFoundError: If the dataset file doesn't exist.
    """
    # Normalize split name
    split_mapping = {
        "train": "train",
        "validation": "valid",
        "valid": "valid",
        "test": "test",
    }

    if split not in split_mapping:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: {list(split_mapping.keys())}"
        )

    normalized_split = split_mapping[split]

    # Get file paths
    paths = get_dataset_paths(base_path)
    file_key = f"bugfixes_{normalized_split}"

    if file_key not in paths:
        raise FileNotFoundError(f"No dataset file found for split '{split}'")

    file_path = paths[file_key]
    return load_pickle_file(file_path)


def load_stable_dataset(
    split: str = "test", base_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load stable dataset for a specific split.

    Args:
        split: Dataset split ('train', 'validation', 'valid', or 'test').
        base_path: Base path to the dataset. If None, uses default path.

    Returns:
        Pandas DataFrame with stable dataset.

    Raises:
        ValueError: If split is not valid.
        FileNotFoundError: If the dataset file doesn't exist.
    """
    # Normalize split name
    split_mapping = {
        "train": "train",
        "validation": "valid",
        "valid": "valid",
        "test": "test",
    }

    if split not in split_mapping:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: {list(split_mapping.keys())}"
        )

    normalized_split = split_mapping[split]

    # Get file paths
    paths = get_dataset_paths(base_path)
    file_key = f"stable_code_{normalized_split}"

    if file_key not in paths:
        raise FileNotFoundError(f"No dataset file found for split '{split}'")

    file_path = paths[file_key]
    return load_pickle_file(file_path)


def load_all_datasets(base_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all available datasets.

    Args:
        base_path: Base path to the dataset. If None, uses default path.

    Returns:
        Dictionary with all loaded datasets.
    """
    paths = get_dataset_paths(base_path)
    datasets = {}

    for name, path in paths.items():
        try:
            datasets[name] = load_pickle_file(path)
            print(f"✓ Loaded {name}: {len(datasets[name])} samples")
        except FileNotFoundError:
            print(f"✗ File not found: {name}")
        except Exception as e:
            print(f"✗ Error loading {name}: {str(e)}")

    return datasets


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about a loaded dataset.

    Args:
        df: Pandas DataFrame to analyze.

    Returns:
        Dictionary with dataset information.
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_rows": df.head(2).to_dict("records") if len(df) > 0 else [],
    }

    # Add specific info for buggy dataset
    if "before_merge" in df.columns and "after_merge" in df.columns:
        info["code_length_stats"] = {
            "before_merge_mean": df["before_merge"].str.len().mean(),
            "before_merge_std": df["before_merge"].str.len().std(),
            "after_merge_mean": df["after_merge"].str.len().mean(),
            "after_merge_std": df["after_merge"].str.len().std(),
        }

    return info


def filter_dataset_by_length(
    df: pd.DataFrame, column: str = "before_merge", max_length: int = 900
) -> pd.DataFrame:
    """
    Filter dataset by code length to avoid context window issues.

    Args:
        df: Pandas DataFrame to filter.
        column: Column name to check for length.
        max_length: Maximum allowed length.

    Returns:
        Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Add length column if it doesn't exist
    length_col = f"{column}_length"
    if length_col not in df.columns:
        df = df.copy()
        df[length_col] = df[column].str.len()

    # Filter by length
    filtered_df = df[df[length_col] < max_length].copy()

    print(f"Original dataset: {len(df)} samples")
    print(f"Filtered dataset: {len(filtered_df)} samples")
    print(f"Removed {len(df) - len(filtered_df)} samples with length >= {max_length}")

    return filtered_df


# Example usage and testing
def example_usage():
    """Example usage of the data loader functions."""
    print("=== BugDetectiveAI Data Loader Example ===\n")

    try:
        # Load buggy test dataset
        print("Loading buggy test dataset...")
        buggy_test = load_buggy_dataset("test")
        print(f"✓ Loaded {len(buggy_test)} samples")

        # Get dataset info
        info = get_dataset_info(buggy_test)
        print(f"\nDataset shape: {info['shape']}")
        print(f"Columns: {info['columns']}")

        # Show sample data
        if len(buggy_test) > 0:
            print(f"\nSample data:")
            sample = buggy_test.iloc[0]
            print(f"Filename: {sample.get('filename', 'N/A')}")
            print(f"Function: {sample.get('function_name', 'N/A')}")
            print(f"Before merge length: {len(sample.get('before_merge', ''))}")
            print(f"After merge length: {len(sample.get('after_merge', ''))}")

        # Load stable dataset
        print("\nLoading stable test dataset...")
        stable_test = load_stable_dataset("test")
        print(f"✓ Loaded {len(stable_test)} samples")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("1. Ensure the dataset files are in the correct location")
        print("2. Update the base_path in get_dataset_paths() if needed")
        print("3. Run the script again")

        # Show available paths
        print("\nExpected file paths:")
        paths = get_dataset_paths()
        for name, path in paths.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {name}: {path}")


if __name__ == "__main__":
    example_usage()
