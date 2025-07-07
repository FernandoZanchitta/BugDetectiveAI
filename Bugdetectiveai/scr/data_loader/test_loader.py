#!/usr/bin/env python3
"""
Simple test script for the data loader functions.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from loader import (
    get_dataset_paths,
    load_pickle_file,
    load_buggy_dataset,
    load_stable_dataset,
    get_dataset_info,
    filter_dataset_by_length,
)


def test_dataset_paths():
    """Test getting dataset paths."""
    print("Testing dataset paths...")
    paths = get_dataset_paths()

    print("Available dataset paths:")
    for name, path in paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")

    return paths


def test_load_buggy_dataset():
    """Test loading buggy dataset."""
    print("\nTesting buggy dataset loading...")

    try:
        # Test loading test split
        df = load_buggy_dataset("test")
        print(f"✓ Successfully loaded buggy test dataset: {len(df)} samples")

        # Get dataset info
        info = get_dataset_info(df)
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {info['columns']}")

        # Show sample
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"  Sample filename: {sample.get('filename', 'N/A')}")
            print(f"  Sample function: {sample.get('function_name', 'N/A')}")

        return df

    except Exception as e:
        print(f"✗ Error loading buggy dataset: {e}")
        return None


def test_load_stable_dataset():
    """Test loading stable dataset."""
    print("\nTesting stable dataset loading...")

    try:
        # Test loading test split
        df = load_stable_dataset("test")
        print(f"✓ Successfully loaded stable test dataset: {len(df)} samples")

        # Get dataset info
        info = get_dataset_info(df)
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {info['columns']}")

        return df

    except Exception as e:
        print(f"✗ Error loading stable dataset: {e}")
        return None


def test_filtering():
    """Test dataset filtering by length."""
    print("\nTesting dataset filtering...")

    try:
        # Load a small sample first
        df = load_buggy_dataset("test")

        if df is not None and len(df) > 0:
            # Filter by length
            filtered_df = filter_dataset_by_length(df, max_length=500)
            print(f"✓ Filtering completed successfully")

            return filtered_df
        else:
            print("✗ No data to filter")
            return None

    except Exception as e:
        print(f"✗ Error during filtering: {e}")
        return None


def main():
    """Run all tests."""
    print("=== BugDetectiveAI Data Loader Tests ===\n")

    # Test 1: Dataset paths
    paths = test_dataset_paths()

    # Test 2: Load buggy dataset
    buggy_df = test_load_buggy_dataset()

    # Test 3: Load stable dataset
    stable_df = test_load_stable_dataset()

    # Test 4: Filtering
    filtered_df = test_filtering()

    # Summary
    print("\n=== Test Summary ===")
    print(
        f"Dataset paths found: {len([p for p in paths.values() if os.path.exists(p)])}/{len(paths)}"
    )
    print(f"Buggy dataset loaded: {'✓' if buggy_df is not None else '✗'}")
    print(f"Stable dataset loaded: {'✓' if stable_df is not None else '✗'}")
    print(f"Filtering test passed: {'✓' if filtered_df is not None else '✗'}")


if __name__ == "__main__":
    main()
