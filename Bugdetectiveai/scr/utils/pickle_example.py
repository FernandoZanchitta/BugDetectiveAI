"""
Simple example of using pickle utilities for saving and loading DataFrames.
"""

import pandas as pd
from data_loader.loader import save_data, load_data


def example_usage():
    """Simple example of saving and loading a DataFrame."""
    
    # Create a sample DataFrame
    df = pd.DataFrame({
        'before_merge': [
            'def add(a, b):\n    return a + b',
            'def multiply(x, y):\n    return x * y'
        ],
        'full_traceback': [
            'TypeError: unsupported operand type(s) for +: \'int\' and \'str\'',
            'NameError: name \'y\' is not defined'
        ],
        'corrected_code': [
            'def add(a, b):\n    return int(a) + int(b)',
            'def multiply(x, y):\n    return x * y'
        ]
    })
    
    print("Original DataFrame:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Save with default settings (auto-generated filename with timestamp)
    saved_path = save_data(df)
    print(f"\nSaved to: {saved_path}")
    
    # Save with custom filename
    custom_path = save_data(df, file_name="my_custom_data")
    print(f"Saved with custom name to: {custom_path}")
    
    # Load the data back
    loaded_df = load_data("my_custom_data")
    print(f"\nLoaded DataFrame:")
    print(loaded_df)
    print(f"Shape: {loaded_df.shape}")
    
    # Verify they're the same
    print(f"\nDataFrames are identical: {df.equals(loaded_df)}")


if __name__ == "__main__":
    example_usage() 