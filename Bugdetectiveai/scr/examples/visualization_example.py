"""
Example usage of visualization functions for BugDetectiveAI.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.visualization import plot_metrics_boxplots, plot_column_distribution


def create_sample_dataset():
    """Create a sample dataset with buggy code and multiple model responses."""
    
    sample_data = {
        "buggy_code": [
            "def add(a, b):\n    return a + b\n",
            "def multiply(x, y):\n    return x * y\n",
            "def divide(a, b):\n    return a / b\n",
            "def subtract(a, b):\n    return a - b\n",
            "def power(base, exp):\n    return base ** exp\n",
        ],
        "response_model_a": [
            "def add(a, b):\n    return a + b\n",  # Same as buggy
            "def multiply(x, y):\n    return x * y\n",  # Same as buggy
            "def divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b\n",  # Fixed
            "def subtract(a, b):\n    return a - b\n",  # Same as buggy
            "def power(base, exp):\n    return base ** exp\n",  # Same as buggy
        ],
        "response_model_b": [
            "def add(a, b):\n    if isinstance(a, str) or isinstance(b, str):\n        return str(a) + str(b)\n    return a + b\n",  # Enhanced
            "def multiply(x, y):\n    return x * y\n",  # Same as buggy
            "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Cannot divide by zero')\n    return a / b\n",  # Fixed differently
            "def subtract(a, b):\n    return a - b\n",  # Same as buggy
            "def power(base, exp):\n    if exp < 0:\n        return 1 / (base ** abs(exp))\n    return base ** exp\n",  # Enhanced
        ],
        "response_model_c": [
            "def add(a, b):\n    return a + b\n",  # Same as buggy
            "def multiply(x, y):\n    return x * y\n",  # Same as buggy
            "def divide(a, b):\n    return a / b\n",  # Same as buggy (no fix)
            "def subtract(a, b):\n    return a - b\n",  # Same as buggy
            "def power(base, exp):\n    return base ** exp\n",  # Same as buggy
        ]
    }
    
    return pd.DataFrame(sample_data)


def main():
    """Demonstrate visualization functions."""
    
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Example 1: Plot metrics boxplots with default settings
    print("\n" + "="*50)
    print("EXAMPLE 1: Plot metrics boxplots (auto-detect response columns)")
    print("="*50)
    
    try:
        plot_metrics_boxplots(df)
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    # Example 2: Plot metrics boxplots with custom settings
    print("\n" + "="*50)
    print("EXAMPLE 2: Plot metrics boxplots (custom settings)")
    print("="*50)
    
    try:
        plot_metrics_boxplots(
            df=df,
            reference_column="buggy_code",
            response_columns=["response_model_a", "response_model_b"],
            figsize=(15, 10),
            title_prefix="Custom Comparison"
        )
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    # Example 3: Plot column distribution
    print("\n" + "="*50)
    print("EXAMPLE 3: Plot column distribution")
    print("="*50)
    
    # Create a sample column with categories for distribution plotting
    df_with_categories = df.copy()
    df_with_categories['model_performance'] = ['good', 'good', 'excellent', 'good', 'excellent']
    
    try:
        plot_column_distribution(df_with_categories, column_name="model_performance", top_n=3)
    except Exception as e:
        print(f"Error in example 3: {e}")


if __name__ == "__main__":
    main() 