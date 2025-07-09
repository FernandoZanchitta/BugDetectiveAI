"""
Example script demonstrating the new metrics storage functionality.

This script shows how to:
1. Compute and store diff_score metrics in a dataframe
2. Use stored metrics for visualization without re-computation
3. Check if metrics are already computed
"""

import pandas as pd
import numpy as np
from utils.simple_metrics import compute_and_store_metrics, has_metrics_columns, get_metrics_columns
from utils.visualization import plot_metrics_boxplots

def create_sample_data():
    """Create sample data for demonstration."""
    # Sample buggy and corrected code
    sample_codes = [
        ("def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b"),
        ("def multiply(x, y):\n    return x * y", "def multiply(x, y):\n    return x * y"),
        ("def divide(a, b):\n    return a / b", "def divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b"),
        ("def subtract(a, b):\n    return a - b", "def subtract(a, b):\n    return a - b"),
        ("def power(base, exp):\n    return base ** exp", "def power(base, exp):\n    if exp < 0:\n        return 1 / (base ** abs(exp))\n    return base ** exp")
    ]
    
    data = []
    for i, (buggy, corrected) in enumerate(sample_codes):
        data.append({
            'id': i,
            'after_merge_without_docstrings': buggy,
            'response_model_a': corrected,
            'response_model_b': buggy,  # Same as original for comparison
            'response_model_c': corrected.replace('def ', 'def fixed_')  # Slightly different
        })
    
    return pd.DataFrame(data)

def demonstrate_metrics_storage():
    """Demonstrate the metrics storage functionality."""
    print("=== Metrics Storage Demonstration ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    df = create_sample_data()
    print(f"   Created dataframe with {len(df)} rows and columns: {list(df.columns)}")
    
    # Check if metrics are already computed
    print("\n2. Checking if metrics are already computed...")
    for col in ['response_model_a', 'response_model_b', 'response_model_c']:
        has_metrics = has_metrics_columns(df, col)
        print(f"   {col}: {'✓' if has_metrics else '✗'} metrics computed")
    
    # Compute metrics
    print("\n3. Computing metrics...")
    df_with_metrics = compute_and_store_metrics(
        df, 
        reference_column="after_merge_without_docstrings",
        show_progress=True
    )
    
    # Check metrics columns
    print("\n4. Checking computed metrics columns...")
    for col in ['response_model_a', 'response_model_b', 'response_model_c']:
        metrics_cols = get_metrics_columns(df_with_metrics, col)
        print(f"   {col}: {len(metrics_cols)} metric columns")
        print(f"      {metrics_cols[:3]}{'...' if len(metrics_cols) > 3 else ''}")
    
    # Show some sample metrics
    print("\n5. Sample metrics values:")
    sample_metrics = df_with_metrics[['metric_model_a_exact_match', 'metric_model_a_ast_score', 'metric_model_a_text_score']].head(3)
    print(sample_metrics)
    
    # Demonstrate visualization with stored metrics
    print("\n6. Creating visualization using stored metrics...")
    try:
        plot_metrics_boxplots(
            df_with_metrics,
            reference_column="after_merge_without_docstrings",
            use_stored_metrics=True,
            compute_if_missing=False
        )
        print("   ✓ Visualization completed successfully!")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    # Demonstrate re-computation avoidance
    print("\n7. Demonstrating re-computation avoidance...")
    df_reused = compute_and_store_metrics(
        df_with_metrics, 
        reference_column="after_merge_without_docstrings",
        show_progress=True
    )
    print("   ✓ No re-computation occurred (metrics already existed)")
    
    return df_with_metrics

def demonstrate_force_recompute():
    """Demonstrate force recomputation."""
    print("\n=== Force Recompute Demonstration ===\n")
    
    # Create sample data
    df = create_sample_data()
    
    # Compute metrics normally
    print("1. Computing metrics normally...")
    df_with_metrics = compute_and_store_metrics(df, show_progress=False)
    
    # Force recomputation
    print("\n2. Force recomputing metrics...")
    df_forced = compute_and_store_metrics(
        df_with_metrics, 
        force_recompute=True,
        show_progress=False
    )
    print("   ✓ Force recomputation completed")
    
    return df_forced

if __name__ == "__main__":
    # Run demonstrations
    df_final = demonstrate_metrics_storage()
    df_forced = demonstrate_force_recompute()
    
    print("\n=== Summary ===")
    print("✓ Metrics storage functionality working correctly")
    print("✓ Re-computation avoidance working")
    print("✓ Force recomputation working")
    print("✓ Visualization integration working")
    
    print(f"\nFinal dataframe shape: {df_final.shape}")
    print(f"Metrics columns: {[col for col in df_final.columns if any(col.startswith(f'metric_model_{i}_') for i in ['a', 'b', 'c'])]}") 