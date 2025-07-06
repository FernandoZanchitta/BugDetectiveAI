import matplotlib.pyplot as plt
import pandas as pd

def plot_column_distribution(df, column_name="traceback_type", top_n=10, figsize=(12, 8)):
    """
    Create a horizontal bar plot showing the distribution of values in a specified column.
    
    Args:
        df (pd.DataFrame): Input dataset
        column_name (str): Name of the column to analyze
        top_n (int): Number of top values to display (default: 10)
        figsize (tuple): Figure size as (width, height) (default: (12, 8))
    """
    # Assert that the column exists in the dataset
    assert column_name in df.columns, f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}"
    
    # Get value counts for the specified column
    value_counts = df[column_name].value_counts()
    
    # Create a visualization of the most common values
    plt.figure(figsize=figsize)
    
    # Get the top N most common values
    top_values = value_counts.head(top_n)
    
    # Create a horizontal bar plot
    bars = plt.barh(range(len(top_values)), top_values.values)
    
    # Customize the plot
    plt.yticks(range(len(top_values)), top_values.index)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Most Common {column_name.replace("_", " ").title()} Values in Dataset', fontsize=14, fontweight='bold')
    
    # Add value labels on the bars
    for i, (bar, value) in enumerate(zip(bars, top_values.values)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 str(value), va='center', fontsize=10)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)
    plt.show()
    
    # Print summary statistics
    print(f"Total unique {column_name} values: {df[column_name].nunique()}")
    print(f"Total samples: {len(df)}")
    print(f"\nTop 5 most common {column_name} values:")
    for i, (value, count) in enumerate(top_values.head().items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i}. {value}: {count} ({percentage:.1f}%)")