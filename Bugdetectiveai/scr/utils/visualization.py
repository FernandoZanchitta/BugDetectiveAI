from utils.simple_metrics import diff_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_column_distribution(
    df, column_name="traceback_type", top_n=10, figsize=(12, 8)
):
    """
    Create a horizontal bar plot showing the distribution of values in a specified column.

    Args:
        df (pd.DataFrame): Input dataset
        column_name (str): Name of the column to analyze
        top_n (int): Number of top values to display (default: 10)
        figsize (tuple): Figure size as (width, height) (default: (12, 8))
    """
    # Assert that the column exists in the dataset
    assert column_name in df.columns, (
        f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}"
    )

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
    plt.xlabel("Count", fontsize=12)
    plt.ylabel(column_name.replace("_", " ").title(), fontsize=12)
    plt.title(
        f"Most Common {column_name.replace('_', ' ').title()} Values in Dataset",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels on the bars
    for i, (bar, value) in enumerate(zip(bars, top_values.values)):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=10,
        )

    # Adjust layout and display
    plt.tight_layout()
    plt.grid(axis="x", alpha=0.3)
    plt.show()

    # Print summary statistics
    print(f"Total unique {column_name} values: {df[column_name].nunique()}")
    print(f"Total samples: {len(df)}")
    print(f"\nTop 5 most common {column_name} values:")
    for i, (value, count) in enumerate(top_values.head().items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i}. {value}: {count} ({percentage:.1f}%)")


def _plot_metric_histograms(
    metrics_dicts, labels, colors, available_metrics, title_prefix=None
):
    """
    Helper to plot histograms for each metric.
    metrics_dicts: list of dicts, each mapping metric name to list of values
    labels: list of legend labels for each dict
    colors: list of colors for each dict
    available_metrics: list of metric names
    title_prefix: optional string to prefix each subplot title
    """
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        for metric_dict, label, color in zip(metrics_dicts, labels, colors):
            axes[row, col].hist(
                metric_dict[metric], bins=20, alpha=0.7, label=label, color=color
            )
        axes[row, col].set_xlabel(metric.replace("_", " ").title())
        axes[row, col].set_ylabel("Frequency")
        title = f"{metric.replace('_', ' ').title()} Comparison"
        if title_prefix:
            title = f"{title_prefix} {title}"
        axes[row, col].set_title(title)
        axes[row, col].legend()
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    plt.tight_layout()
    plt.show()


def _print_metric_statistics(metrics_dicts, labels, available_metrics, stat_title):
    """
    Helper to print summary statistics for each metric and group.
    """
    print(f"=== {stat_title} ===")
    for metric in available_metrics:
        for metric_dict, label in zip(metrics_dicts, labels):
            mean = np.mean(metric_dict[metric])
            std = np.std(metric_dict[metric])
            print(
                f"{metric.replace('_', ' ').title()} - {label}: mean={mean:.3f}, std={std:.3f}"
            )


def compare_metrics_versus_bug_histograms(
    buggy_codes, groundtruth_codes, corrected_codes
):
    """Compare metrics between buggy, corrected, and groundtruth codes using histograms."""
    # Calculate diff scores (returns dict with all metrics)
    bug_vs_corrected_diff = [
        diff_score(bug, corr) for bug, corr in zip(buggy_codes, corrected_codes)
    ]
    bug_vs_groundtruth_diff = [
        diff_score(bug, gt) for bug, gt in zip(buggy_codes, groundtruth_codes)
    ]
    available_metrics = list(bug_vs_corrected_diff[0].keys())
    bug_vs_corrected_metrics = {
        metric: [d[metric] for d in bug_vs_corrected_diff]
        for metric in available_metrics
    }
    bug_vs_groundtruth_metrics = {
        metric: [d[metric] for d in bug_vs_groundtruth_diff]
        for metric in available_metrics
    }
    _plot_metric_histograms(
        [bug_vs_corrected_metrics, bug_vs_groundtruth_metrics],
        labels=["Bug vs Corrected", "Bug vs Groundtruth"],
        colors=["red", "blue"],
        available_metrics=available_metrics,
    )
    _print_metric_statistics(
        [bug_vs_corrected_metrics, bug_vs_groundtruth_metrics],
        labels=["Bug vs Corrected", "Bug vs Groundtruth"],
        available_metrics=available_metrics,
        stat_title="ALL METRICS STATISTICS",
    )


def compare_groundtruth_vs_corrected_histograms(groundtruth_codes, corrected_codes):
    """Compare metrics between groundtruth and corrected codes using histograms."""
    groundtruth_vs_corrected_diff = [
        diff_score(gt, corr) for gt, corr in zip(groundtruth_codes, corrected_codes)
    ]
    available_metrics = list(groundtruth_vs_corrected_diff[0].keys())
    groundtruth_vs_corrected_metrics = {
        metric: [d[metric] for d in groundtruth_vs_corrected_diff]
        for metric in available_metrics
    }
    _plot_metric_histograms(
        [groundtruth_vs_corrected_metrics],
        labels=["Groundtruth vs Corrected"],
        colors=["green"],
        available_metrics=available_metrics,
    )
    _print_metric_statistics(
        [groundtruth_vs_corrected_metrics],
        labels=["Groundtruth vs Corrected"],
        available_metrics=available_metrics,
        stat_title="GROUNDTRUTH VS CORRECTED METRICS STATISTICS",
    )


def plot_metrics_boxplots(
    df, 
    reference_column="after_merge_without_docstrings", 
    response_columns=None, 
    figsize=(20, 12),
    title_prefix="Metrics Comparison"
):
    """
    Create boxplots for all diff_score metrics across multiple response columns.
    
    Args:
        df (pd.DataFrame): Input dataset containing code columns
        reference_column (str): Name of the reference code column (default: "buggy_code")
        response_columns (list): List of response column names to compare against reference.
                               If None, will automatically find all columns starting with "response_"
        figsize (tuple): Figure size as (width, height) (default: (20, 12))
        title_prefix (str): Prefix for the overall title (default: "Metrics Comparison")
    """
    # Validate reference column exists
    assert reference_column in df.columns, (
        f"Reference column '{reference_column}' not found. Available columns: {list(df.columns)}"
    )
    
    # Auto-detect response columns if not provided
    if response_columns is None:
        response_columns = [col for col in df.columns if col.startswith("response_")]
    
    # Validate response columns exist
    missing_columns = [col for col in response_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Response columns not found: {missing_columns}")
    
    if not response_columns:
        raise ValueError("No response columns found. Please specify response_columns or ensure columns start with 'response_'")
    
    print(f"Comparing {reference_column} against {len(response_columns)} response columns: {response_columns}")
    
    # Calculate metrics for each response column
    all_metrics_data = {}
    
    for response_col in response_columns:
        print(f"Calculating metrics for {response_col}...")
        metrics_list = []
        
        for idx, row in df.iterrows():
            try:
                reference_code = str(row[reference_column])
                response_code = str(row[response_col])
                
                # Skip if either code is empty or NaN
                if pd.isna(reference_code) or pd.isna(response_code) or reference_code == "" or response_code == "":
                    continue
                    
                metrics = diff_score(reference_code, response_code)
                metrics_list.append(metrics)
                
            except Exception as e:
                print(f"Error calculating metrics for row {idx}: {e}")
                continue
        
        if metrics_list:
            # Convert list of dicts to dict of lists
            metrics_dict = {}
            for metric_name in metrics_list[0].keys():
                metrics_dict[metric_name] = [m[metric_name] for m in metrics_list]
            all_metrics_data[response_col] = metrics_dict
        else:
            print(f"Warning: No valid metrics calculated for {response_col}")
    
    if not all_metrics_data:
        raise ValueError("No valid metrics data could be calculated for any response column")
    
    # Get all available metrics
    available_metrics = list(next(iter(all_metrics_data.values())).keys())
    
    # Create subplots for each metric
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_metrics == 1:
        axes = axes.reshape(1, 1)
    
    # Create boxplots for each metric
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        
        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        
        for response_col in response_columns:
            if response_col in all_metrics_data and metric in all_metrics_data[response_col]:
                boxplot_data.append(all_metrics_data[response_col][metric])
                labels.append(response_col.replace("response_", "").replace("_", " ").title())
        
        if boxplot_data:
            # Create boxplot
            bp = axes[row, col].boxplot(boxplot_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray'][:len(bp['boxes'])]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize the plot
            axes[row, col].set_title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if len(labels) > 3:
                axes[row, col].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall title
    fig.suptitle(f"{title_prefix}: {reference_column} vs Response Models", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Reference column: {reference_column}")
    print(f"Number of response columns: {len(response_columns)}")
    print(f"Number of metrics: {len(available_metrics)}")
    print(f"Available metrics: {available_metrics}")
    
    for metric in available_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for response_col in response_columns:
            if response_col in all_metrics_data and metric in all_metrics_data[response_col]:
                values = all_metrics_data[response_col][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
                print(f"  {response_col.replace('response_', '').replace('_', ' ').title()}: "
                      f"mean={mean_val:.3f}, std={std_val:.3f}, median={median_val:.3f}")
