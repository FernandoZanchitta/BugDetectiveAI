from utils.simple_metrics import diff_score, compute_and_store_metrics, get_metrics_columns, has_metrics_columns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


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
    axes[row, col].set_title(title, fontsize=13)
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
    buggy_codes, groundtruth_codes, corrected_codes, 
    use_dataframe=False, df=None, reference_column=None, response_columns=None,
    use_stored_metrics=True, compute_if_missing=True
):
    """Compare metrics between buggy, corrected, and groundtruth codes using histograms.
    
    Args:
        buggy_codes: List of buggy code strings (used only if use_dataframe=False)
        groundtruth_codes: List of groundtruth code strings (used only if use_dataframe=False)
        corrected_codes: List of corrected code strings (used only if use_dataframe=False)
        use_dataframe: If True, use dataframe approach with stored metrics (default: False)
        df: DataFrame containing code columns (required if use_dataframe=True)
        reference_column: Name of reference column (required if use_dataframe=True)
        response_columns: List of response column names (required if use_dataframe=True)
        use_stored_metrics: If True, use pre-computed metrics (default: True)
        compute_if_missing: If True, compute missing metrics automatically (default: True)
    """
    if use_dataframe:
        if df is None or reference_column is None or response_columns is None:
            raise ValueError("df, reference_column, and response_columns are required when use_dataframe=True")
        
        # Use stored metrics approach
        if use_stored_metrics:
            # Check if we need to compute metrics
            need_computation = False
            for response_col in response_columns:
                if not has_metrics_columns(df, response_col):
                    need_computation = True
                    break
            
            if need_computation:
                if compute_if_missing:
                    print("Some metrics are missing. Computing them now...")
                    df = compute_and_store_metrics(df, reference_column, response_columns)
                else:
                    print("Some metrics are missing. Set compute_if_missing=True to compute them automatically.")
                    return
            
            # Use stored metrics for visualization
            compare_metrics_histograms_from_stored(df, response_columns, title_prefix="Bug vs Response Models")
            return
    print("Not Using Stored data - Evaluating Metrics...")
    # Original implementation for computing metrics on-the-fly
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


def compare_groundtruth_vs_corrected_histograms(
    groundtruth_codes, corrected_codes,
    use_dataframe=False, df=None, reference_column=None, response_columns=None,
    use_stored_metrics=True, compute_if_missing=True
):
    """Compare metrics between groundtruth and corrected codes using histograms.
    
    Args:
        groundtruth_codes: List of groundtruth code strings (used only if use_dataframe=False)
        corrected_codes: List of corrected code strings (used only if use_dataframe=False)
        use_dataframe: If True, use dataframe approach with stored metrics (default: False)
        df: DataFrame containing code columns (required if use_dataframe=True)
        reference_column: Name of reference column (required if use_dataframe=True)
        response_columns: List of response column names (required if use_dataframe=True)
        use_stored_metrics: If True, use pre-computed metrics (default: True)
        compute_if_missing: If True, compute missing metrics automatically (default: True)
    """
    if use_dataframe:
        if df is None or reference_column is None or response_columns is None:
            raise ValueError("df, reference_column, and response_columns are required when use_dataframe=True")
        
        # Use stored metrics approach
        if use_stored_metrics:
            # Check if we need to compute metrics
            need_computation = False
            for response_col in response_columns:
                if not has_metrics_columns(df, response_col):
                    need_computation = True
                    break
            
            if need_computation:
                if compute_if_missing:
                    print("Some metrics are missing. Computing them now...")
                    df = compute_and_store_metrics(df, reference_column, response_columns)
                else:
                    print("Some metrics are missing. Set compute_if_missing=True to compute them automatically.")
                    return
            
            # Use stored metrics for visualization
            compare_metrics_histograms_from_stored(df, response_columns, title_prefix="Groundtruth vs Response Models")
            return
    
    # Original implementation for computing metrics on-the-fly
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


def compare_metrics_histograms_from_stored(
    df: pd.DataFrame,
    response_columns: List[str],
    title_prefix: str = "Metrics Comparison",
    figsize: Tuple[int, int] = (18, 12)
):
    """
    Create histograms using pre-computed metrics stored in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataset containing pre-computed metrics columns
        response_columns (List[str]): List of response column names
        title_prefix (str): Prefix for the overall title
        figsize (Tuple[int, int]): Figure size as (width, height)
    """
    # Get all available metrics from the first response column
    if not response_columns:
        raise ValueError("No response columns provided")
    
    first_response_col = response_columns[0]
    if not has_metrics_columns(df, first_response_col):
        raise ValueError(f"No metrics columns found for {first_response_col}")
    
    metrics_columns = get_metrics_columns(df, first_response_col)
    # Remove 'response_' prefix from response column name for metric name extraction
    clean_first_response_col = first_response_col.replace("response_", "")
    metric_names = [col.replace(f"metric_{clean_first_response_col}_", "") for col in metrics_columns]
    
    print(f"Creating histograms using stored metrics for {len(response_columns)} response columns...")
    
    # Prepare metrics data for histogram plotting
    metrics_dicts = []
    labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, response_col in enumerate(response_columns):
        metrics_dict = {}
        for metric in metric_names:
            # Remove 'response_' prefix from response column name
            clean_response_col = response_col.replace("response_", "")
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                # Remove NaN values
                values = df[metric_column].dropna().tolist()
                metrics_dict[metric] = values
        
        if metrics_dict:  # Only add if we have data
            metrics_dicts.append(metrics_dict)
            labels.append(response_col.replace("response_", "").replace("_", " ").title())
    
    if not metrics_dicts:
        raise ValueError("No valid metrics data found for any response column")
    
    # Use the existing helper function for plotting
    _plot_metric_histograms(
        metrics_dicts,
        labels=labels,
        colors=colors[:len(metrics_dicts)],
        available_metrics=metric_names,
        title_prefix=title_prefix
    )
    
    # Print statistics using stored metrics
    _print_metric_statistics_from_stored(df, response_columns, metric_names, f"{title_prefix} STATISTICS")


def _print_metric_statistics_from_stored(
    df: pd.DataFrame, 
    response_columns: List[str], 
    metric_names: List[str], 
    stat_title: str
):
    """
    Helper to print summary statistics for stored metrics.
    """
    print(f"=== {stat_title} (FROM STORED METRICS) ===")
    for metric in metric_names:
        for response_col in response_columns:
            # Remove 'response_' prefix from response column name
            clean_response_col = response_col.replace("response_", "")
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                values = df[metric_column].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"{metric.replace('_', ' ').title()} - {response_col.replace('response_', '').replace('_', ' ').title()}: "
                          f"mean={mean_val:.3f}, std={std_val:.3f} (n={len(values)})")


def plot_metrics_boxplots(
    df, 
    reference_column="after_merge_without_docstrings", 
    response_columns=None, 
    figsize=(20, 15),
    title_prefix="Metrics Comparison",
    use_stored_metrics=True,
    compute_if_missing=True
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
        use_stored_metrics (bool): If True, use pre-computed metrics stored in dataframe (default: True)
        compute_if_missing (bool): If True and use_stored_metrics=True, compute missing metrics (default: True)
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
    
    # Use stored metrics if requested
    if use_stored_metrics:
        # Check if we need to compute metrics
        need_computation = False
        for response_col in response_columns:
            if not has_metrics_columns(df, response_col):
                need_computation = True
                break
        
        if need_computation:
            if compute_if_missing:
                print("Some metrics are missing. Computing them now...")
                df = compute_and_store_metrics(df, reference_column, response_columns)
            else:
                print("Some metrics are missing. Set compute_if_missing=True to compute them automatically.")
                return
        
        # Use stored metrics for visualization
        plot_metrics_boxplots_from_stored(df, response_columns, figsize, title_prefix)
        return
    
    # Original implementation for computing metrics on-the-fly
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
                 fontsize=16, fontweight='bold', y=0.99)
    
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


def plot_metrics_boxplots_from_stored(
    df: pd.DataFrame,
    response_columns: List[str],
    figsize: Tuple[int, int] = (20, 15),
    title_prefix: str = "Metrics Comparison"
):
    """
    Create boxplots using pre-computed metrics stored in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataset containing pre-computed metrics columns
        response_columns (List[str]): List of response column names
        figsize (Tuple[int, int]): Figure size as (width, height)
        title_prefix (str): Prefix for the overall title
    """
    # Get all available metrics from the first response column
    if not response_columns:
        raise ValueError("No response columns provided")
    
    first_response_col = response_columns[0]
    if not has_metrics_columns(df, first_response_col):
        raise ValueError(f"No metrics columns found for {first_response_col}")
    
    metrics_columns = get_metrics_columns(df, first_response_col)
    # Remove 'response_' prefix from response column name for metric name extraction
    clean_first_response_col = first_response_col.replace("response_", "")
    metric_names = [col.replace(f"metric_{clean_first_response_col}_", "") for col in metrics_columns]
    
    print(f"Creating boxplots using stored metrics for {len(response_columns)} response columns...")
    
    # Create subplots for each metric
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_metrics == 1:
        axes = axes.reshape(1, 1)
    
    # Create boxplots for each metric
    for i, metric in enumerate(metric_names):
        row = i // n_cols
        col = i % n_cols
        
        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        
        for response_col in response_columns:
            # Remove 'response_' prefix from response column name
            clean_response_col = response_col.replace("response_", "")
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                # Remove NaN values for boxplot
                values = df[metric_column].dropna().tolist()
                if values:  # Only add if we have non-empty data
                    boxplot_data.append(values)
                    labels.append(clean_response_col.replace("_", " ").title())
        
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
    fig.suptitle(f"{title_prefix}: Using Stored Metrics", 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS (FROM STORED METRICS) ===")
    print(f"Number of response columns: {len(response_columns)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Available metrics: {metric_names}")
    
    for metric in metric_names:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for response_col in response_columns:
            # Remove 'response_' prefix from response column name
            clean_response_col = response_col.replace("response_", "")
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                values = df[metric_column].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    median_val = values.median()
                    print(f"  {clean_response_col.replace('_', ' ').title()}: "
                          f"mean={mean_val:.3f}, std={std_val:.3f}, median={median_val:.3f} "
                          f"(n={len(values)})")


def plot_metrics_histograms(
    df: pd.DataFrame,
    reference_column: str = "after_merge_without_docstrings",
    response_columns: Optional[List[str]] = None,
    use_stored_metrics: bool = True,
    compute_if_missing: bool = True,
    figsize: Tuple[int, int] = (18, 12),
    title_prefix: str = "Metrics Comparison"
):
    """
    Create histograms for all diff_score metrics across multiple response columns.
    
    Args:
        df (pd.DataFrame): Input dataset containing code columns
        reference_column (str): Name of the reference code column
        response_columns (List[str]): List of response column names to compare against reference.
                                    If None, will automatically find all columns starting with "response_"
        use_stored_metrics (bool): If True, use pre-computed metrics stored in dataframe (default: True)
        compute_if_missing (bool): If True and use_stored_metrics=True, compute missing metrics (default: True)
        figsize (Tuple[int, int]): Figure size as (width, height)
        title_prefix (str): Prefix for the overall title
    """
    # Validate reference column exists
    if reference_column not in df.columns:
        raise ValueError(f"Reference column '{reference_column}' not found. Available columns: {list(df.columns)}")
    
    # Auto-detect response columns if not provided
    if response_columns is None:
        response_columns = [col for col in df.columns if col.startswith("response_")]
    
    # Validate response columns exist
    missing_columns = [col for col in response_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Response columns not found: {missing_columns}")
    
    if not response_columns:
        raise ValueError("No response columns found. Please specify response_columns or ensure columns start with 'response_'")
    
    # Use stored metrics if requested
    if use_stored_metrics:
        # Check if we need to compute metrics
        need_computation = False
        for response_col in response_columns:
            if not has_metrics_columns(df, response_col):
                need_computation = True
                break
        
        if need_computation:
            if compute_if_missing:
                print("Some metrics are missing. Computing them now...")
                df = compute_and_store_metrics(df, reference_column, response_columns)
            else:
                print("Some metrics are missing. Set compute_if_missing=True to compute them automatically.")
                return
        
        # Use stored metrics for visualization
        compare_metrics_histograms_from_stored(df, response_columns, title_prefix, figsize)
        return
    
    # Original implementation for computing metrics on-the-fly
    print(f"Computing metrics on-the-fly for {reference_column} vs {len(response_columns)} response columns...")
    
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
    
    # Prepare data for histogram plotting
    metrics_dicts = []
    labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, response_col in enumerate(response_columns):
        if response_col in all_metrics_data:
            metrics_dicts.append(all_metrics_data[response_col])
            labels.append(response_col.replace("response_", "").replace("_", " ").title())
    
    # Use the existing helper function for plotting
    _plot_metric_histograms(
        metrics_dicts,
        labels=labels,
        colors=colors[:len(metrics_dicts)],
        available_metrics=available_metrics,
        title_prefix=title_prefix
    )
    
    # Print statistics
    _print_metric_statistics(
        metrics_dicts,
        labels=labels,
        available_metrics=available_metrics,
        stat_title=f"{title_prefix} STATISTICS"
    )


def create_metrics_summary_table(
    df: pd.DataFrame,
    response_columns: List[str],
    title: str = "Metrics Summary Table",
    show_std: bool = True,
    show_count: bool = True,
    round_digits: int = 3
) -> pd.DataFrame:
    """
    Create a summary table with statistics (mean and variance) for each metric across different models.
    
    Args:
        df (pd.DataFrame): Input dataset containing pre-computed metrics columns
        response_columns (List[str]): List of response column names
        title (str): Title for the table (default: "Metrics Summary Table")
        show_std (bool): If True, show standard deviation instead of variance (default: True)
        show_count (bool): If True, show count of valid samples (default: True)
        round_digits (int): Number of decimal places to round results (default: 3)
    
    Returns:
        pd.DataFrame: Summary table with models as rows and metrics as columns
        
    Raises:
        ValueError: If no valid metrics data is found
    """
    if not response_columns:
        raise ValueError("No response columns provided")
    
    # Get all available metrics from the first response column
    first_response_col = response_columns[0]
    if not has_metrics_columns(df, first_response_col):
        raise ValueError(f"No metrics columns found for {first_response_col}")
    
    metrics_columns = get_metrics_columns(df, first_response_col)
    # Remove 'response_' prefix from response column name for metric name extraction
    clean_first_response_col = first_response_col.replace("response_", "")
    metric_names = [col.replace(f"metric_{clean_first_response_col}_", "") for col in metrics_columns]
    
    print(f"Creating metrics summary table for {len(response_columns)} response columns...")
    
    # Initialize summary data
    summary_data = {}
    model_names = []
    
    for response_col in response_columns:
        # Remove 'response_' prefix from response column name
        clean_response_col = response_col.replace("response_", "")
        model_name = clean_response_col.replace("_", " ").title()
        model_names.append(model_name)
        
        model_stats = {}
        
        for metric in metric_names:
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                # Remove NaN values
                values = df[metric_column].dropna()
                
                if len(values) > 0:
                    mean_val = values.mean()
                    if show_std:
                        stat_val = values.std()
                        stat_label = "std"
                    else:
                        stat_val = values.var()
                        stat_label = "var"
                    
                    # Format the statistic string
                    if show_count:
                        stat_str = f"{mean_val:.{round_digits}f} ± {stat_val:.{round_digits}f} (n={len(values)})"
                    else:
                        stat_str = f"{mean_val:.{round_digits}f} ± {stat_val:.{round_digits}f}"
                    
                    model_stats[metric] = stat_str
                else:
                    model_stats[metric] = "N/A"
            else:
                model_stats[metric] = "Missing"
        
        summary_data[model_name] = model_stats
    
    if not summary_data:
        raise ValueError("No valid metrics data found for any response column")
    
    # Create DataFrame from summary data
    summary_df = pd.DataFrame(summary_data).T
    
    # Print the table
    print(f"\n=== {title} ===")
    print(f"Models: {len(model_names)}")
    print(f"Metrics: {len(metric_names)}")
    print(f"Statistics shown: mean ± {stat_label}")
    if show_count:
        print("Sample counts shown in parentheses")
    print()
    
    # Display the table
    print(summary_df.to_string())
    
    # Print additional summary
    print(f"\nTable shape: {summary_df.shape}")
    print(f"Available metrics: {list(metric_names)}")
    print(f"Models analyzed: {list(model_names)}")
    
    return summary_df


def create_metrics_summary_table_detailed(
    df: pd.DataFrame,
    response_columns: List[str],
    title: str = "Detailed Metrics Summary Table",
    round_digits: int = 3
) -> pd.DataFrame:
    """
    Create a detailed summary table with mean, std, min, max, and count for each metric across different models.
    
    Args:
        df (pd.DataFrame): Input dataset containing pre-computed metrics columns
        response_columns (List[str]): List of response column names
        title (str): Title for the table (default: "Detailed Metrics Summary Table")
        round_digits (int): Number of decimal places to round results (default: 3)
    
    Returns:
        pd.DataFrame: Detailed summary table with models as rows and metrics as columns
    """
    if not response_columns:
        raise ValueError("No response columns provided")
    
    # Get all available metrics from the first response column
    first_response_col = response_columns[0]
    if not has_metrics_columns(df, first_response_col):
        raise ValueError(f"No metrics columns found for {first_response_col}")
    
    metrics_columns = get_metrics_columns(df, first_response_col)
    # Remove 'response_' prefix from response column name for metric name extraction
    clean_first_response_col = first_response_col.replace("response_", "")
    metric_names = [col.replace(f"metric_{clean_first_response_col}_", "") for col in metrics_columns]
    
    print(f"Creating detailed metrics summary table for {len(response_columns)} response columns...")
    
    # Initialize summary data
    summary_data = {}
    model_names = []
    
    for response_col in response_columns:
        # Remove 'response_' prefix from response column name
        clean_response_col = response_col.replace("response_", "")
        model_name = clean_response_col.replace("_", " ").title()
        model_names.append(model_name)
        
        model_stats = {}
        
        for metric in metric_names:
            metric_column = f"metric_{clean_response_col}_{metric}"
            if metric_column in df.columns:
                # Remove NaN values
                values = df[metric_column].dropna()
                
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    count_val = len(values)
                    
                    # Format the detailed statistic string
                    stat_str = f"μ={mean_val:.{round_digits}f}, σ={std_val:.{round_digits}f}, min={min_val:.{round_digits}f}, max={max_val:.{round_digits}f}, n={count_val}"
                    
                    model_stats[metric] = stat_str
                else:
                    model_stats[metric] = "N/A"
            else:
                model_stats[metric] = "Missing"
        
        summary_data[model_name] = model_stats
    
    if not summary_data:
        raise ValueError("No valid metrics data found for any response column")
    
    # Create DataFrame from summary data
    summary_df = pd.DataFrame(summary_data).T
    
    # Print the table
    print(f"\n=== {title} ===")
    print(f"Models: {len(model_names)}")
    print(f"Metrics: {len(metric_names)}")
    print("Statistics shown: μ=mean, σ=std, min, max, n=count")
    print()
    
    # Display the table
    print(summary_df.to_string())
    
    # Print additional summary
    print(f"\nTable shape: {summary_df.shape}")
    print(f"Available metrics: {list(metric_names)}")
    print(f"Models analyzed: {list(model_names)}")
    
    return summary_df
