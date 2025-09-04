import ast
import difflib
import logging
from typing import Dict, Tuple, List, Optional
import pandas as pd
from codebleu import calc_codebleu
import numpy as np

# Constants
DEFAULT_CODEBLEU_WEIGHTS = (0.1, 0.4, 0.1, 0.4)
LOG_FILENAME = "codebleu_dataflow_warnings.log"
PY_KEYWORDS = {
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
}

# Configure logging for problematic cases
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def exact_match(a: str, b: str) -> float:
    """Check if two strings are exactly equal.

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        1.0 if strings are identical, 0.0 otherwise
    """
    return 1.0 if a == b else 0.0


def codebleu(
    candidate: str,
    reference: str,
    weights: Tuple[float, float, float, float] = DEFAULT_CODEBLEU_WEIGHTS,
) -> Dict[str, float]:
    """Calculate CodeBLEU metrics between candidate and reference code.

    Args:
        candidate: The generated/fixed code to evaluate
        reference: The original/correct code to compare against
        weights: Tuple of weights for (ngram_match, weighted_ngram_match, syntax_match, dataflow_match)

    Returns:
        Dictionary containing CodeBLEU metrics

    Raises:
        AssertionError: If weights are invalid
    """
    if len(weights) != 4 or sum(weights) != 1.0 or not all(w >= 0 for w in weights):
        raise ValueError(
            "Weights must be a tuple of 4 non-negative floats that sum to 1.0"
        )

    metric = calc_codebleu([reference], [candidate], lang="python", weights=weights)

    # Log dataflow extraction failures for debugging
    if metric.get("dataflow_match", 1.0) == 0.0:
        logging.info("Dataflow extraction failed:")
        logging.info(f"REFERENCE:\n{reference}")
        logging.info(f"CANDIDATE:\n{candidate}")
        logging.info("-" * 60)

    return metric


class Normalizer(ast.NodeTransformer):
    """Normalizes identifiers and constants in an AST to make comparisons robust to naming changes."""

    def __init__(self):
        self.var_count = 0
        self.func_count = 0
        self.class_count = 0
        self.name_mapping = {}

    def _get_or_create_mapping(self, name: str, prefix: str, counter: int) -> str:
        """Helper method to get or create normalized name mapping."""
        if name not in self.name_mapping:
            counter += 1
            self.name_mapping[name] = f"{prefix}_{counter}"
        return self.name_mapping[name]

    def visit_FunctionDef(self, node):
        """Rename function definitions to normalized names."""
        node.name = self._get_or_create_mapping(node.name, "_func", self.func_count)
        self.func_count += 1
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        """Rename class definitions to normalized names."""
        node.name = self._get_or_create_mapping(node.name, "_class", self.class_count)
        self.class_count += 1
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        """Rename variable identifiers to normalized names."""
        node.id = self._get_or_create_mapping(node.id, "_var", self.var_count)
        self.var_count += 1
        return node

    def visit_Constant(self, node):
        """Replace literal constants with placeholder."""
        return ast.copy_location(ast.Constant(value="_const_"), node)


def get_normalized_ast(code: str) -> str:
    """Generate a normalized AST dump for the given code.

    Args:
        code: Python code string to normalize

    Returns:
        Normalized AST dump as string, or empty string if parsing fails
    """
    try:
        parsed = ast.parse(code)
        normalizer = Normalizer()
        normalized_tree = normalizer.visit(parsed)
        ast.fix_missing_locations(normalized_tree)
        return ast.dump(normalized_tree, indent=4)
    except SyntaxError:
        return ""


def _calculate_ast_similarity(before_code: str, after_code: str) -> float:
    """Calculate AST similarity between two code strings.

    Args:
        before_code: Original code
        after_code: Modified code

    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        before_ast = ast.dump(ast.parse(before_code))
        after_ast = ast.dump(ast.parse(after_code))
        return difflib.SequenceMatcher(None, before_ast, after_ast).ratio()
    except SyntaxError:
        return 0.0


def diff_score(before_code: str, after_code: str) -> Dict[str, float]:
    """Calculate comprehensive similarity scores between before and after code.

    This is the main function that calculates multiple similarity metrics:
    - AST structure similarity
    - Normalized AST similarity (ignoring variable/function names)
    - Text similarity
    - CodeBLEU metrics

    Args:
        before_code: Original code string
        after_code: Modified code string

    Returns:
        Dictionary containing all similarity metrics
    """


    # Calculate exact match
    exact_match_score = exact_match(before_code, after_code)

    # Calculate AST similarity
    ast_score = _calculate_ast_similarity(before_code, after_code)

    # Calculate normalized AST similarity
    before_normalized = get_normalized_ast(before_code)
    after_normalized = get_normalized_ast(after_code)
    ast_score_normalized = difflib.SequenceMatcher(
        None, before_normalized, after_normalized
    ).ratio()

    # Calculate text similarity
    text_score = difflib.SequenceMatcher(None, before_code, after_code).ratio()

    # Calculate CodeBLEU metrics
    codebleu_metrics = codebleu(after_code, before_code)

    return {
        # "exact_match": exact_match_score,
        "ast_score": ast_score,
        "text_score": text_score,
        "ast_score_normalized": ast_score_normalized,
        **codebleu_metrics,
    }


def compute_and_store_metrics(
    df: pd.DataFrame,
    reference_column: str = "after_merge_without_docstrings",
    response_columns: Optional[List[str]] = None,
    force_recompute: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:
    """Calculate diff_score metrics and store them as columns in the dataframe.
    
    This function computes metrics between reference code and response code columns,
    storing the results as new columns to avoid re-computation in visualization functions.
    
    Args:
        df (pd.DataFrame): Input dataframe containing code columns
        reference_column (str): Name of the reference code column (default: "after_merge_without_docstrings")
        response_columns (List[str], optional): List of response column names to compare against reference.
                                              If None, will automatically find all columns starting with "response_"
        force_recompute (bool): If True, recompute metrics even if columns already exist (default: False)
        show_progress (bool): If True, show progress with tqdm (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with metrics columns added
        
    Raises:
        ValueError: If reference column or response columns don't exist
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
    
    # Get expected metric names from a sample calculation
    sample_metrics = diff_score("", "")
    metric_names = list(sample_metrics.keys())
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Import tqdm for progress tracking if needed
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Warning: tqdm not available, progress tracking disabled")
            show_progress = False
    
    print(f"Computing metrics for {reference_column} vs {len(response_columns)} response columns...")
    
    for response_col in response_columns:
        print(f"Processing {response_col}...")
        
        # Check if metrics columns already exist for this response column
        clean_response_col = response_col.replace("response_", "")
        existing_columns = [f"metric_{clean_response_col}_{metric}" for metric in metric_names]
        columns_exist = all(col in result_df.columns for col in existing_columns)
        
        if columns_exist and not force_recompute:
            print(f"  Metrics columns already exist for {response_col}, skipping...")
            continue
        
        # Calculate metrics for each row
        metrics_data = {metric: [] for metric in metric_names}
        
        # Create iterator with progress bar if requested
        iterator = result_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(result_df), desc=f"Computing {response_col}")
        
        for idx, row in iterator:
            try:
                reference_code = str(row[reference_column])
                response_code = str(row[response_col])
                
                # Skip if either code is empty or NaN
                if pd.isna(reference_code) or pd.isna(response_code) or reference_code == "" or response_code == "":
                    # Add NaN values for all metrics
                    for metric in metric_names:
                        metrics_data[metric].append(float('nan'))
                    continue
                
                # Calculate metrics
                metrics = diff_score(reference_code, response_code)
                
                # Store each metric
                for metric in metric_names:
                    metrics_data[metric].append(metrics[metric])
                    
            except Exception as e:
                print(f"  Error calculating metrics for row {idx}: {e}")
                # Add NaN values for all metrics on error
                for metric in metric_names:
                    metrics_data[metric].append(float('nan'))
        
        # Add metric columns to dataframe
        for metric in metric_names:
            # Remove 'response_' prefix from response column name
            clean_response_col = response_col.replace("response_", "")
            column_name = f"metric_{clean_response_col}_{metric}"
            result_df[column_name] = metrics_data[metric]
        
        print(f"  Added {len(metric_names)} metric columns for {response_col}")
    
    print(f"Completed! Added metrics for {len(response_columns)} response columns.")
    return result_df


def get_metrics_columns(df: pd.DataFrame, response_column: str) -> List[str]:
    """Get the names of metrics columns for a specific response column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        response_column (str): Name of the response column
        
    Returns:
        List[str]: List of metrics column names for the response column
    """
    # Get expected metric names from a sample calculation
    sample_metrics = diff_score("", "")
    metric_names = list(sample_metrics.keys())
    
    # Remove 'response_' prefix from response column name
    clean_response_col = response_column.replace("response_", "")
    
    # Check which metrics columns exist
    expected_columns = [f"metric_{clean_response_col}_{metric}" for metric in metric_names]
    existing_columns = [col for col in expected_columns if col in df.columns]
    
    return existing_columns


def has_metrics_columns(df: pd.DataFrame, response_column: str) -> bool:
    """Check if metrics columns exist for a specific response column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        response_column (str): Name of the response column
        
    Returns:
        bool: True if all metrics columns exist, False otherwise
    """
    # Get expected metric names from a sample calculation
    sample_metrics = diff_score("", "")
    metric_names = list(sample_metrics.keys())
    
    # Remove 'response_' prefix from response column name
    clean_response_col = response_column.replace("response_", "")
    
    # Check if all expected columns exist
    expected_columns = [f"metric_{clean_response_col}_{metric}" for metric in metric_names]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing metrics columns for {response_column}: {missing_columns}")
        return False
    
    return True


def wilcoxon_test(
    df: pd.DataFrame,
    response_column1: str,
    response_column2: str,
    metric_name: str = "ast_score",
    reference_column: str = "after_merge_without_docstrings",
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test between two response columns for a specific metric.
    
    This function compares the performance of two different response columns (e.g., different LLM models)
    using the Wilcoxon signed-rank test, which is appropriate for paired samples and doesn't assume
    normal distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        response_column1 (str): First response column name (e.g., "response_gpt4")
        response_column2 (str): Second response column name (e.g., "response_claude")
        metric_name (str): Name of the metric to compare (default: "ast_score")
        reference_column (str): Reference column used for metric calculation
        alternative (str): Alternative hypothesis: "two-sided", "greater", or "less" (default: "two-sided")
    
    Returns:
        Dict[str, float]: Dictionary containing test results:
            - statistic: Wilcoxon test statistic
            - pvalue: p-value for the test
            - mean_diff: Mean difference (col2 - col1)
            - median_diff: Median difference (col2 - col1)
            - effect_size: Effect size (Z/sqrt(N))
            - significant: Boolean indicating if p < 0.05
    
    Raises:
        ValueError: If required columns don't exist
        ImportError: If scipy is not available
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy is required for Wilcoxon test. Install with: poetry add scipy")
    
    # Get metric column names
    clean_col1 = response_column1.replace("response_", "")
    clean_col2 = response_column2.replace("response_", "")
    
    metric_col1 = f"metric_{clean_col1}_{metric_name}"
    metric_col2 = f"metric_{clean_col2}_{metric_name}"
    
    # Validate columns exist
    missing_columns = []
    for col in [response_column1, response_column2, metric_col1, metric_col2]:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # Extract metric values, filtering out NaN values
    values1 = df[metric_col1].dropna()
    values2 = df[metric_col2].dropna()
    
    # Ensure we have the same indices for paired comparison
    common_idx = values1.index.intersection(values2.index)
    if len(common_idx) < 3:
        raise ValueError(f"Insufficient paired data: only {len(common_idx)} valid pairs found")
    
    paired_values1 = values1.loc[common_idx]
    paired_values2 = values2.loc[common_idx]
    
    # Perform Wilcoxon test
    statistic, pvalue = stats.wilcoxon(
        paired_values1, 
        paired_values2, 
        alternative=alternative
    )
    
    # Calculate descriptive statistics
    differences = paired_values2 - paired_values1
    mean_diff = differences.mean()
    median_diff = differences.median()
    
    # Calculate effect size (Z/sqrt(N))
    n = len(common_idx)
    effect_size = abs(statistic) / (n * (n + 1) / 2) ** 0.5
    
    # Determine significance
    significant = pvalue < 0.05
    
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "effect_size": effect_size,
        "significant": significant,
        "n_pairs": n,
        "alternative": alternative
    }


def compare_multiple_responses(
    df: pd.DataFrame,
    response_columns: List[str],
    metric_name: str = "ast_score",
    reference_column: str = "after_merge_without_docstrings",
    alpha: float = 0.05
) -> pd.DataFrame:
    """Compare multiple response columns using pairwise Wilcoxon tests.
    
    This function performs pairwise comparisons between all response columns for a given metric,
    creating a comprehensive comparison matrix.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        response_columns (List[str]): List of response column names to compare
        metric_name (str): Name of the metric to compare (default: "ast_score")
        reference_column (str): Reference column used for metric calculation
        alpha (float): Significance level (default: 0.05)
    
    Returns:
        pd.DataFrame: DataFrame with pairwise comparison results
    """
    results = []
    
    for i, col1 in enumerate(response_columns):
        for j, col2 in enumerate(response_columns):
            if i >= j:  # Skip self-comparisons and duplicate pairs
                continue
                
            try:
                test_result = wilcoxon_test(
                    df, col1, col2, metric_name, reference_column
                )
                
                results.append({
                    "response_1": col1,
                    "response_2": col2,
                    "metric": metric_name,
                    "statistic": test_result["statistic"],
                    "pvalue": test_result["pvalue"],
                    "mean_diff": test_result["mean_diff"],
                    "median_diff": test_result["median_diff"],
                    "effect_size": test_result["effect_size"],
                    "significant": test_result["significant"],
                    "n_pairs": test_result["n_pairs"]
                })
                
            except Exception as e:
                print(f"Error comparing {col1} vs {col2}: {e}")
                continue
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def print_wilcoxon_summary(
    df: pd.DataFrame,
    response_columns: List[str],
    metric_name: str = "ast_score",
    reference_column: str = "after_merge_without_docstrings"
):
    """Print a formatted summary of Wilcoxon test results.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        response_columns (List[str]): List of response column names to compare
        metric_name (str): Name of the metric to compare
        reference_column (str): Reference column used for metric calculation
    """
    print(f"\n{'='*60}")
    print(f"WILCOXON TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Metric: {metric_name}")
    print(f"Reference: {reference_column}")
    print(f"Response columns: {len(response_columns)}")
    print(f"{'='*60}")
    
    # Perform pairwise comparisons
    comparison_df = compare_multiple_responses(
        df, response_columns, metric_name, reference_column
    )
    
    if comparison_df.empty:
        print("No valid comparisons found.")
        return
    
    # Print results
    for _, row in comparison_df.iterrows():
        print(f"\n{row['response_1']} vs {row['response_2']}")
        print(f"  Statistic: {row['statistic']:.4f}")
        print(f"  p-value: {row['pvalue']:.6f}")
        print(f"  Mean diff (col2-col1): {row['mean_diff']:.6f}")
        print(f"  Median diff (col2-col1): {row['median_diff']:.6f}")
        print(f"  Effect size: {row['effect_size']:.4f}")
        print(f"  Significant (α=0.05): {'Yes' if row['significant'] else 'No'}")
        print(f"  N pairs: {row['n_pairs']}")
    
    # Summary statistics
    significant_comparisons = comparison_df[comparison_df['significant']].shape[0]
    total_comparisons = comparison_df.shape[0]
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Significant differences: {significant_comparisons}")
    print(f"Significance rate: {significant_comparisons/total_comparisons*100:.1f}%")
    
    # Effect size interpretation
    small_effects = comparison_df[comparison_df['effect_size'] < 0.1].shape[0]
    medium_effects = comparison_df[(comparison_df['effect_size'] >= 0.1) & (comparison_df['effect_size'] < 0.3)].shape[0]
    large_effects = comparison_df[comparison_df['effect_size'] >= 0.3].shape[0]
    
    print(f"\nEffect sizes:")
    print(f"  Small (<0.1): {small_effects}")
    print(f"  Medium (0.1-0.3): {medium_effects}")
    print(f"  Large (≥0.3): {large_effects}")


def aggregate_metrics_summary(
    dataframes: List[pd.DataFrame],
    category_column: str = "prompt",
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Aggregate metrics across multiple DataFrames and return summary statistics.
    
    This function takes a list of DataFrames (e.g., different prompt conditions) and
    aggregates the metrics for each model/category combination, returning mean and
    variance for each metric.
    
    Args:
        dataframes (List[pd.DataFrame]): List of DataFrames to aggregate
        category_column (str): Column name to use as category (default: "prompt")
        model_names (List[str], optional): List of model names to include. 
                                         If None, will auto-detect from metric columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows for each model/category and columns for 
                     metric means and variances
                     
    Raises:
        ValueError: If no valid DataFrames provided or missing required columns
    """
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    
    # Auto-detect model names if not provided
    if model_names is None:
        # Get metric columns from first DataFrame
        metric_cols = [col for col in dataframes[0].columns if col.startswith('metric_')]
        # Extract model names from metric column names
        model_names = list(set([col.split('_')[1] for col in metric_cols]))
    
    # Get metric names from a sample calculation
    sample_metrics = diff_score("", "")
    metric_names = list(sample_metrics.keys())
    
    # Initialize results storage
    results = []
    
    for df in dataframes:
        if category_column not in df.columns:
            print(f"Warning: Category column '{category_column}' not found in DataFrame, skipping...")
            continue
        
        # Get unique categories in this DataFrame
        categories = df[category_column].unique()
        
        for category in categories:
            # Filter data for this category
            category_data = df[df[category_column] == category]
            
            for model in model_names:
                if model not in model_names:
                    continue
                
                # Initialize row data
                row_data = {
                    'model': model,
                    'category': category,
                    'n_samples': len(category_data)
                }
                
                # Calculate mean and variance for each metric
                for metric in metric_names:
                    metric_col = f'metric_{model}_{metric}'
                    
                    if metric_col in category_data.columns:
                        # Get metric values, filtering out NaN
                        metric_values = category_data[metric_col].dropna()
                        
                        if len(metric_values) > 0:
                            row_data[f'{metric}_mean'] = metric_values.mean()
                            row_data[f'{metric}_variance'] = metric_values.var()
                            row_data[f'{metric}_std'] = metric_values.std()
                            row_data[f'{metric}_min'] = metric_values.min()
                            row_data[f'{metric}_max'] = metric_values.max()
                        else:
                            # No valid values
                            row_data[f'{metric}_mean'] = float('nan')
                            row_data[f'{metric}_variance'] = float('nan')
                            row_data[f'{metric}_std'] = float('nan')
                            row_data[f'{metric}_min'] = float('nan')
                            row_data[f'{metric}_max'] = float('nan')
                    else:
                        # Metric column not found
                        row_data[f'{metric}_mean'] = float('nan')
                        row_data[f'{metric}_variance'] = float('nan')
                        row_data[f'{metric}_std'] = float('nan')
                        row_data[f'{metric}_min'] = float('nan')
                        row_data[f'{metric}_max'] = float('nan')
                
                results.append(row_data)
    
    if not results:
        raise ValueError("No valid data found in any DataFrame")
    
    # Create result DataFrame
    result_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    base_cols = ['model', 'category', 'n_samples']
    metric_cols = []
    
    for metric in metric_names:
        metric_cols.extend([
            f'{metric}_mean', f'{metric}_variance', f'{metric}_std',
            f'{metric}_min', f'{metric}_max'
        ])
    
    # Ensure all columns exist (some might be missing if metrics weren't found)
    existing_cols = [col for col in base_cols + metric_cols if col in result_df.columns]
    result_df = result_df[existing_cols]
    
    return result_df


def print_metrics_summary(
    summary_df: pd.DataFrame,
    metric_name: str = "ast_score",
    category_column: str = "category"
):
    """Print a formatted summary of metrics for a specific metric.
    
    Args:
        summary_df (pd.DataFrame): Summary DataFrame from aggregate_metrics_summary
        metric_name (str): Name of the metric to display
        category_column (str): Name of the category column
    """
    print(f"\n{'='*80}")
    print(f"METRICS SUMMARY: {metric_name.upper()}")
    print(f"{'='*80}")
    
    # Filter columns for this metric
    metric_cols = [col for col in summary_df.columns if metric_name in col]
    display_cols = ['model', category_column, 'n_samples'] + metric_cols
    
    # Display summary
    display_df = summary_df[display_cols].copy()
    
    # Round numeric columns for better display
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    
    print(display_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY FOR {metric_name.upper()}")
    print(f"{'='*80}")
    
    # Best performing model/category combination
    mean_col = f'{metric_name}_mean'
    if mean_col in display_df.columns:
        best_idx = display_df[mean_col].idxmax()
        best_row = display_df.loc[best_idx]
        
        print(f"Best Performance:")
        print(f"  Model: {best_row['model']}")
        print(f"  Category: {best_row[category_column]}")
        print(f"  Mean Score: {best_row[mean_col]:.4f}")
        print(f"  Standard Deviation: {best_row[f'{metric_name}_std']:.4f}")
        print(f"  N Samples: {best_row['n_samples']}")
    
    # Performance range
    if mean_col in display_df.columns:
        print(f"\nPerformance Range:")
        print(f"  Min Mean: {display_df[mean_col].min():.4f}")
        print(f"  Max Mean: {display_df[mean_col].max():.4f}")
        print(f"  Overall Mean: {display_df[mean_col].mean():.4f}")
        print(f"  Overall Std: {display_df[mean_col].std():.4f}")


def compare_categories_performance(
    summary_df: pd.DataFrame,
    metric_name: str = "ast_score",
    category_column: str = "category"
) -> pd.DataFrame:
    """Compare performance across different categories for a specific metric.
    
    Args:
        summary_df (pd.DataFrame): Summary DataFrame from aggregate_metrics_summary
        metric_name (str): Name of the metric to compare
        category_column (str): Name of the category column
        
    Returns:
        pd.DataFrame: DataFrame with category comparisons
    """
    mean_col = f'{metric_name}_mean'
    std_col = f'{metric_name}_std'
    
    if mean_col not in summary_df.columns:
        raise ValueError(f"Metric column '{mean_col}' not found in summary DataFrame")
    
    # Group by category and calculate statistics
    category_stats = summary_df.groupby(category_column).agg({
        mean_col: ['mean', 'std', 'min', 'max'],
        'n_samples': 'sum'
    }).round(4)
    
    # Flatten column names
    category_stats.columns = [f'{metric_name}_{col}' for col in category_stats.columns]
    
    # Sort by mean performance
    category_stats = category_stats.sort_values(f'{metric_name}_mean', ascending=False)
    
    return category_stats
