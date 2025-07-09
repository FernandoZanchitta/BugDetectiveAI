# Simple Metrics System for BugDetectiveAI

This module provides a simple and focused metrics system for evaluating code changes, particularly focused on bug fixes. The system is designed to be lightweight and easy to use.

## Overview

The metrics system consists of a single main function:

- **diff_score()**: Comprehensive similarity scoring between before and after code versions

## Usage Examples

### Basic Usage

```python
from simple_metrics import diff_score

# Evaluate code changes
result = diff_score(
    before_code="def add(a, b):\n    return a + b",
    after_code="def add(a, b):\n    return a + b + 1"
)

print(f"AST Score: {result['ast_score']:.3f}")
print(f"Text Score: {result['text_score']:.3f}")
print(f"CodeBLEU Score: {result['codebleu']:.3f}")
```

### Available Metrics

The `diff_score()` function returns a dictionary with the following metrics:

- **ast_score**: Raw AST structure similarity
- **ast_score_normalized**: AST similarity ignoring variable/function names
- **text_score**: Raw text similarity using difflib
- **codebleu**: CodeBLEU metric score
- **ngram_match**: N-gram matching component
- **weighted_ngram_match**: Weighted n-gram matching
- **syntax_match**: Syntax matching component
- **dataflow_match**: Data flow matching component

## Integration with BugDetectiveAI Dataset

The system works with the BugDetectiveAI dataset format:

```python
import pandas as pd
from simple_metrics import diff_score

# Load dataset
df = pd.read_pickle("data/test.pkl")

# Evaluate a single sample
before_code = df.iloc[0]['before_merge']
after_code = df.iloc[0]['after_merge']
predicted_code = "your_llm_prediction_here"

result = diff_score(before_code, predicted_code)
print(f"Similarity: {result['codebleu']:.3f}")
```

## Visualization Functions

The `visualization.py` module provides functions for analyzing and visualizing metrics across multiple model responses:

### plot_metrics_boxplots()

Creates boxplots for all diff_score metrics across multiple response columns.

```python
from utils.visualization import plot_metrics_boxplots
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Auto-detect response columns (those starting with "response_")
plot_metrics_boxplots(df)

# Or specify custom response columns
plot_metrics_boxplots(
    df=df,
    reference_column="buggy_code",
    response_columns=["response_model_a", "response_model_b", "response_model_c"],
    figsize=(20, 12),
    title_prefix="Model Comparison"
)
```

**Parameters:**
- `df`: Input dataset containing code columns
- `reference_column`: Name of the reference code column (default: "buggy_code")
- `response_columns`: List of response column names to compare against reference. If None, automatically finds all columns starting with "response_"
- `figsize`: Figure size as (width, height) (default: (20, 12))
- `title_prefix`: Prefix for the overall title (default: "Metrics Comparison")

**Features:**
- Automatically calculates all diff_score metrics for each response column
- Creates subplots for each metric with colored boxplots
- Provides summary statistics for each metric and model
- Handles missing or invalid data gracefully
- Auto-detects response columns if not specified

### plot_column_distribution()

Creates horizontal bar plots showing the distribution of values in a specified column.

```python
from utils.visualization import plot_column_distribution

# Plot distribution of traceback types
plot_column_distribution(df, column_name="traceback_type", top_n=10)
```

## Dependencies

- Python 3.7+
- codebleu
- difflib (built-in)
- ast (built-in)
- matplotlib
- pandas
- numpy 