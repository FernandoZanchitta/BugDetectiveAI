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

## Dependencies

- Python 3.7+
- codebleu
- difflib (built-in)
- ast (built-in) 