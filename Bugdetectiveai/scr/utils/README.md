# Generic Metrics System for BugDetectiveAI

This module provides a flexible and extensible metrics system for evaluating code changes, particularly focused on bug fixes. The system is designed to be easily extensible with new metrics while maintaining a consistent interface.

## Overview

The metrics system consists of several key components:

- **BaseMetric**: Abstract base class for all metrics
- **MetricsEvaluator**: Main evaluator that manages multiple metrics
- **DiffBasedMetric**: Implementation of diff-based similarity metric
- **Data Structures**: MetricResult and EvaluationResult for consistent data handling

## Architecture

### Core Classes

#### BaseMetric (Abstract)
All metrics must inherit from this class and implement:
- `calculate_single()`: Calculate metric for a single sample
- `calculate_batch()`: Calculate metric for multiple samples (optional override)
- `aggregate_results()`: Aggregate batch results into statistics (optional override)

#### MetricsEvaluator
Main class that manages multiple metrics:
- Add/remove metrics dynamically
- Evaluate single samples or batches
- Aggregate results across multiple metrics
- Provide unified interface for all metrics

#### DiffBasedMetric
Current implementation that:
- Compares unified diffs between code versions
- Uses IoU (Intersection over Union) of diff hunks
- Provides interpretable results for developers

## Usage Examples

### Basic Usage

```python
from metrics import create_evaluator

# Create evaluator with default metrics
evaluator = create_evaluator()

# Evaluate a single sample
result = evaluator.evaluate_single(
    before_code="def add(a, b):\n    return a + b",
    after_code="def add(a, b):\n    return a + b + 1",
    predicted_code="def add(a, b):\n    return a + b + 1",
    sample_id="example_1"
)

print(f"Similarity: {result.metrics['diff_edit_similarity'].score}")
```

### Batch Evaluation

```python
# Evaluate multiple samples
before_codes = ["def add(a, b):\n    return a + b"]
after_codes = ["def add(a, b):\n    return a + b + 1"]
predicted_codes = ["def add(a, b):\n    return a + b + 1"]

results = evaluator.evaluate_batch(before_codes, after_codes, predicted_codes)
aggregated = evaluator.aggregate_results(results)

print(f"Mean similarity: {aggregated['diff_edit_similarity']['mean_diff_edit_similarity']}")
```

### Custom Metrics

To add a new metric, inherit from BaseMetric:

```python
from metrics import BaseMetric, MetricResult

class MyCustomMetric(BaseMetric):
    def __init__(self):
        super().__init__("my_custom_metric")
    
    def calculate_single(self, before_code, after_code, predicted_code, **kwargs):
        # Your metric calculation logic here
        score = 0.5  # Example score
        metadata = {"custom_info": "example"}
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            metadata=metadata
        )

# Register with evaluator
evaluator = MetricsEvaluator()
evaluator.add_metric(MyCustomMetric())
```

## Data Structures

### MetricResult
```python
@dataclass
class MetricResult:
    metric_name: str
    score: float
    metadata: Dict[str, Any]
    details: Optional[Dict[str, Any]] = None
```

### EvaluationResult
```python
@dataclass
class EvaluationResult:
    sample_id: Optional[str] = None
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Diff-Based Metric Details

The current `DiffBasedMetric` implements:

1. **Unified Diff Generation**: Uses Python's `difflib.unified_diff`
2. **Diff Hunk Parsing**: Extracts structured information from diff output
3. **IoU Similarity**: Calculates Intersection over Union of diff hunks
4. **Metadata Extraction**: Provides detailed information about changes

### Metrics Provided
- `diff_edit_similarity`: IoU similarity between ground truth and predicted diffs
- `total_additions`: Number of additions in predicted diff
- `total_deletions`: Number of deletions in predicted diff
- `num_hunks`: Number of diff hunks
- `hunks`: Detailed information about each diff hunk

## Running Tests

```bash
# Run all tests
python ../tests/test_diff_metrics.py

# Run specific test class
python -m unittest tests.test_diff_metrics.TestDiffBasedMetric
```

## Running Examples

```bash
# Run basic example
python metrics.py

# Run dataset evaluation example
python diff_metrics_example.py
```

## Extending the System

### Adding New Metrics

1. Create a new class inheriting from `BaseMetric`
2. Implement the `calculate_single()` method
3. Optionally override `calculate_batch()` or `aggregate_results()`
4. Register the metric with `MetricsEvaluator`

### Example: String Similarity Metric

```python
import difflib

class StringSimilarityMetric(BaseMetric):
    def __init__(self):
        super().__init__("string_similarity")
    
    def calculate_single(self, before_code, after_code, predicted_code, **kwargs):
        # Compare predicted vs ground truth
        similarity = difflib.SequenceMatcher(None, after_code, predicted_code).ratio()
        
        return MetricResult(
            metric_name=self.name,
            score=similarity,
            metadata={"method": "sequence_matcher"}
        )
```

### Example: AST-Based Metric

```python
import ast

class ASTSimilarityMetric(BaseMetric):
    def __init__(self):
        super().__init__("ast_similarity")
    
    def calculate_single(self, before_code, after_code, predicted_code, **kwargs):
        try:
            gt_ast = ast.parse(after_code)
            pred_ast = ast.parse(predicted_code)
            # Implement AST comparison logic
            similarity = 0.8  # Example
        except SyntaxError:
            similarity = 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=similarity,
            metadata={"method": "ast_comparison"}
        )
```

## Integration with BugDetectiveAI Dataset

The system is designed to work with the BugDetectiveAI dataset format:

```python
import pandas as pd
from metrics import create_evaluator

# Load dataset
df = pd.read_pickle("data/test.pkl")

# Extract code snippets
buggy_codes = df['before_merge'].tolist()
gt_fixed_codes = df['after_merge'].tolist()
llm_fixed_codes = simulate_llm_predictions(buggy_codes)  # Your LLM predictions

# Evaluate
evaluator = create_evaluator(["diff_edit_similarity"])
results = evaluator.evaluate_batch(buggy_codes, gt_fixed_codes, llm_fixed_codes)
aggregated = evaluator.aggregate_results(results)
```

## Future Metrics

The system is designed to easily accommodate additional metrics such as:

- **Exact Match (EM)**: Binary match between predicted and ground truth
- **CodeBleu**: String-based similarity for code
- **AST-Based Similarity**: Abstract Syntax Tree comparison
- **Semantic Similarity**: Using code embeddings
- **Execution-Based Metrics**: Runtime behavior comparison

## Contributing

To add a new metric:

1. Create a new file in the metrics directory
2. Implement your metric class inheriting from `BaseMetric`
3. Add tests in the test directory
4. Update the factory function in `metrics.py` to include your metric
5. Update this README with documentation

## Dependencies

- Python 3.7+
- numpy
- pandas (for dataset handling)
- difflib (built-in)
- dataclasses (built-in)
- abc (built-in) 