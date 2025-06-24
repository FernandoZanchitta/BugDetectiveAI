# Data Loader Module

This module provides functions to load and process the BugDetectiveAI dataset from pickle files.

## Overview

The data loader module handles:
- Loading buggy and stable datasets from pickle files
- Path management for different dataset splits
- Dataset information and statistics
- Filtering datasets by code length
- Error handling and validation

## Functions

### Core Loading Functions

#### `load_buggy_dataset(split='test', base_path=None)`
Load buggy dataset for a specific split.

**Parameters:**
- `split` (str): Dataset split ('train', 'validation', 'valid', or 'test')
- `base_path` (str, optional): Base path to the dataset

**Returns:**
- `pd.DataFrame`: Pandas DataFrame with buggy dataset

**Example:**
```python
from data_loader import load_buggy_dataset

# Load test split
df = load_buggy_dataset('test')
print(f"Loaded {len(df)} samples")

# Load with custom path
df = load_buggy_dataset('train', base_path='/path/to/dataset')
```

#### `load_stable_dataset(split='test', base_path=None)`
Load stable dataset for a specific split.

**Parameters:**
- `split` (str): Dataset split ('train', 'validation', 'valid', or 'test')
- `base_path` (str, optional): Base path to the dataset

**Returns:**
- `pd.DataFrame`: Pandas DataFrame with stable dataset

#### `load_pickle_file(file_path)`
Load any pickle file and return a pandas DataFrame.

**Parameters:**
- `file_path` (str): Path to the pickle file

**Returns:**
- `pd.DataFrame`: Pandas DataFrame loaded from pickle

### Utility Functions

#### `get_dataset_paths(base_path=None)`
Get paths to all dataset files.

**Returns:**
- `Dict[str, str]`: Dictionary mapping dataset names to file paths

#### `get_dataset_info(df)`
Get comprehensive information about a loaded dataset.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze

**Returns:**
- `Dict`: Dictionary with dataset statistics and information

#### `filter_dataset_by_length(df, column='before_merge', max_length=900)`
Filter dataset by code length to avoid context window issues.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to filter
- `column` (str): Column name to check for length
- `max_length` (int): Maximum allowed length

**Returns:**
- `pd.DataFrame`: Filtered DataFrame

## Dataset Structure

The buggy dataset contains the following columns:

- `before_merge`: Original buggy code
- `after_merge`: Fixed code (ground truth)
- `filename`: Source file name
- `function_name`: Function/method name
- `url`: Issue URL
- `source code and errors`: Parsed source code and error messages
- `full_traceback`: Complete traceback report
- `traceback_type`: Exception type
- `before_merge_without_docstrings`: Code without docstrings
- `after_merge_without_docstrings`: Fixed code without docstrings
- `before_merge_docstrings`: Docstrings from original code
- `after_merge_docstrings`: Docstrings from fixed code
- `path_to_snippet_before_merge`: Path to buggy snippet file
- `path_to_snippet_after_merge`: Path to fixed snippet file

## Usage Examples

### Basic Usage

```python
from data_loader import load_buggy_dataset, get_dataset_info

# Load test dataset
df = load_buggy_dataset('test')

# Get dataset information
info = get_dataset_info(df)
print(f"Dataset shape: {info['shape']}")
print(f"Columns: {info['columns']}")

# Show sample data
if len(df) > 0:
    sample = df.iloc[0]
    print(f"Filename: {sample['filename']}")
    print(f"Function: {sample['function_name']}")
    print(f"Buggy code: {sample['before_merge'][:100]}...")
    print(f"Fixed code: {sample['after_merge'][:100]}...")
```

### Filtering by Length

```python
from data_loader import load_buggy_dataset, filter_dataset_by_length

# Load and filter dataset
df = load_buggy_dataset('train')
filtered_df = filter_dataset_by_length(df, max_length=500)

print(f"Original: {len(df)} samples")
print(f"Filtered: {len(filtered_df)} samples")
```

### Loading All Datasets

```python
from data_loader import load_all_datasets

# Load all available datasets
datasets = load_all_datasets()

for name, df in datasets.items():
    print(f"{name}: {len(df)} samples")
```

## File Structure

The expected dataset structure is:

```
data/
└── pytracebugs_dataset_v1/
    ├── buggy_dataset/
    │   ├── bugfixes_train.pickle
    │   ├── bugfixes_valid.pickle
    │   └── bugfixes_test.pickle
    └── stable_dataset/
        ├── stable_code_train.pickle
        ├── stable_code_valid.pickle
        └── stable_code_test.pickle
```

## Configuration

### Custom Dataset Path

To use a custom dataset path:

```python
from data_loader import load_buggy_dataset

# Use custom base path
df = load_buggy_dataset('test', base_path='/path/to/your/dataset')
```

### Default Path

The default path is set to:
```
/Users/zanchitta/Developer/BugDetectiveAI/Bugdetectiveai/data/pytracebugs_dataset_v1
```

To change the default path, modify the `get_dataset_paths()` function in `loader.py`.

## Error Handling

The module provides comprehensive error handling:

- **FileNotFoundError**: When dataset files don't exist
- **ValueError**: When invalid parameters are provided
- **TypeError**: When data cannot be loaded as DataFrame

## Testing

Run the test script to verify the data loader works:

```bash
cd Bugdetectiveai/scr/data_loader
python test_loader.py
```

## Integration with Metrics System

The data loader can be easily integrated with the metrics system:

```python
from data_loader import load_buggy_dataset
from utils.metrics import create_evaluator

# Load dataset
df = load_buggy_dataset('test')

# Extract code snippets
buggy_codes = df['before_merge'].tolist()
gt_fixed_codes = df['after_merge'].tolist()

# Your LLM predictions here
llm_fixed_codes = your_llm_predictions(buggy_codes)

# Evaluate with metrics
evaluator = create_evaluator(['diff_edit_similarity'])
results = evaluator.evaluate_batch(buggy_codes, gt_fixed_codes, llm_fixed_codes)
```

## Dependencies

- Python 3.7+
- pandas
- pickle (built-in)
- os (built-in)
- pathlib (built-in) 