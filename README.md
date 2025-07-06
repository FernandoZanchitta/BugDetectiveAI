# BugDetectiveAI

A comprehensive framework for automated program repair (APR) using Large Language Models (LLMs) to detect and fix bugs in Python code, with multi-metric evaluation capabilities.

## Overview

BugDetectiveAI leverages state-of-the-art LLMs to act as intelligent code detectives, analyzing buggy code snippets along with their associated error traces to generate corrected implementations. The framework provides a systematic approach to evaluate repair quality using multiple complementary metrics.

## Key Features

- **Multi-LLM Support**: Integration with various LLM providers through OpenRouter API (Qwen, GPT-4, etc.)
- **Comprehensive Evaluation**: Multiple evaluation metrics including AST-based similarity, text similarity, and CodeBLEU scores
- **Checkpoint System**: Robust checkpointing for long-running experiments with automatic resume capability
- **Structured Prompting**: Jinja2-based template system for consistent and customizable prompts
- **Progress Monitoring**: Clean progress tracking with Tqdm integration
- **Dataset Management**: Efficient loading and processing of buggy code datasets

## Architecture

### Core Components

- **`bug_detective/`**: Main processing pipeline for LLM-based code correction
- **`llm_models/`**: LLM integration layer with OpenRouter support and prompt management
- **`data_loader/`**: Dataset loading utilities for buggy code collections
- **`utils/`**: Evaluation metrics and utility functions
- **`examples/`**: Usage examples and tutorials

### Data Flow

1. **Dataset Loading**: Load buggy code snippets with associated error traces
2. **Prompt Generation**: Create structured prompts using Jinja2 templates
3. **LLM Processing**: Generate corrections using configured LLM models
4. **Evaluation**: Assess repair quality using multiple metrics
5. **Checkpointing**: Save progress for experiment continuity

## Dataset Structure

The framework works with buggy code datasets containing:

- **`before_merge`**: Original buggy code implementation
- **`after_merge`**: Ground truth corrected implementation
- **`full_traceback`**: Complete error traceback information
- **`function_name`**: Name of the function containing the bug
- **`filename`**: Source file containing the bug
- **Additional metadata**: Bug type, description, line ranges, etc.

## Evaluation Metrics

### Primary Metrics

- **Exact Match (EM)**: Direct string comparison between generated and ground truth fixes
- **AST-Based Similarity**: Structural similarity using Abstract Syntax Tree comparison
- **Text-Based Similarity**: String similarity using difflib
- **CodeBLEU**: Multi-dimensional code similarity metric including:
  - N-gram match score
  - Weighted n-gram match score
  - Syntax match score
  - Dataflow match score

### Normalized Scoring

The framework provides both raw and normalized AST scores to account for variable naming differences while preserving structural similarity.

## Usage Example

```python
from data_loader.loader import load_buggy_dataset
from bug_detective.detective import process_prompt_dataset
from llm_models.open_router import create_openrouter_model

# Load dataset
dataset = load_buggy_dataset('train')

# Configure LLM
model = create_openrouter_model(
    model_name="qwen/qwen-2.5-coder-32b-instruct",
    temperature=0.0,
    max_tokens=1000
)

# Process dataset
responses = await process_prompt_dataset(model, dataset)

# Evaluate results
from utils.simple_metrics import diff_score, codebleu
# ... evaluation code
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd BugDetectiveAI

# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell
```

## Dependencies

- Python 3.11+
- pandas, numpy, scikit-learn
- tqdm, ipywidgets, jupyter
- openai, huggingface-hub
- jinja2, codebleu
- tree-sitter-python

## Project Structure

```
BugDetectiveAI/
├── Bugdetectiveai/
│   └── scr/
│       ├── bug_detective/     # Main processing pipeline
│       ├── llm_models/        # LLM integration layer
│       ├── data_loader/       # Dataset utilities
│       ├── utils/             # Evaluation metrics
│       ├── examples/          # Usage examples
│       └── notebooks/         # Jupyter notebooks
├── data/                      # Dataset storage
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Contributing

This project is designed for research in automated program repair. Contributions are welcome for:

- Additional evaluation metrics
- New LLM model integrations
- Dataset processing improvements
- Documentation enhancements

## License

[License information to be added]
