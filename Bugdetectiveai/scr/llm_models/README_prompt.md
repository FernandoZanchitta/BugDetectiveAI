# Prompt Builder Module

This module provides a flexible and powerful way to build prompts using Jinja2 templates for the BugDetectiveAI project.

## Features

- **Template-based prompt generation** using Jinja2
- **Multiple prompt types**: correction, analysis, custom
- **Dynamic content injection** with proper escaping
- **Backward compatibility** with existing code
- **Extensible design** for future prompt types

## Installation

The module requires Jinja2, which is included in the project dependencies:

```bash
poetry install
```

## Usage

### Basic Usage

```python
from llm_models.prompt import build_correction_prompt

# Simple correction prompt
prompt = build_correction_prompt(
    buggy_code="def add(a, b): return a + b",
    traceback_error="TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    instruction_prompt="Please fix the type error."
)
```

### Using the PromptBuilder Class

```python
from llm_models.prompt import PromptBuilder

builder = PromptBuilder()

# Build correction prompt with examples
prompt = builder.build_correction_prompt(
    buggy_code=buggy_code,
    traceback_error=error,
    retrieved_examples=[
        {"buggy_code": "x + 'y'", "corrected_code": "x + str(y)"}
    ],
    instruction_prompt="Fix the type error"
)
```

### Analysis Prompts

```python
# Build analysis prompt
prompt = builder.build_analysis_prompt(
    buggy_code=code,
    traceback_error=error,
    analysis_type="performance",
    additional_context={
        "framework": "Django",
        "environment": "production"
    }
)
```

### Custom Templates

```python
# Use custom template string
template = """
You are a {{ role }} expert. Please help with:

**Code**: {{ code }}
**Error**: {{ error }}

Provide a solution with explanation.
"""

prompt = builder.build_custom_prompt(
    template_string=template,
    role="Python debugging",
    code="result = 5 + '3'",
    error="TypeError: unsupported operand type(s) for +: 'int' and 'str'"
)
```

## Template Variables

### Correction Prompts

- `instruction_prompt`: The main instruction for the LLM
- `buggy_code`: The original buggy code
- `traceback_error`: The error message/traceback
- `retrieved_examples`: List of example dictionaries with `buggy_code` and `corrected_code`
- `include_examples`: Boolean to control example inclusion

### Analysis Prompts

- `analysis_type`: Type of analysis (e.g., "general", "performance", "security")
- `buggy_code`: The code to analyze
- `traceback_error`: The error message
- `additional_context`: Dictionary of additional context variables

## File-based Templates

You can also use template files by setting a template directory:

```python
builder = PromptBuilder(template_dir="./templates")

# Load template from file
prompt = builder.build_structured_prompt(
    template_name="correction_template",
    buggy_code=code,
    traceback_error=error
)
```

Template files should have `.jinja` extension and be placed in the template directory.

## Integration with Existing Code

The module is designed to be a drop-in replacement for the existing prompt construction in `open_router.py`. The `evaluate_correction` function now uses the new prompt builder while maintaining the same interface.

## Testing

Run the test file to see examples of all prompt types:

```bash
cd Bugdetectiveai/scr/llm_models
python test_prompt.py
```

## Future Enhancements

- Support for more prompt types (documentation, testing, etc.)
- Template validation and linting
- Prompt versioning and management
- Integration with prompt optimization tools 