# BugDetectiveAI - Structured Output

A simple, focused implementation for bug detection using OpenAI with structured output.

## Directory Structure

```
scr/
├── llm_models/           # LLM model implementations
│   ├── __init__.py
│   ├── base_model.py     # Abstract base class
│   └── openai_model.py   # OpenAI implementation
├── structured_output/    # Structured output system
│   ├── __init__.py
│   ├── schemas.py        # Schema definitions
│   └── output_processor.py # Output processing
├── bug_detective/        # Main bug detection logic
│   ├── __init__.py
│   └── detective.py      # Main BugDetective class
└── examples/            # Usage examples
    ├── __init__.py
    └── basic_usage.py   # Basic usage example
```

## Features

- **OpenAI Integration**: Uses function calling for structured output
- **Basic & Concise Schemas**: Two levels of detail for bug analysis
- **Simple API**: Easy to use with minimal setup

## Quick Start

```python
import asyncio
from Bugdetectiveai.scr.llm_models.base_model import ModelConfig
from Bugdetectiveai.scr.llm_models.openai_model import OpenAILLMModel
from Bugdetectiveai.scr.bug_detective.detective import BugDetective

async def main():
    config = ModelConfig(
        model_name="gpt-4",
        temperature=0.1,
        api_key="your-openai-api-key"
    )
    
    model = OpenAILLMModel(config)
    detective = BugDetective(model)
    
    code = """
    def divide(a, b):
        return a / b  # No division by zero check
    """
    
    result = await detective.analyze_bug(code, concise=True)
    
    if result.success:
        print(f"Bug: {result.bug_analysis['bug_type']}")
        print(f"Severity: {result.bug_analysis['severity']}")

asyncio.run(main())
```

## Schema Types

### Basic Schema
```json
{
  "bug_type": "string",
  "severity": "low|medium|high|critical",
  "description": "string",
  "location": "string"
}
```

### Concise Schema
```json
{
  "bug_type": "string",
  "severity": "string",
  "description": "string",
  "location": "string",
  "suggested_fix": "string",
  "confidence": "number"
}
```

## Setup

1. Install dependencies:
```bash
pip install openai
```

2. Set environment variable:
```bash
export OPENAI_API_KEY="your-key"
```

3. Run example:
```bash
cd Bugdetectiveai/scr/examples
python basic_usage.py
```

## Extending

To add more LLM providers, implement the `BaseLLMModel` interface:

```python
class NewLLMModel(BaseLLMModel):
    async def generate_structured_output(self, prompt: str, schema: dict) -> StructuredOutput:
        # Your implementation here
        pass
    
    async def generate_basic_output(self, prompt: str) -> str:
        # Your implementation here
        pass
    
    def validate_config(self) -> bool:
        # Your validation here
        pass
``` 