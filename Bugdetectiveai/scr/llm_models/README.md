# LLM Models - OpenRouter Integration

This directory contains LLM model implementations for the BugDetectiveAI project, including OpenRouter integration for accessing multiple AI models through a single API.

## Overview

The OpenRouter integration allows you to access hundreds of AI models from different providers (OpenAI, Anthropic, Google, Meta, etc.) using a single API key and consistent interface.

## Setup

### 1. Install Dependencies

```bash
poetry add openai
```

### 2. Get OpenRouter API Key

1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Navigate to your API keys section
3. Create a new API key
4. Copy the key for use in your application

### 3. Set Environment Variable

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

Or add to your `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Note**: If using Poetry, you can also add environment variables to your `pyproject.toml` or use Poetry's environment management.

## Available Models

OpenRouter provides access to models from multiple providers:

### Anthropic Models
- `anthropic/claude-3.5-sonnet` (Recommended)
- `anthropic/claude-3.5-haiku`
- `anthropic/claude-3-opus`

### OpenAI Models (via OpenRouter)
- `openai/gpt-4`
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`

### Google Models
- `google/gemini-pro`
- `google/gemini-flash-1.5`

### Meta Models
- `meta-llama/llama-3.1-8b-instruct`
- `meta-llama/llama-3.1-70b-instruct`

### Other Providers
- `mistralai/mistral-7b-instruct`
- `nousresearch/nous-hermes-2-mixtral-8x7b-dpo`

## Usage

### Basic Usage

```python
from llm_models.open_router import create_openrouter_model

# Create model instance
model = create_openrouter_model(
    model_name="anthropic/claude-3.5-sonnet",
    temperature=0.1
)

# Generate basic text output
response = await model.generate_basic_output("Hello, how are you?")
print(response)
```

### Progress Monitoring

The OpenRouter model includes clean progress monitoring using tqdm:

```python
# Individual code generation (no verbose output)
result = await model.generate_code_output(prompt, show_progress=True)

# Batch processing with progress bar
prompts = ["prompt1", "prompt2", "prompt3"]
results = await model.generate_batch_outputs(prompts, show_progress=True)
# Shows: Processing prompt 1/3: 100%|██████████| 3/3 [00:30<00:00, 10.0s/prompts]
```

### Advanced Configuration

```python
from llm_models.base_model import ModelConfig
from llm_models.open_router import OpenRouterLLMModel

# Create custom configuration
config = ModelConfig(
    model_name="anthropic/claude-3.5-sonnet",
    api_key="your_api_key",  # Optional if using env var
    temperature=0.2,
    max_tokens=1000
)

# Create model with custom config
model = OpenRouterLLMModel(config)
```

### Structured Output Generation

```python
# Define output schema
schema = {
    "type": "object",
    "properties": {
        "bug_type": {"type": "string", "enum": ["syntax", "logic", "runtime"]},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "description": {"type": "string"},
        "line_number": {"type": "integer"}
    },
    "required": ["bug_type", "severity", "description"]
}

# Generate structured output
result = await model.generate_structured_output(
    prompt="Analyze this Python code for bugs: print('Hello World'",
    output_schema=schema
)

if result.success:
    print(f"Bug Type: {result.content['bug_type']}")
    print(f"Severity: {result.content['severity']}")
    print(f"Description: {result.content['description']}")
else:
    print(f"Error: {result.error_message}")
```

### Error Handling

```python
try:
    response = await model.generate_basic_output("Your prompt here")
    print(response)
except RuntimeError as e:
    print(f"API Error: {e}")
except ValueError as e:
    print(f"Configuration Error: {e}")
```

## Integration with BugDetectiveAI

### Using in Detective Module

```python
from llm_models.open_router import create_openrouter_model
from bug_detective.detective import BugDetective

# Create OpenRouter model
model = create_openrouter_model(
    model_name="anthropic/claude-3.5-sonnet",
    temperature=0.1
)

# Initialize bug detective with OpenRouter model
detective = BugDetective(model=model)

# Analyze code for bugs
result = await detective.analyze_code(
    code="your_python_code_here",
    error_traceback="error_traceback_here"
)
```

### Using in Data Loader

```python
from llm_models.open_router import create_openrouter_model
from data_loader.loader import DataLoader

# Create model for data processing
model = create_openrouter_model(
    model_name="anthropic/claude-3.5-haiku",  # Faster, cheaper for data processing
    temperature=0.0
)

# Use in data loader
loader = DataLoader(model=model)
```

### Dataset Processing with Progress Monitoring

The detective module includes progress monitoring for dataset processing:

```python
from bug_detective.detective import process_prompt_dataset
import pandas as pd

# Create sample dataset
df = pd.DataFrame({
    'before_merge': ['buggy_code_1', 'buggy_code_2'],
    'full_traceback': ['error_1', 'error_2']
})

# Process with progress monitoring
results = await process_prompt_dataset(model, df)
# Shows: Processing sample 1/2: 100%|██████████| 2/2 [00:45<00:00, 22.5s/samples]
```

## Cost Optimization

### Model Selection by Use Case

- **Code Analysis**: `anthropic/claude-3.5-sonnet` (best performance)
- **Data Processing**: `anthropic/claude-3.5-haiku` (faster, cheaper)
- **Quick Tests**: `meta-llama/llama-3.1-8b-instruct` (very cheap)
- **High Accuracy**: `anthropic/claude-3.5-sonnet` (most expensive but highest quality)

### Temperature Settings

- **Structured Output**: `temperature=0.0` (most consistent)
- **Creative Analysis**: `temperature=0.1-0.3` (balanced)
- **Exploration**: `temperature=0.5+` (more creative)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: Invalid configuration
   ```
   Solution: Set `OPENROUTER_API_KEY` environment variable

2. **Model Not Available**
   ```
   Error: Model not found
   ```
   Solution: Check model name spelling and availability on OpenRouter

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   Solution: Implement retry logic or use a different model

4. **Import Error**
   ```
   Error: OpenAI package not installed
   ```
   Solution: Run `poetry add openai`

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create model with debug info
model = create_openrouter_model(
    model_name="anthropic/claude-3.5-sonnet",
    temperature=0.1
)
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple text generation
- `single_test.py` - Individual model testing
- Integration examples in the main project modules

## API Reference

### OpenRouterLLMModel

- `generate_basic_output(prompt: str) -> str`
- `generate_structured_output(prompt: str, output_schema: dict) -> StructuredOutput`
- `validate_config() -> bool`

### create_openrouter_model()

- `model_name: str` - Model identifier (e.g., "anthropic/claude-3.5-sonnet")
- `api_key: Optional[str]` - OpenRouter API key
- `temperature: float` - Sampling temperature (0.0-2.0)
- `max_tokens: Optional[int]` - Maximum tokens to generate

## Support

For issues with:
- **OpenRouter API**: Check [OpenRouter Documentation](https://openrouter.ai/docs)
- **Model Availability**: Visit [OpenRouter Models](https://openrouter.ai/models)
- **Poetry Issues**: Refer to the project's `pyproject.toml` for dependency management
 