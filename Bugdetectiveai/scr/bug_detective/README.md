# Bug Detective Module

This module provides robust dataset processing with automatic checkpoint support to prevent loss of progress and API credits.

## Features

- **Checkpoint Support**: Automatically saves progress to `data/checkpoints/` directory
- **Resume Capability**: Can resume from where it left off if interrupted
- **Error Handling**: Graceful error handling with automatic checkpoint saving
- **Progress Monitoring**: Real-time progress tracking with tqdm

## Usage

### Basic Processing

```python
import asyncio
import pandas as pd
from llm_models.open_router import create_openrouter_model
from bug_detective.detective import process_prompt_dataset

async def main():
    # Create your dataset
    df = pd.DataFrame({
        'before_merge': ['your code here'],
        'full_traceback': ['error message here']
    })
    
    # Create model
    model = create_openrouter_model()
    
    # Process with checkpoint support
    responses = await process_prompt_dataset(
        open_router_model=model,
        prompt_dataset=df,
        dataset_name="my_dataset",  # Optional: custom checkpoint name
        save_frequency=5  # Optional: save every 5 samples
    )
    
    print(f"Generated {len(responses)} responses")

asyncio.run(main())
```

### Checkpoint Management

```python
from utils.checkpoints import list_checkpoints, delete_checkpoint, get_checkpoint_info

# List all available checkpoints
checkpoints = list_checkpoints()
print(f"Available checkpoints: {checkpoints}")

# Get info about a specific checkpoint
info = get_checkpoint_info("my_dataset")
print(f"Checkpoint info: {info}")

# Delete a specific checkpoint
success = delete_checkpoint("my_dataset")
```

## Checkpoint Files

Checkpoints are stored as JSON files in `data/checkpoints/` with the format:
- Filename: `{dataset_name}_responses.json`
- Content: `{"responses": [...], "processed_indices": [...]}`

## Error Recovery

The system handles several types of interruptions:

1. **KeyboardInterrupt (Ctrl+C)**: Saves checkpoint and exits gracefully
2. **API Errors**: Saves checkpoint and continues with next sample
3. **Network Issues**: Saves checkpoint and retries
4. **System Crashes**: Can resume from last saved checkpoint

## Configuration

- `dataset_name`: Custom name for checkpoint file (defaults to dataset hash)
- `save_frequency`: How often to save checkpoint (default: 5 samples)
- Checkpoint directory: `data/checkpoints/` (created automatically)

## Available Checkpoint Utilities

The checkpoint functionality is organized in `utils/checkpoints.py`:

- `get_checkpoint_path()`: Get checkpoint file path
- `load_checkpoint()`: Load existing checkpoint data
- `save_checkpoint()`: Save current progress
- `list_checkpoints()`: List all available checkpoints
- `delete_checkpoint()`: Delete specific checkpoint
- `get_checkpoint_info()`: Get detailed checkpoint information
- `clear_all_checkpoints()`: Remove all checkpoints

## Best Practices

1. Use descriptive `dataset_name` for easy identification
2. Set appropriate `save_frequency` based on dataset size
3. Monitor checkpoint directory for disk space
4. Clean up old checkpoints when no longer needed
5. Use checkpoint utilities from `utils.checkpoints` for management tasks 