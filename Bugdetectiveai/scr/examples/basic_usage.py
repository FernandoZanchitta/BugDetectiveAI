"""
Basic usage example for BugDetectiveAI with checkpoint support.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
from llm_models.open_router import create_openrouter_model
from bug_detective.detective import process_prompt_dataset
from utils.checkpoints import list_checkpoints, delete_checkpoint


async def main():
    """Example usage with checkpoint support."""
    
    # Create sample dataset
    sample_data = {
        'before_merge': [
            'def add(a, b):\n    return a + b\n',
            'def multiply(x, y):\n    return x * y\n'
        ],
        'full_traceback': [
            'TypeError: unsupported operand type(s) for +: \'int\' and \'str\'',
            'NameError: name \'x\' is not defined'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create model
    model = create_openrouter_model(
        model_name="anthropic/claude-3.5-sonnet",
        temperature=0.0
    )
    
    # Process with checkpoint support
    print("Processing dataset with checkpoint support...")
    responses = await process_prompt_dataset(
        open_router_model=model,
        prompt_dataset=df,
        dataset_name="example_dataset",  # Custom name for checkpoint
        save_frequency=2  # Save every 2 samples
    )
    
    print(f"\nGenerated {len(responses)} responses:")
    for i, response in enumerate(responses):
        print(f"Sample {i+1}: {response[:100]}...")
    
    # List available checkpoints
    print(f"\nAvailable checkpoints: {list_checkpoints()}")
    
    # Example: Delete checkpoint if needed
    # delete_checkpoint("example_dataset")


if __name__ == "__main__":
    asyncio.run(main()) 