"""
Bug Detective module for processing datasets with LLM models.
"""

import asyncio
from typing import List, Optional
from tqdm import tqdm
import pandas as pd
from llm_models.open_router import OpenRouterLLMModel
from llm_models.prompt import PromptBuilder
from utils.checkpoints import (
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    list_checkpoints,
    delete_checkpoint,
)


async def process_prompt_dataset(
    open_router_model: OpenRouterLLMModel,
    prompt_dataset: pd.DataFrame,
    dataset_name: Optional[str] = None,
    save_frequency: int = 5,
) -> List[str]:
    """
    Process a prompt dataset using an OpenRouter LLM model with checkpoint support.

    Args:
        open_router_model: Configured OpenRouter LLM model instance
        prompt_dataset: DataFrame containing 'before_merge' and 'full_traceback' columns
        dataset_name: Name for checkpoint file (defaults to dataset hash)
        save_frequency: How often to save checkpoint (every N samples)

    Returns:
        List of model responses for each sample in the dataset

    Raises:
        ValueError: If required columns are missing from the dataset
    """
    # Validate required columns
    required_columns = ["before_merge", "full_traceback"]
    missing_columns = [
        col for col in required_columns if col not in prompt_dataset.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Generate dataset name if not provided
    if dataset_name is None:
        dataset_name = f"dataset_{hash(str(prompt_dataset.columns.tolist()))}"

    # Setup checkpoint
    checkpoint_path = get_checkpoint_path(dataset_name)
    checkpoint_data = load_checkpoint(checkpoint_path)

    responses = checkpoint_data["responses"]
    processed_indices = set(checkpoint_data["processed_indices"])

    # Initialize prompt builder
    prompt_builder = PromptBuilder()

    # Create progress bar
    pbar = tqdm(
        total=len(prompt_dataset), desc="Processing dataset", unit="samples", ncols=100
    )

    # Set initial progress
    pbar.update(len(processed_indices))

    try:
        for index, (_, row) in enumerate(prompt_dataset.iterrows()):
            # Skip already processed samples
            if index in processed_indices:
                continue

            try:
                # Update progress bar description
                pbar.set_description(
                    f"Processing sample {index + 1}/{len(prompt_dataset)}"
                )

                # Build correction prompt
                prompt = prompt_builder.build_correction_prompt(
                    buggy_code=str(row["before_merge"]),
                    traceback_error=str(row["full_traceback"]),
                )

                # Generate response
                response = await open_router_model.generate_basic_output(prompt)
                responses.append(response)
                processed_indices.add(index)

                # Save checkpoint periodically
                if len(processed_indices) % save_frequency == 0:
                    save_checkpoint(checkpoint_path, responses, processed_indices)

                # Update progress bar
                pbar.update(1)

            except Exception as e:
                # Handle errors gracefully and add error message
                error_response = f"Error processing sample {index}: {str(e)}"
                responses.append(error_response)
                processed_indices.add(index)
                pbar.update(1)

                # Save checkpoint on error
                save_checkpoint(checkpoint_path, responses, processed_indices)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint...")
        save_checkpoint(checkpoint_path, responses, processed_indices)
        print(f"Checkpoint saved to {checkpoint_path}")
        raise

    finally:
        # Always save final checkpoint
        save_checkpoint(checkpoint_path, responses, processed_indices)
        pbar.close()

    return responses
