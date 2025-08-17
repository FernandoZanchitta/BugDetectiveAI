"""
Bug Detective module for processing datasets with LLM models.
"""

import asyncio
from typing import List, Optional
from tqdm import tqdm
import pandas as pd
from llm_models.open_router import OpenRouterLLMModel


async def process_prompt_dataset(
    open_router_model: OpenRouterLLMModel,
    prompt_dataset: pd.DataFrame,
    dataset_name: Optional[str] = None,
    save_frequency: int = 5,
    instruction_prompt: str = "You are a helpful assistant that corrects the code based on the traceback error.",
) -> List[str]:
    """
    Process a prompt dataset using an OpenRouter LLM model.

    Args:
        open_router_model: Configured OpenRouter LLM model instance
        prompt_dataset: DataFrame containing 'before_merge' and 'full_traceback' columns
        dataset_name: Name for dataset (optional)
        save_frequency: How often to save (every N samples)
        instruction_prompt: Instruction prompt for the model

    Returns:
        List of model responses for each sample in the dataset

    Raises:
        ValueError: If required columns are missing from the dataset
    """
    # Validate required columns
    required_columns = ["before_merge_without_docstrings", "full_traceback"]
    missing_columns = [
        col for col in required_columns if col not in prompt_dataset.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    responses = []
    processed_indices = set()

    # Create progress bar
    pbar = tqdm(
        total=len(prompt_dataset), desc="Processing dataset", unit="samples", ncols=100
    )

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

                # Build simple correction prompt
                prompt = f"""
{instruction_prompt}

Buggy code:
{str(row["before_merge_without_docstrings"])}

Traceback error:
{str(row["full_traceback"])}

Please provide the corrected code:
"""

                # Generate response
                response = await open_router_model.generate_code_output(prompt)
                responses.append(response)
                processed_indices.add(index)

                # Update progress
                pbar.update(1)

            except Exception as e:
                error_msg = f"Error processing sample {index}: {str(e)}"
                responses.append(error_msg)
                processed_indices.add(index)
                pbar.update(1)

    finally:
        pbar.close()

    return responses
