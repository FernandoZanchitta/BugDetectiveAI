"""
Bug Detective module for processing datasets with LLM models.
"""

import asyncio
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from llm_models.open_router import OpenRouterLLMModel
from llm_models.prompt import PromptBuilder


async def process_prompt_dataset(
    open_router_model: OpenRouterLLMModel,
    prompt_dataset: pd.DataFrame
) -> List[str]:
    """
    Process a prompt dataset using an OpenRouter LLM model.
    
    Args:
        open_router_model: Configured OpenRouter LLM model instance
        prompt_dataset: DataFrame containing 'before_merge' and 'full_traceback' columns
        
    Returns:
        List of model responses for each sample in the dataset
        
    Raises:
        ValueError: If required columns are missing from the dataset
    """
    # Validate required columns
    required_columns = ['before_merge', 'full_traceback']
    missing_columns = [col for col in required_columns if col not in prompt_dataset.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize prompt builder
    prompt_builder = PromptBuilder()
    
    # Process each sample with progress monitoring
    responses = []
    
    # Create progress bar
    pbar = tqdm(
        total=len(prompt_dataset),
        desc="Processing dataset",
        unit="samples",
        ncols=100
    )
    
    for index, (_, row) in enumerate(prompt_dataset.iterrows()):
        try:
            # Update progress bar description
            pbar.set_description(f"Processing sample {index + 1}/{len(prompt_dataset)}")
            
            # Build correction prompt
            prompt = prompt_builder.build_correction_prompt(
                buggy_code=str(row['before_merge']),
                traceback_error=str(row['full_traceback'])
            )
            
            # Generate response
            response = await open_router_model.generate_basic_output(prompt)
            responses.append(response)
            
            # Update progress bar
            pbar.update(1)
            
        except Exception as e:
            # Handle errors gracefully and add error message
            error_response = f"Error processing sample {index}: {str(e)}"
            responses.append(error_response)
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    return responses
