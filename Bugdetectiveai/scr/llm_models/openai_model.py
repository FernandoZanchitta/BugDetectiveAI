"""
OpenAI LLM model implementation with structured output capabilities.
"""

import json
from typing import Dict, Any, Optional
from .base_model import BaseLLMModel, ModelConfig, StructuredOutput
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAILLMModel(BaseLLMModel):
    """OpenAI LLM model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return bool(self.config.api_key and self.config.model_name)
    
    async def _initialize_client(self):
        """Initialize OpenAI client."""
        if not self.client:
            try:
                import openai
                self.client = openai.AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        output_schema: Dict[str, Any]
    ) -> StructuredOutput:
        """Generate structured output using OpenAI function calling."""
        try:
            await self._initialize_client()
            
            if self.client is None:
                raise RuntimeError("OpenAI client not initialized")
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                functions=[{
                    "name": "extract_structured_data",
                    "description": "Extract structured data from the input",
                    "parameters": output_schema
                }],
                function_call={"name": "extract_structured_data"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            function_call = response.choices[0].message.function_call
            if function_call:
                content = json.loads(function_call.arguments)
                return StructuredOutput(success=True, content=content)
            else:
                return StructuredOutput(
                    success=False, 
                    content={}, 
                    error_message="No function call returned"
                )
                
        except Exception as e:
            return StructuredOutput(
                success=False, 
                content={}, 
                error_message=str(e)
            )
    
    async def generate_basic_output(self, prompt: str) -> str:
        """Generate basic text output."""
        try:
            await self._initialize_client()
            
            if self.client is None:
                raise RuntimeError("OpenAI client not initialized")
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("No content returned from OpenAI API")
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") 


def evaluate_correction(prompt, buggy_code, traceback_error, retrieved_examples=None, model="gpt-4"):
    """Evaluate an LLM's ability to correct a piece of code.
    
    Args:
        prompt (str): The instruction prompt for the LLM.
        buggy_code (str): The original buggy code.
        traceback_error (str): The traceback error associated with the code.
        retrieved_examples (list of dict): List of examples in the form 
            [{"buggy_code": str, "corrected_code": str}], optional.
        model (str): The LLM model to use.
    Returns:
        str: The corrected code returned by the LLM.
    """
    # Build the final prompt
    message_parts = []
    message_parts.append(prompt)
    message_parts.append("\n### BUGGY CODE:\n" + buggy_code)
    message_parts.append("\n### ERROR:\n" + traceback_error)

    if retrieved_examples:
        examples_str = ""
        for ex in retrieved_examples:
            examples_str += (
                "#### EXAMPLE:\n"
                f"Buggy:\n{ex['buggy_code']}\nCorrected:\n{ex['corrected_code']}\n\n"
            )
        message_parts.append("\n### RETRIEVED EXAMPLES:\n" + examples_str)

    message_parts.append("\n### RETURN ONLY THE CORRECTED CODE BELOW:\n")

    final_message = "\n".join(message_parts)

    # Call the LLM
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": final_message}],
        temperature=0,
    )
    corrected_code = response.choices[0].message.content

    return corrected_code