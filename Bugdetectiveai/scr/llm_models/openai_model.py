"""
OpenAI LLM model implementation with structured output capabilities.
"""

import json
from typing import Dict, Any, Optional
from .base_model import BaseLLMModel, ModelConfig, StructuredOutput


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
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") 