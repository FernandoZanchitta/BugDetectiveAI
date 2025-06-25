"""
LLM Models package for BugDetectiveAI.
Currently supports OpenAI, designed for easy extension to other providers.
"""

from .base_model import BaseLLMModel, ModelConfig, StructuredOutput
from .openai_model import OpenAILLMModel

__all__ = [
    "BaseLLMModel",
    "ModelConfig", 
    "StructuredOutput",
    "OpenAILLMModel"
] 