"""
LLM Models package for BugDetectiveAI.
Supports OpenAI and OpenRouter models, optimized for code generation.
"""

from .base_model import BaseLLMModel, ModelConfig, StructuredOutput
from .openai_model import OpenAILLMModel, create_openai_model
from .open_router import OpenRouterLLMModel, create_openrouter_model

__all__ = [
    "BaseLLMModel",
    "ModelConfig", 
    "StructuredOutput",
    "OpenAILLMModel",
    "OpenRouterLLMModel",
    "create_openai_model",
    "create_openrouter_model"
] 