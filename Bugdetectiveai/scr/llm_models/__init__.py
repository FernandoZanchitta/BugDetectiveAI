"""
LLM Models package for BugDetectiveAI.
Supports OpenRouter models, optimized for code generation.
"""

from .base_model import BaseLLMModel, ModelConfig, StructuredOutput
from .open_router import OpenRouterLLMModel, create_openrouter_model

__all__ = [
    "BaseLLMModel",
    "ModelConfig", 
    "StructuredOutput",
    "OpenRouterLLMModel",
    "create_openrouter_model"
] 