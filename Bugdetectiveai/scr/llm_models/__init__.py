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
    "create_openrouter_model",
]
models = {
    "qwen": {
        "model_name": "qwen-2.5-coder-32b-instruct",
        "open_router_name": "qwen/qwen-2.5-coder-32b-instruct",
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "open_router_name": "openai/gpt-4o-mini",
    },
    "claude-3.5-sonnet": {
        "model_name": "claude-3.5-sonnet",
        "open_router_name": "anthropic/claude-3.5-sonnet",
    },
}
