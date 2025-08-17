"""
LLM models module for BugDetectiveAI.
"""

from .open_router import OpenRouterLLMModel, create_openrouter_model, create_apr_model

__all__ = [
    "OpenRouterLLMModel",
    "create_openrouter_model",
    "create_apr_model",
]
