"""
Base abstract class for LLM models with structured output capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for LLM models."""

    model_name: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class StructuredOutput:
    """Base structured output format."""

    success: bool
    content: Dict[str, Any]
    error_message: Optional[str] = None


class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate_structured_output(
        self, prompt: str, output_schema: Dict[str, Any]
    ) -> StructuredOutput:
        """Generate structured output based on schema."""
        pass

    @abstractmethod
    async def generate_basic_output(self, prompt: str) -> str:
        """Generate basic text output."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate model configuration."""
        pass
