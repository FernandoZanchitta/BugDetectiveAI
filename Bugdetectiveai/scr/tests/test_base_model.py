"""
Tests for base LLM model classes.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_models.base_model import ModelConfig, StructuredOutput, BaseLLMModel


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig dataclass."""
    
    def test_basic_config(self):
        """Test basic configuration creation."""
        config = ModelConfig(
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.1,
            max_tokens=1000,
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1"
        )
        
        self.assertEqual(config.model_name, "anthropic/claude-3.5-sonnet")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
    
    def test_default_config(self):
        """Test configuration with default values."""
        config = ModelConfig(model_name="anthropic/claude-3.5-sonnet")
        
        self.assertEqual(config.model_name, "anthropic/claude-3.5-sonnet")
        self.assertEqual(config.temperature, 0.1)  # Default
        self.assertIsNone(config.max_tokens)  # Default
        self.assertIsNone(config.api_key)  # Default
        self.assertIsNone(config.base_url)  # Default


class TestStructuredOutput(unittest.TestCase):
    """Test cases for StructuredOutput dataclass."""
    
    def test_successful_output(self):
        """Test successful structured output."""
        content = {"bug_type": "syntax_error", "severity": "high"}
        output = StructuredOutput(success=True, content=content)
        
        self.assertTrue(output.success)
        self.assertEqual(output.content, content)
        self.assertIsNone(output.error_message)
    
    def test_failed_output(self):
        """Test failed structured output."""
        error_msg = "API rate limit exceeded"
        output = StructuredOutput(success=False, content={}, error_message=error_msg)
        
        self.assertFalse(output.success)
        self.assertEqual(output.content, {})
        self.assertEqual(output.error_message, error_msg)
    
    def test_output_without_error_message(self):
        """Test output without error message."""
        output = StructuredOutput(success=False, content={})
        
        self.assertFalse(output.success)
        self.assertIsNone(output.error_message)


class TestBaseLLMModel(unittest.TestCase):
    """Test cases for BaseLLMModel abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseLLMModel is abstract and cannot be instantiated."""
        with self.assertRaises(TypeError):
            BaseLLMModel(ModelConfig(model_name="test"))
    
    def test_concrete_implementation(self):
        """Test concrete implementation of BaseLLMModel."""
        class MockLLMModel(BaseLLMModel):
            async def generate_structured_output(self, prompt: str, output_schema: Dict[str, Any]) -> StructuredOutput:
                return StructuredOutput(success=True, content={"test": "data"})
            
            async def generate_basic_output(self, prompt: str) -> str:
                return "Test response"
            
            def validate_config(self) -> bool:
                return True
        
        config = ModelConfig(model_name="mock-model")
        model = MockLLMModel(config)
        
        self.assertEqual(model.config, config)
        self.assertTrue(model.validate_config())


if __name__ == "__main__":
    unittest.main() 