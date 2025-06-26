"""
Tests for OpenAI LLM model implementation.
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_models.openai_model import OpenAILLMModel
from llm_models.base_model import ModelConfig, StructuredOutput


class TestOpenAILLMModel(unittest.TestCase):
    """Test cases for OpenAILLMModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=1000,
            api_key="test-api-key",
            base_url="https://api.openai.com/v1"
        )
        self.model = OpenAILLMModel(self.config)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        self.assertTrue(self.model.validate_config())
    
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = ModelConfig(model_name="gpt-4", api_key=None)
        model = OpenAILLMModel(config)
        self.assertFalse(model.validate_config())
    
    def test_validate_config_missing_model_name(self):
        """Test configuration validation with missing model name."""
        config = ModelConfig(model_name="", api_key="test-key")
        model = OpenAILLMModel(config)
        self.assertFalse(model.validate_config())
    
    @patch('openai.AsyncOpenAI')
    async def test_initialize_client_success(self, mock_openai):
        """Test successful client initialization."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        await self.model._initialize_client()
        
        mock_openai.assert_called_once_with(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        self.assertEqual(self.model.client, mock_client)
    
    @patch('openai.AsyncOpenAI')
    async def test_initialize_client_already_initialized(self, mock_openai):
        """Test client initialization when already initialized."""
        mock_client = AsyncMock()
        self.model.client = mock_client
        
        await self.model._initialize_client()
        
        # Should not create new client
        mock_openai.assert_not_called()
        self.assertEqual(self.model.client, mock_client)
    
    @patch('openai.AsyncOpenAI')
    async def test_initialize_client_import_error(self, mock_openai):
        """Test client initialization with import error."""
        mock_openai.side_effect = ImportError("OpenAI package not installed")
        
        with self.assertRaises(ImportError) as context:
            await self.model._initialize_client()
        
        self.assertIn("OpenAI package not installed", str(context.exception))
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_structured_output_success(self, mock_openai):
        """Test successful structured output generation."""
        # Mock the client and response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_function_call = Mock()
        mock_function_call.arguments = '{"bug_type": "syntax_error", "severity": "high"}'
        mock_message.function_call = mock_function_call
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        schema = {
            "type": "object",
            "properties": {
                "bug_type": {"type": "string"},
                "severity": {"type": "string"}
            }
        }
        
        result = await self.model.generate_structured_output("Test prompt", schema)
        
        self.assertTrue(result.success)
        self.assertEqual(result.content, {"bug_type": "syntax_error", "severity": "high"})
        self.assertIsNone(result.error_message)
        
        # Verify the API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        self.assertEqual(call_args[1]['temperature'], 0.1)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_structured_output_no_function_call(self, mock_openai):
        """Test structured output generation with no function call."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.function_call = None  # No function call
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        schema = {"type": "object", "properties": {}}
        result = await self.model.generate_structured_output("Test prompt", schema)
        
        self.assertFalse(result.success)
        self.assertEqual(result.content, {})
        self.assertEqual(result.error_message, "No function call returned")
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_structured_output_api_error(self, mock_openai):
        """Test structured output generation with API error."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        schema = {"type": "object", "properties": {}}
        result = await self.model.generate_structured_output("Test prompt", schema)
        
        self.assertFalse(result.success)
        self.assertEqual(result.content, {})
        self.assertEqual(result.error_message, "API Error")
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_basic_output_success(self, mock_openai):
        """Test successful basic output generation."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = await self.model.generate_basic_output("Test prompt")
        
        self.assertEqual(result, "This is a test response")
        
        # Verify the API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        self.assertEqual(call_args[1]['temperature'], 0.1)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_basic_output_api_error(self, mock_openai):
        """Test basic output generation with API error."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with self.assertRaises(RuntimeError) as context:
            await self.model.generate_basic_output("Test prompt")
        
        self.assertIn("OpenAI API error: API Error", str(context.exception))


def run_async_test(test_func):
    """Helper function to run async tests."""
    return asyncio.run(test_func())


if __name__ == "__main__":
    # Create test suite for async tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add async tests
    test_cases = [
        TestOpenAILLMModel('test_initialize_client_success'),
        TestOpenAILLMModel('test_initialize_client_already_initialized'),
        TestOpenAILLMModel('test_initialize_client_import_error'),
        TestOpenAILLMModel('test_generate_structured_output_success'),
        TestOpenAILLMModel('test_generate_structured_output_no_function_call'),
        TestOpenAILLMModel('test_generate_structured_output_api_error'),
        TestOpenAILLMModel('test_generate_basic_output_success'),
        TestOpenAILLMModel('test_generate_basic_output_api_error'),
    ]
    
    for test_case in test_cases:
        suite.addTest(test_case)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite) 