"""
Tests for tqdm progress monitoring features.
"""

import unittest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from llm_models.open_router import OpenRouterLLMModel, create_openrouter_model
from llm_models.base_model import ModelConfig
from bug_detective.detective import process_prompt_dataset


class TestProgressMonitoring(unittest.TestCase):
    """Test progress monitoring features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_name="test-model",
            api_key="test-key",
            temperature=0.0
        )
        self.model = OpenRouterLLMModel(self.config)
    
    @patch('llm_models.open_router.openai')
    async def test_individual_progress_monitoring(self, mock_openai):
        """Test individual code generation with progress monitoring."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def test():\n    return True"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test with progress monitoring enabled
        with patch('builtins.print') as mock_print:
            result = await self.model.generate_code_output(
                "Write a test function", 
                show_progress=True
            )
        
        # Verify progress messages were printed
        expected_messages = [
            "🔄 Initializing OpenRouter client...",
            "📝 Building enhanced prompt...",
            "🚀 Generating code with model: test-model",
            "🧹 Cleaning and extracting code...",
            "✅ Code generation completed!"
        ]
        
        for msg in expected_messages:
            mock_print.assert_any_call(msg)
        
        # Verify the result
        self.assertIn("def test()", result)
    
    @patch('llm_models.open_router.openai')
    async def test_batch_progress_monitoring(self, mock_openai):
        """Test batch processing with progress monitoring."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def test():\n    return True"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test batch processing
        prompts = ["prompt1", "prompt2", "prompt3"]
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            
            results = await self.model.generate_batch_outputs(
                prompts, 
                show_progress=True
            )
        
        # Verify tqdm was called correctly
        mock_tqdm.assert_called_once_with(
            total=3,
            desc="Generating batch outputs",
            unit="prompts",
            ncols=100
        )
        
        # Verify progress bar methods were called
        self.assertEqual(mock_pbar.update.call_count, 3)
        mock_pbar.close.assert_called_once()
        
        # Verify results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("def test()", result)
    
    @patch('llm_models.open_router.openai')
    async def test_dataset_progress_monitoring(self, mock_openai):
        """Test dataset processing with progress monitoring."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def fixed():\n    return True"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create test dataset
        df = pd.DataFrame({
            'before_merge': ['def buggy():\n    return False', 'def another():\n    return None'],
            'full_traceback': ['AssertionError', 'TypeError']
        })
        
        # Test dataset processing
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            
            results = await process_prompt_dataset(self.model, df)
        
        # Verify tqdm was called correctly
        mock_tqdm.assert_called_once_with(
            total=2,
            desc="Processing dataset",
            unit="samples",
            ncols=100
        )
        
        # Verify progress bar methods were called
        self.assertEqual(mock_pbar.update.call_count, 2)
        mock_pbar.close.assert_called_once()
        
        # Verify results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("def fixed()", result)
    
    def test_progress_disabled(self):
        """Test that progress monitoring can be disabled."""
        # Test that no tqdm is imported when progress is disabled
        # This is more of a configuration test
        self.assertTrue(hasattr(self.model, 'generate_batch_outputs'))
        self.assertTrue(hasattr(self.model, 'generate_code_output'))


class TestProgressMonitoringIntegration(unittest.TestCase):
    """Integration tests for progress monitoring."""
    
    def test_create_model_with_progress(self):
        """Test that created models support progress monitoring."""
        model = create_openrouter_model(
            model_name="test-model",
            api_key="test-key"
        )
        
        # Verify the model has progress monitoring methods
        self.assertTrue(hasattr(model, 'generate_code_output'))
        self.assertTrue(hasattr(model, 'generate_batch_outputs'))
        
        # Verify the methods accept show_progress parameter
        import inspect
        sig = inspect.signature(model.generate_code_output)
        self.assertIn('show_progress', sig.parameters)
        
        sig = inspect.signature(model.generate_batch_outputs)
        self.assertIn('show_progress', sig.parameters)


if __name__ == '__main__':
    # Run the tests
    unittest.main() 