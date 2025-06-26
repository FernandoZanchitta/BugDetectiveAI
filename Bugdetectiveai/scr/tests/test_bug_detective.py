"""
Tests for BugDetective class.
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bug_detective.detective import BugDetective, DetectionResult
from llm_models.base_model import ModelConfig, StructuredOutput, BaseLLMModel
from structured_output.schemas import BugAnalysisSchema


class MockLLMModel(BaseLLMModel):
    """Mock LLM model for testing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    async def generate_structured_output(self, prompt: str, output_schema: Dict[str, Any]) -> StructuredOutput:
        """Mock structured output generation."""
        if "syntax error" in prompt.lower():
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "syntax_error",
                    "severity": "high",
                    "description": "Missing semicolon",
                    "location": "line 5"
                }
            )
        else:
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "logic_error",
                    "severity": "medium",
                    "description": "Incorrect calculation",
                    "location": "line 10"
                }
            )
    
    async def generate_basic_output(self, prompt: str) -> str:
        """Mock basic output generation."""
        return "Mock analysis response"
    
    def validate_config(self) -> bool:
        """Mock config validation."""
        return True


class TestBugDetective(unittest.TestCase):
    """Test cases for BugDetective class."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(model_name="test-model")
        self.mock_model = MockLLMModel(config)
        self.detective = BugDetective(self.mock_model)
    
    def test_initialization(self):
        """Test BugDetective initialization."""
        self.assertEqual(self.detective.model, self.mock_model)
        self.assertIsNotNone(self.detective.output_processor)
    
    async def test_analyze_bug_basic_success(self):
        """Test successful basic bug analysis."""
        code = "def add(a, b):\n    return a + b"
        
        result = await self.detective.analyze_bug(code)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
        if result.bug_analysis:
            self.assertIn("bug_type", result.bug_analysis)
            self.assertIn("severity", result.bug_analysis)
            self.assertIn("description", result.bug_analysis)
            self.assertIn("location", result.bug_analysis)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.model_used, "test-model")
    
    async def test_analyze_bug_concise_success(self):
        """Test successful concise bug analysis."""
        code = "def add(a, b):\n    return a + b"
        
        result = await self.detective.analyze_bug(code, concise=True)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
        self.assertIn("bug_type", result.bug_analysis)
        self.assertIn("severity", result.bug_analysis)
        self.assertIn("description", result.bug_analysis)
        self.assertIn("location", result.bug_analysis)
        self.assertIn("suggested_fix", result.bug_analysis)
        self.assertIn("confidence", result.bug_analysis)
        self.assertIsNone(result.error_message)
    
    async def test_analyze_bug_with_context(self):
        """Test bug analysis with context."""
        code = "def add(a, b):\n    return a + b"
        context = "This function is used in a calculator application"
        
        result = await self.detective.analyze_bug(code, context=context)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
    
    async def test_analyze_bug_model_error(self):
        """Test bug analysis when model fails."""
        # Create a mock model that raises an exception
        class ErrorModel(MockLLMModel):
            async def generate_structured_output(self, prompt: str, output_schema: Dict[str, Any]) -> StructuredOutput:
                raise Exception("Model error")
        
        error_model = ErrorModel(ModelConfig(model_name="error-model"))
        detective = BugDetective(error_model)
        
        code = "def add(a, b):\n    return a + b"
        result = await detective.analyze_bug(code)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.bug_analysis)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Model error", result.error_message)
        self.assertEqual(result.model_used, "error-model")
    
    async def test_batch_analyze(self):
        """Test batch analysis of multiple code samples."""
        code_samples = [
            "def add(a, b):\n    return a + b",
            "def sub(a, b):\n    return a - b",
            "def mul(a, b):\n    return a * b"
        ]
        
        results = await self.detective.batch_analyze(code_samples)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.bug_analysis)
            self.assertEqual(result.model_used, "test-model")
    
    async def test_batch_analyze_concise(self):
        """Test batch analysis with concise mode."""
        code_samples = [
            "def add(a, b):\n    return a + b",
            "def sub(a, b):\n    return a - b"
        ]
        
        results = await self.detective.batch_analyze(code_samples, concise=True)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.success)
            self.assertIn("suggested_fix", result.bug_analysis)
            self.assertIn("confidence", result.bug_analysis)
    
    def test_create_prompt_basic(self):
        """Test prompt creation for basic analysis."""
        code = "def add(a, b):\n    return a + b"
        prompt = self.detective._create_prompt(code, None, False)
        
        self.assertIn("Analyze this code for bugs", prompt)
        self.assertIn(code, prompt)
        self.assertIn("detailed analysis", prompt)
    
    def test_create_prompt_concise(self):
        """Test prompt creation for concise analysis."""
        code = "def add(a, b):\n    return a + b"
        prompt = self.detective._create_prompt(code, None, True)
        
        self.assertIn("Analyze this code for bugs", prompt)
        self.assertIn(code, prompt)
        self.assertIn("concise analysis", prompt)
        self.assertIn("confidence", prompt)
    
    def test_create_prompt_with_context(self):
        """Test prompt creation with context."""
        code = "def add(a, b):\n    return a + b"
        context = "This is a calculator function"
        prompt = self.detective._create_prompt(code, context, False)
        
        self.assertIn("Analyze this code for bugs", prompt)
        self.assertIn(code, prompt)
        self.assertIn("Context: This is a calculator function", prompt)


class TestDetectionResult(unittest.TestCase):
    """Test cases for DetectionResult dataclass."""
    
    def test_successful_result(self):
        """Test successful detection result."""
        bug_analysis = {
            "bug_type": "syntax_error",
            "severity": "high",
            "description": "Missing semicolon",
            "location": "line 5"
        }
        
        result = DetectionResult(
            success=True,
            bug_analysis=bug_analysis,
            model_used="gpt-4"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.bug_analysis, bug_analysis)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.model_used, "gpt-4")
    
    def test_failed_result(self):
        """Test failed detection result."""
        result = DetectionResult(
            success=False,
            error_message="Model error",
            model_used="gpt-4"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.bug_analysis)
        self.assertEqual(result.error_message, "Model error")
        self.assertEqual(result.model_used, "gpt-4")
    
    def test_result_with_defaults(self):
        """Test detection result with default values."""
        result = DetectionResult(success=True)
        
        self.assertTrue(result.success)
        self.assertIsNone(result.bug_analysis)
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.model_used)


def run_async_test(test_func):
    """Helper function to run async tests."""
    return asyncio.run(test_func())


if __name__ == "__main__":
    # Create test suite for async tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add async tests
    test_cases = [
        TestBugDetective('test_analyze_bug_basic_success'),
        TestBugDetective('test_analyze_bug_concise_success'),
        TestBugDetective('test_analyze_bug_with_context'),
        TestBugDetective('test_analyze_bug_model_error'),
        TestBugDetective('test_batch_analyze'),
        TestBugDetective('test_batch_analyze_concise'),
    ]
    
    for test_case in test_cases:
        suite.addTest(test_case)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite) 