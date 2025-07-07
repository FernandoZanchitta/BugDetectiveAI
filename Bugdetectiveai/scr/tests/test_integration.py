"""
Integration tests for the complete BugDetectiveAI workflow.
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bug_detective.detective import BugDetective
from llm_models.base_model import ModelConfig, StructuredOutput, BaseLLMModel
from structured_output.schemas import BugAnalysisSchema


class MockLLMModel(BaseLLMModel):
    """Mock LLM model for integration testing."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    async def generate_structured_output(
        self, prompt: str, output_schema: Dict[str, Any]
    ) -> StructuredOutput:
        """Mock structured output generation with realistic responses."""
        # Simulate different responses based on prompt content
        if "syntax" in prompt.lower():
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "syntax_error",
                    "severity": "high",
                    "description": "Missing semicolon at end of statement",
                    "location": "line 3, column 15",
                },
            )
        elif "logic" in prompt.lower():
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "logic_error",
                    "severity": "medium",
                    "description": "Incorrect calculation in loop condition",
                    "location": "line 7",
                },
            )
        elif "concise" in prompt.lower():
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "runtime_error",
                    "severity": "critical",
                    "description": "Division by zero in calculation",
                    "location": "line 12",
                    "suggested_fix": "Add null check before division",
                    "confidence": 0.95,
                },
            )
        else:
            return StructuredOutput(
                success=True,
                content={
                    "bug_type": "potential_bug",
                    "severity": "low",
                    "description": "Unused variable detected",
                    "location": "line 5",
                },
            )

    async def generate_basic_output(self, prompt: str) -> str:
        """Mock basic output generation."""
        return "Mock analysis: Code appears to have potential issues."

    def validate_config(self) -> bool:
        """Mock config validation."""
        return True


class TestBugDetectiveIntegration(unittest.TestCase):
    """Integration tests for complete BugDetective workflow."""

    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(model_name="integration-test-model")
        self.mock_model = MockLLMModel(config)
        self.detective = BugDetective(self.mock_model)

    async def test_complete_workflow_basic_analysis(self):
        """Test complete workflow for basic bug analysis."""
        code = """
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)  # Potential division by zero
"""

        result = await self.detective.analyze_bug(code)

        # Verify the complete workflow
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
        self.assertIn("bug_type", result.bug_analysis)
        self.assertIn("severity", result.bug_analysis)
        self.assertIn("description", result.bug_analysis)
        self.assertIn("location", result.bug_analysis)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.model_used, "integration-test-model")

    async def test_complete_workflow_concise_analysis(self):
        """Test complete workflow for concise bug analysis."""
        code = """
def divide_numbers(a, b):
    return a / b  # Division by zero risk
"""

        result = await self.detective.analyze_bug(code, concise=True)

        # Verify concise analysis includes additional fields
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
        self.assertIn("bug_type", result.bug_analysis)
        self.assertIn("severity", result.bug_analysis)
        self.assertIn("description", result.bug_analysis)
        self.assertIn("location", result.bug_analysis)
        self.assertIn("suggested_fix", result.bug_analysis)
        self.assertIn("confidence", result.bug_analysis)
        self.assertIsNone(result.error_message)

    async def test_complete_workflow_with_context(self):
        """Test complete workflow with context information."""
        code = """
def process_user_data(user_input):
    data = user_input.split(',')
    return data[0]  # Potential index error
"""
        context = "This function processes user input from a web form"

        result = await self.detective.analyze_bug(code, context=context)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)
        self.assertIn("bug_type", result.bug_analysis)
        self.assertIn("severity", result.bug_analysis)
        self.assertIn("description", result.bug_analysis)
        self.assertIn("location", result.bug_analysis)

    async def test_complete_workflow_batch_analysis(self):
        """Test complete workflow for batch analysis."""
        code_samples = [
            """
def add_numbers(a, b):
    return a + b  # No obvious bugs
""",
            """
def multiply_numbers(a, b):
    result = a * b
    return result  # No obvious bugs
""",
            """
def divide_numbers(a, b):
    return a / b  # Potential division by zero
""",
        ]

        results = await self.detective.batch_analyze(code_samples)

        # Verify all results are successful
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.bug_analysis)
            self.assertIn("bug_type", result.bug_analysis)
            self.assertIn("severity", result.bug_analysis)
            self.assertIn("description", result.bug_analysis)
            self.assertIn("location", result.bug_analysis)
            self.assertEqual(result.model_used, "integration-test-model")

    async def test_workflow_with_schema_validation(self):
        """Test workflow with schema validation."""
        # Test with valid schema
        schema = BugAnalysisSchema.get_basic_schema()

        # Verify schema structure
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)

        properties = schema["properties"]
        self.assertIn("bug_type", properties)
        self.assertIn("severity", properties)
        self.assertIn("description", properties)
        self.assertIn("location", properties)

        # Test that the model can work with this schema
        code = "def test(): pass"
        result = await self.detective.analyze_bug(code)

        self.assertTrue(result.success)
        # Verify the output matches the schema
        self.assertIn("bug_type", result.bug_analysis)
        self.assertIn("severity", result.bug_analysis)
        self.assertIn("description", result.bug_analysis)
        self.assertIn("location", result.bug_analysis)

    async def test_workflow_error_handling(self):
        """Test workflow error handling."""

        # Create a model that fails
        class FailingModel(MockLLMModel):
            async def generate_structured_output(
                self, prompt: str, output_schema: Dict[str, Any]
            ) -> StructuredOutput:
                raise Exception("Simulated model failure")

        failing_model = FailingModel(ModelConfig(model_name="failing-model"))
        detective = BugDetective(failing_model)

        code = "def test(): pass"
        result = await detective.analyze_bug(code)

        self.assertFalse(result.success)
        self.assertIsNone(result.bug_analysis)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Simulated model failure", result.error_message)
        self.assertEqual(result.model_used, "failing-model")

    async def test_workflow_with_invalid_output(self):
        """Test workflow with invalid model output."""

        # Create a model that returns invalid output
        class InvalidOutputModel(MockLLMModel):
            async def generate_structured_output(
                self, prompt: str, output_schema: Dict[str, Any]
            ) -> StructuredOutput:
                return StructuredOutput(
                    success=True,
                    content={
                        "bug_type": "syntax_error"
                        # Missing required fields: severity, description, location
                    },
                )

        invalid_model = InvalidOutputModel(ModelConfig(model_name="invalid-model"))
        detective = BugDetective(invalid_model)

        code = "def test(): pass"
        result = await detective.analyze_bug(code)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Missing required field", result.error_message)
        self.assertEqual(result.model_used, "invalid-model")


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow tests."""

    async def test_simple_bug_detection_workflow(self):
        """Test a simple end-to-end bug detection workflow."""
        config = ModelConfig(model_name="e2e-test-model")
        mock_model = MockLLMModel(config)
        detective = BugDetective(mock_model)

        # Test code with obvious issues
        code = """
def calculate_percentage(value, total):
    return (value / total) * 100  # Division by zero risk
"""

        # Run analysis
        result = await detective.analyze_bug(code, concise=True)

        # Verify complete workflow
        self.assertTrue(result.success)
        self.assertIsNotNone(result.bug_analysis)

        # Verify all expected fields are present
        expected_fields = [
            "bug_type",
            "severity",
            "description",
            "location",
            "suggested_fix",
            "confidence",
        ]
        for field in expected_fields:
            self.assertIn(field, result.bug_analysis)

        # Verify field types
        self.assertIsInstance(result.bug_analysis["bug_type"], str)
        self.assertIsInstance(result.bug_analysis["severity"], str)
        self.assertIsInstance(result.bug_analysis["description"], str)
        self.assertIsInstance(result.bug_analysis["location"], str)
        self.assertIsInstance(result.bug_analysis["suggested_fix"], str)
        self.assertIsInstance(result.bug_analysis["confidence"], (int, float))

        # Verify confidence is in valid range
        self.assertGreaterEqual(result.bug_analysis["confidence"], 0)
        self.assertLessEqual(result.bug_analysis["confidence"], 1)

    async def test_multiple_analysis_types(self):
        """Test different types of analysis in the same workflow."""
        config = ModelConfig(model_name="multi-test-model")
        mock_model = MockLLMModel(config)
        detective = BugDetective(mock_model)

        test_cases = [
            ("def test(): pass", False),  # Basic analysis
            ("def test(): pass", True),  # Concise analysis
        ]

        for code, concise in test_cases:
            result = await detective.analyze_bug(code, concise=concise)

            self.assertTrue(result.success)
            self.assertIsNotNone(result.bug_analysis)

            if concise:
                # Should have additional fields
                self.assertIn("suggested_fix", result.bug_analysis)
                self.assertIn("confidence", result.bug_analysis)
            else:
                # Should not have additional fields
                self.assertNotIn("suggested_fix", result.bug_analysis)
                self.assertNotIn("confidence", result.bug_analysis)


def run_async_test(test_func):
    """Helper function to run async tests."""
    return asyncio.run(test_func())


if __name__ == "__main__":
    # Create test suite for async tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add integration tests
    integration_tests = [
        TestBugDetectiveIntegration("test_complete_workflow_basic_analysis"),
        TestBugDetectiveIntegration("test_complete_workflow_concise_analysis"),
        TestBugDetectiveIntegration("test_complete_workflow_with_context"),
        TestBugDetectiveIntegration("test_complete_workflow_batch_analysis"),
        TestBugDetectiveIntegration("test_workflow_with_schema_validation"),
        TestBugDetectiveIntegration("test_workflow_error_handling"),
        TestBugDetectiveIntegration("test_workflow_with_invalid_output"),
    ]

    # Add end-to-end tests
    e2e_tests = [
        TestEndToEndWorkflow("test_simple_bug_detection_workflow"),
        TestEndToEndWorkflow("test_multiple_analysis_types"),
    ]

    for test_case in integration_tests + e2e_tests:
        suite.addTest(test_case)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
