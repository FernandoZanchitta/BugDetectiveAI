"""
Tests for structured output processing and schemas.
"""

import sys
import os
import unittest
from unittest.mock import Mock
from typing import Dict, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from structured_output.output_processor import StructuredOutputProcessor, ProcessedOutput
from structured_output.schemas import BugAnalysisSchema, CodeReviewSchema, ErrorAnalysisSchema
from llm_models.base_model import StructuredOutput


class TestStructuredOutputProcessor(unittest.TestCase):
    """Test cases for StructuredOutputProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = StructuredOutputProcessor()
    
    def test_process_output_success(self):
        """Test successful output processing."""
        output = StructuredOutput(
            success=True,
            content={
                "bug_type": "syntax_error",
                "severity": "high",
                "description": "Missing semicolon",
                "location": "line 5"
            }
        )
        
        schema = BugAnalysisSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data, output.content)
        self.assertEqual(result.validation_errors, [])
        self.assertEqual(result.raw_output, output)
    
    def test_process_output_failed_generation(self):
        """Test processing of failed output generation."""
        output = StructuredOutput(
            success=False,
            content={},
            error_message="API rate limit exceeded"
        )
        
        schema = BugAnalysisSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.data, {})
        self.assertEqual(result.validation_errors, ["API rate limit exceeded"])
        self.assertEqual(result.raw_output, output)
    
    def test_process_output_missing_required_fields(self):
        """Test processing with missing required fields."""
        output = StructuredOutput(
            success=True,
            content={
                "bug_type": "syntax_error",
                "severity": "high"
                # Missing "description" and "location"
            }
        )
        
        schema = BugAnalysisSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.data, output.content)
        self.assertIn("Missing required field: description", result.validation_errors)
        self.assertIn("Missing required field: location", result.validation_errors)
    
    def test_process_output_invalid_field_types(self):
        """Test processing with invalid field types."""
        output = StructuredOutput(
            success=True,
            content={
                "bug_type": 123,  # Should be string
                "severity": "high",
                "description": "Test",
                "location": "line 5"
            }
        )
        
        schema = BugAnalysisSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertIn("bug_type: Expected string type", result.validation_errors)
    
    def test_process_output_invalid_enum_value(self):
        """Test processing with invalid enum value."""
        output = StructuredOutput(
            success=True,
            content={
                "bug_type": "syntax_error",
                "severity": "invalid_severity",  # Not in enum
                "description": "Test",
                "location": "line 5"
            }
        )
        
        schema = BugAnalysisSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertIn("severity: Value must be one of ['low', 'medium', 'high', 'critical']", result.validation_errors)
    
    def test_process_output_invalid_number_range(self):
        """Test processing with invalid number range."""
        output = StructuredOutput(
            success=True,
            content={
                "bug_type": "syntax_error",
                "severity": "high",
                "description": "Test",
                "location": "line 5",
                "confidence": 1.5  # Should be <= 1.0
            }
        )
        
        schema = BugAnalysisSchema.get_concise_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertIn("confidence: Value must be <= 1", result.validation_errors)
    
    def test_process_output_invalid_array(self):
        """Test processing with invalid array."""
        output = StructuredOutput(
            success=True,
            content={
                "overall_quality": "good",
                "issues": "not_an_array",  # Should be array
                "suggestions": ["suggestion1", "suggestion2"]
            }
        )
        
        schema = CodeReviewSchema.get_basic_schema()
        result = self.processor.process_output(output, schema)
        
        self.assertFalse(result.is_valid)
        self.assertIn("issues: Expected array type", result.validation_errors)
    
    def test_extract_key_fields(self):
        """Test extracting specific fields from processed data."""
        data = {
            "bug_type": "syntax_error",
            "severity": "high",
            "description": "Missing semicolon",
            "location": "line 5",
            "confidence": 0.9
        }
        
        fields = ["bug_type", "severity", "confidence"]
        extracted = self.processor.extract_key_fields(data, fields)
        
        expected = {
            "bug_type": "syntax_error",
            "severity": "high",
            "confidence": 0.9
        }
        self.assertEqual(extracted, expected)
    
    def test_extract_key_fields_missing_fields(self):
        """Test extracting fields that don't exist."""
        data = {
            "bug_type": "syntax_error",
            "severity": "high"
        }
        
        fields = ["bug_type", "severity", "nonexistent_field"]
        extracted = self.processor.extract_key_fields(data, fields)
        
        expected = {
            "bug_type": "syntax_error",
            "severity": "high"
        }
        self.assertEqual(extracted, expected)
    
    def test_format_output_valid(self):
        """Test formatting valid output."""
        processed_output = ProcessedOutput(
            is_valid=True,
            data={"bug_type": "syntax_error", "severity": "high"},
            validation_errors=[],
            raw_output=Mock()
        )
        
        formatted = self.processor.format_output(processed_output)
        
        self.assertIn("bug_type", formatted)
        self.assertIn("syntax_error", formatted)
        self.assertIn("severity", formatted)
        self.assertIn("high", formatted)
    
    def test_format_output_invalid(self):
        """Test formatting invalid output."""
        processed_output = ProcessedOutput(
            is_valid=False,
            data={},
            validation_errors=["Missing required field: description"],
            raw_output=Mock()
        )
        
        formatted = self.processor.format_output(processed_output)
        
        self.assertIn("Invalid output", formatted)
        self.assertIn("Missing required field: description", formatted)


class TestBugAnalysisSchema(unittest.TestCase):
    """Test cases for BugAnalysisSchema."""
    
    def test_get_basic_schema(self):
        """Test basic schema structure."""
        schema = BugAnalysisSchema.get_basic_schema()
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        properties = schema["properties"]
        self.assertIn("bug_type", properties)
        self.assertIn("severity", properties)
        self.assertIn("description", properties)
        self.assertIn("location", properties)
        
        # Check severity enum
        severity_prop = properties["severity"]
        self.assertEqual(severity_prop["type"], "string")
        self.assertIn("enum", severity_prop)
        self.assertEqual(severity_prop["enum"], ["low", "medium", "high", "critical"])
        
        # Check required fields
        required = schema["required"]
        self.assertIn("bug_type", required)
        self.assertIn("severity", required)
        self.assertIn("description", required)
        self.assertIn("location", required)
    
    def test_get_concise_schema(self):
        """Test concise schema structure."""
        schema = BugAnalysisSchema.get_concise_schema()
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        properties = schema["properties"]
        self.assertIn("bug_type", properties)
        self.assertIn("severity", properties)
        self.assertIn("description", properties)
        self.assertIn("location", properties)
        self.assertIn("suggested_fix", properties)
        self.assertIn("confidence", properties)
        
        # Check confidence number constraints
        confidence_prop = properties["confidence"]
        self.assertEqual(confidence_prop["type"], "number")
        self.assertEqual(confidence_prop["minimum"], 0)
        self.assertEqual(confidence_prop["maximum"], 1)


class TestCodeReviewSchema(unittest.TestCase):
    """Test cases for CodeReviewSchema."""
    
    def test_get_basic_schema(self):
        """Test basic code review schema structure."""
        schema = CodeReviewSchema.get_basic_schema()
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        properties = schema["properties"]
        self.assertIn("overall_quality", properties)
        self.assertIn("issues", properties)
        self.assertIn("suggestions", properties)
        
        # Check overall_quality enum
        quality_prop = properties["overall_quality"]
        self.assertEqual(quality_prop["type"], "string")
        self.assertIn("enum", quality_prop)
        self.assertEqual(quality_prop["enum"], ["poor", "fair", "good", "excellent"])
        
        # Check arrays
        issues_prop = properties["issues"]
        self.assertEqual(issues_prop["type"], "array")
        self.assertIn("items", issues_prop)
        self.assertEqual(issues_prop["items"]["type"], "string")
    
    def test_get_concise_schema(self):
        """Test concise code review schema structure."""
        schema = CodeReviewSchema.get_concise_schema()
        
        properties = schema["properties"]
        self.assertIn("overall_quality", properties)
        self.assertIn("issues", properties)
        self.assertIn("suggestions", properties)
        self.assertIn("complexity_score", properties)
        self.assertIn("maintainability_score", properties)
        self.assertIn("security_concerns", properties)
        
        # Check number fields
        complexity_prop = properties["complexity_score"]
        self.assertEqual(complexity_prop["type"], "number")


class TestErrorAnalysisSchema(unittest.TestCase):
    """Test cases for ErrorAnalysisSchema."""
    
    def test_get_basic_schema(self):
        """Test basic error analysis schema structure."""
        schema = ErrorAnalysisSchema.get_basic_schema()
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        properties = schema["properties"]
        self.assertIn("error_type", properties)
        self.assertIn("root_cause", properties)
        self.assertIn("impact", properties)
        
        required = schema["required"]
        self.assertIn("error_type", required)
        self.assertIn("root_cause", required)
        self.assertIn("impact", required)
    
    def test_get_concise_schema(self):
        """Test concise error analysis schema structure."""
        schema = ErrorAnalysisSchema.get_concise_schema()
        
        properties = schema["properties"]
        self.assertIn("error_type", properties)
        self.assertIn("root_cause", properties)
        self.assertIn("impact", properties)
        self.assertIn("fix_priority", properties)
        self.assertIn("affected_components", properties)
        self.assertIn("estimated_fix_time", properties)
        
        # Check array field
        components_prop = properties["affected_components"]
        self.assertEqual(components_prop["type"], "array")
        self.assertEqual(components_prop["items"]["type"], "string")


if __name__ == "__main__":
    unittest.main() 