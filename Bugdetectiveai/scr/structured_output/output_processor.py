"""
Output processor for handling structured outputs from LLM models.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ..llm_models.base_model import StructuredOutput


@dataclass
class ProcessedOutput:
    """Processed structured output with validation."""
    is_valid: bool
    data: Dict[str, Any]
    validation_errors: List[str]
    raw_output: StructuredOutput


class StructuredOutputProcessor:
    """Processor for structured outputs from LLM models."""
    
    def __init__(self):
        self.validation_cache = {}
    
    def process_output(
        self, 
        output: StructuredOutput, 
        expected_schema: Dict[str, Any]
    ) -> ProcessedOutput:
        """Process and validate structured output."""
        if not output.success:
            return ProcessedOutput(
                is_valid=False,
                data={},
                validation_errors=[output.error_message or "Output generation failed"],
                raw_output=output
            )
        
        validation_errors = self._validate_against_schema(
            output.content, 
            expected_schema
        )
        
        return ProcessedOutput(
            is_valid=len(validation_errors) == 0,
            data=output.content,
            validation_errors=validation_errors,
            raw_output=output
        )
    
    def _validate_against_schema(
        self, 
        data: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> List[str]:
        """Validate data against JSON schema."""
        errors = []
        
        # Basic type validation
        if schema.get("type") == "object":
            if not isinstance(data, dict):
                errors.append("Expected object type")
                return errors
            
            # Check required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate properties
            properties = schema.get("properties", {})
            for field, value in data.items():
                if field in properties:
                    field_errors = self._validate_field(value, properties[field])
                    errors.extend([f"{field}: {error}" for error in field_errors])
        
        return errors
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any]) -> List[str]:
        """Validate a single field against its schema."""
        errors = []
        
        field_type = field_schema.get("type")
        
        if field_type == "string":
            if not isinstance(value, str):
                errors.append("Expected string type")
            elif "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(f"Value must be one of {field_schema['enum']}")
        
        elif field_type == "number":
            if not isinstance(value, (int, float)):
                errors.append("Expected number type")
            else:
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    errors.append(f"Value must be >= {field_schema['minimum']}")
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    errors.append(f"Value must be <= {field_schema['maximum']}")
        
        elif field_type == "array":
            if not isinstance(value, list):
                errors.append("Expected array type")
            else:
                items_schema = field_schema.get("items", {})
                for i, item in enumerate(value):
                    item_errors = self._validate_field(item, items_schema)
                    errors.extend([f"item[{i}]: {error}" for error in item_errors])
        
        return errors
    
    def extract_key_fields(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Extract specific fields from processed data."""
        return {field: data.get(field) for field in fields if field in data}
    
    def format_output(self, processed_output: ProcessedOutput) -> str:
        """Format processed output for display."""
        if not processed_output.is_valid:
            return f"Invalid output: {', '.join(processed_output.validation_errors)}"
        
        return json.dumps(processed_output.data, indent=2) 