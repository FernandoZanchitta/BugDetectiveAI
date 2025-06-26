"""
Schema definitions for structured output in BugDetectiveAI.
"""

from typing import Dict, Any
from dataclasses import dataclass


class BugAnalysisSchema:
    """Schema for bug analysis structured output."""
    
    @staticmethod
    def get_basic_schema() -> Dict[str, Any]:
        """Get basic schema for bug analysis."""
        return {
            "type": "object",
            "properties": {
                "bug_type": {
                    "type": "string",
                    "description": "Type of bug identified"
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Severity level of the bug"
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the bug"
                },
                "location": {
                    "type": "string",
                    "description": "Location of the bug in the code"
                }
            },
            "required": ["bug_type", "severity", "description", "location"]
        }
    
    @staticmethod
    def get_concise_schema() -> Dict[str, Any]:
        """Get concise schema for bug analysis."""
        return {
            "type": "object",
            "properties": {
                "bug_type": {"type": "string"},
                "severity": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "suggested_fix": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["bug_type", "severity", "description", "location"]
        }


@dataclass
class CodeReviewSchema:
    """Schema for code review structured output."""
    
    @staticmethod
    def get_basic_schema() -> Dict[str, Any]:
        """Get basic schema for code review."""
        return {
            "type": "object",
            "properties": {
                "overall_quality": {
                    "type": "string",
                    "enum": ["poor", "fair", "good", "excellent"],
                    "description": "Overall code quality assessment"
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified issues"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of improvement suggestions"
                }
            },
            "required": ["overall_quality", "issues", "suggestions"]
        }
    
    @staticmethod
    def get_concise_schema() -> Dict[str, Any]:
        """Get concise schema for code review."""
        return {
            "type": "object",
            "properties": {
                "overall_quality": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "suggestions": {"type": "array", "items": {"type": "string"}},
                "complexity_score": {"type": "number"},
                "maintainability_score": {"type": "number"},
                "security_concerns": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["overall_quality", "issues", "suggestions"]
        }


@dataclass
class ErrorAnalysisSchema:
    """Schema for error analysis structured output."""
    
    @staticmethod
    def get_basic_schema() -> Dict[str, Any]:
        """Get basic schema for error analysis."""
        return {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "description": "Type of error identified"
                },
                "root_cause": {
                    "type": "string",
                    "description": "Root cause of the error"
                },
                "impact": {
                    "type": "string",
                    "description": "Impact of the error"
                }
            },
            "required": ["error_type", "root_cause", "impact"]
        }
    
    @staticmethod
    def get_concise_schema() -> Dict[str, Any]:
        """Get concise schema for error analysis."""
        return {
            "type": "object",
            "properties": {
                "error_type": {"type": "string"},
                "root_cause": {"type": "string"},
                "impact": {"type": "string"},
                "fix_priority": {"type": "string"},
                "affected_components": {"type": "array", "items": {"type": "string"}},
                "estimated_fix_time": {"type": "string"}
            },
            "required": ["error_type", "root_cause", "impact"]
        } 