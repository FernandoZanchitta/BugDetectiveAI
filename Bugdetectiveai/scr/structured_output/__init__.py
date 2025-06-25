"""
Structured Output package for BugDetectiveAI.
Provides basic and concise schemas for bug analysis.
"""

from .schemas import BugAnalysisSchema
from .output_processor import StructuredOutputProcessor

__all__ = [
    "BugAnalysisSchema",
    "StructuredOutputProcessor"
] 