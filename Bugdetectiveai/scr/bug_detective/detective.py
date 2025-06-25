"""
Main BugDetective class for bug detection using LLM models.
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..llm_models.base_model import BaseLLMModel
from ..structured_output.schemas import BugAnalysisSchema
from ..structured_output.output_processor import StructuredOutputProcessor


@dataclass
class DetectionResult:
    """Result of bug detection analysis."""
    success: bool
    bug_analysis: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    model_used: Optional[str] = None


class BugDetective:
    """Main class for bug detection using LLM models."""
    
    def __init__(self, model: BaseLLMModel):
        self.model = model
        self.output_processor = StructuredOutputProcessor()
    
    async def analyze_bug(
        self, 
        code: str, 
        context: Optional[str] = None,
        concise: bool = False
    ) -> DetectionResult:
        """Analyze code for bugs."""
        try:
            schema = BugAnalysisSchema.get_concise_schema() if concise else BugAnalysisSchema.get_basic_schema()
            prompt = self._create_prompt(code, context, concise)
            
            output = await self.model.generate_structured_output(prompt, schema)
            processed = self.output_processor.process_output(output, schema)
            
            return DetectionResult(
                success=processed.is_valid,
                bug_analysis=processed.data if processed.is_valid else None,
                error_message=None if processed.is_valid else ", ".join(processed.validation_errors),
                model_used=self.model.config.model_name
            )
            
        except Exception as e:
            return DetectionResult(
                success=False,
                error_message=str(e),
                model_used=self.model.config.model_name
            )
    
    def _create_prompt(self, code: str, context: Optional[str], concise: bool) -> str:
        """Create analysis prompt."""
        prompt = f"Analyze this code for bugs:\n\n{code}"
        
        if context:
            prompt += f"\n\nContext: {context}"
        
        if concise:
            prompt += "\n\nProvide concise analysis with bug type, severity, description, location, suggested fix, and confidence."
        else:
            prompt += "\n\nProvide detailed analysis with bug type, severity, description, and location."
        
        return prompt
    
    async def batch_analyze(
        self, 
        code_samples: list[str], 
        concise: bool = False
    ) -> list[DetectionResult]:
        """Analyze multiple code samples."""
        tasks = [self.analyze_bug(code, concise=concise) for code in code_samples]
        return await asyncio.gather(*tasks) 