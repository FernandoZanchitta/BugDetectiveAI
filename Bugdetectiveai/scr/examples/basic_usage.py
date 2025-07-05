"""
Basic usage example for BugDetectiveAI.
"""

import asyncio
import os

from ..llm_models.base_model import ModelConfig
from ..llm_models.open_router import OpenRouterLLMModel
from ..bug_detective.detective import BugDetective


async def main():
    """Basic example using OpenRouter."""
    print("BugDetectiveAI - Basic Example")
    print("=" * 30)
    
    # Check for API key
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        print("❌ Please set OPEN_ROUTER_KEY environment variable")
        return
    
    # Configure model
    config = ModelConfig(
        model_name="anthropic/claude-3.5-sonnet",
        temperature=0.1,
        api_key=api_key
    )
    
    # Initialize model and detective
    model = OpenRouterLLMModel(config)
    detective = BugDetective(model)
    
    # Sample buggy code
    buggy_code = """
    def divide(a, b):
        return a / b  # No division by zero check
    """
    
    print("Analyzing code...")
    print(f"Code: {buggy_code.strip()}")
    
    # Analyze with basic schema
    result = await detective.analyze_bug(buggy_code, concise=False)
    
    if result.success:
        print("\n✅ Analysis successful!")
        print(f"Bug Type: {result.bug_analysis['bug_type']}")
        print(f"Severity: {result.bug_analysis['severity']}")
        print(f"Description: {result.bug_analysis['description']}")
        print(f"Location: {result.bug_analysis['location']}")
    else:
        print(f"\n❌ Analysis failed: {result.error_message}")
    
    # Analyze with concise schema
    print("\n" + "=" * 30)
    print("Analyzing with concise schema...")
    
    result = await detective.analyze_bug(buggy_code, concise=True)
    
    if result.success:
        print("✅ Analysis successful!")
        print(f"Bug Type: {result.bug_analysis['bug_type']}")
        print(f"Severity: {result.bug_analysis['severity']}")
        print(f"Description: {result.bug_analysis['description']}")
        print(f"Location: {result.bug_analysis['location']}")
        if 'suggested_fix' in result.bug_analysis:
            print(f"Suggested Fix: {result.bug_analysis['suggested_fix']}")
        if 'confidence' in result.bug_analysis:
            print(f"Confidence: {result.bug_analysis['confidence']}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main()) 