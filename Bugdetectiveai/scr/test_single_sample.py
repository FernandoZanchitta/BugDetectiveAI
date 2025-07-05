#!/usr/bin/env python3
"""
Simple test script for BugDetectiveAI with one sample.
"""

import asyncio
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_models.base_model import ModelConfig
from llm_models.open_router import OpenRouterLLMModel
from bug_detective.detective import BugDetective


async def test_single_sample():
    """Test the system with one sample."""
    print("🐛 BugDetectiveAI - Single Sample Test")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        print("❌ Error: OPEN_ROUTER_KEY environment variable not set")
        print("Please set it with: export OPEN_ROUTER_KEY='your-key-here'")
        return False
    
    # Configure model
    config = ModelConfig(
        model_name="anthropic/claude-3.5-sonnet",
        temperature=0.1,
        api_key=api_key
    )
    
    # Initialize model and detective
    model = OpenRouterLLMModel(config)
    detective = BugDetective(model)
    
    # Single test sample - a classic division by zero bug
    test_code = """
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count  # Potential division by zero if numbers is empty
"""
    
    print("📝 Test Code:")
    print(test_code.strip())
    print("\n🔍 Analyzing...")
    
    try:
        # Run the analysis
        result = await detective.analyze_bug(test_code, concise=True)
        
        if result.success:
            print("✅ Test PASSED!")
            print("\n📊 Analysis Results:")
            print(f"   Bug Type: {result.bug_analysis['bug_type']}")
            print(f"   Severity: {result.bug_analysis['severity']}")
            print(f"   Description: {result.bug_analysis['description']}")
            print(f"   Location: {result.bug_analysis['location']}")
            print(f"   Suggested Fix: {result.bug_analysis['suggested_fix']}")
            print(f"   Confidence: {result.bug_analysis['confidence']}")
            print(f"   Model Used: {result.model_used}")
            return True
        else:
            print("❌ Test FAILED!")
            print(f"   Error: {result.error_message}")
            return False
            
    except Exception as e:
        print("❌ Test FAILED with exception!")
        print(f"   Exception: {str(e)}")
        return False


def main():
    """Main function to run the test."""
    try:
        success = asyncio.run(test_single_sample())
        if success:
            print("\n🎉 Single sample test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Single sample test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 