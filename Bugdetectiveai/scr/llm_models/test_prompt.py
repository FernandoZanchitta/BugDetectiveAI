"""
Test file to demonstrate the prompt builder functionality.
"""

from prompt import PromptBuilder, build_correction_prompt


def test_basic_prompt_building():
    """Test basic prompt building functionality."""
    
    # Test the convenience function
    buggy_code = """
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, "3")
print(result)
"""
    
    traceback_error = """
TypeError: unsupported operand type(s) for +: 'int' and 'str'
"""
    
    prompt = build_correction_prompt(
        buggy_code=buggy_code,
        traceback_error=traceback_error,
        instruction_prompt="Please fix the type error in this code."
    )
    
    print("=== Basic Prompt ===")
    print(prompt)
    print("\n" + "="*50 + "\n")


def test_prompt_with_examples():
    """Test prompt building with retrieved examples."""
    
    builder = PromptBuilder()
    
    buggy_code = """
def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 0)
"""
    
    traceback_error = """
ZeroDivisionError: division by zero
"""
    
    retrieved_examples = [
        {
            "buggy_code": "result = 5 / 0",
            "corrected_code": "result = 5 / 1 if 1 != 0 else 0"
        },
        {
            "buggy_code": "x = 10 / 0",
            "corrected_code": "x = 10 / 1 if 1 != 0 else 0"
        }
    ]
    
    prompt = builder.build_correction_prompt(
        buggy_code=buggy_code,
        traceback_error=traceback_error,
        retrieved_examples=retrieved_examples,
        instruction_prompt="Please fix the division by zero error."
    )
    
    print("=== Prompt with Examples ===")
    print(prompt)
    print("\n" + "="*50 + "\n")


def test_analysis_prompt():
    """Test analysis prompt building."""
    
    builder = PromptBuilder()
    
    buggy_code = """
import requests

def fetch_data(url):
    response = requests.get(url)
    return response.json()

data = fetch_data("https://api.example.com/data")
"""
    
    traceback_error = """
requests.exceptions.ConnectionError: Failed to establish a new connection
"""
    
    prompt = builder.build_analysis_prompt(
        buggy_code=buggy_code,
        traceback_error=traceback_error,
        analysis_type="error handling",
        additional_context={
            "framework": "Flask",
            "environment": "production"
        }
    )
    
    print("=== Analysis Prompt ===")
    print(prompt)
    print("\n" + "="*50 + "\n")


def test_custom_prompt():
    """Test custom prompt building with template string."""
    
    builder = PromptBuilder()
    
    template_string = """
You are a {{ role }} expert. Please help with the following issue:

**Problem**: {{ problem }}
**Code**: 
```python
{{ code }}
```
**Error**: {{ error }}

Please provide:
1. A brief explanation of the issue
2. The corrected code
3. Best practices to avoid this in the future

Format your response as:
**Explanation**: [your explanation]
**Corrected Code**: [your code]
**Best Practices**: [your recommendations]
"""
    
    prompt = builder.build_custom_prompt(
        template_string=template_string,
        role="Python debugging",
        problem="Type conversion error in mathematical operation",
        code="result = 5 + '3'",
        error="TypeError: unsupported operand type(s) for +: 'int' and 'str'"
    )
    
    print("=== Custom Prompt ===")
    print(prompt)


if __name__ == "__main__":
    test_basic_prompt_building()
    test_prompt_with_examples()
    test_analysis_prompt()
    test_custom_prompt() 