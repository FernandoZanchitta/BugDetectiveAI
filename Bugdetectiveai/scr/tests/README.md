# BugDetectiveAI Test Suite

This directory contains comprehensive tests for the BugDetectiveAI project. The tests are organized into different categories and cover both synchronous and asynchronous functionality.

## Test Structure

### Test Files

1. **`test_base_model.py`** - Tests for base LLM model classes
   - `ModelConfig` dataclass
   - `StructuredOutput` dataclass
   - `BaseLLMModel` abstract class

2. **`test_openai_model.py`** - Tests for OpenAI LLM model implementation
   - Configuration validation
   - Client initialization
   - Structured output generation
   - Basic output generation
   - Error handling

3. **`test_bug_detective.py`** - Tests for the main BugDetective class
   - Bug analysis functionality
   - Batch analysis
   - Prompt creation
   - Error handling

4. **`test_structured_output.py`** - Tests for structured output processing
   - Output validation
   - Schema validation
   - Field extraction
   - Output formatting

5. **`test_integration.py`** - Integration and end-to-end tests
   - Complete workflow testing
   - Error scenarios
   - Multiple analysis types

6. **`test_diff_metrics.py`** - Tests for diff-based metrics (existing)
   - Metric calculations
   - Evaluator functionality
   - Factory functions

7. **`run_all_tests.py`** - Test runner script
   - Runs all tests
   - Supports running specific test categories
   - Provides detailed test summaries

## Running Tests

### Prerequisites

Make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

### Running All Tests

To run all tests:

```bash
cd Bugdetectiveai/scr/tests
python run_all_tests.py
```

### Running Specific Test Categories

You can run tests for specific categories:

```bash
# Base model tests
python run_all_tests.py base

# OpenAI model tests
python run_all_tests.py openai

# BugDetective tests
python run_all_tests.py detective

# Structured output tests
python run_all_tests.py structured

# Integration tests
python run_all_tests.py integration

# Metrics tests
python run_all_tests.py metrics
```

### Running Individual Test Files

You can also run individual test files:

```bash
# Run base model tests
python test_base_model.py

# Run OpenAI model tests
python test_openai_model.py

# Run bug detective tests
python test_bug_detective.py

# Run structured output tests
python test_structured_output.py

# Run integration tests
python test_integration.py
```

## Test Categories

### Synchronous Tests
- **Base Model Tests**: Configuration, data structures, abstract class behavior
- **Structured Output Tests**: Validation, processing, schema handling
- **Metrics Tests**: Diff-based metric calculations and evaluation

### Asynchronous Tests
- **OpenAI Model Tests**: API interactions, client management, error handling
- **BugDetective Tests**: Main analysis functionality, batch processing
- **Integration Tests**: End-to-end workflows, error scenarios

## Test Coverage

### Core Functionality
- ✅ Model configuration and validation
- ✅ Structured output generation and validation
- ✅ Bug analysis with different schemas
- ✅ Batch processing capabilities
- ✅ Error handling and recovery
- ✅ Schema validation and processing

### Edge Cases
- ✅ Missing required fields
- ✅ Invalid data types
- ✅ API failures and timeouts
- ✅ Configuration errors
- ✅ Network issues

### Integration Scenarios
- ✅ Complete bug detection workflow
- ✅ Multiple analysis types (basic vs concise)
- ✅ Context-aware analysis
- ✅ Batch processing with mixed results
- ✅ Error propagation through the system

## Test Patterns

### Mocking Strategy
- **LLM Models**: Mocked to avoid actual API calls
- **API Clients**: Mocked for controlled testing
- **Network Calls**: Simulated with realistic responses

### Async Testing
- Uses `asyncio.run()` for async test execution
- Proper async/await patterns throughout
- Mocked async functions for controlled testing

### Validation Testing
- Schema validation with various data types
- Required field validation
- Enum value validation
- Number range validation
- Array type validation

## Expected Test Results

When all tests pass, you should see output like:

```
BugDetectiveAI Test Suite
============================================================
============================================================
RUNNING SYNCHRONOUS TESTS
============================================================
test_basic_config (test_base_model.TestModelConfig) ... ok
test_default_config (test_base_model.TestModelConfig) ... ok
...

============================================================
RUNNING ASYNCHRONOUS TESTS
============================================================
test_validate_config_valid (test_openai_model.TestOpenAILLMModel) ... ok
test_generate_structured_output_success (test_openai_model.TestOpenAILLMModel) ... ok
...

============================================================
TEST SUMMARY
============================================================
Synchronous tests: 25 run, 0 failures, 0 errors
Asynchronous tests: 15 run, 0 failures, 0 errors

Total: 40 tests run, 0 failures, 0 errors

🎉 ALL TESTS PASSED! 🎉
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running tests from the correct directory
2. **Async Test Failures**: Ensure proper async/await usage in test methods
3. **Mock Issues**: Check that mocks are properly configured for async functions

### Debug Mode

To run tests with more verbose output:

```bash
python -m unittest discover -v
```

### Running Specific Test Methods

To run a specific test method:

```bash
python -m unittest test_bug_detective.TestBugDetective.test_analyze_bug_basic_success
```

## Contributing

When adding new functionality:

1. **Add corresponding tests** for new features
2. **Update existing tests** if interfaces change
3. **Ensure test coverage** for error conditions
4. **Use descriptive test names** that explain what is being tested
5. **Follow the existing patterns** for mocking and async testing

## Test Maintenance

- **Regular Updates**: Update tests when functionality changes
- **Mock Maintenance**: Keep mocks in sync with actual implementations
- **Schema Validation**: Update schema tests when schemas change
- **Integration Testing**: Ensure end-to-end workflows continue to work 