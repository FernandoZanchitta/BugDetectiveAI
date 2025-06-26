"""
Test runner for all BugDetectiveAI tests.
"""

import sys
import os
import unittest
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
from test_base_model import TestModelConfig, TestStructuredOutput, TestBaseLLMModel
from test_openai_model import TestOpenAILLMModel
from test_bug_detective import TestBugDetective, TestDetectionResult
from test_structured_output import (
    TestStructuredOutputProcessor, 
    TestBugAnalysisSchema, 
    TestCodeReviewSchema, 
    TestErrorAnalysisSchema
)
from test_integration import TestBugDetectiveIntegration, TestEndToEndWorkflow
from test_diff_metrics import TestDiffBasedMetric, TestMetricsEvaluator, TestFactoryFunction


def run_sync_tests():
    """Run all synchronous tests."""
    print("=" * 60)
    print("RUNNING SYNCHRONOUS TESTS")
    print("=" * 60)
    
    # Create test suite for sync tests
    sync_suite = unittest.TestSuite()
    
    # Add sync test classes
    sync_test_classes = [
        TestModelConfig,
        TestStructuredOutput,
        TestBaseLLMModel,
        TestDetectionResult,
        TestStructuredOutputProcessor,
        TestBugAnalysisSchema,
        TestCodeReviewSchema,
        TestErrorAnalysisSchema,
        TestDiffBasedMetric,
        TestMetricsEvaluator,
        TestFactoryFunction,
    ]
    
    for test_class in sync_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        sync_suite.addTests(tests)
    
    # Run sync tests
    runner = unittest.TextTestRunner(verbosity=2)
    sync_result = runner.run(sync_suite)
    
    return sync_result


async def run_async_tests():
    """Run all asynchronous tests."""
    print("\n" + "=" * 60)
    print("RUNNING ASYNCHRONOUS TESTS")
    print("=" * 60)
    
    # Create test suite for async tests
    async_suite = unittest.TestSuite()
    
    # Add async test classes
    async_test_classes = [
        TestOpenAILLMModel,
        TestBugDetective,
        TestBugDetectiveIntegration,
        TestEndToEndWorkflow,
    ]
    
    for test_class in async_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        async_suite.addTests(tests)
    
    # Run async tests
    runner = unittest.TextTestRunner(verbosity=2)
    async_result = runner.run(async_suite)
    
    return async_result


def run_specific_test_category(category):
    """Run tests for a specific category."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING {category.upper()} TESTS")
    print(f"{'=' * 60}")
    
    suite = unittest.TestSuite()
    
    if category == "base":
        test_classes = [TestModelConfig, TestStructuredOutput, TestBaseLLMModel]
    elif category == "openai":
        test_classes = [TestOpenAILLMModel]
    elif category == "detective":
        test_classes = [TestBugDetective, TestDetectionResult]
    elif category == "structured":
        test_classes = [TestStructuredOutputProcessor, TestBugAnalysisSchema, TestCodeReviewSchema, TestErrorAnalysisSchema]
    elif category == "integration":
        test_classes = [TestBugDetectiveIntegration, TestEndToEndWorkflow]
    elif category == "metrics":
        test_classes = [TestDiffBasedMetric, TestMetricsEvaluator, TestFactoryFunction]
    else:
        print(f"Unknown test category: {category}")
        return None
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def print_test_summary(sync_result, async_result):
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    if sync_result:
        total_tests += sync_result.testsRun
        total_failures += len(sync_result.failures)
        total_errors += len(sync_result.errors)
        print(f"Synchronous tests: {sync_result.testsRun} run, {len(sync_result.failures)} failures, {len(sync_result.errors)} errors")
    
    if async_result:
        total_tests += async_result.testsRun
        total_failures += len(async_result.failures)
        total_errors += len(async_result.errors)
        print(f"Asynchronous tests: {async_result.testsRun} run, {len(async_result.failures)} failures, {len(async_result.errors)} errors")
    
    print(f"\nTotal: {total_tests} tests run, {total_failures} failures, {total_errors} errors")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n❌ {total_failures + total_errors} tests failed")
    
    return total_failures + total_errors == 0


def main():
    """Main test runner function."""
    print("BugDetectiveAI Test Suite")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category in ["base", "openai", "detective", "structured", "integration", "metrics"]:
            result = run_specific_test_category(category)
            if result:
                success = len(result.failures) + len(result.errors) == 0
                print(f"\nCategory '{category}' tests: {'PASSED' if success else 'FAILED'}")
            return
        else:
            print(f"Unknown test category: {category}")
            print("Available categories: base, openai, detective, structured, integration, metrics")
            return
    
    # Run all tests
    try:
        # Run synchronous tests
        sync_result = run_sync_tests()
        
        # Run asynchronous tests
        async_result = asyncio.run(run_async_tests())
        
        # Print summary
        success = print_test_summary(sync_result, async_result)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 