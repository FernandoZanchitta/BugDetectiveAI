"""
Simple test runner for BugDetectiveAI tests.
"""

import sys
import os
import unittest
import asyncio

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_sync_tests():
    """Run all synchronous tests."""
    print("=" * 60)
    print("RUNNING SYNCHRONOUS TESTS")
    print("=" * 60)

    # Import test modules
    from test_base_model import TestModelConfig, TestStructuredOutput, TestBaseLLMModel
    from test_structured_output import (
        TestStructuredOutputProcessor,
        TestBugAnalysisSchema,
        TestCodeReviewSchema,
        TestErrorAnalysisSchema,
    )
    from test_diff_metrics import (
        TestDiffBasedMetric,
        TestMetricsEvaluator,
        TestFactoryFunction,
    )

    # Create test suite
    suite = unittest.TestSuite()

    # Add sync test classes
    sync_test_classes = [
        TestModelConfig,
        TestStructuredOutput,
        TestBaseLLMModel,
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
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def run_simple_async_tests():
    """Run simple async tests."""
    print("\n" + "=" * 60)
    print("RUNNING SIMPLE ASYNC TESTS")
    print("=" * 60)

    # No async tests for now - removed OpenAI tests
    print("No async tests to run")

    # Create empty test suite
    suite = unittest.TestSuite()

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def main():
    """Main test runner function."""
    print("BugDetectiveAI Simple Test Suite")
    print("=" * 60)

    try:
        # Run synchronous tests
        sync_result = run_sync_tests()

        # Run simple async tests
        async_result = run_simple_async_tests()

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = sync_result.testsRun + async_result.testsRun
        total_failures = len(sync_result.failures) + len(async_result.failures)
        total_errors = len(sync_result.errors) + len(async_result.errors)

        print(
            f"Synchronous tests: {sync_result.testsRun} run, {len(sync_result.failures)} failures, {len(sync_result.errors)} errors"
        )
        print(
            f"Asynchronous tests: {async_result.testsRun} run, {len(async_result.failures)} failures, {len(async_result.errors)} errors"
        )
        print(
            f"\nTotal: {total_tests} tests run, {total_failures} failures, {total_errors} errors"
        )

        if total_failures == 0 and total_errors == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n❌ {total_failures + total_errors} tests failed")

        return total_failures + total_errors == 0

    except Exception as e:
        print(f"\n\nError running tests: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
