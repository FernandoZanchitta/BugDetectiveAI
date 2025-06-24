import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import unittest
from metrics import DiffBasedMetric, MetricsEvaluator, create_evaluator, MetricResult, EvaluationResult


class TestDiffBasedMetric(unittest.TestCase):
    """Test cases for the DiffBasedMetric class."""
    
    def setUp(self):
        self.metric = DiffBasedMetric()
    
    def test_metric_name(self):
        """Test that the metric has the correct name."""
        self.assertEqual(self.metric.name, "diff_edit_similarity")
    
    def test_perfect_match(self):
        """Test perfect match scenario."""
        before_code = "def add(a, b):\n    return a + b"
        after_code = "def add(a, b):\n    return a + b + 1"
        predicted_code = "def add(a, b):\n    return a + b + 1"
        
        result = self.metric.calculate_single(before_code, after_code, predicted_code)
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.metric_name, "diff_edit_similarity")
        self.assertEqual(result.score, 1.0)  # Perfect match
        self.assertIn('total_additions', result.metadata)
        self.assertIn('total_deletions', result.metadata)
    
    def test_no_match(self):
        """Test no match scenario."""
        before_code = "def add(a, b):\n    return a + b"
        after_code = "def add(a, b):\n    return a + b + 1"
        predicted_code = "def add(a, b):\n    return a + b"  # No change
        
        result = self.metric.calculate_single(before_code, after_code, predicted_code)
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.metric_name, "diff_edit_similarity")
        self.assertEqual(result.score, 0.0)  # No match


class TestMetricsEvaluator(unittest.TestCase):
    """Test cases for the MetricsEvaluator class."""
    
    def setUp(self):
        self.evaluator = MetricsEvaluator()
        self.diff_metric = DiffBasedMetric()
        self.evaluator.add_metric(self.diff_metric)
    
    def test_evaluate_single(self):
        """Test single sample evaluation."""
        before_code = "def add(a, b):\n    return a + b"
        after_code = "def add(a, b):\n    return a + b + 1"
        predicted_code = "def add(a, b):\n    return a + b + 1"
        
        result = self.evaluator.evaluate_single(before_code, after_code, predicted_code, "test_1")
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.sample_id, "test_1")
        self.assertIn("diff_edit_similarity", result.metrics)
        self.assertEqual(result.metrics["diff_edit_similarity"].score, 1.0)
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        before_codes = [
            "def add(a, b):\n    return a + b",
            "def sub(a, b):\n    return a - b"
        ]
        after_codes = [
            "def add(a, b):\n    return a + b + 1",
            "def sub(a, b):\n    return a - b - 1"
        ]
        predicted_codes = [
            "def add(a, b):\n    return a + b + 1",
            "def sub(a, b):\n    return a - b - 1"
        ]
        
        results = self.evaluator.evaluate_batch(before_codes, after_codes, predicted_codes)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("diff_edit_similarity", result.metrics)
            self.assertEqual(result.metrics["diff_edit_similarity"].score, 1.0)
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        before_codes = ["def add(a, b):\n    return a + b"]
        after_codes = ["def add(a, b):\n    return a + b + 1"]
        predicted_codes = ["def add(a, b):\n    return a + b + 1"]
        
        results = self.evaluator.evaluate_batch(before_codes, after_codes, predicted_codes)
        aggregated = self.evaluator.aggregate_results(results)
        
        self.assertIn("diff_edit_similarity", aggregated)
        self.assertIn("mean_diff_edit_similarity", aggregated["diff_edit_similarity"])
        self.assertEqual(aggregated["diff_edit_similarity"]["mean_diff_edit_similarity"], 1.0)


class TestFactoryFunction(unittest.TestCase):
    """Test cases for the factory function."""
    
    def test_create_evaluator_default(self):
        """Test creating evaluator with default metrics."""
        evaluator = create_evaluator()
        
        self.assertIsInstance(evaluator, MetricsEvaluator)
        self.assertIn("diff_edit_similarity", evaluator.get_metric_names())
    
    def test_create_evaluator_specific(self):
        """Test creating evaluator with specific metrics."""
        evaluator = create_evaluator(["diff_edit_similarity"])
        
        self.assertIsInstance(evaluator, MetricsEvaluator)
        self.assertEqual(evaluator.get_metric_names(), ["diff_edit_similarity"])


def run_simple_examples():
    """Run simple examples to demonstrate the system."""
    print("=== Simple Examples ===\n")
    
    # Example 1: Perfect fix
    print("Example 1: Perfect fix")
    evaluator = create_evaluator(["diff_edit_similarity"])
    
    before_code = "def add(a, b):\n    return a + b"
    after_code = "def add(a, b):\n    return a + b + 1"
    predicted_code = "def add(a, b):\n    return a + b + 1"
    
    result = evaluator.evaluate_single(before_code, after_code, predicted_code, "perfect_fix")
    print(f"Similarity: {result.metrics['diff_edit_similarity'].score:.3f}")
    print(f"Metadata: {result.metrics['diff_edit_similarity'].metadata}\n")
    
    # Example 2: Wrong fix
    print("Example 2: Wrong fix")
    wrong_predicted = "def add(a, b):\n    return a + b"  # No change
    
    result = evaluator.evaluate_single(before_code, after_code, wrong_predicted, "wrong_fix")
    print(f"Similarity: {result.metrics['diff_edit_similarity'].score:.3f}")
    print(f"Metadata: {result.metrics['diff_edit_similarity'].metadata}\n")
    
    # Example 3: Batch evaluation
    print("Example 3: Batch evaluation")
    batch_results = evaluator.evaluate_batch(
        [before_code, before_code],
        [after_code, after_code],
        [predicted_code, wrong_predicted],
        ["perfect", "wrong"]
    )
    
    aggregated = evaluator.aggregate_results(batch_results)
    print(f"Mean similarity: {aggregated['diff_edit_similarity']['mean_diff_edit_similarity']:.3f}")
    print(f"Std similarity: {aggregated['diff_edit_similarity']['std_diff_edit_similarity']:.3f}")


if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    print("\n" + "="*50 + "\n")
    
    # Run examples
    run_simple_examples() 