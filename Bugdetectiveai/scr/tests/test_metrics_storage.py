"""
Tests for the metrics storage functionality.
"""

import unittest
import pandas as pd
import numpy as np
from utils.simple_metrics import (
    compute_and_store_metrics, 
    has_metrics_columns, 
    get_metrics_columns,
    diff_score
)
from utils.visualization import (
    plot_metrics_histograms,
    compare_metrics_histograms_from_stored,
    _print_metric_statistics_from_stored
)

class TestMetricsStorage(unittest.TestCase):
    """Test cases for metrics storage functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'after_merge_without_docstrings': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    return a / b'
            ],
            'response_model_a': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    if b == 0:\n        raise ValueError("Division by zero")\n    return a / b'
            ],
            'response_model_b': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    return a / b'
            ]
        })
    
    def test_compute_and_store_metrics(self):
        """Test that metrics are computed and stored correctly."""
        # Compute metrics
        result_df = compute_and_store_metrics(
            self.sample_data, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a", "response_model_b"],
            show_progress=False
        )
        
        # Check that metrics columns were added
        expected_metrics = ['exact_match', 'ast_score', 'text_score', 'ast_score_normalized']
        for response_col in ["response_model_a", "response_model_b"]:
            for metric in expected_metrics:
                column_name = f"{response_col}_{metric}"
                self.assertIn(column_name, result_df.columns)
        
        # Check that metrics values are reasonable
        for response_col in ["response_model_a", "response_model_b"]:
            # Remove 'response_' prefix for metric column name
            clean_response_col = response_col.replace("response_", "")
            exact_match_col = f"metric_{clean_response_col}_exact_match"
            self.assertIn(exact_match_col, result_df.columns)
            values = result_df[exact_match_col].dropna()
            self.assertTrue(all(0 <= val <= 1 for val in values))
    
    def test_has_metrics_columns(self):
        """Test has_metrics_columns function."""
        # Initially no metrics
        self.assertFalse(has_metrics_columns(self.sample_data, "response_model_a"))
        
        # After computing metrics
        result_df = compute_and_store_metrics(
            self.sample_data, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a"],
            show_progress=False
        )
        self.assertTrue(has_metrics_columns(result_df, "response_model_a"))
    
    def test_get_metrics_columns(self):
        """Test get_metrics_columns function."""
        # Initially no metrics
        columns = get_metrics_columns(self.sample_data, "response_model_a")
        self.assertEqual(len(columns), 0)
        
        # After computing metrics
        result_df = compute_and_store_metrics(
            self.sample_data, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a"],
            show_progress=False
        )
        columns = get_metrics_columns(result_df, "response_model_a")
        self.assertGreater(len(columns), 0)
        
        # Check that all columns start with the metric prefix and clean response column name
        for col in columns:
            self.assertTrue(col.startswith("metric_model_a_"))
    
    def test_force_recompute(self):
        """Test force recomputation functionality."""
        # Compute metrics normally
        result_df = compute_and_store_metrics(
            self.sample_data, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a"],
            show_progress=False
        )
        
        # Store original values
        original_values = result_df["metric_model_a_exact_match"].copy()
        
        # Force recompute
        result_df_forced = compute_and_store_metrics(
            result_df, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a"],
            force_recompute=True,
            show_progress=False
        )
        
        # Values should be the same (same input data)
        pd.testing.assert_series_equal(
            original_values, 
            result_df_forced["metric_model_a_exact_match"]
        )
    
    def test_auto_detect_response_columns(self):
        """Test automatic detection of response columns."""
        # Add a non-response column
        test_df = self.sample_data.copy()
        test_df['other_column'] = 'test'
        
        result_df = compute_and_store_metrics(
            test_df, 
            reference_column="after_merge_without_docstrings",
            show_progress=False
        )
        
        # Should only process response_model_a and response_model_b
        expected_columns = ['response_model_a', 'response_model_b']
        for col in expected_columns:
            self.assertTrue(has_metrics_columns(result_df, col))
        
        # Should not process other_column
        self.assertFalse(has_metrics_columns(result_df, 'other_column'))
    
    def test_empty_or_nan_values(self):
        """Test handling of empty or NaN values."""
        test_df = self.sample_data.copy()
        test_df.loc[1, 'response_model_a'] = ''
        test_df.loc[2, 'response_model_a'] = np.nan
        
        result_df = compute_and_store_metrics(
            test_df, 
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a"],
            show_progress=False
        )
        
        # Should have NaN values for empty/NaN inputs
        exact_match_col = "metric_model_a_exact_match"
        self.assertTrue(pd.isna(result_df.loc[1, exact_match_col]))
        self.assertTrue(pd.isna(result_df.loc[2, exact_match_col]))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent reference column
        with self.assertRaises(ValueError):
            compute_and_store_metrics(
                self.sample_data, 
                reference_column="non_existent_column",
                show_progress=False
            )
        
        # Test with non-existent response columns
        with self.assertRaises(ValueError):
            compute_and_store_metrics(
                self.sample_data, 
                reference_column="after_merge_without_docstrings",
                response_columns=["non_existent_response"],
                show_progress=False
            )


class TestHistogramVisualization(unittest.TestCase):
    """Test cases for histogram visualization with stored metrics."""
    
    def setUp(self):
        """Set up test data with metrics."""
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'after_merge_without_docstrings': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    return a / b'
            ],
            'response_model_a': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    if b == 0:\n        raise ValueError("Division by zero")\n    return a / b'
            ],
            'response_model_b': [
                'def add(a, b):\n    return a + b',
                'def multiply(x, y):\n    return x * y',
                'def divide(a, b):\n    return a / b'
            ]
        })
        
        # Compute metrics for testing
        self.df_with_metrics = compute_and_store_metrics(
            self.sample_data,
            reference_column="after_merge_without_docstrings",
            response_columns=["response_model_a", "response_model_b"],
            show_progress=False
        )
    
    def test_compare_metrics_histograms_from_stored(self):
        """Test histogram creation from stored metrics."""
        # This should not raise an error
        try:
            compare_metrics_histograms_from_stored(
                self.df_with_metrics,
                response_columns=["response_model_a", "response_model_b"],
                title_prefix="Test Comparison"
            )
        except Exception as e:
            self.fail(f"compare_metrics_histograms_from_stored raised an exception: {e}")
    
    def test_compare_metrics_histograms_from_stored_no_metrics(self):
        """Test histogram creation with no stored metrics."""
        with self.assertRaises(ValueError):
            compare_metrics_histograms_from_stored(
                self.sample_data,  # No metrics computed
                response_columns=["response_model_a"]
            )
    
    def test_compare_metrics_histograms_from_stored_empty_response_columns(self):
        """Test histogram creation with empty response columns."""
        with self.assertRaises(ValueError):
            compare_metrics_histograms_from_stored(
                self.df_with_metrics,
                response_columns=[]
            )
    
    def test_print_metric_statistics_from_stored(self):
        """Test printing statistics from stored metrics."""
        # This should not raise an error
        try:
            metric_names = ['exact_match', 'ast_score']
            _print_metric_statistics_from_stored(
                self.df_with_metrics,
                response_columns=["response_model_a", "response_model_b"],
                metric_names=metric_names,
                stat_title="Test Statistics"
            )
        except Exception as e:
            self.fail(f"_print_metric_statistics_from_stored raised an exception: {e}")
    
    def test_plot_metrics_histograms_stored_metrics(self):
        """Test plot_metrics_histograms with stored metrics."""
        # This should not raise an error
        try:
            plot_metrics_histograms(
                self.df_with_metrics,
                reference_column="after_merge_without_docstrings",
                response_columns=["response_model_a", "response_model_b"],
                use_stored_metrics=True,
                compute_if_missing=False
            )
        except Exception as e:
            self.fail(f"plot_metrics_histograms with stored metrics raised an exception: {e}")
    
    def test_plot_metrics_histograms_missing_metrics(self):
        """Test plot_metrics_histograms with missing metrics."""
        # Should compute missing metrics automatically
        try:
            plot_metrics_histograms(
                self.sample_data,  # No metrics computed
                reference_column="after_merge_without_docstrings",
                response_columns=["response_model_a"],
                use_stored_metrics=True,
                compute_if_missing=True
            )
        except Exception as e:
            self.fail(f"plot_metrics_histograms with auto-computation raised an exception: {e}")
    
    def test_plot_metrics_histograms_onthefly(self):
        """Test plot_metrics_histograms with on-the-fly computation."""
        # This should not raise an error
        try:
            plot_metrics_histograms(
                self.sample_data,
                reference_column="after_merge_without_docstrings",
                response_columns=["response_model_a"],
                use_stored_metrics=False
            )
        except Exception as e:
            self.fail(f"plot_metrics_histograms with on-the-fly computation raised an exception: {e}")


if __name__ == '__main__':
    unittest.main() 