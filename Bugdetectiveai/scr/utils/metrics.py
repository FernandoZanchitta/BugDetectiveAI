import difflib
import re
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class DiffHunk:
    """Represents a single diff hunk with its metadata."""
    start_line: int
    end_line: int
    lines: List[str]
    operation: str  # 'add', 'remove', 'modify'
    content: str


@dataclass
class MetricResult:
    """Generic container for metric evaluation results."""
    metric_name: str
    score: float
    metadata: Dict[str, Any]
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Container for complete evaluation results."""
    sample_id: Optional[str] = None
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    
    All metrics should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_single(self, 
                        before_code: str, 
                        after_code: str, 
                        predicted_code: str,
                        **kwargs) -> MetricResult:
        """
        Calculate metric for a single sample.
        
        Args:
            before_code: Original code (e.g., buggy code)
            after_code: Target code (e.g., ground truth fixed code)
            predicted_code: Predicted code (e.g., LLM-generated fix)
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            MetricResult with the calculated score and metadata
        """
        pass
    
    def calculate_batch(self, 
                       before_codes: List[str],
                       after_codes: List[str], 
                       predicted_codes: List[str],
                       **kwargs) -> List[MetricResult]:
        """
        Calculate metric for a batch of samples.
        
        Args:
            before_codes: List of original codes
            after_codes: List of target codes
            predicted_codes: List of predicted codes
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            List of MetricResult objects
        """
        if len(before_codes) != len(after_codes) != len(predicted_codes):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for before, after, predicted in zip(before_codes, after_codes, predicted_codes):
            result = self.calculate_single(before, after, predicted, **kwargs)
            results.append(result)
        
        return results
    
    def aggregate_results(self, results: List[MetricResult]) -> Dict[str, float]:
        """
        Aggregate batch results into summary statistics.
        
        Args:
            results: List of MetricResult objects
            
        Returns:
            Dictionary with aggregated statistics
        """
        scores = [result.score for result in results]
        
        return {
            f'mean_{self.name}': float(np.mean(scores)),
            f'std_{self.name}': float(np.std(scores)),
            f'median_{self.name}': float(np.median(scores)),
            f'min_{self.name}': float(np.min(scores)),
            f'max_{self.name}': float(np.max(scores)),
            f'count_{self.name}': len(scores)
        }


class DiffBasedMetric(BaseMetric):
    """
    Diff-based metric for evaluating code changes.
    
    Compares the unified diff between:
    - Ground truth: before_code → after_code
    - Prediction: before_code → predicted_code
    
    Measures how similar the changes are using IoU of diff hunks.
    """
    
    def __init__(self):
        super().__init__("diff_edit_similarity")
    
    def parse_unified_diff(self, diff_lines: List[str]) -> List[DiffHunk]:
        """
        Parse unified diff output into structured diff hunks.
        
        Args:
            diff_lines: List of lines from unified diff
            
        Returns:
            List of DiffHunk objects
        """
        hunks = []
        current_hunk = None
        current_lines = []
        
        for line in diff_lines:
            # Hunk header: @@ -start,count +start,count @@
            if line.startswith('@@'):
                # Save previous hunk if exists
                if current_hunk and current_lines:
                    current_hunk.lines = current_lines
                    hunks.append(current_hunk)
                
                # Parse hunk header
                match = re.match(r'^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    current_hunk = DiffHunk(
                        start_line=old_start,
                        end_line=old_start + old_count - 1,
                        lines=[],
                        operation='modify',
                        content=''
                    )
                    current_lines = []
            
            elif line.startswith('+') and not line.startswith('+++'):
                # Addition
                if current_hunk:
                    current_lines.append(line)
                    if current_hunk.operation == 'modify':
                        current_hunk.operation = 'add'
            elif line.startswith('-') and not line.startswith('---'):
                # Deletion
                if current_hunk:
                    current_lines.append(line)
                    if current_hunk.operation == 'modify':
                        current_hunk.operation = 'remove'
            elif line.startswith(' '):
                # Context line
                if current_hunk:
                    current_lines.append(line)
        
        # Add final hunk
        if current_hunk and current_lines:
            current_hunk.lines = current_lines
            current_hunk.content = '\n'.join(current_lines)
            hunks.append(current_hunk)
        
        return hunks
    
    def get_unified_diff(self, before_code: str, after_code: str) -> List[str]:
        """
        Generate unified diff between two code versions.
        
        Args:
            before_code: Original code
            after_code: Modified code
            
        Returns:
            List of diff lines
        """
        before_lines = before_code.splitlines(keepends=True)
        after_lines = after_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile='before',
            tofile='after',
            lineterm=''
        )
        
        return list(diff)
    
    def extract_diff_hunks(self, before_code: str, after_code: str) -> List[DiffHunk]:
        """
        Extract diff hunks from code comparison.
        
        Args:
            before_code: Original code
            after_code: Modified code
            
        Returns:
            List of DiffHunk objects
        """
        diff_lines = self.get_unified_diff(before_code, after_code)
        return self.parse_unified_diff(diff_lines)
    
    def calculate_hunk_similarity(self, hunk1: DiffHunk, hunk2: DiffHunk) -> float:
        """
        Calculate similarity between two diff hunks.
        
        Args:
            hunk1: First diff hunk
            hunk2: Second diff hunk
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check if hunks overlap in line ranges
        overlap_start = max(hunk1.start_line, hunk2.start_line)
        overlap_end = min(hunk1.end_line, hunk2.end_line)
        
        if overlap_start > overlap_end:
            return 0.0
        
        # Calculate content similarity using sequence matcher
        matcher = difflib.SequenceMatcher(None, hunk1.content, hunk2.content)
        return matcher.ratio()
    
    def calculate_iou_similarity(self, gt_hunks: List[DiffHunk], 
                                pred_hunks: List[DiffHunk]) -> float:
        """
        Calculate Intersection over Union (IoU) similarity between two sets of diff hunks.
        
        Args:
            gt_hunks: Ground truth diff hunks
            pred_hunks: Predicted diff hunks
            
        Returns:
            IoU similarity score between 0 and 1
        """
        if not gt_hunks and not pred_hunks:
            return 1.0  # Both empty, perfect match
        
        if not gt_hunks or not pred_hunks:
            return 0.0  # One empty, no match
        
        # Calculate pairwise similarities
        similarities = []
        for gt_hunk in gt_hunks:
            for pred_hunk in pred_hunks:
                similarity = self.calculate_hunk_similarity(gt_hunk, pred_hunk)
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Use maximum similarity for each hunk (best match)
        max_similarities = []
        used_pred_indices = set()
        
        for i, gt_hunk in enumerate(gt_hunks):
            best_similarity = 0.0
            best_pred_idx = -1
            
            for j, pred_hunk in enumerate(pred_hunks):
                if j not in used_pred_indices:
                    similarity = self.calculate_hunk_similarity(gt_hunk, pred_hunk)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pred_idx = j
            
            if best_pred_idx >= 0:
                used_pred_indices.add(best_pred_idx)
                max_similarities.append(best_similarity)
        
        # Calculate IoU
        intersection = sum(max_similarities)
        union = len(gt_hunks) + len(pred_hunks) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_single(self, 
                        before_code: str,
                        after_code: str,
                        predicted_code: str,
                        **kwargs) -> MetricResult:
        """
        Calculate diff edit similarity between ground truth and predicted fixes.
        
        Args:
            before_code: Original code (e.g., buggy code)
            after_code: Target code (e.g., ground truth fixed code)
            predicted_code: Predicted code (e.g., LLM-generated fix)
            **kwargs: Additional arguments (not used in this metric)
            
        Returns:
            MetricResult with similarity score and metadata
        """
        # Extract diff hunks for ground truth
        gt_hunks = self.extract_diff_hunks(before_code, after_code)
        
        # Extract diff hunks for prediction
        pred_hunks = self.extract_diff_hunks(before_code, predicted_code)
        
        # Calculate IoU similarity
        iou_similarity = self.calculate_iou_similarity(gt_hunks, pred_hunks)
        
        # Calculate statistics
        total_additions = sum(1 for hunk in pred_hunks if hunk.operation == 'add')
        total_deletions = sum(1 for hunk in pred_hunks if hunk.operation == 'remove')
        total_modifications = sum(1 for hunk in pred_hunks if hunk.operation == 'modify')
        
        metadata = {
            'total_additions': total_additions,
            'total_deletions': total_deletions,
            'total_modifications': total_modifications,
            'num_hunks': len(pred_hunks),
            'gt_num_hunks': len(gt_hunks),
            'hunks': [
                {
                    'start_line': hunk.start_line,
                    'end_line': hunk.end_line,
                    'operation': hunk.operation,
                    'content': hunk.content
                }
                for hunk in pred_hunks
            ]
        }
        
        return MetricResult(
            metric_name=self.name,
            score=iou_similarity,
            metadata=metadata
        )


class MetricsEvaluator:
    """
    Main evaluator class that manages multiple metrics.
    
    This class provides a unified interface for evaluating code
    using multiple metrics simultaneously.
    """
    
    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        """
        Initialize the evaluator with a list of metrics.
        
        Args:
            metrics: List of metric instances to use for evaluation
        """
        self.metrics = metrics or []
        self._metric_dict = {metric.name: metric for metric in self.metrics}
    
    def add_metric(self, metric: BaseMetric):
        """
        Add a metric to the evaluator.
        
        Args:
            metric: Metric instance to add
        """
        self.metrics.append(metric)
        self._metric_dict[metric.name] = metric
    
    def remove_metric(self, metric_name: str):
        """
        Remove a metric from the evaluator.
        
        Args:
            metric_name: Name of the metric to remove
        """
        if metric_name in self._metric_dict:
            metric = self._metric_dict[metric_name]
            self.metrics.remove(metric)
            del self._metric_dict[metric_name]
    
    def evaluate_single(self, 
                       before_code: str,
                       after_code: str,
                       predicted_code: str,
                       sample_id: Optional[str] = None,
                       **kwargs) -> EvaluationResult:
        """
        Evaluate a single sample using all registered metrics.
        
        Args:
            before_code: Original code
            after_code: Target code
            predicted_code: Predicted code
            sample_id: Optional identifier for the sample
            **kwargs: Additional arguments passed to metrics
            
        Returns:
            EvaluationResult with all metric scores
        """
        result = EvaluationResult(sample_id=sample_id)
        
        for metric in self.metrics:
            metric_result = metric.calculate_single(
                before_code, after_code, predicted_code, **kwargs
            )
            result.metrics[metric.name] = metric_result
        
        return result
    
    def evaluate_batch(self, 
                      before_codes: List[str],
                      after_codes: List[str],
                      predicted_codes: List[str],
                      sample_ids: Optional[List[str]] = None,
                      **kwargs) -> List[EvaluationResult]:
        """
        Evaluate a batch of samples using all registered metrics.
        
        Args:
            before_codes: List of original codes
            after_codes: List of target codes
            predicted_codes: List of predicted codes
            sample_ids: Optional list of sample identifiers
            **kwargs: Additional arguments passed to metrics
            
        Returns:
            List of EvaluationResult objects
        """
        if len(before_codes) != len(after_codes) != len(predicted_codes):
            raise ValueError("All input lists must have the same length")
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(before_codes))]
        
        results = []
        for i, (before, after, predicted) in enumerate(zip(before_codes, after_codes, predicted_codes)):
            sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
            result = self.evaluate_single(before, after, predicted, sample_id, **kwargs)
            results.append(result)
        
        return results
    
    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate batch results into summary statistics for all metrics.
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary mapping metric names to their aggregated statistics
        """
        if not results:
            return {}
        
        # Group results by metric
        metric_results = {}
        for result in results:
            for metric_name, metric_result in result.metrics.items():
                if metric_name not in metric_results:
                    metric_results[metric_name] = []
                metric_results[metric_name].append(metric_result)
        
        # Aggregate each metric
        aggregated = {}
        for metric_name, metric_result_list in metric_results.items():
            if metric_name in self._metric_dict:
                metric = self._metric_dict[metric_name]
                aggregated[metric_name] = metric.aggregate_results(metric_result_list)
        
        return aggregated
    
    def get_metric_names(self) -> List[str]:
        """Get list of registered metric names."""
        return list(self._metric_dict.keys())
    
    def get_metric(self, metric_name: str) -> Optional[BaseMetric]:
        """Get a specific metric by name."""
        return self._metric_dict.get(metric_name)


# Factory function for creating common metric combinations
def create_evaluator(metric_names: Optional[List[str]] = None) -> MetricsEvaluator:
    """
    Create an evaluator with common metrics.
    
    Args:
        metric_names: List of metric names to include. If None, includes all available.
        
    Returns:
        MetricsEvaluator instance
    """
    available_metrics = {
        'diff_edit_similarity': DiffBasedMetric()
    }
    
    if metric_names is None:
        metric_names = list(available_metrics.keys())
    
    metrics: List[BaseMetric] = [available_metrics[name] for name in metric_names if name in available_metrics]
    
    return MetricsEvaluator(metrics)


# Example usage
def example_usage():
    """Example usage of the generic metrics system."""
    # Create evaluator with diff-based metric
    evaluator = create_evaluator(['diff_edit_similarity'])
    
    # Example data
    before_code = """
def calculate_sum(a, b):
    return a + b  # Bug: should be a + b + 1
"""
    
    after_code = """
def calculate_sum(a, b):
    return a + b + 1  # Fixed: added + 1
"""
    
    predicted_code = """
def calculate_sum(a, b):
    return a + b + 1  # Fixed the bug
"""
    
    # Evaluate single sample
    result = evaluator.evaluate_single(before_code, after_code, predicted_code, "example_1")
    
    print("Single Sample Evaluation:")
    for metric_name, metric_result in result.metrics.items():
        print(f"  {metric_name}: {metric_result.score:.3f}")
        print(f"    Metadata: {metric_result.metadata}")
    
    # Evaluate batch
    batch_results = evaluator.evaluate_batch(
        [before_code], [after_code], [predicted_code], ["example_1"]
    )
    
    # Aggregate results
    aggregated = evaluator.aggregate_results(batch_results)
    
    print("\nBatch Evaluation:")
    for metric_name, stats in aggregated.items():
        print(f"  {metric_name}:")
        for stat_name, value in stats.items():
            print(f"    {stat_name}: {value:.3f}")


if __name__ == "__main__":
    example_usage()
