#!/usr/bin/env python3
"""
Example script demonstrating how to use the generic metrics system
with the BugDetectiveAI dataset format.

This script shows how to:
1. Load data from the buggy dataset using the data loader
2. Apply multiple metrics to evaluate LLM fixes
3. Generate comprehensive evaluation reports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from typing import List, Dict, Any, Optional
from metrics import MetricsEvaluator, create_evaluator, DiffBasedMetric
from data_loader.loader import load_buggy_dataset, get_dataset_info, filter_dataset_by_length


def simulate_llm_fixes(buggy_codes: List[str], gt_fixed_codes: List[str]) -> List[str]:
    """
    Simulate LLM-generated fixes for demonstration purposes.
    In practice, this would be replaced with actual LLM predictions.
    
    Args:
        buggy_codes: List of buggy code snippets
        gt_fixed_codes: List of ground truth fixed code snippets
        
    Returns:
        List of simulated LLM fixes
    """
    # For demonstration, we'll simulate different quality levels
    llm_fixes = []
    
    for i, (buggy, gt_fixed) in enumerate(zip(buggy_codes, gt_fixed_codes)):
        # Simulate different scenarios:
        # 1. Perfect fix (50% of cases)
        # 2. Partial fix (30% of cases) 
        # 3. Wrong fix (20% of cases)
        
        import random
        scenario = random.random()
        
        if scenario < 0.5:
            # Perfect fix
            llm_fixes.append(gt_fixed)
        elif scenario < 0.8:
            # Partial fix - add some noise
            if "return" in gt_fixed and "return" in buggy:
                # Simulate partial fix by keeping some of the bug
                llm_fixes.append(buggy.replace("return", "return # partial fix"))
            else:
                llm_fixes.append(gt_fixed)
        else:
            # Wrong fix - return original buggy code
            llm_fixes.append(buggy)
    
    return llm_fixes


def evaluate_dataset_with_metrics(split: str = 'test',
                                max_samples: int = 10,
                                metric_names: Optional[List[str]] = None,
                                base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a dataset using the generic metrics system.
    
    Args:
        split: Dataset split to evaluate
        max_samples: Maximum number of samples to evaluate
        metric_names: List of metric names to use. If None, uses all available.
        base_path: Base path to the dataset
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading {split} dataset...")
    
    # Load dataset using the data loader
    df = load_buggy_dataset(split, base_path)
    
    # Get dataset info
    info = get_dataset_info(df)
    print(f"Dataset shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    
    # Limit samples for demonstration
    if len(df) > max_samples:
        df = df.head(max_samples)
    
    print(f"Evaluating {len(df)} samples...")
    
    # Extract code snippets
    buggy_codes = df['before_merge'].tolist()
    gt_fixed_codes = df['after_merge'].tolist()
    
    # Simulate LLM fixes (replace with actual LLM predictions)
    llm_fixed_codes = simulate_llm_fixes(buggy_codes, gt_fixed_codes)
    
    # Create evaluator with specified metrics
    evaluator = create_evaluator(metric_names)
    
    # Generate sample IDs
    sample_ids = [f"{split}_{i}" for i in range(len(buggy_codes))]
    
    # Evaluate batch
    batch_results = evaluator.evaluate_batch(
        buggy_codes, gt_fixed_codes, llm_fixed_codes, sample_ids
    )
    
    # Aggregate results
    aggregated_results = evaluator.aggregate_results(batch_results)
    
    # Add dataset metadata
    results = {
        'dataset_info': {
            'split': split,
            'total_samples': len(df),
            'evaluated_samples': len(buggy_codes),
            'metrics_used': evaluator.get_metric_names(),
            'dataset_shape': info['shape'],
            'columns': info['columns']
        },
        'aggregated_metrics': aggregated_results,
        'individual_results': []
    }
    
    # Add individual sample details
    for i, result in enumerate(batch_results):
        # Get filename if available
        filename = df.iloc[i].get('bug filename', f'sample_{i}') if i < len(df) else f'sample_{i}'
        
        sample_details = {
            'sample_id': result.sample_id,
            'filename': filename,
            'metrics': {},
            'buggy_code_length': len(buggy_codes[i]),
            'gt_fixed_code_length': len(gt_fixed_codes[i]),
            'llm_fixed_code_length': len(llm_fixed_codes[i])
        }
        
        # Add metric results
        for metric_name, metric_result in result.metrics.items():
            sample_details['metrics'][metric_name] = {
                'score': metric_result.score,
                'metadata': metric_result.metadata
            }
        
        results['individual_results'].append(sample_details)
    
    return results


def print_evaluation_report(results: Dict[str, Any]):
    """
    Print a formatted evaluation report.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("GENERIC METRICS EVALUATION REPORT")
    print("="*60)
    
    # Dataset info
    dataset_info = results['dataset_info']
    print(f"\nDataset: {dataset_info['split']}")
    print(f"Total Samples: {dataset_info['total_samples']}")
    print(f"Evaluated Samples: {dataset_info['evaluated_samples']}")
    print(f"Metrics Used: {', '.join(dataset_info['metrics_used'])}")
    print(f"Dataset Shape: {dataset_info['dataset_shape']}")
    
    # Overall metrics
    aggregated = results['aggregated_metrics']
    print(f"\nOverall Metrics:")
    for metric_name, stats in aggregated.items():
        print(f"  {metric_name}:")
        for stat_name, value in stats.items():
            print(f"    {stat_name}: {value:.3f}")
    
    # Sample details
    print(f"\nSample Details:")
    if results['individual_results']:
        # Get all metric names from first sample
        first_sample = results['individual_results'][0]
        metric_names = list(first_sample['metrics'].keys())
        
        # Print header
        header = f"{'ID':<8} {'Filename':<20}"
        for metric_name in metric_names:
            header += f" {metric_name[:8]:<8}"
        print(header)
        print("-" * (8 + 20 + 8 * len(metric_names)))
        
        # Print sample results
        for sample in results['individual_results']:
            row = f"{sample['sample_id']:<8} {sample['filename'][:19]:<20}"
            for metric_name in metric_names:
                score = sample['metrics'][metric_name]['score']
                row += f" {score:<8.3f}"
            print(row)
    
    # Summary statistics
    if results['individual_results']:
        print(f"\nSummary:")
        for metric_name in metric_names:
            scores = [s['metrics'][metric_name]['score'] for s in results['individual_results']]
            perfect_fixes = sum(1 for s in scores if s == 1.0)
            good_fixes = sum(1 for s in scores if s >= 0.8)
            poor_fixes = sum(1 for s in scores if s < 0.5)
            
            print(f"  {metric_name}:")
            print(f"    Perfect fixes (score = 1.0): {perfect_fixes}/{len(scores)} ({perfect_fixes/len(scores)*100:.1f}%)")
            print(f"    Good fixes (score ≥ 0.8): {good_fixes}/{len(scores)} ({good_fixes/len(scores)*100:.1f}%)")
            print(f"    Poor fixes (score < 0.5): {poor_fixes}/{len(scores)} ({poor_fixes/len(scores)*100:.1f}%)")


def main():
    """Main function to run the generic metrics evaluation."""
    print("BugDetectiveAI - Generic Metrics Evaluation with Data Loader")
    print("="*60)
    
    # Configuration
    split = "test"
    max_samples = 10
    metric_names = ["diff_edit_similarity"]  # Can be extended with more metrics
    
    try:
        # Run evaluation
        results = evaluate_dataset_with_metrics(
            split, max_samples, metric_names
        )
        
        # Print report
        print_evaluation_report(results)
        
        # Save results
        output_file = f"generic_metrics_results_{split}.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def run_dummy_example():
    """Run example with dummy data when real dataset is not available."""
    print("\n" + "="*50)
    print("DUMMY DATA EXAMPLE")
    print("="*50)
    
    # Create dummy dataset
    buggy_codes = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def divide(a, b):\n    return a / b"
    ]
    
    gt_fixed_codes = [
        "def add(a, b):\n    return a + b + 1",
        "def multiply(x, y):\n    return x * y * 2",
        "def divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b"
    ]
    
    llm_fixed_codes = [
        "def add(a, b):\n    return a + b + 1",  # Perfect fix
        "def multiply(x, y):\n    return x * y",   # Wrong fix
        "def divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b"  # Perfect fix
    ]
    
    # Create evaluator
    evaluator = create_evaluator(["diff_edit_similarity"])
    
    # Evaluate
    results = evaluator.evaluate_batch(buggy_codes, gt_fixed_codes, llm_fixed_codes)
    aggregated = evaluator.aggregate_results(results)
    
    print(f"Mean Diff Edit Similarity: {aggregated['diff_edit_similarity']['mean_diff_edit_similarity']:.3f}")
    print(f"Individual similarities: {[results[i].metrics['diff_edit_similarity'].score for i in range(3)]}")
    
    print("\nThis demonstrates how the generic metrics system works:")
    print("- Sample 1: Perfect fix (similarity = 1.0)")
    print("- Sample 2: Wrong fix (similarity = 0.0)")
    print("- Sample 3: Perfect fix (similarity = 1.0)")
    
    print("\nThe system is designed to be easily extensible:")
    print("- Add new metrics by inheriting from BaseMetric")
    print("- Register metrics with the MetricsEvaluator")
    print("- Evaluate multiple metrics simultaneously")


if __name__ == "__main__":
    main() 