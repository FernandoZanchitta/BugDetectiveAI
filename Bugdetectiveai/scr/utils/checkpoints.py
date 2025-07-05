"""
Checkpoint utilities for BugDetectiveAI.
Provides robust checkpoint saving and loading functionality.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def get_checkpoint_path(dataset_name: str) -> Path:
    """Get checkpoint file path for a dataset.
    
    Args:
        dataset_name: Name of the dataset for checkpoint identification
        
    Returns:
        Path to the checkpoint file
    """
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{dataset_name}_responses.json"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load existing checkpoint if available.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing responses and processed indices
    """
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {"responses": [], "processed_indices": set()}


def save_checkpoint(checkpoint_path: Path, responses: List[str], processed_indices: set):
    """Save current progress to checkpoint.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        responses: List of generated responses
        processed_indices: Set of processed sample indices
    """
    try:
        checkpoint_data = {
            "responses": responses,
            "processed_indices": list(processed_indices)
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def list_checkpoints() -> List[str]:
    """List all available checkpoints.
    
    Returns:
        List of dataset names that have checkpoints
    """
    checkpoint_dir = Path("data/checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for file in checkpoint_dir.glob("*_responses.json"):
        dataset_name = file.stem.replace("_responses", "")
        checkpoints.append(dataset_name)
    
    return checkpoints


def delete_checkpoint(dataset_name: str) -> bool:
    """Delete a specific checkpoint.
    
    Args:
        dataset_name: Name of the dataset checkpoint to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    checkpoint_path = get_checkpoint_path(dataset_name)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting checkpoint: {e}")
            return False
    return False


def get_checkpoint_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a specific checkpoint.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with checkpoint information (file size, last modified, etc.)
    """
    checkpoint_path = get_checkpoint_path(dataset_name)
    if not checkpoint_path.exists():
        return {}
    
    try:
        stat = checkpoint_path.stat()
        checkpoint_data = load_checkpoint(checkpoint_path)
        
        return {
            "file_size": stat.st_size,
            "last_modified": stat.st_mtime,
            "total_responses": len(checkpoint_data.get("responses", [])),
            "processed_samples": len(checkpoint_data.get("processed_indices", set())),
            "file_path": str(checkpoint_path)
        }
    except Exception as e:
        print(f"Error getting checkpoint info: {e}")
        return {}


def clear_all_checkpoints() -> int:
    """Clear all checkpoints.
    
    Returns:
        Number of checkpoints deleted
    """
    checkpoint_dir = Path("data/checkpoints")
    if not checkpoint_dir.exists():
        return 0
    
    deleted_count = 0
    for file in checkpoint_dir.glob("*_responses.json"):
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    return deleted_count 