"""
Postprocessing utilities for segmentation outputs.
"""

import numpy as np
from typing import Optional


def postprocess_segmentation(segmentation: np.ndarray, 
                             min_size: int = 100,
                             fill_holes: bool = True) -> np.ndarray:
    """
    Postprocess segmentation mask.
    
    Args:
        segmentation: Segmentation mask [H, W] with class indices
        min_size: Minimum size for connected components (remove smaller regions)
        fill_holes: Whether to fill holes in segmentation masks
        
    Returns:
        Postprocessed segmentation mask
    """
    from scipy import ndimage
    
    processed = segmentation.copy()
    
    # Fill holes if requested
    if fill_holes:
        for class_id in np.unique(processed):
            if class_id == 0:  # Skip background
                continue
            mask = (processed == class_id).astype(np.uint8)
            filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
            processed[filled == 1] = class_id
    
    # Remove small connected components
    if min_size > 0:
        for class_id in np.unique(processed):
            if class_id == 0:  # Skip background
                continue
            mask = (processed == class_id).astype(np.uint8)
            labeled, num_features = ndimage.label(mask)
            
            for label_id in range(1, num_features + 1):
                component_size = np.sum(labeled == label_id)
                if component_size < min_size:
                    processed[labeled == label_id] = 0
    
    return processed


def get_segmentation_statistics(segmentation: np.ndarray) -> dict:
    """
    Calculate statistics for segmentation mask.
    
    Args:
        segmentation: Segmentation mask [H, W] or [H, W, D]
        
    Returns:
        Dictionary with statistics
    """
    unique_classes, counts = np.unique(segmentation, return_counts=True)
    
    stats = {
        "classes": unique_classes.tolist(),
        "class_counts": counts.tolist(),
        "total_pixels": int(segmentation.size),
        "class_percentages": {}
    }
    
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / segmentation.size) * 100
        stats["class_percentages"][int(class_id)] = float(percentage)
    
    return stats
