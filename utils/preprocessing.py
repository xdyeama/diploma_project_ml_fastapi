"""
Preprocessing utilities for different input types.
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple


def normalize_image(image: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
        min_val: Minimum value for normalization (if None, uses image min)
        max_val: Maximum value for normalization (if None, uses image max)
        
    Returns:
        Normalized image
    """
    if min_val is None:
        min_val = image.min()
    if max_val is None:
        max_val = image.max()
    
    if max_val == min_val:
        return np.zeros_like(image)
    
    normalized = (image - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.
    
    Args:
        image: PIL Image (grayscale)
        target_size: Target size for resizing (height, width)
        
    Returns:
        Preprocessed tensor ready for model input [1, 1, H, W]
    """
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize
    img_array = normalize_image(img_array)
    
    # Resize if needed
    if image.size != target_size[::-1]:  # PIL size is (width, height)
        image_resized = image.resize(target_size[::-1], Image.LANCZOS)
        img_array = np.array(image_resized, dtype=np.float32)
        img_array = normalize_image(img_array)
    
    # Add batch and channel dimensions: [1, 1, H, W]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def preprocess_nifti(volume: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess a 3D NIfTI volume.
    
    Args:
        volume: 3D numpy array [H, W, D]
        target_size: Target size for resizing slices (height, width)
        
    Returns:
        Preprocessed volume [H, W, D]
    """
    processed_volume = np.zeros((target_size[0], target_size[1], volume.shape[2]), dtype=np.float32)
    
    for i in range(volume.shape[2]):
        slice_data = volume[:, :, i]
        
        # Normalize slice
        slice_normalized = normalize_image(slice_data)
        
        # Resize slice
        from scipy.ndimage import zoom
        if slice_data.shape != target_size:
            zoom_factors = (target_size[0] / slice_data.shape[0], target_size[1] / slice_data.shape[1])
            slice_normalized = zoom(slice_normalized, zoom_factors, order=1)
        
        processed_volume[:, :, i] = slice_normalized
    
    return processed_volume


def preprocess_nifti_slice(slice_data: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Preprocess a single NIfTI slice for model inference.
    
    Args:
        slice_data: 2D numpy array [H, W]
        target_size: Target size for resizing (height, width)
        
    Returns:
        Preprocessed tensor ready for model input [1, 1, H, W]
    """
    # Normalize
    slice_normalized = normalize_image(slice_data)
    
    # Resize if needed
    if slice_data.shape != target_size:
        from scipy.ndimage import zoom
        zoom_factors = (target_size[0] / slice_data.shape[0], target_size[1] / slice_data.shape[1])
        slice_normalized = zoom(slice_normalized, zoom_factors, order=1)
    
    # Add batch and channel dimensions
    slice_tensor = torch.from_numpy(slice_normalized).unsqueeze(0).unsqueeze(0).float()
    
    return slice_tensor
