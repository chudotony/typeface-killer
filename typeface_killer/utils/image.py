"""
Image utility functions.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

def filter_by_size(
    images: List[np.ndarray],
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    return_indices: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
    """
    Filter images based on their dimensions.
    
    Args:
        images: List of images as numpy arrays
        min_height: Minimum height in pixels
        max_height: Maximum height in pixels
        min_width: Minimum width in pixels
        max_width: Maximum width in pixels
        return_indices: If True, also return indices of kept images
        
    Returns:
        List of filtered images, or tuple of (filtered images, kept indices) if return_indices is True
    """
    kept_images = []
    kept_indices = []
    
    for idx, img in enumerate(images):
        height, width = img.shape[:2]
        keep = True
        
        # Check height constraints
        if min_height is not None and height < min_height:
            keep = False
        if max_height is not None and height > max_height:
            keep = False
            
        # Check width constraints
        if min_width is not None and width < min_width:
            keep = False
        if max_width is not None and width > max_width:
            keep = False
            
        if keep:
            kept_images.append(img)
            kept_indices.append(idx)
    
    if return_indices:
        return kept_images, kept_indices
    return kept_images 