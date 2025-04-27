"""
Letter segmentation implementation using Hi-SAM.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Add Hi-SAM directory to Python path
hi_sam_path = os.path.join(os.path.dirname(__file__), "Hi-SAM")
if hi_sam_path not in sys.path:
    sys.path.insert(0, hi_sam_path)

try:
    # Import all from demo_hisam.py
    from demo_hisam import *
except ImportError as e:
    logging.error(f"Failed to import Hi-SAM modules: {str(e)}")
    logging.error(f"Hi-SAM path: {hi_sam_path}")
    logging.error(f"Current sys.path: {sys.path}")
    raise

class LetterSegmenter:
    """
    Segments letters in word images using Hi-SAM.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the letter segmenter.
        
        Args:
            config: Configuration dictionary containing letter segmentation settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load Hi-SAM configuration
        self.model_type = config["letter_segmentation"]["model_type"]
        self.checkpoint = config["letter_segmentation"]["checkpoint"]
        self.device = config["letter_segmentation"]["device"]
        self.patch_mode = config["letter_segmentation"]["patch_mode"]
        self.input_size = config["letter_segmentation"]["input_size"]
        
        # Verify checkpoint exists
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"Hi-SAM checkpoint not found at {self.checkpoint}. "
                "Please download the model checkpoint first."
            )
        
        # Initialize Hi-SAM model
        self._initialize_model()
        
        self.logger.info(f"Initialized LetterSegmenter with model type: {self.model_type}")
    
    def _initialize_model(self):
        """Initialize the Hi-SAM model and predictor."""
        # Create args parser with default values
        args = self._get_args_parser()
        
        # Load model
        self.model = model_registry[self.model_type](args)
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize predictor
        self.predictor = SamPredictor(self.model)
    
    def _get_args_parser(self):
        """Create and return an args parser with current configuration."""
        from demo_hisam import get_args_parser
        
        return get_args_parser(
            checkpoint=self.checkpoint,
            model_type=self.model_type,
            device=self.device,
            patch_mode=self.patch_mode,
            input_size=self.input_size
        )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for Hi-SAM.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image in RGB format
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize if needed
        if self.input_size is not None:
            h, w = image.shape[:2]
            if max(h, w) > self.input_size:
                scale = self.input_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
        
        return image
    
    def segment_letters(self, word_image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for letters in the word image.
        
        Args:
            word_image: Word image as numpy array (RGB format)
            
        Returns:
            Binary mask where True indicates letter pixels
        """
        try:
            # Preprocess image
            word_image = self._preprocess_image(word_image)
            
            # Set image in predictor
            self.predictor.set_image(word_image)
            
            if self.patch_mode:
                # Use sliding window approach for large images
                ori_size = word_image.shape[:2]
                patch_list, h_slice_list, w_slice_list = patchify_sliding(
                    word_image, 
                    patch_size=512, 
                    stride=384
                )
                
                mask_512 = []
                for patch in patch_list:
                    self.predictor.set_image(patch)
                    _, hr_mask, _, _ = self.predictor.predict(multimask_output=False)
                    mask_512.append(hr_mask[0])
                
                # Combine patches
                mask = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
                assert mask.shape[-2:] == ori_size
            else:
                # Process entire image at once
                _, hr_mask, _, _ = self.predictor.predict(multimask_output=False)
                mask = hr_mask[0]
            
            # Convert to binary mask using model's default threshold
            binary_mask = mask > self.model.mask_threshold
            
            # Post-process mask
            binary_mask = self._postprocess_mask(binary_mask)
            
            return binary_mask
            
        except Exception as e:
            self.logger.error(f"Failed to segment letters: {str(e)}")
            raise
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process the binary mask to improve quality.
        
        Args:
            mask: Binary mask to process
            
        Returns:
            Processed binary mask
        """
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8) * 255
        )
        
        # Filter out small components
        min_area = 50  # Minimum area threshold
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                mask[labels == label] = False
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(bool) 