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
import argparse

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
    
    def __init__(self, config: Dict, output_dir: str):
        """
        Initialize the letter segmenter.
        
        Args:
            config: Configuration dictionary containing letter segmentation settings
            output_dir: Directory to save segmentation results
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Load Hi-SAM configuration
        self.model_type = config["letter_segmentation"]["model_type"]
        self.device = config["letter_segmentation"]["device"]
        self.patch_mode = config["letter_segmentation"]["patch_mode"]
        self.input_size = config["letter_segmentation"]["input_size"]
        
        # Get checkpoint path relative to Hi-SAM directory
        self.checkpoint = os.path.join("pretrained_checkpoint", "sam_tss_h_textseg.pth")
        
        # Debug logging
        self.logger.debug(f"Current working directory: {os.getcwd()}")
        self.logger.debug(f"Hi-SAM path: {hi_sam_path}")
        self.logger.debug(f"Checkpoint path: {self.checkpoint}")
        self.logger.debug(f"Checkpoint exists: {os.path.exists(self.checkpoint)}")
        
        # Verify checkpoint exists
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"Hi-SAM checkpoint not found at {self.checkpoint}. "
                "Please download the model checkpoint and place it in the Hi-SAM/pretrained_checkpoint directory."
            )
        
        # Initialize Hi-SAM model
        self._initialize_model()
        
        self.logger.info(f"Initialized LetterSegmenter with model type: {self.model_type}")
    
    def _initialize_model(self):
        """Initialize the Hi-SAM model and predictor."""
        # Create args parser with default values
        args = self.get_args_parser()
        
        # Load model
        self.model = model_registry[self.model_type](args)
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize predictor
        self.predictor = SamPredictor(self.model)
    
    def get_args_parser(self, input_paths=["dummy_input"], checkpoint=None, output=None, model_type=None,
                    device=None, hier_det=False, input_size=[1024,1024],
                    patch_mode=False, attn_layers=1, prompt_len=12):
        parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

        parser.add_argument("--input", type=str, nargs="+", default=input_paths,
                            help="Path to the input image")
        parser.add_argument("--output", type=str, default=str(self.output_dir),
                            help="A file or directory to save output visualizations.")
        parser.add_argument("--model-type", type=str, default=self.model_type,
                            help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
        parser.add_argument("--checkpoint", type=str, default=self.checkpoint,
                            help="The path to the SAM checkpoint to use for mask generation.")
        parser.add_argument("--device", type=str, default=self.device,
                            help="The device to run generation on.")
        parser.add_argument("--hier_det", action='store_true', default=hier_det,
                            help="If False, only text stroke segmentation.")
        parser.add_argument('--input_size', default=input_size, type=list)
        parser.add_argument('--patch_mode', action='store_true', default=self.patch_mode)

        # Self-prompting arguments
        parser.add_argument('--attn_layers', default=attn_layers, type=int,
                            help='The number of image to token cross attention layers in model_aligner')
        parser.add_argument('--prompt_len', default=prompt_len, type=int,
                            help='The number of prompt token')

        # Parse with empty list to avoid SystemExit
        return parser.parse_args([])
    
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
        Generate a mask for letters in the word image, matching the notebook logic.
        Args:
            word_image: Word image as numpy array (BGR format from OpenCV)
        Returns:
            Mask (either thresholded or raw, depending on patch mode)
        """
        try:
            # Always convert BGR to RGB before prediction
            word_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2RGB)
            word_image = self._preprocess_image(word_image)
            self.predictor.set_image(word_image)
            if self.patch_mode:
                ori_size = word_image.shape[:2]
                patch_list, h_slice_list, w_slice_list = patchify_sliding(word_image, 512, 384)
                mask_512 = []
                for patch in patch_list:
                    self.predictor.set_image(patch)
                    _, hr_mask, _, _ = self.predictor.predict(multimask_output=False, return_logits=True)
                    mask_512.append(hr_mask[0])
                mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
                assert mask_512.shape[-2:] == ori_size
                mask = mask_512 > self.model.mask_threshold
            else:
                _, hr_mask, _, _ = self.predictor.predict(multimask_output=False)
                mask = hr_mask  # Do NOT threshold here!
            return mask
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