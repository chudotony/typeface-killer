"""
Word detection implementation using EasyOCR.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError

from typeface_killer.utils.image import resize_if_large

class WordDetector:
    """
    Detects words in images using EasyOCR with specific parameters for typographic analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the word detector.
        
        Args:
            config: Configuration dictionary containing word detection settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize EasyOCR reader with specified languages
        self.reader = easyocr.Reader(
            lang_list=config["word_detection"]["languages"],
            gpu=True  # Use GPU if available
        )
        
        # Load configuration parameters
        self.min_confidence = config["word_detection"]["min_confidence"]
        self.ocr_params = config["word_detection"]["ocr_params"]
        
        self.logger.info(f"Initialized WordDetector with languages: {config['word_detection']['languages']}")
    
    def detect_words(self, image_path: str) -> List[Dict]:
        """
        Detect words in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of dictionaries containing word information:
            {
                'x': int,          # x-coordinate of top-left corner
                'y': int,          # y-coordinate of top-left corner
                'width': int,      # width of the word bounding box
                'height': int,     # height of the word bounding box
                'text': str,       # detected text
                'confidence': float # confidence score
            }
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize image if needed to avoid DecompressionBombError
        image = resize_if_large(image)
        
        try:
            # Save resized image temporarily for EasyOCR
            temp_path = str(Path(image_path).with_name(f"temp_{Path(image_path).name}"))
            cv2.imwrite(temp_path, image)
            
            # Detect words with configured parameters
            results = self.reader.readtext(
                temp_path,
                **self.ocr_params
            )
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
        except Exception as e:
            self.logger.error(f"Error during word detection: {str(e)}")
            raise
        
        detected_words = []
        low_confidence_count = 0
        
        for bbox, text, confidence in results:
            # Filter by confidence
            if confidence < self.min_confidence:
                low_confidence_count += 1
                continue
            
            # Get bounding box coordinates
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            x, y = top_left
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            
            detected_words.append({
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'text': text,
                'confidence': confidence,
            })
        
        # Log if no words were detected
        if len(detected_words) == 0:
            self.logger.warning(f"No words detected in {image_path}")
        
        return detected_words