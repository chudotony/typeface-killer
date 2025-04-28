"""
Letter detection implementation using Tesseract and EasyOCR.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw
from tesserocr import PyTessBaseAPI, PSM, OEM, RIL

class LetterDetector:
    """
    Detects letters in word images using Tesseract and EasyOCR.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the letter detector.
        
        Args:
            config: Configuration dictionary containing letter detection settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(
            lang_list=config["word_detection"]["languages"],
            gpu=True  # Use GPU if available
        )
        
        # Tesseract configuration
        tesseract_config = config["letter_detection"]["tesseract"]
        self.tessdata_path = tesseract_config["tessdata_path"]
        self.languages = tesseract_config["languages"]
        self.min_region_size = tesseract_config["min_region_size"]
        self.large_region_threshold = tesseract_config["large_region_threshold"]
        self.small_region_padding = tesseract_config["small_region_padding"]
        self.large_region_padding = tesseract_config["large_region_padding"]
        
        # EasyOCR configuration
        easyocr_config = config["letter_detection"]["easyocr"]
        self.allowed_letters = easyocr_config["allowlist"]
        self.min_confidence = easyocr_config["min_confidence"]
        self.decoder = easyocr_config["decoder"]
        
        self.logger.info("Initialized LetterDetector")
    
    def _detect_letter_regions(self, pil_img: Image.Image) -> List[Dict]:
        """
        Detect potential letter regions using Tesseract.
        
        Args:
            pil_img: PIL Image object
            
        Returns:
            List of dictionaries containing region information
        """
        regions = []
        
        try:
            with PyTessBaseAPI(
                psm=PSM.SINGLE_BLOCK,
                oem=OEM.LSTM_ONLY,
                path=self.tessdata_path,
                lang=self.languages
            ) as api:
                api.SetImage(pil_img)
                api.SetVariable("lstm_use_matrix", "0")
                api.SetVariable("tessedit_char_whitelist", self.allowed_letters)
                api.SetVariable("classify_bln_numeric_mode", "0")
                api.SetVariable("segment_nonalphabetic_script", "0")
                
                boxes = api.GetComponentImages(RIL.SYMBOL, True, True)
                self.logger.info(f"Tesseract found {len(boxes)} potential regions")
                
                for i, (_, box, _, _) in enumerate(boxes):
                    if box is None:
                        self.logger.debug(f"Region {i}: box is None")
                        continue
                    
                    x, y, w, h = box['x'], box['y'], box['w'], box['h']
                    self.logger.debug(f"Region {i}: x={x}, y={y}, w={w}, h={h}")
                    
                    # Skip too small regions
                    if w * h < self.min_region_size:
                        self.logger.debug(f"Region {i}: too small (size={w*h}, min={self.min_region_size})")
                        continue
                    
                    # Determine padding based on region size
                    gap = self.large_region_padding if w * h > self.large_region_threshold else self.small_region_padding
                    bbox = {
                        'x': x - gap,
                        'y': y - gap,
                        'w': w + gap * 2,
                        'h': h + gap * 2
                    }
                    # Adjust bbox if it's outside the image
                    img_w, img_h = pil_img.size
                    if bbox['x']<0:
                        bbox['x'] = 0
                    if bbox['y']<0:
                        bbox['y'] = 0
                    if bbox['x']+bbox['w']>img_w:
                        bbox['w'] = img_w-bbox['x']
                    if bbox['y']+bbox['h']>img_h:
                        bbox['h'] = img_h-bbox['y']

                    try:
                        # Crop region with padding
                        region = pil_img.crop((x-gap, y-gap, x+w+gap, y+h+gap))
                        
                        regions.append({
                            'region': region,
                            'bbox': bbox
                        })
                        self.logger.debug(f"Region {i}: added with padding {gap}")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to crop region {i}: {str(e)}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract: {str(e)}")
            return []
        
        self.logger.info(f"Detected {len(regions)} valid regions")
        return regions
    
    def _identify_letters(self, regions: List[Dict]) -> List[Dict]:
        """
        Identify letters in regions using EasyOCR.
        
        Args:
            regions: List of region dictionaries from _detect_letter_regions
            
        Returns:
            List of dictionaries containing letter information
        """
        letters = []
        
        for i, region_info in enumerate(regions):
            try:
                # Convert region to numpy array for EasyOCR
                char_array = np.asarray(region_info['region'])
                
                # Detect character using EasyOCR
                result = self.reader.readtext(
                    char_array,
                    allowlist=self.allowed_letters,
                    decoder=self.decoder
                )
                self.logger.debug(f"Region {i}: EasyOCR found {len(result)} detections")
                
                for detection in result:
                    _, text, conf = detection
                    self.logger.debug(f"Region {i}: detected '{text}' with confidence {conf}")
                    
                    if conf >= self.min_confidence and len(text) == 1:
                        letters.append({
                            'char': text,
                            'bbox': region_info['bbox']
                        })
                        self.logger.debug(f"Region {i}: added letter '{text}'")
            
            except Exception as e:
                self.logger.warning(f"Failed to identify letter in region {i}: {str(e)}")
                continue
        
        self.logger.info(f"Identified {len(letters)} letters")
        return letters
    
    def detect_letters(self, word_image: np.ndarray) -> List[Dict]:
        """
        Detect and identify letters in a word image.
        
        Args:
            word_image: Word image as numpy array
            
        Returns:
            List of dictionaries containing letter information
        """
        # Convert to PIL Image
        pil_img = Image.fromarray(word_image)
        
        # Step 1: Detect potential letter regions
        regions = self._detect_letter_regions(pil_img)
        self.logger.info(f"Detected {len(regions)} potential letter regions")
        
        # Step 2: Identify letters in regions
        letters = self._identify_letters(regions)
        self.logger.info(f"Identified {len(letters)} letters")
        
        return letters 