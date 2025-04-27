#!/usr/bin/env python3
"""
Main script for the Typeface Killer pipeline.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import PIL.Image as Image
import potrace

from typeface_killer.word_detection import WordDetector
from typeface_killer.letter_detection import LetterDetector
from typeface_killer.letter_segmentation import LetterSegmenter
from typeface_killer.vectorization import Vectorizer
from typeface_killer.utils.config import load_config
from typeface_killer.utils.logging import setup_logging
from typeface_killer.utils.image import filter_by_size

# Import save_binary_mask from Hi-SAM
import sys
hi_sam_path = os.path.join(os.path.dirname(__file__), "letter_segmentation", "Hi-SAM")
if hi_sam_path not in sys.path:
    sys.path.insert(0, hi_sam_path)
from demo_hisam import save_binary_mask

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Typeface Killer - Typographic Feature Extraction Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    return parser

def detect_words_for_all_images(image_paths: List[str], config: Dict, output_dir: str) -> Dict[str, List[Dict]]:
    """Detect words in all images using EasyOCR."""
    word_detector = WordDetector(config)
    all_words = {}
    
    # Create words directory in output
    words_dir = Path(output_dir) / "words"
    words_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in image_paths:
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image: {image_path}")
            continue
            
        words = word_detector.detect_words(image_path)
        
        # Get word images for filtering
        word_images = []
        for word in words:
            x, y = word['x'], word['y']
            width, height = word['width'], word['height']
            word_image = image[y:y+height, x:x+width]
            word_images.append(word_image)
        
        # Filter word images by size
        filtered_images, kept_indices = filter_by_size(
            word_images,
            min_height=config["word_detection"]["min_height"],
            min_width=config["word_detection"]["min_width"],
            return_indices=True
        )
        
        # Get filtered words and save their images
        filtered_words = [words[idx] for idx in kept_indices]
        for idx, word in enumerate(filtered_words):
            # Save the cropped word image with original filename + 4-digit number
            word_filename = f"{Path(image_path).stem}_{idx:04d}.png"
            word_path = str(words_dir / word_filename)
            cv2.imwrite(word_path, filtered_images[idx])
        
        all_words[image_path] = filtered_words
    
    # Clean up EasyOCR
    del word_detector
    return all_words

def detect_letters_for_all_words(all_words: Dict[str, List[Dict]], config: Dict, output_dir: str) -> Dict[str, List[Dict]]:
    """Detect letters in all words using Tesseract."""
    letter_detector = LetterDetector(config)
    all_letters = {}
    
    # Get words directory path
    words_dir = Path(output_dir) / "words"
    
    for image_path, words in all_words.items():
        image_letters = []
        for word_idx, word in enumerate(words):
            # Read the word image from file
            word_filename = f"{Path(image_path).stem}_{word_idx:04d}.png"
            word_path = str(words_dir / word_filename)
            word_image = cv2.imread(word_path)
            
            if word_image is None:
                logging.error(f"Could not read word image: {word_path}")
                continue
                
            # Detect letters in the word image
            letters = letter_detector.detect_letters(word_image)
            # Add word index to each letter
            for letter in letters:
                letter["word_idx"] = word_idx
                letter["word_path"] = word_path
            image_letters.extend(letters)
            logging.info(f"Detected {len(letters)} letters in word {word_path}")
        
        all_letters[image_path] = image_letters
        logging.info(f"Total letters for image {image_path}: {len(image_letters)}")
    
    # Clean up Tesseract
    del letter_detector
    return all_letters

def segment_letters_for_all_images(all_letters: Dict[str, List[Dict]], config: Dict, output_dir: str) -> Dict[str, List[Dict]]:
    """Segment all words using Hi-SAM."""
    # Create words_segmented directory in output
    segments_dir = Path(output_dir) / "words_segmented"
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize letter segmenter
    letter_segmenter = LetterSegmenter(config=config, output_dir=str(segments_dir))
    
    # Process each unique word image
    processed_words = set()
    for image_path, letters in all_letters.items():
        logging.info(f"Processing image {image_path} with {len(letters)} letters")
        for letter in letters:
            word_path = letter["word_path"]
            if word_path in processed_words:
                continue
                
            # Read and segment the word
            word_image = cv2.imread(word_path)
            if word_image is None:
                logging.error(f"Could not read word image: {word_path}")
                continue
            
            segmented_mask = letter_segmenter.segment_letters(word_image)
            if segmented_mask is None:
                logging.error(f"Failed to segment word: {word_path}")
                continue
            
            # Save the segmented word
            segment_path = str(segments_dir / Path(word_path).name)
            try:
                save_binary_mask(segmented_mask, segment_path)
                processed_words.add(word_path)
                logging.info(f"Saved segmented word: {segment_path}")
            except Exception as e:
                logging.error(f"Error saving mask to {segment_path}: {str(e)}")
    
    # Clean up
    del letter_segmenter
    return all_letters  # Return original letters with bboxes

def vectorize_all_letters(all_letters: Dict[str, List[Dict]], output_dir: str, config: Dict) -> Dict[str, Dict]:
    """Vectorize all segmented letters."""
    results = {}
    
    # Create vectors directory
    vectors_dir = Path(output_dir) / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Get segments directory
    segments_dir = Path(output_dir) / "words_segmented"
    
    for image_path, letters in all_letters.items():
        logging.info(f"Vectorizing {len(letters)} letters for image {image_path}")
        image_results = []
        for idx, letter in enumerate(letters):
            # Get the segmented word image
            word_path = letter["word_path"]
            segment_path = str(segments_dir / Path(word_path).name)
            
            try:
                word_image = Image.open(segment_path).convert('L')
            except Exception as e:
                logging.error(f"Could not read segmented word: {segment_path}")
                continue
            
            # Get letter bbox
            bbox = letter["bbox"]
            x, y = bbox["x"], bbox["y"]
            w, h = bbox["w"], bbox["h"]
            
            # Validate bbox
            if x < 0 or y < 0 or x + w > word_image.width or y + h > word_image.height:
                logging.warning(f"Invalid bbox for letter {idx} in {word_path}: {bbox}")
                continue
            
            # Crop the letter region
            crop = word_image.crop((x, y, x+w, y+h))
            
            # Check if crop is empty
            crop_array = np.asarray(crop)
            if np.all(crop_array == 0):
                logging.warning(f"Empty crop for letter {idx} in {word_path}")
                continue
            
            # Convert to numpy array and invert
            crop_array = 255 - crop_array  # Invert the image
            
            # Create bitmap and trace
            bitmap = potrace.Bitmap(crop_array)
            path = bitmap.trace(opttolerance=config["vectorization"]["opttolerance"])
            
            # Generate SVG path
            svg_filename = f"{Path(image_path).stem}_{idx:04d}.svg"
            svg_path = str(vectors_dir / svg_filename)
            
            # Write SVG file
            with open(svg_path, "w") as f:
                f.write(f'<svg viewBox="0 0 {w} {h}">')
                f.write('<path d="')
                
                for curve in path:
                    f.write('M {},{}'.format(curve.start_point.x, curve.start_point.y))
                    for segment in curve:
                        if segment.is_corner:
                            f.write(f' L {segment.c.x},{segment.c.y} L {segment.end_point.x},{segment.end_point.y}')
                        else:
                            f.write(f' C {segment.c1.x},{segment.c1.y} {segment.c2.x},{segment.c2.y} {segment.end_point.x},{segment.end_point.y}')
                    f.write(' Z ')
                
                f.write('" fill-rule="evenodd" />')
                f.write("</svg>")
            
            # Store letter information
            image_results.append({
                "char": letter.get("char", ""),
                "bbox": bbox,
                "svg_path": svg_path
            })
            logging.info(f"Vectorized letter {idx} for {word_path}")
        
        # Get the original metadata from the dataset
        with open(config["dataset_path"], 'r') as f:
            dataset = json.load(f)
            original_filename = Path(image_path).name
            source_info = dataset[original_filename].get("source", {})
        
        results[image_path] = {
            "letters": image_results,
            "source": source_info
        }
        logging.info(f"Completed vectorization for {image_path}: {len(image_results)} letters")
    
    return results

def process_dataset(dataset_path: str, output_dir: str, config: Dict) -> Dict:
    """
    Process all images in the dataset efficiently.
    
    Args:
        dataset_path: Path to the dataset JSON file
        output_dir: Directory to save outputs
        config: Configuration dictionary
    
    Returns:
        Dictionary containing processed data for all images
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Add dataset path to config for later use
    config["dataset_path"] = dataset_path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "vectors").mkdir(exist_ok=True)
    
    # Get list of image paths with input directory prepended
    input_dir = Path("input/images")
    image_paths = [str(input_dir / filename) for filename in dataset.keys()]
    
    # Process in stages to manage memory efficiently
    logging.info("Stage 1: Word detection")
    all_words = detect_words_for_all_images(image_paths, config, output_dir)
    
    logging.info("Stage 2: Letter detection")
    all_letters = detect_letters_for_all_words(all_words, config, output_dir)
    del all_words  # Free memory
    
    logging.info("Stage 3: Letter segmentation")
    all_letters = segment_letters_for_all_images(all_letters, config, output_dir)
    
    logging.info("Stage 4: Vectorization")
    results = vectorize_all_letters(all_letters, output_dir, config)
    
    return results

def main():
    """Main entry point of the pipeline."""
    # Setup
    parser = setup_argparse()
    args = parser.parse_args()
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process dataset
    results = process_dataset(args.dataset, args.output, config)
    
    # Save results
    output_path = Path(args.output) / "features.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Pipeline completed. Results saved to {output_path}")

if __name__ == "__main__":
    main() 