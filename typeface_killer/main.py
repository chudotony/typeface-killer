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

from typeface_killer.word_detection import WordDetector
from typeface_killer.letter_detection import LetterDetector
from typeface_killer.letter_segmentation import LetterSegmenter
from typeface_killer.vectorization import Vectorizer
from typeface_killer.utils.config import load_config
from typeface_killer.utils.logging import setup_logging
from typeface_killer.utils.image import filter_by_size

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
            image_letters.extend(letters)
        all_letters[image_path] = image_letters
    
    # Clean up Tesseract
    del letter_detector
    return all_letters

def segment_letters_for_all_images(all_letters: Dict[str, List[Dict]], config: Dict) -> Dict[str, List[Dict]]:
    """Segment all letters using Hi-SAM."""
    letter_segmenter = LetterSegmenter(config)
    all_segmented = {}
    
    for image_path, letters in all_letters.items():
        segmented_letters = []
        for letter in letters:
            segmented = letter_segmenter.segment_letter(letter)
            segmented_letters.append(segmented)
        all_segmented[image_path] = segmented_letters
    
    # Clean up Hi-SAM
    del letter_segmenter
    return all_segmented

def vectorize_all_letters(all_segmented: Dict[str, List[Dict]], output_dir: str, config: Dict) -> Dict[str, Dict]:
    """Vectorize all segmented letters."""
    vectorizer = Vectorizer(config)
    results = {}
    
    for image_path, segmented_letters in all_segmented.items():
        image_letters = []
        for idx, segmented in enumerate(segmented_letters):
            # Vectorize the letter
            vector = vectorizer.vectorize(segmented)
            
            # Generate SVG path
            svg_filename = f"{Path(image_path).stem}_{idx:04d}.svg"
            svg_path = str(Path(output_dir) / "vectors" / svg_filename)
            
            # Store letter information
            image_letters.append({
                "char": segmented.get("char", ""),
                "bbox": {
                    "x": segmented["position"]["x"],
                    "y": segmented["position"]["y"],
                    "w": segmented["position"]["width"],
                    "h": segmented["position"]["height"]
                },
                "svg_path": svg_path
            })
        
        # Get the original metadata from the dataset
        with open(config["dataset_path"], 'r') as f:
            dataset = json.load(f)
            source_info = dataset[image_path].get("source", {})
        
        results[image_path] = {
            "letters": image_letters,
            "source": source_info
        }
    
    # Clean up vectorizer
    del vectorizer
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
    all_segmented = segment_letters_for_all_images(all_letters, config)
    del all_letters  # Free memory
    
    logging.info("Stage 4: Vectorization")
    results = vectorize_all_letters(all_segmented, output_dir, config)
    
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