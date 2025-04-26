#!/usr/bin/env python3
"""
Main script for the Typeface Killer pipeline.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

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
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    return parser

def process_image(image_path: str, output_dir: str, config: Dict) -> Dict:
    """
    Process a single image through the pipeline.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        config: Configuration dictionary
    
    Returns:
        Dictionary containing extracted features
    """
    # Initialize components
    word_detector = WordDetector(config)
    letter_detector = LetterDetector(config)
    letter_segmenter = LetterSegmenter(config)
    vectorizer = Vectorizer(config)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process pipeline
    logging.info(f"Processing image: {image_path}")
    
    # 1. Word detection
    words = word_detector.detect_words(image_path)
    logging.info(f"Detected {len(words)} words")
    
    # Filter words by size
    word_images = [word["image"] for word in words]
    filtered_images, kept_indices = filter_by_size(
        word_images,
        min_height=config["word_detection"].get("min_height", 20),
        max_height=config["word_detection"].get("max_height", 200),
        min_width=config["word_detection"].get("min_width", 20),
        max_width=config["word_detection"].get("max_width", 1000),
        return_indices=True
    )
    
    # Update words list with filtered results
    words = [words[idx] for idx in kept_indices]
    logging.info(f"After size filtering: {len(words)} words")
    
    features = {}
    for word_idx, word in enumerate(words):
        # 2. Letter detection
        letters = letter_detector.detect_letters(word)
        logging.info(f"Word {word_idx}: Detected {len(letters)} letters")
        
        word_features = []
        for letter_idx, letter in enumerate(letters):
            # 3. Letter segmentation
            segmented_letter = letter_segmenter.segment_letter(letter)
            
            # 4. Vectorization
            vector = vectorizer.vectorize(segmented_letter)
            
            # Store features
            word_features.append({
                "letter": letter_idx,
                "vector": vector,
                "position": letter["position"]
            })
        
        features[f"word_{word_idx}"] = word_features
    
    return features

def main():
    """Main entry point of the pipeline."""
    # Setup
    parser = setup_argparse()
    args = parser.parse_args()
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process image
    features = process_image(args.input, args.output, config)
    
    # Save results
    output_path = Path(args.output) / "features.json"
    import json
    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)
    
    logging.info(f"Pipeline completed. Results saved to {output_path}")

if __name__ == "__main__":
    main() 