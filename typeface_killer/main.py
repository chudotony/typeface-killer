"""
Main pipeline for processing typographic images.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from typeface_killer.word_detection import WordDetector
from typeface_killer.letter_detection import LetterDetector
from typeface_killer.letter_segmentation import LetterSegmenter
from typeface_killer.vectorization import Vectorizer
from typeface_killer.utils.config import load_config
from typeface_killer.utils.image import filter_by_size

def process_dataset(
    dataset_path: Union[str, Path],
    config_path: Union[str, Path] = "config/default.yaml",
    input_dir: Union[str, Path] = "data/input",
    output_dir: Union[str, Path] = "data/output"
) -> None:
    """
    Process a dataset of typographic images through the pipeline.
    
    Args:
        dataset_path: Path to the JSON dataset file
        config_path: Path to the configuration file
        input_dir: Directory containing input images
        output_dir: Directory for storing output files
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    output_dir = Path(output_dir)
    words_dir = output_dir / "words"
    segmented_dir = output_dir / "segmented"
    vectors_dir = output_dir / "vectors"
    
    for dir_path in [words_dir, segmented_dir, vectors_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Step 1: Word Detection for all images
    print("Step 1: Detecting words in all images...")
    word_detector = WordDetector(config)
    word_results = {}
    
    for filename, data in dataset.items():
        print(f"  Processing {filename}...")
        image_path = Path(input_dir) / filename
        words = word_detector.detect_words(str(image_path))
        
        # Filter words by size
        words = filter_by_size(
            words,
            min_height=70,
            min_width=70
        )
        
        # Save word images and store results
        word_results[filename] = []
        for i, word in enumerate(words, 1):
            word_num = f"{i:04d}"
            word_filename = f"{Path(filename).stem}_{word_num}{Path(filename).suffix}"
            word_path = words_dir / word_filename
            
            # Save word image
            Image.fromarray(word["image"]).save(word_path)
            word_results[filename].append({
                "path": str(word_path),
                "image": word["image"],
                "text": word["text"],
                "confidence": word["confidence"],
                "position": word["position"]
            })
    
    # Clean up word detector
    del word_detector
    
    # Step 2: Letter Detection for all words
    print("\nStep 2: Detecting letters in all words...")
    letter_detector = LetterDetector(config)
    letter_results = {}
    
    for filename, words in word_results.items():
        print(f"  Processing {filename}...")
        letter_results[filename] = []
        
        for word in words:
            letters = letter_detector.detect_letters(word["image"])
            letter_results[filename].append(letters)
            
            # Update dataset with letter bboxes
            if "letters" not in dataset[filename]:
                dataset[filename]["letters"] = []
    
    # Clean up letter detector
    del letter_detector
    
    # Step 3: Letter Segmentation for all words
    print("\nStep 3: Segmenting all words...")
    segmenter = LetterSegmenter(config)
    segmentation_results = {}
    
    for filename, words in word_results.items():
        print(f"  Processing {filename}...")
        segmentation_results[filename] = []
        
        for i, word in enumerate(words, 1):
            word_num = f"{i:04d}"
            segmented_mask = segmenter.segment_letters(word["image"])
            segmentation_results[filename].append(segmented_mask)
            
            # Save segmented word
            segmented_filename = f"{Path(filename).stem}_{word_num}_segmented{Path(filename).suffix}"
            segmented_path = segmented_dir / segmented_filename
            Image.fromarray(segmented_mask.astype(np.uint8) * 255).save(segmented_path)
    
    # Clean up segmenter
    del segmenter
    
    # Step 4: Letter Vectorization
    print("\nStep 4: Vectorizing all letters...")
    vectorizer = Vectorizer(config)
    
    for filename, words in word_results.items():
        print(f"  Processing {filename}...")
        
        for i, (word, letters, segmented_mask) in enumerate(zip(
            words,
            letter_results[filename],
            segmentation_results[filename]
        ), 1):
            word_num = f"{i:04d}"
            
            for j, letter in enumerate(letters, 1):
                letter_num = f"{j:04d}"
                
                # Crop letter from segmented mask
                bbox = letter["bbox"]
                letter_mask = segmented_mask[bbox["y"]:bbox["y"]+bbox["h"], bbox["x"]:bbox["x"]+bbox["w"]]
                
                # Vectorize letter
                svg_filename = f"{Path(filename).stem}_{word_num}_{letter_num}.svg"
                svg_path = vectorizer.vectorize_letter(letter_mask, str(vectors_dir / svg_filename))
                
                # Update dataset with letter information
                dataset[filename]["letters"].append({
                    "char": letter["char"],
                    "bbox": bbox,
                    "svg_path": str(svg_path),
                    "features": {}  # To be filled in later
                })
    
    # Clean up vectorizer
    del vectorizer
    
    # Save updated dataset
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process typographic images through the pipeline.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSON file")
    parser.add_argument("--config", default="config/default.yaml", help="Path to the configuration file")
    parser.add_argument("--input", default="data/input", help="Directory containing input images")
    parser.add_argument("--output", default="data/output", help="Directory for storing output files")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_path=args.dataset,
        config_path=args.config,
        input_dir=args.input,
        output_dir=args.output
    )

if __name__ == "__main__":
    main() 