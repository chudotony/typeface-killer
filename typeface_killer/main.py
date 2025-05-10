#!/usr/bin/env python3
"""
Main script for the Typeface Killer pipeline.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

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
from typeface_killer.utils.image import filter_by_size, resize_if_large

# Import save_binary_mask from Hi-SAM
import sys
hi_sam_path = os.path.join(os.path.dirname(__file__), "letter_segmentation", "Hi-SAM")
if hi_sam_path not in sys.path:
    sys.path.insert(0, hi_sam_path)
from demo_hisam import save_binary_mask

from PIL.Image import DecompressionBombError
from enum import Enum

class PipelineModule(Enum):
    WORD_DETECTION = "word_detection"
    LETTER_DETECTION = "letter_detection"
    LETTER_SEGMENTATION = "letter_segmentation"
    VECTORIZATION = "vectorization"
    ALL = "all"

class ModularPipeline:
    def __init__(self, dataset_path: str, output_dir: str, config: Dict):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dict:
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
            
    def _load_intermediate_results(self, module: PipelineModule) -> Optional[Dict]:
        """Load results from previous module execution."""
        result_path = self.output_dir / f"{module.value}_results.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                return json.load(f)
        return None
        
    def _save_intermediate_results(self, results: Dict, module: PipelineModule):
        """Save module results for future use."""
        result_path = self.output_dir / f"{module.value}_results.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _save_dataset_state(self, module: PipelineModule):
        """Save current state of dataset after module completion."""
        dataset_path = self.output_dir / f"dataset_{module.value}.json"
        with open(dataset_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        logging.info(f"Saved dataset state after {module.value} to {dataset_path}")
            
    def validate_module_dependencies(self, module: PipelineModule) -> bool:
        """Check if required previous module results exist."""
        dependencies = {
            PipelineModule.LETTER_DETECTION: [PipelineModule.WORD_DETECTION],
            PipelineModule.LETTER_SEGMENTATION: [PipelineModule.LETTER_DETECTION],
            PipelineModule.VECTORIZATION: [PipelineModule.LETTER_SEGMENTATION]
        }
        
        if module in dependencies:
            for dep in dependencies[module]:
                if not (self.output_dir / f"{dep.value}_results.json").exists():
                    logging.error(f"Missing required results from {dep.value}")
                    return False
        return True
        
    def run_module(self, module: PipelineModule) -> Dict:
        """Run a specific pipeline module."""
        if module != PipelineModule.WORD_DETECTION and not self.validate_module_dependencies(module):
            raise ValueError(f"Cannot run {module.value}: missing dependencies")
            
        input_dir = Path("input/images")
        image_paths = [str(input_dir / filename) for filename in self.dataset.keys()]
        
        results = None
        if module == PipelineModule.WORD_DETECTION:
            results = detect_words_for_all_images(image_paths, self.config, str(self.output_dir))
            # Update dataset with word information
            for image_path, words in results.items():
                image_name = next((k for k in self.dataset.keys() if Path(image_path).name == k), None)
                if image_name:
                    self.dataset[image_name]["words"] = words
            
        elif module == PipelineModule.LETTER_DETECTION:
            words = self._load_intermediate_results(PipelineModule.WORD_DETECTION)
            results = detect_letters_for_all_words(words, self.config, str(self.output_dir))
            # Update dataset with letter detection results
            for image_path, letters in results.items():
                image_name = next((k for k in self.dataset.keys() if Path(image_path).name == k), None)
                if image_name:
                    self.dataset[image_name]["detected_letters"] = letters
            
        elif module == PipelineModule.LETTER_SEGMENTATION:
            letters = self._load_intermediate_results(PipelineModule.LETTER_DETECTION)
            results = segment_letters_for_all_images(letters, self.config, str(self.output_dir))
            # Update dataset with segmentation results
            for image_path, letters in results.items():
                image_name = next((k for k in self.dataset.keys() if Path(image_path).name == k), None)
                if image_name:
                    self.dataset[image_name]["segmented_letters"] = letters
            
        elif module == PipelineModule.VECTORIZATION:
            letters = self._load_intermediate_results(PipelineModule.LETTER_SEGMENTATION)
            results = vectorize_all_letters(letters, self.output_dir, self.config)
            # Update dataset with vectorization results (already handled in _update_dataset_with_vectors)
            self._update_dataset_with_vectors(results)
            
        else:
            raise ValueError(f"Unknown module: {module}")
            
        # Save both intermediate results and dataset state
        self._save_intermediate_results(results, module)
        self._save_dataset_state(module)
        
        return results
        
    def run(self, module: PipelineModule = PipelineModule.ALL) -> Dict:
        """Run the pipeline in modular fashion."""
        if module == PipelineModule.ALL:
            modules = [m for m in PipelineModule if m != PipelineModule.ALL]
        else:
            modules = [module]
            
        results = None
        for mod in modules:
            logging.info(f"Running module: {mod.value}")
            results = self.run_module(mod)
            
        # Update dataset with final results if vectorization was run
        if modules[-1] == PipelineModule.VECTORIZATION:
            self._update_dataset_with_vectors(results)
            
        return self.dataset
        
    def _update_dataset_with_vectors(self, vectorized_letters: Dict):
        """Update dataset with vectorization results."""
        for image_path, letters in vectorized_letters.items():
            base_name = Path(image_path).stem
            matching_key = next((key for key in self.dataset.keys() if key.startswith(base_name)), None)
            if matching_key:
                self.dataset[matching_key]["letters"] = letters
                logging.info(f"Added {len(letters)} letters to {matching_key}")
            else:
                logging.warning(f"No matching image found in dataset for {base_name}")

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Typeface Killer - Typographic Feature Extraction Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument(
        "--module",
        type=str,
        choices=[m.value for m in PipelineModule],
        default="all",
        help="Pipeline module to run"
    )
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
        logging.info(f"Processing image {image_path} with {len(words)} words")
        
        for word_idx, word in enumerate(words):
            # Read the word image from file
            word_filename = f"{Path(image_path).stem}_{word_idx:04d}.png"
            word_path = str(words_dir / word_filename)
            word_image = cv2.imread(word_path)
            
            if word_image is None:
                logging.error(f"Could not read word image: {word_path}")
                continue
                
            # Validate word image
            if word_image.size == 0:
                logging.error(f"Empty word image: {word_path}")
                continue
                
            # Detect letters in the word image
            letters = letter_detector.detect_letters(word_image)
            if not letters:
                logging.warning(f"No letters detected in word {word_path}")
                continue
                
            # Add word index and validate each letter
            valid_letters = []
            for letter in letters:
                # Validate bbox
                bbox = letter.get("bbox", {})
                if not all(k in bbox for k in ["x", "y", "w", "h"]):
                    logging.warning(f"Invalid bbox in letter from {word_path}")
                    continue
                    
                # Validate character
                if not letter.get("char"):
                    logging.warning(f"Empty character in letter from {word_path}")
                    continue
                    
                letter["word_idx"] = word_idx
                letter["word_path"] = word_path
                valid_letters.append(letter)
                logging.debug(f"Detected letter '{letter['char']}' at {bbox} in {word_path}")
            
            image_letters.extend(valid_letters)
            logging.info(f"Detected {len(valid_letters)} valid letters in word {word_path}")
        
        all_letters[image_path] = image_letters
        logging.info(f"Total valid letters for image {image_path}: {len(image_letters)}")
    
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
            
            # Validate word image
            if word_image.size == 0:
                logging.error(f"Empty word image: {word_path}")
                continue
                
            segmented_mask = letter_segmenter.segment_letters(word_image)
            if segmented_mask is None:
                logging.error(f"Failed to segment word: {word_path}")
                continue
                
            # Validate segmented mask
            if segmented_mask.size == 0:
                logging.error(f"Empty segmented mask for word: {word_path}")
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

def vectorize_all_letters(all_letters: Dict[str, List[Dict]], output_dir: Path, config: Dict) -> Dict[str, List[Dict]]:
    """
    Vectorize all segmented letters.
    
    Args:
        all_letters: Dictionary mapping image paths to lists of letter dictionaries
        output_dir: Directory to save vectorized letters
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping image paths to lists of vectorized letter dictionaries
    """
    logging.info("Starting letter vectorization...")
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    
    vectorizer = Vectorizer(output_dir=vectors_dir, config=config)
    vectorized_letters = {}
    
    # Group letters by their word path and original image path
    word_letters = {}
    original_paths = {}  # Track original image path for each word
    for image_path, letters in all_letters.items():
        for letter in letters:
            if "word_path" in letter:
                word_filename = Path(letter["word_path"]).name
                if word_filename not in word_letters:
                    word_letters[word_filename] = []
                    # Store the original image path for this word
                    original_paths[word_filename] = image_path
                word_letters[word_filename].append(letter)

    # Process each word's letters
    for word_filename, letters in word_letters.items():
        if not letters:
            logging.warning(f"No letters found for word {word_filename}")
            continue

        # Get the original image path for this word
        original_image_path = original_paths[word_filename]
        original_name = Path(original_image_path).stem
            
        logging.info(f"Processing {len(letters)} letters from word {word_filename}")
        image_letters = []
        
        # Get the segmented word path
        word_path = output_dir / "words_segmented" / word_filename
        if not word_path.exists():
            logging.warning(f"Segmented word image not found: {word_path}")
            continue
            
        try:
            word_image = Image.open(word_path).convert('L')
        except Exception as e:
            logging.error(f"Failed to load segmented word image {word_path}: {str(e)}")
            continue

        # Extract word index from filename (assuming format name_XXXX.ext)
        word_idx = int(word_filename.split('_')[-1].split('.')[0])
        
        for letter_idx, letter in enumerate(letters):
            try:
                # Get bbox and validate
                bbox = letter["bbox"]
                if not isinstance(bbox, dict):
                    logging.warning(f"Invalid bbox format: {bbox}")
                    continue
                    
                # Convert bbox to tuple format (x, y, w, h)
                x, y = bbox["x"], bbox["y"]
                w, h = bbox["w"], bbox["h"]
                
                # Validate bbox values
                if not all(isinstance(v, (int, float)) for v in [x, y, w, h]):
                    logging.warning(f"Invalid bbox values: {bbox}")
                    continue
                    
                # Crop letter region
                try:
                    letter_image = word_image.crop((x, y, x + w, y + h))
                except Exception as e:
                    logging.warning(f"Failed to crop letter: {str(e)}")
                    continue
                    
                if letter_image.size[0] == 0 or letter_image.size[1] == 0:
                    logging.warning(f"Empty crop for letter: {letter['char']}")
                    continue
                    
                # Generate output filename using original image name, word index, and letter index
                output_filename = f"{original_name}_{word_idx:04d}_{letter_idx:04d}.svg"
                
                # Vectorize letter
                svg_path = vectorizer.vectorize_letter(letter_image, output_filename)
                if svg_path is None:
                    logging.warning(f"Skipping letter '{letter['char']}' due to empty or invalid vectorization")
                    continue
                    
                # Add vectorized letter to results only if vectorization was successful
                vectorized_letter = {
                    "char": letter["char"],
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "svg_path": output_filename  # Store relative path
                }
                image_letters.append(vectorized_letter)
                logging.info(f"Vectorized letter: {letter['char']}")
                
            except Exception as e:
                logging.error(f"Error processing letter: {str(e)}")
                continue
                
        # Only add letters to results if there are valid vectorized letters
        if image_letters:
            if original_image_path not in vectorized_letters:
                vectorized_letters[original_image_path] = []
            vectorized_letters[original_image_path].extend(image_letters)
            logging.info(f"Vectorized {len(image_letters)} letters for word {word_filename}")
            
    return vectorized_letters

def process_dataset(dataset_path: str, output_dir: str, config: Dict) -> Dict:
    """
    Process all images in the dataset efficiently.
    
    Args:
        dataset_path: Path to the dataset JSON file
        output_dir: Directory to save outputs
        config: Configuration dictionary
    
    Returns:
        Dictionary containing the original dataset enriched with letter information
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
    vectorized_letters = vectorize_all_letters(all_letters, output_path, config)
    
    # Enrich the original dataset with letter information
    logging.info(f"Dataset keys: {list(dataset.keys())}")
    for image_path, letters in vectorized_letters.items():
        # Get the base image name without the word index
        base_name = Path(image_path).stem[:-5]  # Remove the _XXXX suffix
        
        # Find the matching key in dataset to get the correct suffix
        matching_key = next((key for key in dataset.keys() if key.startswith(base_name)), None)
        if matching_key:
            image_name = matching_key
            logging.info(f"Processing image: {image_name}")
            
            # Filter out any letters that don't have an SVG file
            valid_letters = []
            for letter in letters:
                svg_path = Path(output_path) / "vectors" / letter["svg_path"]
                if svg_path.exists():
                    valid_letters.append(letter)
                else:
                    logging.warning(f"Skipping letter '{letter['char']}' - SVG file not found: {svg_path}")
            
            dataset[image_name]["letters"] = valid_letters
            logging.info(f"Added {len(valid_letters)} valid letters to {image_name}")
        else:
            logging.warning(f"No matching image found in dataset for {base_name}")
    
    return dataset

def main():
    """Main entry point of the pipeline."""
    parser = setup_argparse()
    args = parser.parse_args()
    setup_logging()
    
    config = load_config(args.config)
    
    pipeline = ModularPipeline(args.dataset, args.output, config)
    results = pipeline.run(PipelineModule(args.module))
    
    # Save final results
    output_path = Path(args.output) / "dataset.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Pipeline completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()