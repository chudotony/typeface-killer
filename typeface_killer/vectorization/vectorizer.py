"""
Vectorization implementation using Potrace.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import potrace

class Vectorizer:
    """
    Converts letter images to SVG paths using Potrace.
    """
    
    def __init__(self, config: Dict, output_dir: Optional[str] = None):
        """
        Initialize the vectorizer.
        
        Args:
            config: Configuration dictionary containing vectorization settings
            output_dir: Optional output directory to save SVGs
        """
        self.config = config
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif "output" in config and "path" in config["output"]:
            self.output_dir = Path(config["output"]["path"])
        else:
            raise ValueError("Output directory must be provided either as an argument or in the config under ['output']['path']")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Potrace parameters
        self.opttolerance = config["vectorization"]["opttolerance"]
    
    def vectorize_letter(self, letter: Dict) -> str:
        """
        Convert a letter image to an SVG path.
        
        Args:
            letter: Dictionary containing letter information including position and image data
            
        Returns:
            Path to the generated SVG file
        """
        # Extract position coordinates
        x, y = letter["position"]["x"], letter["position"]["y"]
        w, h = letter["position"]["width"], letter["position"]["height"]
        
        # Get the image data
        if "image" in letter:
            image = letter["image"]
        else:
            # If no image data, try to load from the original image path
            image_path = letter.get("image_path")
            if not image_path:
                raise ValueError("No image data or path provided for letter")
            image = Image.open(image_path)
        
        # Crop the letter region
        crop = image.crop((x, y, x+w, y+h))
        
        # Convert to numpy array and invert
        crop_array = np.asarray(crop)
        if len(crop_array.shape) == 3:  # If RGB, convert to grayscale
            crop_array = np.mean(crop_array, axis=2)
        crop_array = 255 - crop_array  # Invert the image
        
        # Create bitmap and trace
        bitmap = potrace.Bitmap(crop_array)
        path = bitmap.trace(opttolerance=self.opttolerance)
        
        # Generate output path
        output_path = self._generate_output_path()
        
        # Write SVG file
        with open(output_path, "w") as f:
            # Write SVG header
            f.write(f'<svg viewBox="0 0 {w} {h}">')
            
            # Create a combined path with proper winding
            f.write('<path d="')
            
            for curve in path:
                # Start a new subpath
                f.write('M {},{}'.format(curve.start_point.x, curve.start_point.y))
                
                for segment in curve:
                    if segment.is_corner:
                        f.write(f' L {segment.c.x},{segment.c.y} L {segment.end_point.x},{segment.end_point.y}')
                    else:
                        f.write(f' C {segment.c1.x},{segment.c1.y} {segment.c2.x},{segment.c2.y} {segment.end_point.x},{segment.end_point.y}')
                
                # Close the subpath
                f.write(' Z ')
            
            f.write('" fill-rule="evenodd" />')
            f.write("</svg>")
        
        return output_path
    
    def _generate_output_path(self) -> str:
        """Generate a unique output path for an SVG file."""
        counter = 0
        while True:
            path = self.output_dir / f"letter_{counter}.svg"
            if not path.exists():
                return str(path)
            counter += 1 