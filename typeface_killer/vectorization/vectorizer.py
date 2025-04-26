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
    
    def __init__(self, config: Dict):
        """
        Initialize the vectorizer.
        
        Args:
            config: Configuration dictionary containing vectorization settings
        """
        self.config = config
        self.output_dir = Path(config["output"]["path"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Potrace parameters
        self.opttolerance = config["vectorization"]["opttolerance"]
    
    def vectorize_letter(self, image: np.ndarray, output_path: Optional[str] = None) -> str:
        """
        Convert a letter image to an SVG path.
        
        Args:
            image: Letter image as numpy array
            output_path: Optional path to save the SVG file. If None, a unique name will be generated.
            
        Returns:
            Path to the generated SVG file
        """
        # Convert image to grayscale and invert
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        image = 255 - image
        
        # Create bitmap
        bitmap = potrace.Bitmap(image)
        
        # Trace the bitmap
        path = bitmap.trace(opttolerance=self.opttolerance)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path()
        
        # Write SVG file
        with open(output_path, "w") as f:
            # Write SVG header
            f.write(f'<svg viewBox="0 0 {image.shape[1]} {image.shape[0]}">')
            
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