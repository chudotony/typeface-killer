"""
Vectorization implementation using Potrace.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import potrace
import logging

class Vectorizer:
    """
    Converts letter images to SVG paths using Potrace.
    """
    
    def __init__(self, output_dir: str, config: Optional[Dict] = None):
        """
        Initialize the vectorizer.
        
        Args:
            output_dir: Directory to save SVGs
            config: Optional configuration dictionary containing vectorization settings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default config values
        self.config = config or {}
        self.opttolerance = self.config.get("vectorization", {}).get("opttolerance", 0.05)
    
    def vectorize_letter(self, letter_image: Image.Image, output_filename: str) -> Optional[str]:
        """
        Convert a letter image to an SVG path.
        
        Args:
            letter_image: PIL Image object containing the letter
            output_filename: Name of the output SVG file
            
        Returns:
            Path to the generated SVG file, or None if vectorization failed
        """
        try:
            # Convert to numpy array and invert
            crop = np.asarray(letter_image)
            crop = 255 - crop  # Invert the image
            
            # Create bitmap and trace
            bitmap = potrace.Bitmap(crop)
            path = bitmap.trace(opttolerance=self.opttolerance)
            
            # Generate SVG path
            svg_path = str(self.output_dir / output_filename)
            
            # Write SVG file
            with open(svg_path, "w") as f:
                f.write(f'<svg viewBox="0 0 {crop.shape[1]} {crop.shape[0]}">')
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
            
            return svg_path
            
        except Exception as e:
            logging.error(f"Error vectorizing letter: {str(e)}")
            return None 