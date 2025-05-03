"""
Vectorization implementation using Potrace.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        self.border_threshold = self.config.get("vectorization", {}).get("border_threshold", 0.3)
    
    def _touches_border(self, curve, width: int, height: int) -> bool:
        """
        Check if a curve touches the border of the image.
        
        Args:
            curve: Potrace curve object
            width: Image width
            height: Image height
            
        Returns:
            True if curve touches border, False otherwise
        """
        border_margin = 1.0  # Pixels from edge to consider as border
        
        # Check start point of the curve
        if (curve.start_point.x <= border_margin or 
            curve.start_point.x >= width - border_margin or
            curve.start_point.y <= border_margin or
            curve.start_point.y >= height - border_margin):
            return True
            
        # Check each segment's end point
        for segment in curve:
            if (segment.end_point.x <= border_margin or 
                segment.end_point.x >= width - border_margin or
                segment.end_point.y <= border_margin or
                segment.end_point.y >= height - border_margin):
                return True
                
            # For bezier curves, also check control points
            if not segment.is_corner:
                if (segment.c1.x <= border_margin or segment.c1.x >= width - border_margin or
                    segment.c1.y <= border_margin or segment.c1.y >= height - border_margin or
                    segment.c2.x <= border_margin or segment.c2.x >= width - border_margin or
                    segment.c2.y <= border_margin or segment.c2.y >= height - border_margin):
                    return True
            else:
                if (segment.c.x <= border_margin or segment.c.x >= width - border_margin or
                    segment.c.y <= border_margin or segment.c.y >= height - border_margin):
                    return True
                    
        return False
    
    def _get_contour_dimensions(self, curve) -> Tuple[float, float]:
        """
        Calculate width and height of a contour.
        
        Args:
            curve: Potrace curve object
            
        Returns:
            Tuple of (width, height) of the contour
        """
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        # Check start point of the curve
        min_x = min(min_x, curve.start_point.x)
        max_x = max(max_x, curve.start_point.x)
        min_y = min(min_y, curve.start_point.y)
        max_y = max(max_y, curve.start_point.y)
        
        # Check each segment
        for segment in curve:
            # Check end point
            min_x = min(min_x, segment.end_point.x)
            max_x = max(max_x, segment.end_point.x)
            min_y = min(min_y, segment.end_point.y)
            max_y = max(max_y, segment.end_point.y)
            
            # For bezier curves, also check control points
            if not segment.is_corner:
                min_x = min(min_x, segment.c1.x, segment.c2.x)
                max_x = max(max_x, segment.c1.x, segment.c2.x)
                min_y = min(min_y, segment.c1.y, segment.c2.y)
                max_y = max(max_y, segment.c1.y, segment.c2.y)
            else:
                min_x = min(min_x, segment.c.x)
                max_x = max(max_x, segment.c.x)
                min_y = min(min_y, segment.c.y)
                max_y = max(max_y, segment.c.y)
                
        return (max_x - min_x, max_y - min_y)
    
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
            
            # Image dimensions
            height, width = crop.shape
            
            # Filter the contours
            filtered_curves = []
            for curve in path:
                # Check if curve touches the border
                if self._touches_border(curve, width, height):
                    # Check dimensions relative to image size
                    contour_width, contour_height = self._get_contour_dimensions(curve)
                    
                    # Keep only if larger than threshold percentage of image dimensions
                    if (contour_width > self.border_threshold * width and 
                        contour_height > self.border_threshold * height):
                        filtered_curves.append(curve)
                else:
                    # Keep all contours that don't touch the border
                    filtered_curves.append(curve)
            
            # Generate SVG path
            svg_path = str(self.output_dir / output_filename)
            
            # Write SVG file
            with open(svg_path, "w") as f:
                f.write(f'<svg viewBox="0 0 {width} {height}">')
                f.write('<path d="')
                
                for curve in filtered_curves:
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