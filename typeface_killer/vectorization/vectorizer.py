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
        self.border_threshold = self.config.get("vectorization", {}).get("border_threshold", 0.5)
    
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
            
            # Debug: log initial number of curves
            logging.debug(f"Initial curves for {output_filename}: {len(path.curves)}")
            
            # Filter the contours
            filtered_curves = []
            for curve in path.curves:
                if self._touches_border(curve, width, height):
                    contour_width, contour_height = self._get_contour_dimensions(curve)
                    if (contour_width > self.border_threshold * width and 
                        contour_height > self.border_threshold * height):
                        filtered_curves.append(curve)
                else:
                    filtered_curves.append(curve)
            
            # Debug: log number of filtered curves
            logging.debug(f"Filtered curves for {output_filename}: {len(filtered_curves)}")
            
            if not filtered_curves:
                logging.warning(f"No valid strokes found for {output_filename}")
                return None
            
            # Build path data first to validate it
            path_data = ""
            stroke_count = 0
            
            for curve in filtered_curves:
                if not hasattr(curve, 'start_point') or not hasattr(curve, 'segments'):
                    logging.warning(f"Invalid curve object in {output_filename}")
                    continue
                    
                curve_data = ""
                # Validate start point coordinates
                if any(not isinstance(coord, (int, float)) for coord in [curve.start_point.x, curve.start_point.y]):
                    logging.warning(f"Invalid start point coordinates in {output_filename}")
                    continue
                    
                curve_data += 'M {:.1f},{:.1f}'.format(curve.start_point.x, curve.start_point.y)
                
                has_segments = False
                has_valid_points = True
                
                for segment in curve:
                    # Validate segment coordinates
                    if segment.is_corner:
                        if not all(hasattr(segment, attr) for attr in ['c', 'end_point']):
                            has_valid_points = False
                            break
                        if any(not isinstance(getattr(point, coord), (int, float)) 
                              for point in [segment.c, segment.end_point] 
                              for coord in ['x', 'y']):
                            has_valid_points = False
                            break
                        curve_data += f' L {segment.c.x:.1f},{segment.c.y:.1f}'
                        curve_data += f' L {segment.end_point.x:.1f},{segment.end_point.y:.1f}'
                    else:
                        if not all(hasattr(segment, attr) for attr in ['c1', 'c2', 'end_point']):
                            has_valid_points = False
                            break
                        if any(not isinstance(getattr(point, coord), (int, float)) 
                              for point in [segment.c1, segment.c2, segment.end_point] 
                              for coord in ['x', 'y']):
                            has_valid_points = False
                            break
                        curve_data += f' C {segment.c1.x:.1f},{segment.c1.y:.1f}'
                        curve_data += f' {segment.c2.x:.1f},{segment.c2.y:.1f}'
                        curve_data += f' {segment.end_point.x:.1f},{segment.end_point.y:.1f}'
                    has_segments = True
                
                # Only add curve if it has valid segments and points
                if has_segments and has_valid_points:
                    curve_data += ' Z'
                    path_data += curve_data
                    stroke_count += 1
                    logging.debug(f"Added valid stroke {stroke_count} to {output_filename}")
            
            # Final validation before writing file
            if stroke_count == 0 or not path_data.strip():
                logging.warning(f"No valid path data for {output_filename}")
                return None
            
            # Generate SVG path and write file
            svg_path = str(self.output_dir / output_filename)
            with open(svg_path, "w") as f:
                f.write(f'<svg viewBox="0 0 {width} {height}">')
                f.write(f'<path d="{path_data}" fill-rule="evenodd" />')
                f.write("</svg>")
            
            logging.info(f"Successfully created SVG with {stroke_count} strokes: {output_filename}")
            return svg_path
            
        except Exception as e:
            logging.error(f"Error vectorizing letter: {str(e)}")
            return None