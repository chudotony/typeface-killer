import svgpathtools
import cv2
import numpy as np
from pathlib import Path
from xml.dom import minidom
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

from skimage.draw import polygon as sk_polygon
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt


from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt
from functools import lru_cache

def sample_path_parallel(path, num_points=300):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Sample segments in parallel
        future_points = [
            executor.submit(lambda s: [s.point(t) for t in np.linspace(0, 1, num_points)], seg)
            for seg in path
        ]
        points = []
        for future in future_points:
            points.extend(future.result())
    return [(p.real, p.imag) for p in points]

def load_normalized_svg(svg_path):
    """
    读取 SVG 文件并解析所有 path 元素，采样路径点后返回 Roorda 风格归一化后的二维点集合。
    """
    paths, attributes = svgpathtools.svg2paths(str(svg_path))

    def sample_path_to_points(path, num_points=300):
        points = []
        for seg in path:
            pts = [seg.point(t) for t in np.linspace(0, 1, num_points)]
            points.extend(pts)
        return [(p.real, p.imag) for p in points]

    # Use parallel sampling for paths with many segments
    all_contours = []
    for p in paths:
        if p:
            if len(p) > 10:  # Only parallelize for complex paths
                points = sample_path_parallel(p)
            else:
                points = sample_path_to_points(p)
            all_contours.append(points)
    
    if not all_contours:
        return []

    return normalize_geometry_roorda(all_contours)

# Add path caching
@lru_cache(maxsize=1000)
def get_cached_paths(svg_path):
    return svgpathtools.svg2paths(str(svg_path))[0]

def normalize_geometry_roorda(contours):
    """Optimized Roorda normalization using vectorized operations"""
    if not contours:
        return []
        
    # Convert to numpy array once
    contours_array = [np.array(c) for c in contours]
    all_points = np.vstack(contours_array)
    
    # Vectorized min/max
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    height = maxs[1] - mins[1]
    scale = 1.0 / height if height > 0 else 1.0
    
    # Vectorized transformation
    return [(c - mins) * scale for c in contours_array]

def compute_medial_axis(contours, canvas_size=256):
    """
    将归一化后的轮廓渲染为二值图像，然后提取中轴线（skeleton）并结合距离变换生成中轴点集。
    返回: (medial_points, skeleton_mask)
    """
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)

    for contour in contours:
        contour_np = np.array(contour)
        scaled = (contour_np * (canvas_size - 1)).astype(int)
        poly_mask = polygon2mask((canvas_size, canvas_size), scaled)
        mask |= poly_mask

    skeleton = skeletonize(mask)
    dist_transform = distance_transform_edt(mask)

    medial_points = []
    ys, xs = np.nonzero(skeleton)
    for x, y in zip(xs, ys):
        radius = dist_transform[y, x]
        medial_points.append((x / canvas_size, y / canvas_size, radius / canvas_size))

    return medial_points, skeleton
    
    
# debug版本
def compute_medial_axis_consistent(contours, canvas_size=256, pad=10):
    """Optimized medial axis computation"""
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    
    # Pre-allocate arrays
    scaled_contours = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Vectorized scaling
        scaled = np.array(contour) * (canvas_size - 2 * pad) + pad
        scaled_contours.append(scaled)
    
    # Batch process polygons
    for scaled in scaled_contours:
        try:
            rr, cc = sk_polygon(scaled[:, 1], scaled[:, 0], shape=(canvas_size, canvas_size))
            mask[rr, cc] = 1
        except Exception as e:
            continue

    if not np.any(mask):
        raise ValueError("Empty mask generated")

    # Compute skeleton and distance transform once
    skeleton = medial_axis(mask)
    if not np.any(skeleton):
        raise ValueError("Empty skeleton")

    dist_transform = distance_transform_edt(mask)
    
    # Vectorized medial point extraction
    ys, xs = np.nonzero(skeleton)
    radii = dist_transform[ys, xs]
    medial_points = list(zip(xs, ys, radii))

    return medial_points, skeleton, dist_transform

def compute_medial_axis_serif(paths, image_size=256):
    all_pts = []
    for path in paths:
        for seg in path:
            pts = [seg.point(t) for t in np.linspace(0, 1, 50)]
            all_pts.extend(pts)

    all_pts = np.array([[pt.real, pt.imag] for pt in all_pts])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    scale = image_size / max(x_max - x_min, y_max - y_min)

    mask = np.zeros((image_size, image_size), dtype=bool)
    for path in paths:
        contour = []
        for seg in path:
            pts = [seg.point(t) for t in np.linspace(0, 1, 50)]
            for pt in pts:
                x, y = (np.array([pt.real, pt.imag]) - [x_min, y_min]) * scale
                contour.append((y, x))  # 注意 y,x 顺序
        poly_mask = polygon2mask((image_size, image_size), np.array(contour))
        mask |= poly_mask

    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=2).astype(bool)
    skeleton = skeletonize(mask)
    return [], skeleton, (x_min, y_min), scale

# Cache graph construction
@lru_cache(maxsize=100)
def skeleton_to_graph_cached(skeleton_bytes):
    """Internal cached version that works with hashable bytes"""
    skeleton = np.frombuffer(skeleton_bytes, dtype=bool).reshape(256, 256)  # Use known shape
    G = nx.Graph()
    ys, xs = np.nonzero(skeleton)
    
    # Add nodes in bulk
    G.add_nodes_from(zip(xs, ys))
    
    # Vectorized edge detection
    for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        shifted_xs = xs + dx
        shifted_ys = ys + dy
        mask = (shifted_xs >= 0) & (shifted_xs < skeleton.shape[1]) & \
               (shifted_ys >= 0) & (shifted_ys < skeleton.shape[0])
        valid_points = skeleton[shifted_ys[mask], shifted_xs[mask]]
        edges = zip(zip(xs[mask], ys[mask]), 
                   zip(shifted_xs[mask], shifted_ys[mask]))
        G.add_edges_from((e[0], e[1]) for e, v in zip(edges, valid_points) if v)
    
    return G

def skeleton_to_graph(skeleton):
    """Main function that handles conversion to hashable format"""
    skeleton_bytes = skeleton.tobytes()  # Convert ndarray to bytes for hashing
    return skeleton_to_graph_cached(skeleton_bytes)
