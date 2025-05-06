import svgpathtools
import numpy as np
from pathlib import Path
from xml.dom import minidom


from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
from scipy.ndimage import distance_transform_edt

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

    all_contours = [sample_path_to_points(p) for p in paths if p]
    if not all_contours:
        return []

    return normalize_geometry_roorda(all_contours)

def normalize_geometry_roorda(contours):
    """
    Roorda 风格归一化：对齐原点，高度统一为 1，同时确保轮廓为逆时针方向。
    """
    def is_clockwise(polygon):
        x, y = np.array(polygon)[:, 0], np.array(polygon)[:, 1]
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0

    def ensure_counter_clockwise(contour):
        return contour[::-1] if is_clockwise(contour) else contour

    # 确保轮廓方向一致
    contours = [ensure_counter_clockwise(c) for c in contours]

    all_points = np.vstack(contours)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    height = max_y - min_y
    scale = 1.0 / height if height > 0 else 1.0

    normalized = []
    for pts in contours:
        pts_shifted = np.array(pts) - np.array([min_x, min_y])
        pts_scaled = pts_shifted * scale
        normalized.append(pts_scaled)
    return normalized


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
