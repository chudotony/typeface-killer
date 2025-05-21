import svgpathtools
import cv2
import numpy as np
from pathlib import Path
from xml.dom import minidom

from skimage.draw import polygon as sk_polygon
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt


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

#def compute_medial_axis(contours, canvas_size=256):
#    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
#
#    all_points = np.vstack(contours)
#    min_x, min_y = np.min(all_points, axis=0)
#    max_x, max_y = np.max(all_points, axis=0)
#    scale = canvas_size / max(max_x - min_x, max_y - min_y)
#
#    for contour in contours:
#        scaled = (np.array(contour) - [min_x, min_y]) * scale
#        poly_mask = polygon2mask((canvas_size, canvas_size), scaled)
#        mask |= poly_mask
#
#    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=2).astype(bool)
#    skeleton = skeletonize(mask)
#    dist = distance_transform_edt(mask)
#    medial_points = [(x, y, dist[y, x]) for y, x in zip(*np.nonzero(skeleton))]
#
#    return medial_points, skeleton, (min_x, min_y), scale


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
    """
    使用 notebook 逻辑、且经过缩放修复的中轴线提取函数。
    - 输入必须是标准化后的 contours（归一到0~1）
    - 会将其缩放至 canvas_size 区域，避免 mask 过小失效
    返回:
        - medial_points: [(x, y, radius)] 以像素为单位
        - skeleton: skeleton mask
        - dist_transform: 距离变换图
    """
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)

    for i, contour in enumerate(contours):
        contour_np = np.array(contour)
        if len(contour_np) < 3:
            continue
        try:
            # 将归一化轮廓缩放至 canvas，居中并保留边距
            scaled = (contour_np * (canvas_size - 2 * pad)) + pad
            x, y = scaled[:, 0], scaled[:, 1]
            rr, cc = sk_polygon(y, x, shape=(canvas_size, canvas_size))
            mask[rr, cc] = 1
        except Exception as e:
            print(f"[Polygon error] Contour #{i} → {e}")
            continue

    if np.sum(mask) == 0:
        raise ValueError("⚠️ Empty mask generated — check contours or normalization")

    skeleton = medial_axis(mask)
    if not np.any(skeleton):
        raise ValueError("⚠️ Medial axis skeleton is empty")

    dist_transform = distance_transform_edt(mask)

    medial_points = []
    ys, xs = np.nonzero(skeleton)
    for x, y in zip(xs, ys):
        radius = dist_transform[y, x]
        medial_points.append((x, y, radius))

    return medial_points, skeleton, dist_transform
#
#def compute_medial_axis_consistent(contours, canvas_size=256, scale=1.5, pad=10):
#    """
#    使用 notebook 的逻辑实现的中轴线提取函数，基于归一化轮廓。
#    返回：
#        - medial_points: [(x, y, radius)]，未归一化（以 raster 为基准的）中轴点
#        - skeleton: 布尔数组，中轴线掩码
#    """
#    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
#
#    for contour in contours:
#        contour_np = np.array(contour)
#        if len(contour_np) == 0:
#            continue
#        # 放缩+加边距以避免裁切
#        scaled = (contour_np - contour_np.min(axis=0)) * scale + pad
#        x, y = scaled[:, 0], scaled[:, 1]
#        height = int(np.ceil(y.max())) + pad
#        width = int(np.ceil(x.max())) + pad
#        rr, cc = sk_polygon(y, x, shape=(height, width))
#        canvas = np.zeros((height, width), dtype=bool)
#        canvas[rr, cc] = 1
#        mask[:height, :width] |= canvas
#
#    skeleton = medial_axis(mask)
#    dist_transform = distance_transform_edt(mask)
#
#    medial_points = []
#    ys, xs = np.nonzero(skeleton)
#    for x, y in zip(xs, ys):
#        radius = dist_transform[y, x]
#        medial_points.append((x, y, radius))
#
#    return medial_points, skeleton, dist_transform

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

def skeleton_to_graph(skeleton):
    """
    Convert a skeleton binary image into a NetworkX graph.
    Nodes are (x, y) pixel positions.
    """
    from skimage.morphology import skeletonize
    import networkx as nx

    G = nx.Graph()
    height, width = skeleton.shape

    for y in range(height):
        for x in range(width):
            if not skeleton[y, x]:
                continue
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < width and 0 <= ny_ < height:
                        if skeleton[ny_, nx_]:
                            G.add_edge((x, y), (nx_, ny_))
    return G
