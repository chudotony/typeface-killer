# ---- 动态加载 svg_utils.py 模块 ----
import importlib.util
import sys
from pathlib import Path

svg_utils_path = Path(__file__).parent / "svg_utils.py"
spec = importlib.util.spec_from_file_location("svg_utils", svg_utils_path)
svg_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svg_utils)

# 提取需要用的函数
load_normalized_svg = svg_utils.load_normalized_svg
compute_medial_axis = svg_utils.compute_medial_axis


from sklearn.decomposition import PCA
import networkx as nx
import numpy as np

def extract_features_from_svg(svg_path):
    """
    Extract three typographic features from a single SVG file:
    - proportion
    - weight
    - slant

    Pipeline:
    1. Normalize the outline.
    2. Compute the medial axis transform (MAT).
    3. Extract features based on MAT and outline.
    """
    outline = load_normalized_svg(svg_path)
    if not outline:
        raise ValueError("Failed to load SVG or no paths found")

    medial_axis, skeleton_mask = compute_medial_axis(outline)
    if not medial_axis:
        raise ValueError("Medial axis computation failed")

    return {
        "proportion": compute_proportion(outline, medial_axis),
        "weight": compute_weight(outline, medial_axis),
        "slant": compute_slant(outline, medial_axis, skeleton_mask),
        # "serif": compute_serif(outline, medial_axis, skeleton_mask),
    }


# -------------------
# Feature definitions
# -------------------

def compute_proportion(outline, medial_axis):
    # 示例：宽高比，使用 bbox
    x_coords = [pt[0] for contour in outline for pt in contour]
    y_coords = [pt[1] for contour in outline for pt in contour]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return round(height / width, 3) if width != 0 else 0.0

def compute_weight(outline, medial_axis):
    # 示例：MAT 半径最大值代表粗细
    if not medial_axis:
        return 0.0
    return round(max([r for (_, _, r) in medial_axis]), 3)

def compute_slant(outline, medial_axis, skeleton_mask):
    """
    基于 skeleton graph 的 longest_path 主干拟合，提取字体倾斜角。
    """
    if skeleton_mask is None:
        return 0.0

    G = build_skeleton_graph_from_mask(skeleton_mask)
    longest_path = find_longest_vertical_path(G)
    if not longest_path or len(longest_path) < 5:
        return 0.0

    coords = np.array(longest_path)
    if len(coords) < 10:
        return 0.0

    # 1. PCA 拟合方向
    pca = PCA(n_components=2)
    pca.fit(coords)
    direction = pca.components_[0]
    center = coords.mean(axis=0)

    # 2. 投影主轴排序
    vectors = coords - center
    projections = vectors @ direction
    sorted_idx = np.argsort(projections)
    coords_sorted = coords[sorted_idx]

    # 3. 滤除前后10%
    n = len(coords_sorted)
    spine_coords = coords_sorted[int(n * 0.1): int(n * 0.9)]
    if len(spine_coords) < 5:
        return 0.0

    # 4. 从主干中取中段两点（按 y 排序）
    sorted_y = spine_coords[np.argsort(spine_coords[:, 1])]
    i1 = int(len(sorted_y) * 0.2)
    i2 = int(len(sorted_y) * 0.8)
    if i2 <= i1:
        return 0.0
    p1, p2 = sorted_y[i1], sorted_y[i2]
    delta = p2 - p1

    slant_angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return round(float(slant_angle), 2)



def find_longest_vertical_path(G):
    """
    在中轴骨架图 G 中寻找最垂直的主干路径。
    使用自定义的 verticality_score 来优先选择垂直跨度大的路径。
    """
    import numpy as np

    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    def verticality_score(path):
        coords = np.array(path)
        dy = coords[:, 1].ptp()
        dx = coords[:, 0].ptp()
        return dy - 0.5 * dx

    best_path = None
    best_score = -np.inf
    for s in endpoints:
        for t in endpoints:
            if s == t:
                continue
            try:
                path = list(nx.shortest_path(G, s, t))
                score = verticality_score(path)
                if score > best_score:
                    best_score = score
                    best_path = path
            except:
                continue

    return best_path



def build_skeleton_graph_from_mask(skeleton_mask):
    """
    将 skeleton 二值图转换为 NetworkX 图结构。
    每个像素为一个节点，相邻像素之间建立边（8 邻域）。
    """
    import numpy as np
    import networkx as nx

    skeleton_coords = np.column_stack(np.nonzero(skeleton_mask))  # (y, x)
    G = nx.Graph()

    for y, x in skeleton_coords:
        G.add_node((x, y))  # 注意转为 (x, y)

    for x, y in G.nodes:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in G:
                    G.add_edge((x, y), neighbor)

    return G
