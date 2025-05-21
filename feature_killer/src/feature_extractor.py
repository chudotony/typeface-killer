# ---- 动态加载 svg_utils.py 模块 ----
import importlib.util
from pathlib import Path
from svgpathtools import svg2paths2
import numpy as np
import networkx as nx

svg_utils_path = Path(__file__).parent / "svg_utils.py"
spec = importlib.util.spec_from_file_location("svg_utils", svg_utils_path)
svg_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svg_utils)

# 提取需要用的函数
load_normalized_svg = svg_utils.load_normalized_svg
compute_medial_axis = svg_utils.compute_medial_axis
compute_medial_axis_serif = svg_utils.compute_medial_axis_serif
skeleton_to_graph = svg_utils.skeleton_to_graph
compute_medial_axis_consistent = svg_utils.compute_medial_axis_consistent

from sklearn.decomposition import PCA
import networkx as nx
import numpy as np

def extract_features_from_svg(svg_path):
    """Extract typographic features from a single SVG file"""
    # Load SVG only once
    outline = load_normalized_svg(svg_path)
    if not outline:
        raise ValueError("Failed to load SVG or no paths found")

    # Get MAT results
    medial_axis, skeleton_mask, dist_transform = compute_medial_axis_consistent(outline)
    
    # Build graph once and reuse
    G = skeleton_to_graph(skeleton_mask)
    
    # For Serif - reuse existing paths
    paths = svg_utils.get_cached_paths(svg_path)
    _, skeleton_orig, bbox_origin, scale_origin = compute_medial_axis_serif(paths)

    is_serif, num_filtered_points = compute_serif(svg_path, skeleton_orig, bbox_origin, scale_origin)

    if not medial_axis:
        raise ValueError("Medial axis computation failed")

    return {
        "proportion": compute_proportion(outline, medial_axis),
        "weight": compute_weight(skeleton_mask, dist_transform, canvas_size=256),
        "slant": compute_slant_from_skeleton(G),
        "serif": is_serif,
        # "serif count": num_filtered_points
        # is_serif, _ = compute_serif_from_segments(svg_path, skeleton_mask, bbox_origin, scale)
    }


# -------------------
# Featurs
# -------------------

# Proportion
# -------------------

def compute_proportion(outline, medial_axis):
    # 示例：宽高比，使用 bbox
    x_coords = [pt[0] for contour in outline for pt in contour]
    y_coords = [pt[1] for contour in outline for pt in contour]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return round(height / width, 3) if width != 0 else 0.0
    

# Weight
# -------------------


"""
Weight is defined as the average stroke thickness of a glyph, estimated from the skeleton derived via the medial axis transform (MAT). To ensure consistency across different glyphs and typefaces, each vector outline is first normalized to unit height, then rasterized onto a fixed-size canvas (e.g., 256×256 pixels). The weight is computed as the average Euclidean distance from skeleton pixels to the nearest contour boundary (via distance transform), multiplied by 2. Finally, this raw stroke thickness is divided by the canvas height to yield a unitless relative weight ranging approximately from 0 to 1, making it comparable across different shapes.
"""

# 平均值
def compute_weight(skeleton, distance, canvas_size=256):
    """
    计算平均 stroke 粗细（即 skeleton 上的 avg radius × 2）
    """
    if not np.any(skeleton):
        return 0.0
    thickness = distance[skeleton].mean() * 2
    relative = thickness / canvas_size
    return round(float(relative), 4)


# MAT 半径最大值
#def compute_weight(outline, medial_axis):
#    if not medial_axis:
#        return 0.0
#    return round(max([r for (_, _, r) in medial_axis]), 3)


# Slant
# -------------------

# 新版 + 加入了 当没有端点（封闭时）
def compute_slant_from_skeleton(G):
    """
    更健壮的 Slant 检测方法：
    - 优先使用端点对找竖直干线
    - 如失败，则 fallback 到所有节点对找最长路径
    """
    if nx.number_connected_components(G) > 1:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    def vertical_score(path):
        coords = np.array(path)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        return abs(dy) - 0.5 * abs(dx)

    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    paths = []
    if len(endpoints) >= 2:
        for s in endpoints:
            for t in endpoints:
                if s != t:
                    try:
                        path = list(nx.shortest_path(G, s, t))
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

    # fallback: use all node pairs if not enough endpoint paths
    if len(paths) == 0:
        nodes = list(G.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    path = list(nx.shortest_path(G, nodes[i], nodes[j]))
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue

    if not paths:
        return None

    longest_path = max(paths, key=vertical_score)

    coords = np.array(longest_path)
    ys = coords[:, 1]
    y_med = np.median(ys)
    coords = coords[np.abs(ys - y_med) < 10]

    xs = coords[:, 0]
    x_med = np.median(xs)
    spine_coords = coords[np.abs(xs - x_med) < 10]

    if len(spine_coords) >= 2:
        sorted_spine = spine_coords[np.argsort(spine_coords[:, 1])]
        i1 = int(len(sorted_spine) * 0.2)
        i2 = int(len(sorted_spine) * 0.8)
        p1 = sorted_spine[i1]
        p2 = sorted_spine[i2]
        delta = p2 - p1
        slant_angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return round(float(slant_angle), 2)
    else:
        return None



# 原新版，没加PCA
# 多连通时取了最大
#
#def compute_slant_from_skeleton(G):
#    """
#    计算 slant（笔画倾斜角），只使用 skeleton 的最大连通子图。
#    """
#    if nx.number_connected_components(G) > 1:
#        # 仅保留最大连通子图（通常是主干）
#        largest_cc = max(nx.connected_components(G), key=len)
#        G = G.subgraph(largest_cc).copy()
#
#    endpoints = [n for n in G.nodes if G.degree[n] == 1]
#
#    def vertical_score(path):
#        coords = np.array(path)
#        dx = coords[-1][0] - coords[0][0]
#        dy = coords[-1][1] - coords[0][1]
#        return abs(dy) - 0.5 * abs(dx)
#
#    # 路径集合（添加容错处理）
#    paths = []
#    for s in endpoints:
#        for t in endpoints:
#            if s == t:
#                continue
#            try:
#                path = list(nx.shortest_path(G, s, t))
#                paths.append(path)
#            except nx.NetworkXNoPath:
#                continue
#
#    if not paths:
#        return None  # 没有有效路径
#
#    longest_path = max(paths, key=vertical_score)
#
#    coords = np.array(longest_path)
#    ys = coords[:, 1]
#    y_med = np.median(ys)
#    coords = coords[np.abs(ys - y_med) < 10]
#
#    xs = coords[:, 0]
#    x_med = np.median(xs)
#    spine_coords = coords[np.abs(xs - x_med) < 10]
#
#    if len(spine_coords) >= 2:
#        sorted_spine = spine_coords[np.argsort(spine_coords[:, 1])]
#        i1 = int(len(sorted_spine) * 0.2)
#        i2 = int(len(sorted_spine) * 0.8)
#        p1 = sorted_spine[i1]
#        p2 = sorted_spine[i2]
#        delta = p2 - p1
#        slant_angle = np.degrees(np.arctan2(delta[1], delta[0]))
#        return round(float(slant_angle), 2)
#    else:
#        return None



# 完美Serif
def compute_serif(svg_path, skeleton_mask, bbox_origin, scale, height_norm=1.0,
                                rotation_thresh=30, length_thresh=30, dist_thresh=10):
    """
    保留原始的 serif segment 检测逻辑，并使用 MAT 的端点进一步过滤。
    返回布尔值（是否为 serif 字体）和 serif 点坐标列表。
    """
    from sklearn.neighbors import KDTree
    from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier
    import numpy as np

    def svg_to_paths(svg_path):
        paths, *_ = svg2paths2(str(svg_path))
        return paths

    def sample_path_points(path_segment, num_samples=20):
        return [path_segment.point(t / num_samples) for t in range(num_samples + 1)]

    def compute_total_rotation(points):
        tangents = np.angle([points[i + 1] - points[i] for i in range(len(points) - 1)])
        diff = np.diff(tangents)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(np.sum(np.abs(diff)))

    def compute_total_length(points):
        return np.sum([abs(points[i + 1] - points[i]) for i in range(len(points) - 1)])

    def detect_serif_segments(paths, rotation_thresh=30, length_thresh=30, samples=20):
        serif_segments = []
        segment_centers = []
        for glyph_path in paths:
            for seg in glyph_path:
                if isinstance(seg, (Line, CubicBezier, QuadraticBezier)):
                    pts = sample_path_points(seg, num_samples=samples)
                    rot = compute_total_rotation(pts)
                    arc_len = compute_total_length(pts)
                    
                    # Early length filtering before expensive rotation calculation
                    if arc_len < length_thresh:
                        if rot > rotation_thresh:
                            serif_segments.append(seg)
                            center_pt = seg.point(0.5)
                            segment_centers.append(center_pt)
        return serif_segments, segment_centers

    def build_skeleton_graph(skeleton):
        import networkx as nx
        # Pre-allocate the graph size
        G = nx.Graph()
        rows, cols = skeleton.shape
        coords = np.column_stack(np.nonzero(skeleton))
        
        # Create nodes in bulk
        G.add_nodes_from(map(tuple, coords[:, ::-1]))  # Reverse coords to (x,y)
        
        # Vectorized edge creation
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            shifted = coords + [dy, dx]
            mask = (shifted[:,0] >= 0) & (shifted[:,0] < rows) & \
                  (shifted[:,1] >= 0) & (shifted[:,1] < cols)
            valid_shifts = shifted[mask]
            valid_coords = coords[mask]
            
            # Filter points that exist in skeleton
            exists_mask = skeleton[valid_shifts[:,0], valid_shifts[:,1]]
            edges = zip(map(tuple, valid_coords[exists_mask, ::-1]), 
                       map(tuple, valid_shifts[exists_mask, ::-1]))
            G.add_edges_from(edges)
            
        return G

    def filter_serif_segments_by_mat(serif_segments, segment_centers, skeleton_mask, bbox_origin, scale, height_norm=1.0, dist_thresh=10):
        # Build graph once
        G = build_skeleton_graph(skeleton_mask)
        leaf_nodes = [pt for pt in G.nodes if G.degree[pt] == 1]
        leaf_arr = np.array(leaf_nodes)

        if len(leaf_arr) == 0:
            return []

        # Build KDTree once
        tree = KDTree(leaf_arr)
        
        # Vectorized conversion of all segments at once
        points = np.array([[pt.real, pt.imag] for pt in segment_centers])
        pixel_xy = ((points * height_norm) - bbox_origin) * scale
        
        # Batch query KDTree
        dists, _ = tree.query(pixel_xy, k=1)
        mask = dists.flatten() <= dist_thresh
        
        # Filter segments using mask
        filtered_segments = [seg for seg, m in zip(serif_segments, mask) if m]
        return [seg.point(0.5) for seg in filtered_segments]

    paths = svg_to_paths(svg_path)
    serif_segments, segment_centers = detect_serif_segments(paths, rotation_thresh, length_thresh)
    filtered_points = filter_serif_segments_by_mat(serif_segments, segment_centers, skeleton_mask, bbox_origin, scale, height_norm, dist_thresh)
    # print(f"[DEBUG] Total serif segments before MAT filter: {len(serif_segments)}")
    # print(f"[DEBUG] Total serif segments after  MAT filter: {len(filtered_points)}")
    return len(filtered_points) >= 3, len(filtered_points)
    # return len(filtered_points) >= 4
