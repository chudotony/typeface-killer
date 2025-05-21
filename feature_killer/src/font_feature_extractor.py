from pathlib import Path
import json
from feature_extractor import extract_features_from_svg
#
#def process_json_file(json_path, vector_dir="data/test_vectors"):
#    import json
#    from pathlib import Path
#
#    with open(json_path, 'r', encoding='utf-8') as f:
#        data = json.load(f)
#
#    #base_dir = Path("data/test_vectors")  # <-- 明确指定 SVG 文件的真实路径（文件夹的名称）
#    base_dir = Path(vector_dir)
#
#    for img_name, entry in data.items():
#        for letter in entry.get("letters", []):
#            svg_rel_path = letter.get("svg_path")
#            svg_path = base_dir / svg_rel_path if svg_rel_path else None
#
#            if svg_path and svg_path.exists():
#                try:
#                    features = extract_features_from_svg(svg_path)
#                    letter.setdefault("features", {}).update(features)
#                    print(f"Processed: {svg_path.name}   serif = {features.get('serif')}")
#
#                except Exception as e:
#                    print(f"[Error] {svg_path}: {e}")
#
#    # base_output_dir = Path("data/output")
#    # output_path = base_output_dir / Path(json_path).with_name(f"{Path(json_path).stem}_with_features.json")
#
#    output_path = Path(json_path).with_name(f"{Path(json_path).stem}_with_features.json")
#    with open(output_path, 'w', encoding='utf-8') as f:
#        json.dump(data, f, indent=2)
#    print(f"[Done] Results saved to: {output_path}")
    
def process_json_file(json_path, vector_dir="data/test_vectors"):
    import json
    from pathlib import Path

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    base_dir = Path(vector_dir)
    # output_path = Path(json_path).with_name(f"{Path(json_path).stem}_with_features.json")
    input_path = Path(json_path)
    if input_path.stem.endswith("_with_features"):
        output_path = input_path  # already a results file, just overwrite it
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_features.json")


    for img_name, entry in data.items():
        for letter in entry.get("letters", []):
            svg_rel_path = letter.get("svg_path")
            svg_path = base_dir / svg_rel_path if svg_rel_path else None

            if "features" in letter:
                continue

            if svg_path and svg_path.exists():
                try:
                    features = extract_features_from_svg(svg_path)
                    letter.setdefault("features", {}).update(features)
                    print(f"Processed: {svg_path.name}, serif = {features.get('serif')}")
                except Exception as e:
                    print(f"[Error] {svg_path}: {e}")

                # 每处理一个就写一次
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
    print(f"[Done] All letters processed and saved incrementally to: {output_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to input JSON file")
    parser.add_argument("--vector_dir", default="data/test_vectors", help="Base directory for SVG vectors")
    args = parser.parse_args()
    process_json_file(args.json_path, args.vector_dir)
