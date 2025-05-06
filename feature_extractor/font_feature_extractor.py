from pathlib import Path
import json
from feature_extractor import extract_features_from_svg
    
def process_json_file(json_path):
    import json
    from pathlib import Path

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    base_dir = Path("output/vectors")  # <-- path to SVG

    for img_name, entry in data.items():
        for letter in entry.get("letters", []):
            svg_rel_path = letter.get("svg_path")
            svg_path = base_dir / svg_rel_path if svg_rel_path else None

            if svg_path and svg_path.exists():
                try:
                    features = extract_features_from_svg(svg_path)
                    letter.setdefault("features", {}).update(features)
                except Exception as e:
                    print(f"[Error] {svg_path}: {e}")

    # base_output_dir = Path("data/output")
    # output_path = base_output_dir / Path(json_path).with_name(f"{Path(json_path).stem}_with_features.json")
    output_path = Path(json_path).with_name(f"{Path(json_path).stem}_with_features.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[Done] Results saved to: {output_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to input JSON file")
    args = parser.parse_args()
    process_json_file(args.json_path)
