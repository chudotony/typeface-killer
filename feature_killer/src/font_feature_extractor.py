from pathlib import Path
import json
from feature_extractor import extract_features_from_svg
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def process_letter(args):
    svg_path, base_dir = args
    try:
        features = extract_features_from_svg(svg_path)
        print(f"Processed: {svg_path.name}, serif = {features.get('serif')}")
        return svg_path, features
    except Exception as e:
        print(f"[Error] {svg_path}: {e}")
        return svg_path, None

def process_json_file(json_path, vector_dir="data/test_vectors"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    base_dir = Path(vector_dir)
    input_path = Path(json_path)
    output_path = input_path.with_name(f"{input_path.stem}_with_features.json") if not input_path.stem.endswith("_with_features") else input_path

    # Collect all SVG paths to process
    tasks = []
    path_to_letter = {}
    for img_name, entry in data.items():
        for letter in entry.get("letters", []):
            if "features" in letter:
                continue
            svg_rel_path = letter.get("svg_path")
            if svg_rel_path:
                svg_path = base_dir / svg_rel_path
                if svg_path.exists():
                    tasks.append((svg_path, base_dir))
                    path_to_letter[str(svg_path)] = letter

    # Process letters in parallel
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(process_letter, tasks)
        
        # Update features and save incrementally
        for svg_path, features in results:
            if features and str(svg_path) in path_to_letter:
                letter = path_to_letter[str(svg_path)]
                letter.setdefault("features", {}).update(features)
                # Save after each batch
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
