# Typeface Killer 
## Qu'est-ce que c'est

A Python-based pipeline for extracting typographic features from images.

## Project Description

This project implements a pipeline for analyzing typographic features in images. The pipeline consists of several steps:

1. Word Detection: Detects and crops words from images using EasyOCR
2. Letter Detection: Identifies individual letters within words using EasyOCR and Tesseract
3. Letter Segmentation: Uses Hi-SAM (Hierarchical Segment Anything Model) to segment lettershapes
4. Vectorization: Converts letter images into vector graphics using Potrace
5. Feature Extraction: Extracts typographic features from the vectorized letters *(TBD)*

## Features

- Word detection using EasyOCR
- Letter segmentation using Hi-SAM
- Letter detection using EasyOCR and Tesseract
- Vectorization of letter images
- Typographic feature extraction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chudotony/typo-killer.git
cd typo-killer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Download required models:
```bash
# Download Hi-SAM model
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

## Project Structure

```
typeface_killer/
├── config/                   # Configuration files
│   └── default.yaml          # Default configuration
├── data/                     # Data directory
│   ├── input/                # Input images
│   ├── output/               # Processed outputs
│   │   ├── words/            # Detected word images
│   │   ├── segmented/        # Segmented word masks
│   │   └── vectors/          # Vectorized letter SVGs
│   └── models/               # Model weights
├── typeface_killer/          # Main package
│   ├── word_detection/       # Word detection module
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── letter_detection/     # Letter detection module
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── letter_segmentation/  # Letter segmentation module
│   │   ├── __init__.py
│   │   └── segmenter.py
│   ├── vectorization/        # Vectorization module
│   │   ├── __init__.py
│   │   └── vectorizer.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── config.py         # Configuration utilities
│       ├── logging.py        # Logging utilities
│       └── image.py          # Image processing utilities
├── tests/                    # Test files
├── requirements.txt          # Project dependencies
└── main.py                   # Main pipeline script
```

## Dataset Structure

As an example, the pipeline processes a JSON dataset with the following structure:

```json
{
    "filename.png": {
        "letters": [
            {
                "char": "A",
                "bbox": {
                    "x": 100,
                    "y": 100,
                    "w": 50,
                    "h": 50
                },
                "svg_path": "data/output/vectors/filename_0001_0001.svg",
                "features": {}
            }
        ],
        "source": {
            "source_name": "Poster Museum",
            "country": "Switzerland",
            "year": 1925,
            "designer": "John Doe",
            "medium": "Poster"
        }
    }
}
```

The `source` information should be provided in the dataset, while the `letters` array will be populated by the pipeline.

## Pipeline Workflow

The pipeline processes images in the following steps:

1. **Word Detection**:
   - Detects words in the input image using EasyOCR
   - Filters words by size (minimum 70x70 pixels)
   - Saves word images to `output/words/` with format: `filename_0001.png`

2. **Letter Detection**:
   - Detects letters within each word
   - Stores letter bounding boxes in the dataset

3. **Letter Segmentation**:
   - Segments each word using Hi-SAM
   - Saves segmented masks to `output/segmented/` with format: `filename_0001_segmented.png`

4. **Letter Vectorization**:
   - Crops letters from segmented masks using detected bounding boxes
   - Converts letter masks to SVG using Potrace
   - Saves SVG files to `output/vectors/` with format: `filename_0001_0001.svg`
   - Updates the dataset with letter information

The pipeline is memory-efficient, loading ML models only when needed and cleaning them up after use.

## Configuration

The system is configured through `config/default.yaml`. Key settings include:

- Word detection parameters
- Letter segmentation settings
- Letter detection parameters
- Vectorization options

### Letter Segmentation

The letter segmentation module uses Hi-SAM for precise letter mask generation. Configuration options include:

```yaml
letter_segmentation:
  model_type: "vit_h"  # Model variant (vit_h, vit_l, vit_b)
  checkpoint: "checkpoints/sam_vit_h_4b8939.pth"
  device: "cuda"  # Processing device
  patch_mode: true  # Use sliding window for large images
  input_size: 1024  # Input image size
```

## Usage

1. Prepare your dataset JSON file with source information
2. Place your input images in the `data/input` directory
3. Configure the system in `config/default.yaml`
4. Run the pipeline:
```bash
python main.py --dataset path/to/dataset.json --input data/input --output data/output
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Note: This project integrates several third-party components, each with their own licenses:
- EasyOCR: Apache License 2.0
- Hi-SAM: Apache License 2.0
- Tesseract OCR: Apache License 2.0
- Potrace: GNU General Public License

Please refer to the respective license files of these components for their specific terms and conditions.

## Acknowledgments

- EasyOCR for word and letter detection
- Hi-SAM for letter segmentation
- Tesseract OCR for letter detection
