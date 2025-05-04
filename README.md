# Typeface Killer 
## Qu'est-ce que c'est

A Python-based pipeline for extracting typographic features from images. The pipeline consists of several steps:

1. Word Detection: Detects and crops words from images using EasyOCR
2. Letter Detection: Identifies individual letters within words using EasyOCR and Tesseract
3. Letter Segmentation: Uses Hi-SAM (Hierarchical Segment Anything Model) to segment lettershapes
4. Vectorization: Converts letter images into vector graphics using Potrace
5. Feature Extraction *(TBD)*: Extracts typographic features from the vectorized letters:
    - Serifs (boolean)
    - Slant (float)
    - Contrast (float)
    - Weight (float)

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/chudotony/typeface-killer.git
cd typeface-killer
```

**2. Set up Hi-SAM:**

We used [SAM-TS-H trained with TextSeg](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnY28xejlzZFVpMXZYQ3NLZ0E%5FZT1VM1dQSnk&cid=E534267B85818129&id=E534267B85818129%2125909&parId=E534267B85818129%2125901&o=OneUp), and the corresponding [ViT-H SAM model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). Other options are available in the [Hi-SAM repo](https://github.com/ymy-k/Hi-SAM?tab=readme-ov-file), following steps reproduce our set-up:

  1. Clone the Hi-SAM repository into the correct directory:
    ```bash
    git clone https://github.com/ymy-k/Hi-SAM.git typeface_killer/letter_segmentation/Hi-SAM
    ```
  2. Create `pretrained_checkpoint` directory and add  ViT-H SAM model. 
    ```bash
    mkdir -p pretrained_checkpoint
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O pretrained_checkpoint/sam_vit_h_4b8939.pth
    ```
  3. Download [SAM-TS-H trained with TextSeg](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnY28xejlzZFVpMXZYQ3NLZ0E%5FZT1VM1dQSnk&cid=E534267B85818129&id=E534267B85818129%2125909&parId=E534267B85818129%2125901&o=OneUp) and put it into the `pretrained_checkpoint` directory.

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Install Tesseract OCR:**
- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

You should also install languages for tesseract, we used `tesseract-ocr-deu`, `tesseract-ocr-eng`, `tesseract-ocr-ita`, `tesseract-ocr-fra`.

## Project Structure

```
typeface-killer/
├── config/                   # Configuration files
│   └── default.yaml
├── input/                    # Input images and dataset
│   ├── images/
│   └── dataset.json
├── output/                   # Processed outputs
│   ├── vectors/              # Letter SVGs
│   ├── words/                # Word images
│   └── words_segmented/      # Word masks
├── pretrained_checkpoint/    # SAM and Hi-SAM weights
├── typeface_killer/          # Main package
│   ├── letter_detection/     # Letter detection module
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── letter_segmentation/  # Letter segmentation module
│   │   ├── Hi-SAM/           # Hi-SAM repo
│   │   │   └── <...>
│   │   ├── __init__.py
│   │   └── segmenter.py
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── image.py
│   │   └── logging.py
│   ├── vectorization/        # Vectorization module
│   │   ├── __init__.py
│   │   └── vectorizer.py
│   ├── word_detection/       # Word detection module
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── __init__.py
│   └── main.py               # Main pipeline script
└── requirements.txt          # Project dependencies
```

## Dataset Structure

As an example, the pipeline processes a JSON dataset with into following structure:

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

The `source` information is specific to our research project and optional for an input dataset, while the `letters` list is filled by the pipeline.

## Pipeline Workflow

The pipeline processes images in the following steps:

**1. Word Detection**:
   - Detects words in the input image using EasyOCR
   - Filters words by size (minimum 70x70 pixels)
   - Saves word images to `output/words/` with format: `filename_<word_id>.png`

**2. Letter Detection**:
   - Detects letters within each word using Tesseract
   - Identify the letter using EasyOCR
   - Filters letters by size (minimum 35x35 pixels)
   - Stores letter bounding boxes in the dataset

**3. Letter Segmentation**:
   - Segments each word using Hi-SAM
   - Saves segmented masks to `output/segmented/` with format: `filename_<word_id>.png`

**4. Letter Vectorization**:
   - Uses as input the segmented masks (from step 3) cropped with detected bounding boxes (from step 2)
   - Converts letter masks to SVG using Potrace
   - Saves SVG files to `output/vectors/` with format: `filename_<word_id>_<letter_id>.svg`
   - Updates the dataset with letter information

## Usage

Run the pipeline with:
```bash
python -m typeface_killer.main --dataset=input/dataset.json --output output 
```