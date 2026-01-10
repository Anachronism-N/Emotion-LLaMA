# HERO Data Specification (Feature-Only)

Due to storage limitations, the HERO framework supports a "Feature-Only" training strategy. This document defines the expected directory structure and file formats.

## 1. Directory Structure

Organize your data as follows:

```
dataset_root/
├── features/                  # Directory containing .npy files for each video
│   ├── video_001.npy
│   ├── video_002.npy
│   └── ...
├── labels.json                # Annotation file
└── classes.json               # (Optional) List of class names
```

## 2. Feature File Format (`.npy`)

Each `.npy` file should contain a **Python Dictionary** with the following keys. These are generated automatically by `scripts/extract_features.py`.

*   **`video_mae`**: `[SeqLen, 1024]` (Visual Motion features)
*   **`clip`**: `[N_Patches, 1408]` (Visual Global features)
*   **`hubert`**: `[SeqLen, 1024]` (Audio features)
*   **`bert`**: `[SeqLen, 768]` (Text Embeddings)
*   **`transcript`**: `str` (ASR Text Transcript)
*   **`au`**: `[SeqLen, 1024]` (Action Units, Optional, defaults to zeros if missing)

**Note**: `SeqLen` can vary between modalities (e.g., Audio might have more frames than Video). The HERO model handles this via `ModalityQFormer`.

## 3. Label File Format (`labels.json`)

A JSON list of dictionaries. Each item represents a sample.

```json
[
  {
    "video_id": "video_001",      # Matches filename in features/ (without .npy)
    "emotion": "happy",           # Emotion Label
    "sentiment": "positive",      # (Optional) Sentiment Label
    "text": "I am so happy today." # Ground truth text (if available, else uses ASR)
  },
  {
    "video_id": "video_002",
    "emotion": "sad",
    "sentiment": "negative",
    "text": "I feel terrible."
  }
]
```

## 4. How to Create This Data

### Step 1: Feature Extraction
Run the provided script on your raw video directory:
```bash
python scripts/extract_features.py --video_path /path/to/raw_video.mp4 --output_dir dataset_root/features
```

### Step 2: Create Labels
Create `labels.json` manually or convert from existing CSVs (e.g., MER2024 labels.csv).

## 5. Dataset Class Implementation
The code uses `minigpt4/datasets/datasets/hero_dataset.py` to load this structure.
It expects the config param `feature_root` to point to `dataset_root/features` and `ann_path` to `dataset_root/labels.json`.
