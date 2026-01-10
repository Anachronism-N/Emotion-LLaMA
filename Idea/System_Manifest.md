# HERO System Manifest & Usage Guide
**Date**: 2026-01-10
**Version**: 2.1 (Feature-Only Ready)

This document provides a comprehensive overview of the HERO framework's file structure, component roles, and data flow, as implemented for the "Feature-Only" training strategy.

---

## 1. File Inventory (Change Log)

### A. Core Model (The Brain)
| File | Status | Role | Key Changes |
| :--- | :--- | :--- | :--- |
| `minigpt4/models/hero/hero_model.py` | **Modified** | Main Model | Integrated `ObservationExpertLayer`, `IntegrationLayer`, Auxiliary Losses (`STMIL`, `SCCL`), and CoT `generate` logic. Handles stacked feature input slicing. |
| `minigpt4/models/hero/observation_experts.py` | **Refactored** | Pillar 1 | Implemented 6 Experts (Visual, Motion, Audio, AU, Text, Synergy). Added `EvidenceDecoder` mechanism. |
| `minigpt4/models/hero/integration_layer.py` | **Refactored** | Pillar 2 | Implemented **Audio-Guided Panoramic Attention**. Replaced simple fusion with query-based Integration. |
| `minigpt4/models/hero/hero_loss.py` | **New** | Training | Implemented `STMIL_Loss` (Disentanglement) and `SCCL_Loss` (Alignment). |

### B. Data Pipeline (The Feeder)
| File | Status | Role | Key Changes |
| :--- | :--- | :--- | :--- |
| `scripts/extract_features.py` | **New** | Pre-processing | Script to convert Raw Video -> `.npy` Feature Dictionaries. Features: CLIP, VideoMAE, HuBERT, Whisper+BERT. |
| `minigpt4/models/hero/feature_extractors.py` | **New** | Pre-processing | Helper classes wrapping HuggingFace models for extraction. |
| `minigpt4/datasets/datasets/hero_dataset.py` | **New** | Loader | `HERODataset` class. Loads `.npy` dicts, pads diverse modalities to `1024` dim, stacks them for the model. |
| `minigpt4/datasets/builders/hero_builder.py` | **New** | Config | Registry builder for HERODataset. |

### C. Configuration & Execution
| File | Status | Role | Key Changes |
| :--- | :--- | :--- | :--- |
| `configs/models/hero.yaml` | **New** | Model Config | Hyperparameters for Experts (dims 1024/768), Loss weights (`lambda_stmil`), and Architecture flags. |
| `configs/datasets/hero_dataset.yaml` | **New** | Data Config | Points to feature directories. |
| `train_hero.py` | **New** | Launcher | Training entry point. |

---

## 2. Data Flow (Pipeline Overview)

The system is designed to minimize storage/memory usage by pre-computing high-dimensional features.

### Phase 1: offline Extraction
**Input**: Raw Video (`video.mp4`)
```mermaid
graph LR
    V[Video] --> |VideoMAE| Motion[Motion Feats (768)]
    V --> |CLIP| Global[Global Feats (1024)]
    V --> |HuBERT| Audio[Audio Feats (1024)]
    V --> |OpenFace| AU[AU Feats (1024)]
    V --> |Whisper+BERT| Text[Text Feats (768)]
    Motion & Global & Audio & AU & Text --> |Save| NPY[video_id.npy]
```
**Output**: A dictionary stored in `.npy` format.

### Phase 2: Loading (HERODataset)
**Input**: `.npy` File
1.  **Load**: Reads dictionary.
2.  **Pad**: Pads all feature tensors to user-defined `max_dim=1024` (zeros padded).
3.  **Stack**: Stacks them into a single tensor `[5, SeqLen, 1024]`.
    *   Index 0: Motion
    *   Index 1: Audio
    *   Index 2: AU
    *   Index 3: Text
    *   Index 4: Visual Global
4.  **Batch**: Returns `video_features` tensor to DataLoader.

### Phase 3: Forward Pass (HEROModel)
**Input**: `video_features` `[B, 5, Seq, 1024]`
1.  **Unpack**: `hero_model.py` detects stacked input.
2.  **Slice**: Slices each index and un-pads based on `ObservationExpertLayer` configs (e.g., Motion is sliced to 768).
3.  **Experts**:
    *   Each modality goes to its specific `ModalityQFormer`.
    *   **Output**: Compressed Features (32 queries) + Text Evidence (for SCCL).
4.  **Integration**:
    *   Audio Features serve as the "Anchor" query.
    *   Cross-attends to other modalities.
    *   **Output**: Integrated Context.
5.  **Reasoning**:
    *   LLM takes Integrated Context + User Prompt.
    *   Generates CoT Response.

---

## 3. Usage Guide

### Step 1: Extract Features
```bash
python scripts/extract_features.py --video_dir /path/to/videos --output_dir /path/to/features
```

### Step 2: Configure
Edit `configs/datasets/hero_dataset.yaml`:
```yaml
build_info:
  images:
    storage: "/path/to/features" # Point to your .npy folder
```

### Step 3: Run Training
```bash
python train_hero.py --cfg-path configs/models/hero.yaml
```

### Step 4: Verify
Check `output/hero_pretrain` for logs. Ensure `loss_stmil` and `loss_sccl` are decaying.
