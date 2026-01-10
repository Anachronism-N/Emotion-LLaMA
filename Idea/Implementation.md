# HERO Implementation Documentation
# HERO 框架实现文档

**Version**: 2.1 (Feature-Only Optimized)
**Date**: 2026-01-10
**Status**: Core Implementation & Data Pipeline Ready

## 1. Overview (项目概述)

HERO (Hierarchical Evidence-based Reasoning and Observation) is a multimodal emotion understanding framework built upon the **Emotion-LLaMA** baseline. It aims to address the limitations of "black-box" fusion by introducing a transparent, hierarchical evidence chain.

### Core Architecture Upgrade (vs Baseline)

| Component | Emotion-LLaMA | HERO (v2.0 Implemented) | Impact |
| :--- | :--- | :--- | :--- |
| **Observation** | Single Linear Proj | **Dual-Output Experts** (Tensor + Text Evidence) | Enables interpretability & alignment |
| **Experts** | Visual, Audio, AU | **6 Experts**: Vis-Global, Vis-Motion, Audio, AU, Text(ASR), Synergy | Granular semantic capture |
| **Integration** | Concat + LLaMA | **Audio-Guided Panoramic Attention** | Prioritizes audio truth, context-aware fusion |
| **Reasoning** | Unstructured | **Hierarchical CoT** (Evidence->Rationale->Prediction) | Chain-of-Thought driven Logic |
| **Loss** | CE Loss | **CE + STMIL + SCCL** | Ensures evidence is disentangled & aligned |

---

## 2. Technical Architecture (技术架构)

### 2.1 Pillar 1: Observation Expert Layer (观测专家层)
*Located in: `minigpt4/models/hero/observation_experts.py`*

**Structure:** 6 Specialized Experts, each with `ModalityQFormer`.
1.  **Visual Global Expert**: Inputs CLIP-Large features (`1024 dim`). Extracts scene context.
2.  **Visual Motion Expert**: Inputs VideoMAE-Base features (`768 dim`). Extracts action dynamics.
3.  **Audio Expert**: Inputs HuBERT-Large features (`1024 dim`). Extracts paralinguistic cues.
4.  **AU Expert**: Inputs OpenFace AU features (`1024 dim` aligned). Extracts micro-expressions.
5.  **Text Expert**: Inputs BERT embeddings (`768 dim`). Extracts verbal content.
6.  **Synergy Expert**: Bi-directional Audio-Visual attention. Detects sarcasm/conflict.

**Mechanism:**
*   **Input**: Stacked Feature Tensor `[B, 5, Seq, 1024]` (Padded)
    *   Model handles slicing/unpadding automatically.
*   **Q-Former**: Compresses L -> 32 Queries.
*   **Dual Output**:
    *   `Feature Tensor`: `[B, 32, 768]` for Integration Layer.
    *   `Semantic Evidence`: Text description (e.g., "Slight frown detected") via `Evidence Decoder`.

### 2.2 Pillar 2: Evidence Integration Layer (证据整合层)
*Located in: `minigpt4/models/hero/integration_layer.py`*

**Audio-Guided Cross-Examination Strategy**:
Instead of equal-weight fusion, we treat **Audio** as the "Anchor of Truth" (as defined in Idea.md).

1.  **Audio-Guided Query Generator**:
    *   Uses **Audio Summary** as the primary Query seed.
    *   Cross-attends to other modalities to generate the `Global Query`.
    *   *Intuition*: "I heard checking in the voice, now let me assume the face is masking it."
    
2.  **Panoramic Dynamic Guided Attention**:
    *   **Query**: Global Query + Output Queries.
    *   **Context**: **K-Bank** (Concatenation of all 6 expert tensors).
    *   **Output**: `Integrated Context` `[B, 64, 4096]` for LLM.

### 2.3 Pillar 3: Hierarchical Reasoning Layer (分层推理层)
*Located in: `minigpt4/models/hero/hero_model.py`*

**Structured Thinking Process**:
*   Enforced via `COT_PROMPT_TEMPLATE`.
*   Reasoning Flow:
    1.  **Evidence Analysis**: List specific cues from Expert Outputs.
    2.  **Rationale**: Synthesize contradictions (Synergy) and matches.
    3.  **Prediction**: JSON output.

### 2.4 Loss Functions (损失函数)
*Located in: `minigpt4/models/hero/hero_loss.py`*

1.  **STMIL (Soft-Truncated Mutual Information Learning)**:
    *   Minimizes MI between Emotion Features and Identity/Content.
    *   Ensures purely affective features.
2.  **SCCL (Supervised Contrastive Continuum Learning)**:
    *   Aligns Expert Feature Tensors with Ground Truth Text descriptions.

---

## 3. Implementation Status (实施状态)

| Module | Status | Verification | Notes |
| :--- | :--- | :--- | :--- |
| **Observation Experts** | ✅ Complete | `test_hero_refactor.py` | Includes Text & Synergy experts. |
| **Integration Layer** | ✅ Complete | `test_hero_refactor.py` | Audio-Guided Attention verified. |
| **HERO Model** | ✅ Complete | `test_hero_refactor.py` | End-to-end forward pass works. |
| **Loss Functions** | ✅ Implemented | `test_hero_refactor.py` | Aux losses ready for training loop. |
| **Inference Pipeline** | ✅ Complete | `test_cot.py` | CoT generation & parsing active. |
| **Training Pipeline** | ⚠️ Partial | N/A | `train.py` needs update to use `hero_loss.py`. |

## 4. Data Pipeline (Feature-Only Strategy)
Due to storage constraints, training uses pre-extracted features.

**Steps**:
1.  **Extract**: `scripts/extract_features.py` -> `.npy` files.
    *   Uses OpenAI CLIP, VideoMAE, HuBERT, Whisper, BERT.
2.  **Load**: `HERODataset` (`hero_dataset.py`)
    *   Reads dictionary from `.npy`.
    *   Pads all modalities to `1024` dim (max).
    *   Stacks into `video_features`.
3.  **Forward**: `HEROModel` (`hero_model.py`)
    *   Detects stacked input.
    *   Slices and un-pads to correct dimensions (`768` for Motion, etc.).
    *   *Note*: Config `hero.yaml` sets Visual=1024, Video=768, Audio=1024 to match extractors.

## 5. Next Steps (Evaluation & Refinement)
*   **Run Feature Extraction**: Process full dataset.
*   **Start Training**: Use `train_hero.py`.
*   **Evaluate**: Monitor STMIL (disentanglement) and SCCL (alignment) losses.
