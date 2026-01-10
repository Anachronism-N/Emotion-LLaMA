# HERO 系统清单与架构全解 (System Manifest & Architecture Guide)

**更新日期**: 2026-01-10
**版本**: 3.0 (Adaptive Attention Ready)

本文档旨在全面说明 HERO 框架的文件构成、数据流向、以及当前系统的完整度与已知待办事项。

---

## 1. 文件清单 (File Inventory & Status)

### A. 核心模型层 (Brain of HERO)
| 文件路径 | 状态 | 来源 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `minigpt4/models/hero/hero_model.py` | **核心重构** | 新建 | 模型的总入口。继承自 `MiniGPTBase`。实现了：<br>1. **特征切片**: 将 Stacked Tensor (1024 dim) 拆解并切片回原始维度。<br>2. **专家初始化**: 加载 6 个 Observation Experts。<br>3. **CoT 生成**: 包含 `structural_chain_of_thought` Prompt 模板与生成逻辑。<br>4. **Loss 整合**: `forward` 中计算 STMIL & SCCL。 |
| `minigpt4/models/hero/observation_experts.py` | **核心重构** | 新建 | **Pillar 1 (底层专家)**。实现了 `VisualExpert`, `AudioExpert`, `TextExpert`, `SynergyExpert`。每个专家包含一个 `ModalityQFormer` 和 `EvidenceDecoder` (用于语义对齐)。 |
| `minigpt4/models/hero/integration_layer.py` | **核心重构** | 新建 | **Pillar 2 (整合层)**。实现了 `AdaptiveQueryGenerator` (动态锚定/音频锚定) 和 `PanoramicGuidedAttention`。负责生成 Global Query 并查询 K-Bank。 |
| `minigpt4/models/hero/hero_loss.py` | **新增** | 新建 | **辅助损失**。实现了 `STMIL_Loss` (模态解耦/互信息最小化) 和 `SCCL_Loss` (语义证据对齐/对比学习)。 |
| `minigpt4/models/hero/__init__.py` | **新增** | 新建 | 模块导出与注册。 |

### B. 数据管道 (Data Pipeline)
| 文件路径 | 状态 | 来源 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `scripts/extract_features.py` | **新增** | 新建 | **预处理脚本**。从 MP4 视频提取 CLIP, VideoMAE, HuBERT, Whisper 序列特征，保存为 `.npy` 字典。 |
| `minigpt4/models/hero/feature_extractors.py` | **新增** | 新建 | **提取器封装**。包含 `StandardVideoMAE`, `StandardHuBERT`, `StandardCLIP` 等类，负责加载 HF 模型并执行 forward。 |
| `minigpt4/datasets/datasets/hero_dataset.py` | **新增** | 新建 | **数据集加载器**。读取 `.npy` 字典，执行 Modal-Specific Truncation (如 Audio 截断至 1024 dim, Video 截断至 768 dim)，然后统一 Pad 到 1024 dim 并 Stack 起来。 |
| `minigpt4/datasets/builders/hero_builder.py` | **新增** | 新建 | 数据集构建器。用于注册配置。 |
| `configs/datasets/hero_dataset.yaml` | **新增** | 新建 | 数据集路径配置。 |

### C. 遗留/基础模块 (Legacy / Shared)
| 文件路径 | 状态 | 来源 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `minigpt4/models/minigpt_base.py` | 沿用 | EmotionLLaMA | 提供基础的 LLaMA 加载、Tokenizer 处理与 Projector 定义。HERO 继承此类。 |
| `minigpt4/models/Qformer.py` | 沿用 | EmotionLLaMA | BLIP-2 Q-Former 的基础实现。HERO 的 `ModalityQFormer` 复用了部分权重逻辑。 |
| `minigpt4/tasks/` | 沿用 | EmotionLLaMA | 训练任务循环逻辑 (Runner)。 |

---

## 2. 数据流详解 (Data Flow & Formats)

### 阶段 1: 预处理 (Pre-processing)
*   **输入**: 原始视频 `video.mp4`
*   **处理**: `extract_features.py`
    *   VideoMAE (Base) -> Flow: `[Seq, 768]`
    *   CLIP (Large) -> Global: `[Seq, 1024]`
    *   HuBERT (Large) -> Audio: `[Seq, 1024]`
    *   OpenFace -> AU: `[Seq, 1024]` (placeholder if missing)
    *   Whisper+BERT -> Text: `[Seq, 768]`
*   **输出**: `video_id.npy` (字典)

### 阶段 2: 训练时加载 (Training / Forward)

#### Step 2.1: Dataset Loading (`hero_dataset.py`)
1.  读取 `.npy`。
2.  **Stacking & Padding**:
    *   为了 Batch 处理，所有模态特征被 Pad 到 `[Max_Seq, 1024]`。
    *   堆叠由一个 Tensor: `[Batch, 6, Max_Seq, 1024]` (6 代表 6 个专家插槽位)。
    *   生成 `modality_mask`: 标记哪些模态是真实的，哪些是 Padding。

#### Step 2.2: Model Encoding (`hero_model.py` -> `observation_experts.py`)
1.  **Slicing**: 模型接收 Stacked Tensor。
    *   *Motion Slot*: 取出 `[:, 0, :, :768]` (切除 Padding，恢复 768 维)。
    *   *Audio Slot*: 取出 `[:, 2, :, :]` (保留 1024 维)。
2.  **Expert Processing**:
    *   各专家 Q-Former 将长序列压缩由 32 个 Query Embeddings。
    *   Evidence Decoder 生成语义向量 (供 SCCL Loss 使用)。

#### Step 2.3: Integration (`integration_layer.py`)
1.  **Adaptive Query Gen**:
    *   **Dynamic Mode (Default)**: Scorer 给各模态打分 -> 加权生成 `hybrid_anchor` -> Self-Attention Refinement -> `global_query`。
    *   *Audio Mode*: 取 Audio Summary 直接作为 Query。
2.  **Panoramic Attention**:
    *   `global_query` Cross-Attend 到含所有模态的 Key-Value Bank。
    *   输出 `integrated_context` `[Batch, 64, 4096]` (映射到 LLM 维度)。

#### Step 2.4: LLM Reasoning
1.  Input: `Integrated Context` + `CoT Prompt` ("...Analyze the evidence...").
2.  LLM Output: `{"evidence": "...", "prediction": "sad"}`.

#### Step 2.5: Loss Calculation
*   **L_Task**: CrossEntropy (预测 Token vs 真实 Text)。
*   **L_STMIL**: 最小化各专家特征互信息 (解耦)。
*   **L_SCCL**: 拉近 专家特征 与 文本语义证据 (对齐)。
*   **Total**: `L_Task + w1*L_STMIL + w2*L_SCCL`.

---

## 3. 差距分析 (Gap Analysis & TODOs)

尽管核心架构已完成，但要实现 **SOTA 效果**，以下部分通常需要进一步工作：

1.  **EvidenceDecoder 的训练数据 (Missing)**:
    *   *现状*: `EvidenceDecoder` 目前生成的是 Placeholder 文本或未经监督的向量。
    *   *需求*: 为了让 SCCL Loss 生效，理想情况下需要有对每个模态的“细粒度文本描述”作为标签（例如 "声音颤抖", "眉毛上扬"）。目前大多数据集缺乏此类标注。
    *   *临时方案*: 可以暂时使用 Emotion Label 广播作为弱监督信号，或者冻结 SCCL 权重。

2.  **OpenFace 集成 (Partially Implemented)**:
    *   *现状*: `extract_features.py` 中有 OpenFace 的逻辑槽位，但在无 Docker 环境下 OpenFace 安装极其困难。
    *   *对策*: 目前代码支持 AU 特征缺失（全 0 填充）。如果需要 AU 专家发挥作用，建议使用 py-feat 等替代库或外挂 OpenFace 容器。

3.  **LLM 词表对齐**:
    *   *现状*: 使用标准 LLaMA Tokenizer。
    *   *风险*: 情感相关的专用词汇（如微表情术语）可能被切分得太碎。
---

## 附录：Git 提交流程 (Git Operations Guide)

为了规范代码提交并确保远程仓库同步，请遵循以下步骤：

### 1. 检查状态 (Review Changes)
在提交前，确认当前修改的文件：
```bash
git status
```

### 2. 暂存更改 (Stage Changes)
将修改后的代码添加到暂存区。建议排除 `__pycache__` 等临时文件（已由 `.gitignore` 处理）：
```bash
git add .
```

### 3. 提交记录 (Commit)
编写简洁明了的提交信息，描述核心变更：
```bash
git commit -m "Your descriptive message here"
```

### 4. 推送到远程 (Push)
将本地 `main` 分支推送到 GitHub 上的 `origin`：
```bash
git push origin main
```

> [!TIP]
> **关于环境**: 提交前建议先运行 `tests/test_hero_refactor.py` 确保核心逻辑未被损坏。

4.  **Batch Size 限制**:
    *   *现状*: 只是 Feature-Only，显存占用仍较大 (Q-Former * 6 + LLM)。可能需要使用 Gradient Checkpointing 或 LoRA (已计划)。

---

## 4. 总结
当前代码库是一个 **完整、可运行、符合 Idea 设计** 的 Feature-Only 训练框架。
*   **核心逻辑**: 动态锚定 (Adaptive Query) 与 证据链 (CoT) 均已就绪。
*   **推荐操作**: 先在 MER2023 (Feature-Only) 上跑通流程，验证 Adaptive Strategy 的权重变化是否符合预期（即噪声大时 Audio 权重是否升高）。
