# HERO 数据规范（仅特征版）
由于存储资源的限制，HERO 框架支持「仅特征」的训练策略。本文档定义了该策略下**要求遵循的目录结构**和**文件格式规范**。

## 1. 目录结构
请按如下结构组织你的数据集：
```
dataset_root/
├── features/                  # 存放各视频特征的 .npy 文件目录
│   ├── video_001.npy
│   ├── video_002.npy
│   └── ...
├── labels.json                # 标注信息文件
└── classes.json               # （可选）类别名称清单
```

## 2. 特征文件格式（`.npy`）
每个 `.npy` 文件中都需要存储一个**Python 字典**，字典包含下述指定键值对。这类文件可通过脚本 `scripts/extract_features.py` 自动生成。

*   **`video_mae`**: 维度为 `[SeqLen, 1024]` → 视觉运动特征
*   **`clip`**: 维度为 `[N_Patches, 1408]` → 视觉全局特征
*   **`hubert`**: 维度为 `[SeqLen, 1024]` → 音频特征
*   **`bert`**: 维度为 `[SeqLen, 768]` → 文本嵌入特征
*   **`transcript`**: 数据类型 `str` → 语音转文字（ASR）的文本转录内容
*   **`au`**: 维度为 `[SeqLen, 1024]` → 面部动作单元特征（可选项，若缺失则默认填充全0）

**注意**：`序列长度(SeqLen)` 在不同数据模态中可能存在数值差异（例如，音频的帧数可能会多于视频的帧数）。HERO 模型会通过 `ModalityQFormer` 组件处理该维度差异问题。

## 3. 标签文件格式（`labels.json`）
该文件的格式为**由字典组成的JSON数组**，数组中的每一个元素都代表一条样本数据。

```json
[
  {
    "video_id": "video_001",      # 与 features/ 目录下的文件名一致（无需带 .npy 后缀）
    "emotion": "happy",           # 情绪标签
    "sentiment": "positive",      # （可选）情感倾向标签
    "text": "I am so happy today." # 真实标注文本（如有则填，无则使用语音转文字的内容）
  },
  {
    "video_id": "video_002",
    "emotion": "sad",
    "sentiment": "negative",
    "text": "I feel terrible."
  }
]
```

## 4. 数据集制作方法
### 步骤1：特征提取
在原始视频目录下，运行官方提供的特征提取脚本即可生成特征文件：
```bash
python scripts/extract_features.py --video_path /path/to/raw_video.mp4 --output_dir dataset_root/features
```

### 步骤2：制作标签文件
手动创建 `labels.json` 文件，也可以从已有的CSV文件（例如 MER2024 的 labels.csv）转换生成该文件。

## 5. 数据集类的实现方式
项目代码中通过文件 `minigpt4/datasets/datasets/hero_dataset.py` 加载上述数据集结构。
该类需要配置两个核心参数：`feature_root` 指向 `dataset_root/features` 目录，`ann_path` 指向 `dataset_root/labels.json` 文件。

## 5. 推荐数据集 (Recommended Datasets for HERO)

根据 **Audio-Guided (音频引导)** 和 **CoT (推理链)** 的架构特点，我们推荐以下包含强多模态信号的数据集：

| 数据集 | 类型 | 规模 | 协议 | 适用性分析 |
| :--- | :--- | :--- | :--- | :--- |
| **MER2023** | 真实世界/噪声视频 | ~67h (含半监督) | CC-BY-NC 4.0 | **极高**。包含噪声与不确定性，非常有利用于验证 HERO 的抗噪性和 Audio 锚点机制。 |
| **CMU-MOSEI** | 独白/演讲 | ~23k 片段 | Apache / CC | **高**。标准 Benchmark，语音清晰，适合作为基准训练数据。 |
| **IEMOCAP** | 表演/交互 | ~12h | 申请许可 | **中**。质量极高但规模小。适合验证捕捉细微表情 (AU Expert) 和语音情感的能力。 |
| **MELD** | 电视剧 (Friends) | ~13k 语句 | CC0 / Open | **中高**。包含多轮对话上下文，适合评估 HERO 的 Reasoning Layer (上下文推理能力)。 |

### 使用建议
*   **预训练 (Pre-training)**: 推荐使用 **MER2023** (含大量未标注数据) 或 **CMU-MOSEI** 构建底层特征对齐能力。
*   **微调 (Fine-tuning)**: 推荐使用 **IEMOCAP** 或 **MELD** 进行特定场景的情感理解评估。