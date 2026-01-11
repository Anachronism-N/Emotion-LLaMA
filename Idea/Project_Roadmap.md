# HERO 项目路线图 (Project Roadmap)

> **基于 Emotion-LLaMA 的 HERO 框架实现**
> 面向多模态情感理解的分层式证据推理与观察框架
> (Hierarchical Evidence-based Reasoning and Observation for Multimodal Emotion Understanding)

---

## 📋 项目概览

### 核心目标
将 Emotion-LLaMA 逐步改造为 HERO 框架，实现：
1. **观测专家层 (Observation Expert Layer)** - MoE 架构的多模态特征提取
2. **证据整合层 (Evidence Integration Layer)** - 全景动态引导注意力
3. **分层推理层 (Hierarchical Reasoning Layer)** - 结构化 CoT 推理

### 当前进度概览

| 阶段 | 状态 | 说明 |
|:-----|:----:|:-----|
| Phase 0: 环境搭建与基线理解 | ✅ 完成 | AU 特征接入已实现 |
| Phase 1: 观测专家层扩展 | ✅ 完成 | 6 Experts + EvidenceDecoder |
| Phase 2: 证据整合层实现 | ✅ 完成 | **AdaptiveQueryGenerator** (3 strategies) |
| Phase 3: 分层推理层改造 | ✅ 完成 | CoT Prompt + Structured Output |
| Phase 4: 训练与评估 | 🟡 就绪 | Feature Extraction Script Ready |
| **Phase 5: 优化与扩展** | ✅ 完成 | 见下方已完成功能列表 |
| **Phase 6: 进阶功能** | ⏳ 待规划 | Multi-Scale Fusion, Augmentation |

#### Phase 5 已完成功能清单

| 功能 | 文件位置 | 说明 |
|:-----|:--------|:-----|
| Evidence Imputation | `evidence_imputation.py` | 缺失模态估计与置信度输出 |
| Temperature Scaling | `integration_layer.py` | 可学习温度参数 |
| MultiModal Contrastive Loss | `hero_loss.py` | 4 对模态对比学习 |
| Modality Entropy Regularizer | `hero_loss.py` | 防止单模态过度依赖 |
| Interpretability Module | `interpretability.py` | 可视化 + CoT 日志 |
| Smart Gradient Checkpointing | `optimization_utils.py` | 冻结 Encoder 兼容 |
| FlashAttention V2 / SDPA | `optimization_utils.py` | 高效注意力计算 |
| QLoRA Setup | `optimization_utils.py` | 4-bit 量化 + LoRA |
| Mixed Precision (AMP) | `optimization_utils.py` | BFloat16 训练 |
| torch.compile | `optimization_utils.py` | PyTorch 2.x 编译优化 |
| **Distributed Training** | `distributed.py` | DDP, FSDP, DeepSpeed |
| **Distributed Inference** | `distributed.py` | 多 GPU 推理引擎 |

---


## 🏗️ 详细实施计划

### Phase 0: 环境搭建与基线理解 ✅

**已完成工作 (详见 [Implementation_Log.md](./Implementation_Log.md)):**

- [x] 1. 接入 AU 特征到 MER2024 数据流
- [x] 2. 让模型支持 4 路特征输入 (`feats_llama_proj1-4`)
- [x] 3. 配置项补充 (`au_feature_dir`)
- [x] 4. AU 特征缺失的鲁棒性处理
- [x] 5. AU 特征维度对齐
- [x] 6. 模态缺失鲁棒性：模态丢弃训练

**当前模型架构理解:**

```
输入: image, video_features (3-4路)
  ↓
visual_encoder (EVA ViT) → image_embeds → llama_proj → image_inputs_llama
  ↓
video_features → feats_llama_proj[1-4] → video_feats
  ↓
concat(image_inputs_llama, video_feats, cls_tk_feats)
  ↓
LLaMA-2-7B → 情感预测
```

---

### Phase 1: 观测专家层扩展 🔄

**目标:** 实现 HERO 的双输出专家结构 (Feature Tensor + Semantic Evidence)

#### 1.1 Q-Former 模块引入 [优先级: 高]

- [ ] **任务 1.1.1**: 为视觉模态添加 Q-Former
  - 参考 BLIP-2/SECap 架构
  - 使用可学习查询向量压缩特征
  - 输出: `[B, 32, 768]` 的 Feature Tensor
  - 文件位置: `minigpt4/models/Qformer.py` (已存在)

- [ ] **任务 1.1.2**: 为音频模态添加 Q-Former
  - 复用相同架构，独立参数
  - 处理 HuBERT 输出的变长序列

- [ ] **任务 1.1.3**: Q-Former 输出头设计
  - Head 1: 特征投影头 (Linear → LLM embedding dim)
  - Head 2: 证据解码头 (轻量级 Transformer Decoder)

#### 1.2 证据解码器实现 [优先级: 高]

- [ ] **任务 1.2.1**: 实现共享的 Evidence Decoder
  - 使用 Task Token 区分模态: `<visual_task>`, `<audio_task>`
  - 输出自然语言描述的语义证据
  - 示例: "视觉证据 (VE-01): 观察到人物面部出现微笑表情 (AU12)"

- [ ] **任务 1.2.2**: 证据模板设计
  - 设计标准化的证据输出格式
  - 包含 AU 编号、置信度等结构化信息

#### 1.3 Synergy Expert (协同专家) [优先级: 中]

- [ ] **任务 1.3.1**: 实现音画协同感知模块
  - 检测音画同步性/冲突
  - 用于识别反讽、苦笑等情况
  - 输出: 同步性分数 + 冲突标记

**涉及文件:**
- `minigpt4/models/minigpt_v2.py` - 主要修改
- `minigpt4/models/observation_experts.py` - **新建**
- `minigpt4/models/evidence_decoder.py` - **新建**

---

### Phase 2: 证据整合层实现 ⏳

**目标:** 实现全景动态引导注意力 (Panoramic Dynamic Guided Attention)

#### 2.1 全局查询生成 [优先级: 高]

- [ ] **任务 2.1.1**: 实现 Summary 向量聚合
  - 收集各专家的 `[CLS]` Token 或 Global Average Pooling 结果
  - 包含: `v_vis_global`, `v_vis_motion`, `v_vis_au`, `v_audio`, `v_text`, `v_synergy`

- [ ] **任务 2.1.2**: 生成全局查询向量 $Q_{global}$
  - 轻量级 Transformer Encoder 融合所有 summary 向量
  - 输出: 包含全模态信息的统一查询

#### 2.2 K-Bank 构建与注意力计算 [优先级: 高]

- [ ] **任务 2.2.1**: 构建细节特征库 K-Bank
  - 拼接所有单模态专家的完整特征序列
  - 形状: `[B, Total_Seq_Len, Dim]`

- [ ] **任务 2.2.2**: 实现 Panoramic-Guided Attention
  - 使用 $Q_{global}$ 检索 K-Bank 中的关键信息
  - 生成: `Integrated_Context_Tensor` + `Dynamic_Attention_Weights`

#### 2.3 模态缺失鲁棒性机制 [优先级: 中]

- [ ] **任务 2.3.1**: 训练时 - 隐式表征对齐
  - 实现 Teacher-Student 架构
  - 多模态融合专家作为"教师"
  - 使用 Modality Dropout + KL 散度损失

- [ ] **任务 2.3.2**: 推理时 - 显式证据补全
  - 轻量级证据补全模块
  - 基于可用证据推断缺失证据

**涉及文件:**
- `minigpt4/models/integration_layer.py` - **新建**
- `minigpt4/models/minigpt_v2.py` - 调用整合层

---

### Phase 3: 分层推理层改造 ⏳

**目标:** 实现结构化 CoT 推理与 JSON 输出

#### 3.1 混合式输入嵌入 [优先级: 高]

- [ ] **任务 3.1.1**: 特征投影与注入
  - Linear Projector: `Integrated_Context_Tensor` → LLM embedding
  - 作为 `<visual_audio_token>` 插入 Prompt 开头

- [ ] **任务 3.1.2**: 文本证据拼接
  - 按注意力权重排序语义证据
  - 拼接在特征 Token 之后

#### 3.2 结构化指令微调 [优先级: 高]

- [ ] **任务 3.2.1**: 设计 JSON 输出格式
  ```json
  {
    "emotion_caption": "情感描述",
    "evidence_summary": ["证据1", "证据2"],
    "reasoning_process": "推理逻辑",
    "final_emotion": "情感标签"
  }
  ```

- [ ] **任务 3.2.2**: 构造指令微调数据
  - 扩展 MERR 数据集
  - 添加 CoT 推理标注

#### 3.3 LLM 基座可选替换 [优先级: 低]

- [ ] **任务 3.3.1**: 支持 Qwen-2-7B-Instruct (可选)
  - 替换 LLaMA-2-7B
  - 调整 Tokenizer 和生成配置

**涉及文件:**
- `minigpt4/models/minigpt_base.py` - 修改 `preparing_embedding`, `forward`
- `minigpt4/conversation/conversation.py` - Prompt 模板

---

### Phase 4: 训练与评估 ⏳

**目标:** 实现三阶段渐进式训练

#### 4.1 Stage 1: 模态解纠缠与表征对齐

- [ ] **任务 4.1.1**: 实现 ITC 损失 (Image/Audio-Text Contrastive)
- [ ] **任务 4.1.2**: 实现 STMIL 损失 (Speech-Text Mutual Information Learning)
- [ ] **任务 4.1.3**: 实现 Synergy 预训练损失

```math
\mathcal{L}_{Stage1} = \mathcal{L}_{ITC} + \lambda_1 \mathcal{L}_{STMIL} + \lambda_2 \mathcal{L}_{Synergy}
```

#### 4.2 Stage 2: 生成式情感预训练

- [ ] **任务 4.2.1**: 实现 Caption Generation 损失
- [ ] **任务 4.2.2**: 实现 SCCL 损失 (Speech-Caption Contrastive)
- [ ] **任务 4.2.3**: 实现 KL 散度损失 (鲁棒性)

```math
\mathcal{L}_{Stage2} = \mathcal{L}_{Gen} + \lambda_3 \mathcal{L}_{SCCL} + \lambda_4 \mathcal{L}_{KL}
```

#### 4.3 Stage 3: 全监督指令微调

- [ ] **任务 4.3.1**: 结构化 CoT 推理训练
- [ ] **任务 4.3.2**: 反事实样本训练 (音画冲突)

```math
\mathcal{L}_{Stage3} = \mathcal{L}_{Struct\_Gen}
```

#### 4.4 评估指标实现

- [ ] **任务 4.4.1**: 性能指标 (WAF, Accuracy, F1)
- [ ] **任务 4.4.2**: 鲁棒性指标 (Noise Drop Rate, Sarcasm Detection)
- [ ] **任务 4.4.3**: 生成质量指标 (CIDEr, SPICE, LLM-as-a-Judge)

**涉及文件:**
- `minigpt4/common/hero_losses.py` - **新建**
- `minigpt4/runners/runner_base.py` - 训练逻辑修改
- `eval_hero.py` - **新建**

---

## 📁 项目文件结构 (预期)

```
Emotion-LLaMA/
├── minigpt4/
│   ├── models/
│   │   ├── minigpt_v2.py          # [修改] 主模型入口
│   │   ├── observation_experts.py # [新建] 观测专家层
│   │   ├── evidence_decoder.py    # [新建] 证据解码器
│   │   ├── integration_layer.py   # [新建] 证据整合层
│   │   ├── hero_model.py          # [新建] HERO 主模型
│   │   └── Qformer.py             # [现有] Q-Former 模块
│   ├── common/
│   │   └── hero_losses.py         # [新建] 损失函数
│   └── datasets/
│       └── datasets/
│           └── mer2024.py         # [修改] 数据加载
├── Idea/
│   ├── Idea.md                    # 原始 idea 文档
│   ├── Project_Roadmap.md         # 本文档
│   └── Implementation_Log.md      # 实现记录
├── train_configs/
│   └── hero_*.yaml                # [新建] HERO 训练配置
└── eval_hero.py                   # [新建] HERO 评估脚本
```

---

## ⏱️ 时间线估算

| 阶段 | 预计时间 | 里程碑 |
|:-----|:--------:|:-------|
| Phase 1 | 2-3 周 | Q-Former + Evidence Decoder 完成 |
| Phase 2 | 2 周 | 证据整合层完成 |
| Phase 3 | 1-2 周 | 结构化推理改造完成 |
| Phase 4 | 3-4 周 | 三阶段训练 + 评估 |
| **总计** | **8-11 周** | HERO v1.0 完成 |

---

## 🔗 相关资源

- **Idea 原文**: [Idea.md](./Idea.md)
- **实现记录**: [Implementation_Log.md](./Implementation_Log.md)
- **Emotion-LLaMA 论文**: [arXiv](https://arxiv.org/pdf/2406.11161)
- **BLIP-2 论文**: [arXiv](https://arxiv.org/abs/2301.12597)
- **MER2024 Challenge**: [官方网站](http://merchallenge.cn/)

---

*最后更新: 2026-01-10*

---

## 🚀 Phase 5: 优化与扩展计划 (未实现功能 & 潜在改进)

本节基于 `Idea.md` 与当前实现的对齐检查结果，列出尚未实现的功能以及可优化方向。

### 5.1 未实现功能 (Gap Analysis)

| 功能 | Idea.md 位置 | 当前状态 | 优先级 |
| :--- | :--- | :--- | :--- |
| **显式证据补全 (Evidence Imputation)** | Line 188 | 未实现 | 🔴 高 |
| **EvidenceDecoder 的细粒度文本训练数据** | Pillar 1 | 缺少监督信号 | 🔴 高 |
| **OpenFace AU 特征实时集成** | Pillar 1 | Placeholder (Zeros) | 🟡 中 |
| **LLM 情感词汇扩展 (Tokenizer)** | Pillar 3 | 使用默认 LLaMA | 🟢 低 |

### 5.2 可优化方向 (Optimization Proposals)

#### A. 动态阈值更新 (Dynamic Threshold for Scorer)
*   **现状**: `AdaptiveQueryGenerator (dynamic)` 中的 Scorer 输出的是绝对分数。
*   **优化**: 引入 **Temperature Scaling** (如 `softmax(scores / T)`) 或者 **Top-K Gating**，允许模型只关注前 K 个最信任的模态。
*   **预期收益**: 提升在极端噪声场景下的鲁棒性。

#### B. 梯度检查点 (Gradient Checkpointing for Memory)
*   **现状**: 6 个 Q-Former + LLM 可能导致显存打满。
*   **优化**: 在 `ModalityQFormer` 和 `HEROModel` 中启用 `torch.utils.checkpoint.checkpoint` 来换时间与内存。
*   **预期收益**: Batch Size 可以增加 2-3 倍。

#### C. LoRA 微调集成
*   **现状**: LLM 全参数训练。
*   **优化**: 集成 PEFT 库，对 LLM 的 QKV 层应用 LoRA。
*   **预期收益**: 训练时间缩短 40% 以上，显存占用降低。

#### D. 证据补全模块 (Implement Evidence Imputation)
*   **现状**: 未实现。
*   **实现方案**:
    1.  训练一个轻量级 Transformer Decoder，以其他模态的 Summary 为输入。
    2.  输出: 缺失模态的估计 Summary 向量。
    3.  可还可以输出说明文本，如 `[IAE-01]: 推断音频情感上扬`。
*   **训练数据**: 使用 Modality Dropout 生成的 (Teacher-Output, Dropped-Input) 对进行监督。

---
