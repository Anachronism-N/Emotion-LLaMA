"""
HERO: Hierarchical Evidence-based Reasoning and Observation
面向多模态情感理解的分层式证据推理与观察框架

This package implements the HERO framework for multimodal emotion understanding.
"""

from minigpt4.models.hero.observation_experts import (
    ObservationExpertLayer,
    ModalityQFormer,
    SynergyExpert,
    ExpertOutput,
)

from minigpt4.models.hero.integration_layer import (
    EvidenceIntegrationLayer,
    AdaptiveQueryGenerator,
    PanoramicGuidedAttention,
    ModalityDropoutTrainer,
    IntegrationOutput,
)

from minigpt4.models.hero.hero_model import (
    HEROModel,
    HEROOutput,
)

from minigpt4.models.hero.evidence_imputation import (
    EvidenceImputationModule,
    ImputationLoss,
    ImputationOutput,
)

from minigpt4.models.hero.hero_loss import (
    STMIL_Loss,
    SCCL_Loss,
    MultiModalContrastiveLoss,
    ModalityEntropyRegularizer,
)

from minigpt4.models.hero.optimization_utils import (
    SmartCheckpointWrapper,
    enable_smart_checkpointing,
    ScaledDotProductAttention,
    setup_qlora,
    AMPTrainingContext,
    compile_model,
    setup_hero_optimizations,
)

from minigpt4.models.hero.interpretability import (
    InterpretabilityModule,
    PredictionRecord,
    AttentionVisualizerHook,
)

from minigpt4.models.hero.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_ddp,
    wrap_model_fsdp,
    get_deepspeed_config,
    initialize_deepspeed,
    DistributedInferenceEngine,
    get_distributed_sampler,
)


__all__ = [
    # Observation Expert Layer
    "ObservationExpertLayer",
    "ModalityQFormer", 
    "SynergyExpert",
    "ExpertOutput",
    
    # Integration Layer
    "EvidenceIntegrationLayer",
    "AdaptiveQueryGenerator",
    "PanoramicGuidedAttention",
    "ModalityDropoutTrainer",
    "IntegrationOutput",
    
    # Evidence Imputation
    "EvidenceImputationModule",
    "ImputationLoss",
    "ImputationOutput",
    
    # Loss Functions
    "STMIL_Loss",
    "SCCL_Loss",
    "MultiModalContrastiveLoss",
    "ModalityEntropyRegularizer",
    
    # Engineering Optimizations
    "SmartCheckpointWrapper",
    "enable_smart_checkpointing",
    "ScaledDotProductAttention",
    "setup_qlora",
    "AMPTrainingContext",
    "compile_model",
    "setup_hero_optimizations",
    
    # Interpretability
    "InterpretabilityModule",
    "PredictionRecord",
    "AttentionVisualizerHook",
    
    # Distributed Training & Inference
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "get_deepspeed_config",
    "initialize_deepspeed",
    "DistributedInferenceEngine",
    "get_distributed_sampler",
    
    # Main Model
    "HEROModel",
    "HEROOutput",
]
