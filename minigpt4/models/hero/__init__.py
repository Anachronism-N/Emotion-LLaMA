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
    
    # Main Model
    "HEROModel",
    "HEROOutput",
]
