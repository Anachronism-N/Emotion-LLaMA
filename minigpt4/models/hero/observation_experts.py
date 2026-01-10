"""
HERO Observation Expert Layer 
观测专家层实现 (Refactored Phase 1 alignment)

This module implements the Observation Expert Layer for the HERO framework,
strictly aligned with Idea.md Section 2.1.

Key Components:
1. ModalityQFormer: Extracts Feature Tensor and Pooled Feature.
2. EvidenceDecoder: Translates features into Semantic Evidence (Text).
3. Experts:
    - Visual Global (CLIP)
    - Visual Motion (VideoMAE)
    - Visual AU (OpenFace)
    - Audio (HuBERT)
    - Text (ASR)
    - Synergy (Audio-Visual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from minigpt4.models.Qformer import BertConfig, BertLMHeadModel

@dataclass
class ExpertOutput:
    """Output from a single observation expert."""
    feature_tensor: torch.Tensor           # [B, num_queries, hidden_dim]
    pooled_feature: torch.Tensor           # [B, hidden_dim]
    semantic_evidence: Optional[List[str]] = None # Textual evidence description
    attention_weights: Optional[torch.Tensor] = None

class EvidenceDecoder(nn.Module):
    """
    Lightweight decoder to generate Semantic Evidence text from features.
    Simulates a small generative model (e.g., T5-small or Decoder-only).
    For Phase 4, this will be trained to caption the features.
    """
    def __init__(self, input_dim: int, vocab_size: int = 30522, max_len: int = 32):
        super().__init__()
        self.max_len = max_len
        # Simple projection to vocab logits for now (placeholder for full decoder)
        self.decoder = nn.Linear(input_dim, vocab_size)
    
    def forward(self, feature_tensor: torch.Tensor) -> List[str]:
        """
        Generate semantic evidence string.
        Since we don't have a tokenizer here, we return a placeholder string
        during the pre-training phase or when not utilizing a real tokenizer.
        """
        batch_size = feature_tensor.size(0)
        # In a real implementation, this would generate tokens -> decode -> string.
        # For now, we return a structured placeholder that will be filled by the training loop/evaluator.
        return [f"Evidence placeholder for batch {i}" for i in range(batch_size)]

class ModalityQFormer(nn.Module):
    """
    Q-Former + Evidence Decoder.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        cross_attention_freq: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Q-Former config
        qformer_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            vocab_size=30522,
            add_cross_attention=True,
            cross_attention_freq=cross_attention_freq,
            encoder_width=hidden_dim,
        )
        self.qformer = BertLMHeadModel(qformer_config)
        self.output_ln = nn.LayerNorm(hidden_dim)
        
        # Evidence Decoder Head
        self.evidence_decoder = EvidenceDecoder(hidden_dim)
        
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> ExpertOutput:
        
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        
        # Project and Norm
        encoder_hidden_states = self.input_proj(encoder_hidden_states)
        encoder_hidden_states = self.input_ln(encoder_hidden_states)
        
        # Expand queries and mask
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=torch.long, device=device
            )
        
        # Q-Former Forward
        outputs = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        
        # Extract features
        query_output = outputs.last_hidden_state[:, :self.num_queries, :]
        query_output = self.output_ln(query_output)
        pooled_feature = query_output.mean(dim=1)
        
        # Generate Text Evidence
        evidence_text = self.evidence_decoder(pooled_feature)
        
        attn_weights = None
        if outputs.cross_attentions is not None and len(outputs.cross_attentions) > 0:
            attn_weights = outputs.cross_attentions[-1]
        
        return ExpertOutput(
            feature_tensor=query_output,
            pooled_feature=pooled_feature,
            semantic_evidence=evidence_text,
            attention_weights=attn_weights,
        )

class SynergyExpert(nn.Module):
    """
    Synergy Expert (Audio-Visual Conflict Detection).
    """
    def __init__(self, audio_dim: int=768, visual_dim: int=768, hidden_dim: int=768, num_heads: int=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.audio_to_visual_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.visual_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_ln = nn.LayerNorm(hidden_dim)
        self.conflict_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.evidence_decoder = EvidenceDecoder(hidden_dim)
        
    def forward(self, audio_features: torch.Tensor, visual_features: torch.Tensor) -> ExpertOutput:
        # Cross Attention
        audio_attended, attn_a2v = self.audio_to_visual_attn(audio_features, visual_features, visual_features)
        visual_attended, attn_v2a = self.visual_to_audio_attn(visual_features, audio_features, audio_features)
        
        # Pooling & Fusion
        audio_pooled = audio_attended.mean(dim=1)
        visual_pooled = visual_attended.mean(dim=1)
        fused = torch.cat([audio_pooled, visual_pooled], dim=-1)
        fused = self.fusion_ffn(fused)
        fused = self.output_ln(fused)
        
        # Conflict Score & Evidence
        conflict_score = self.conflict_head(fused) # Can be used in loss
        feature_tensor = fused.unsqueeze(1) # [B, 1, H]
        evidence_text = self.evidence_decoder(fused)
        
        return ExpertOutput(
            feature_tensor=feature_tensor,
            pooled_feature=fused,
            semantic_evidence=evidence_text,
            attention_weights=attn_a2v,
        )

class ObservationExpertLayer(nn.Module):
    """
    Refactored Observation Expert Layer aligned with Idea.md.
    Experts:
    1. Visual Global (CLIP)
    2. Visual Motion (VideoMAE)
    3. Visual AU (OpenFace)
    4. Audio (HuBERT)
    5. Text (ASR)
    6. Synergy
    """
    def __init__(
        self,
        visual_dim: int = 1408, # CLIP L/14
        video_dim: int = 1024,  # VideoMAE
        audio_dim: int = 1024,  # HuBERT
        au_dim: int = 1024,     # OpenFace mapped
        text_dim: int = 768,    # ASR/BERT emb
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_qformer_layers: int = 2,
        llm_hidden_dim: int = 4096,
        include_synergy: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_synergy = include_synergy
        
        # Store dims for external access (e.g. HEROModel slicing)
        self.visual_dim = visual_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.au_dim = au_dim
        self.text_dim = text_dim
        
        # 1. Visual Global Expert (CLIP)
        self.visual_global_expert = ModalityQFormer(visual_dim, hidden_dim, num_queries, num_qformer_layers)
        self.visual_global_proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # 2. Visual Motion Expert (VideoMAE)
        self.visual_motion_expert = ModalityQFormer(video_dim, hidden_dim, num_queries, num_qformer_layers)
        self.visual_motion_proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # 3. AU Expert
        self.au_expert = ModalityQFormer(au_dim, hidden_dim, num_queries, num_qformer_layers)
        self.au_proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # 4. Audio Expert
        self.audio_expert = ModalityQFormer(audio_dim, hidden_dim, num_queries, num_qformer_layers)
        self.audio_proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # 5. Text Expert (ASR)
        self.text_expert = ModalityQFormer(text_dim, hidden_dim, num_queries, num_qformer_layers)
        self.text_proj = nn.Linear(hidden_dim, llm_hidden_dim)
        
        # 6. Synergy Expert
        if include_synergy:
            self.synergy_expert = SynergyExpert(hidden_dim, hidden_dim, hidden_dim)
            self.synergy_proj = nn.Linear(hidden_dim, llm_hidden_dim)
            
    def forward(
        self,
        vis_global_feats: torch.Tensor,
        vis_motion_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        au_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, ExpertOutput]:
        
        outputs = {}
        
        # Process basic experts
        outputs['vis_global'] = self.visual_global_expert(vis_global_feats)
        outputs['vis_motion'] = self.visual_motion_expert(vis_motion_feats)
        outputs['audio'] = self.audio_expert(audio_feats)
        
        if au_feats is not None:
            outputs['au'] = self.au_expert(au_feats)
            
        if text_feats is not None:
            outputs['text'] = self.text_expert(text_feats)
            
        # Synergy
        if self.include_synergy:
            # Synergy inputs: aligned features from Motion and Audio experts (or raw features if dim matches)
            # Here we reuse the Feature Tensors from experts as "raw" input to synergy might be too high dim
            # But SynergyExpert expects raw inputs usually. For now, let's use the Expert Features (projected to hidden_dim)
            # actually SynergyExpert expects `hidden_dim` inputs in its constructor.
            # So we pass the Q-Former outputs of Motion and Audio.
            
            outputs['synergy'] = self.synergy_expert(
                audio_features=outputs['audio'].feature_tensor,
                visual_features=outputs['vis_motion'].feature_tensor
            )
            
        return outputs

    def project_to_llm(self, expert_outputs: Dict[str, ExpertOutput]) -> Dict[str, torch.Tensor]:
        projected = {}
        # Simple mapping
        mapping = {
            'vis_global': self.visual_global_proj,
            'vis_motion': self.visual_motion_proj,
            'au': self.au_proj,
            'audio': self.audio_proj,
            'text': self.text_proj,
            'synergy': self.synergy_proj if self.include_synergy else None
        }
        
        for name, proj_layer in mapping.items():
            if name in expert_outputs and proj_layer is not None:
                projected[name] = proj_layer(expert_outputs[name].feature_tensor)
                
        return projected

    def get_summary_vectors(self, expert_outputs: Dict[str, ExpertOutput]) -> torch.Tensor:
        # Order matters!
        keys = ['vis_global', 'vis_motion', 'audio', 'au', 'text', 'synergy']
        summaries = []
        for k in keys:
            if k in expert_outputs:
                summaries.append(expert_outputs[k].pooled_feature)
        return torch.stack(summaries, dim=1) if summaries else torch.empty(0)
