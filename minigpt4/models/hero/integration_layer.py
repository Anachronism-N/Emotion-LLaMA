"""
HERO Evidence Integration Layer
证据整合层实现 (Refactored Phase 1 alignment)

This module implements the Evidence Integration Layer for the HERO framework,
strictly aligned with Idea.md Section 2.2.

Key Components:
1. AudioGuidedAttention: Implementation of "Audio Anchoring" where Audio features 
   guide the query generation to "cross-examine" other modalities.
2. PanoramicDynamicGuidedAttention: The core fusion mechanism.
3. ModalityDropoutTrainer: Robustness training.

Reference: HERO Idea.md Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IntegrationOutput:
    """Output from the Evidence Integration Layer."""
    integrated_context: torch.Tensor  # [B, num_queries, llm_dim]
    attention_weights: torch.Tensor   # [B, total_seq_len]
    global_query: torch.Tensor        # [B, hidden_dim]
    modality_importance: Dict[str, torch.Tensor]  # Per-modality attention scores


class AudioGuidedQueryGenerator(nn.Module):
    """
    Audio-Guided Query Generator (Implements 'Audio Anchoring').
    
    Instead of a flat self-attention over all summaries, this module uses 
    Audio summary as the primary 'Anchor' query to attend to other modalities.
    This aligns with the idea that audio leaks the most 'truthful' emotion.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        max_experts: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Audio serves as Query
        # Visual/Text/AU etc. serve as Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_ln = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        summary_vectors: torch.Tensor,
        audio_idx: int,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            summary_vectors: [B, num_experts, H]
            audio_idx: Index of the audio expert in the summary vectors
        """
        # Extract Audio Anchor
        audio_anchor = summary_vectors[:, audio_idx:audio_idx+1, :] # [B, 1, H]
        
        # Others (including audio itself is fine, or exclude it? usually keep it for self-reinforcement)
        evidence_bank = summary_vectors # [B, N, H]
        
        key_padding_mask = None
        if modality_mask is not None:
             key_padding_mask = (modality_mask == 0)

        # Audio 'interrogates' the evidence bank
        attn_out, attn_weights = self.cross_attn(
            query=audio_anchor,
            key=evidence_bank,
            value=evidence_bank,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + Norm
        x = self.ln(audio_anchor + attn_out)
        
        # FFN
        x = self.output_ln(x + self.ffn(x))
        
        # Squeeze to get global query vector [B, H]
        global_query = x.squeeze(1)
        
        # Attention weights [B, 1, N] -> [B, N]
        expert_attention = attn_weights.squeeze(1)
        
        return global_query, expert_attention


class PanoramicGuidedAttention(nn.Module):
    """
    Panoramic Dynamic Guided Attention (全景动态引导注意力)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        llm_dim: int = 4096,
        num_heads: int = 8,
        num_output_queries: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.llm_dim = llm_dim
        self.num_output_queries = num_output_queries
        
        # Learnable output queries
        self.output_queries = nn.Parameter(
            torch.zeros(1, num_output_queries, hidden_dim)
        )
        nn.init.normal_(self.output_queries, std=0.02)
        
        # Global query integration
        self.global_query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-attention: output queries attend to K-bank
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Final projection to LLM dimension
        self.output_proj = nn.Linear(hidden_dim, llm_dim)
        
    def forward(
        self,
        global_query: torch.Tensor,
        k_bank: torch.Tensor,
        k_bank_mask: Optional[torch.Tensor] = None,
        modality_boundaries: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size = global_query.shape[0]
        
        # Expand output queries
        queries = self.output_queries.expand(batch_size, -1, -1)
        
        # Integrate global query into output queries
        global_query_expanded = self.global_query_proj(global_query).unsqueeze(1)
        queries = queries + global_query_expanded
        
        # Cross-attention to K-bank
        key_padding_mask = None
        if k_bank_mask is not None:
            key_padding_mask = (k_bank_mask == 0)

        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=k_bank,
            value=k_bank,
            key_padding_mask=key_padding_mask,
        )
        queries = self.ln1(queries + attn_out)
        
        # Self-attention for refinement
        self_attn_out, _ = self.self_attn(
            query=queries, key=queries, value=queries,
        )
        queries = self.ln2(queries + self_attn_out)
        
        # FFN
        queries = self.ln3(queries + self.ffn(queries))
        
        # Project to LLM dimension
        integrated_context = self.output_proj(queries)
        
        # Compute attention weights
        attention_weights = attn_weights.mean(dim=1)  # [B, total_seq_len]
        
        # Compute per-modality importance
        modality_importance = {}
        if modality_boundaries is not None:
            # Order must match EvidenceIntegrationLayer.modality_order
            modality_names = ['vis_global', 'vis_motion', 'audio', 'au', 'text', 'synergy']
            start_idx = 0
            for i, end_idx in enumerate(modality_boundaries):
                if i < len(modality_names):
                    modality_attn = attention_weights[:, start_idx:end_idx].sum(dim=1)
                    modality_importance[modality_names[i]] = modality_attn
                start_idx = end_idx
        
        return integrated_context, attention_weights, modality_importance


class EvidenceIntegrationLayer(nn.Module):
    """
    Evidence Integration Layer with Audio-Guided Attention.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        llm_dim: int = 4096,
        num_heads: int = 8,
        num_output_queries: int = 64,
        max_experts: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.modality_order = ['vis_global', 'vis_motion', 'audio', 'au', 'text', 'synergy']
        
        # Audio Guided Query Generator
        self.query_generator = AudioGuidedQueryGenerator(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_experts=max_experts,
            dropout=dropout,
        )
        
        # Panoramic guided attention
        self.panoramic_attention = PanoramicGuidedAttention(
            hidden_dim=hidden_dim,
            llm_dim=llm_dim,
            num_heads=num_heads,
            num_output_queries=num_output_queries,
            dropout=dropout,
        )
        
    def forward(
        self,
        expert_features: Dict[str, torch.Tensor],
        summary_vectors: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> IntegrationOutput:
        
        # Identify Audio Index
        audio_idx = 2 # Default position in order
        if 'audio' in self.modality_order:
             audio_idx = self.modality_order.index('audio')
             
        # Generate global query (Audio Guided)
        global_query, expert_attention = self.query_generator(
            summary_vectors, audio_idx, modality_mask
        )
        
        # Build K-bank
        k_bank_parts = []
        boundaries = []
        current_len = 0
        
        for modality in self.modality_order:
            if modality in expert_features:
                feat = expert_features[modality]
                k_bank_parts.append(feat)
                current_len += feat.shape[1]
                boundaries.append(current_len)
        
        if not k_bank_parts:
             # Handle empty case (should not happen with proper mask)
             device = summary_vectors.device
             k_bank = torch.zeros(summary_vectors.shape[0], 1, self.query_generator.hidden_dim, device=device)
             k_bank_mask = None
        else:
            k_bank = torch.cat(k_bank_parts, dim=1)
            
            # Generate k_bank_mask
            k_bank_mask = None
            if modality_mask is not None:
                batch_size = modality_mask.shape[0]
                mask_parts = []
                for i, modality in enumerate(self.modality_order):
                    if modality in expert_features:
                        feat_len = expert_features[modality].shape[1]
                        m = modality_mask[:, i:i+1]
                        mask_parts.append(m.expand(-1, feat_len))
                if mask_parts:
                    k_bank_mask = torch.cat(mask_parts, dim=1)
        
        # Apply panoramic attention
        integrated_context, attention_weights, modality_importance = self.panoramic_attention(
            global_query=global_query,
            k_bank=k_bank,
            k_bank_mask=k_bank_mask,
            modality_boundaries=boundaries,
        )
        
        return IntegrationOutput(
            integrated_context=integrated_context,
            attention_weights=attention_weights,
            global_query=global_query,
            modality_importance=modality_importance,
        )


class ModalityDropoutTrainer(nn.Module):
    """
    Modality Dropout Training Wrapper.
    """
    def __init__(self, integration_layer: EvidenceIntegrationLayer, dropout_prob: float = 0.3):
        super().__init__()
        self.integration_layer = integration_layer
        self.dropout_prob = dropout_prob
        
    def forward(
        self,
        expert_features: Dict[str, torch.Tensor],
        summary_vectors: torch.Tensor,
        training: bool = True,
    ) -> Tuple[IntegrationOutput, Optional[torch.Tensor]]:
        
        batch_size = summary_vectors.shape[0]
        num_experts = summary_vectors.shape[1]
        device = summary_vectors.device
        
        if training and self.dropout_prob > 0:
            # Teacher
            teacher_output = self.integration_layer(expert_features, summary_vectors, modality_mask=None)
            
            # Student (Dropout)
            dropout_mask = torch.ones(batch_size, num_experts, device=device)
            # Ensure Audio (Anchor) is kept? Idea.md implies Audio is crucial.
            # But robust training might want to drop Audio too.
            # Let's simple random drop for now.
            for b in range(batch_size):
                num_to_drop = 0
                for i in range(num_experts):
                    if torch.rand(1).item() < self.dropout_prob:
                        num_to_drop += 1
                if num_to_drop >= num_experts:
                    num_to_drop = num_experts - 1
                drop_indices = torch.randperm(num_experts)[:num_to_drop]
                dropout_mask[b, drop_indices] = 0
                
            # Mask features
            dropped_features = {}
            # We need to map index back to key using integration_layer.modality_order
            order = self.integration_layer.modality_order
            
            for idx, key in enumerate(order):
                if key in expert_features:
                    mask = dropout_mask[:, idx:idx+1, None].expand_as(expert_features[key])
                    dropped_features[key] = expert_features[key] * mask
            
            dropped_summaries = summary_vectors * dropout_mask.unsqueeze(-1)
            
            student_output = self.integration_layer(
                dropped_features, dropped_summaries, modality_mask=dropout_mask
            )
            
            # KL Loss
            teacher_logits = teacher_output.integrated_context
            student_logits = student_output.integrated_context
            kl_loss = F.mse_loss(student_logits, teacher_logits.detach())
            
            return student_output, kl_loss
            
        else:
            return self.integration_layer(expert_features, summary_vectors, modality_mask=None), None
