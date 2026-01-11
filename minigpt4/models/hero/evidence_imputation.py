"""
HERO Evidence Imputation Module
证据补全模块实现

This module provides the ability to estimate missing modality features
based on available modalities during inference. It uses a lightweight
Transformer Encoder to model cross-modality relationships.

Reference: HERO Idea.md Section 2.2 (Explicit Evidence Imputation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImputationOutput:
    """Output from the Evidence Imputation Module."""
    imputed_summaries: torch.Tensor  # [B, N_missing, Dim]
    confidence_scores: torch.Tensor  # [B, N_missing]
    imputed_indices: List[int]       # Which modality indices were imputed


class EvidenceImputationModule(nn.Module):
    """
    Evidence Imputation Module for handling missing modalities.
    
    When a modality is missing during inference, this module estimates
    what that modality's summary vector would have been based on the
    available modalities.
    
    Architecture:
        1. Modality Positional Embeddings (learnable)
        2. Transformer Encoder (cross-modality context modeling)
        3. Imputation Head (projects context to missing modality space)
        4. Confidence Head (estimates reliability of imputation)
    
    Args:
        hidden_dim: Dimension of summary vectors.
        num_modalities: Total number of possible modalities.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_modalities: int = 6,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Learnable modality position embeddings
        # Each modality gets a unique embedding to identify it
        self.modality_embedding = nn.Embedding(num_modalities, hidden_dim)
        
        # Transformer Encoder to model cross-modality relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Imputation head: generates the imputed summary
        self.impute_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Confidence head: estimates how reliable the imputation is
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Output layer norm
        self.output_ln = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        available_summaries: torch.Tensor,
        available_indices: List[int],
        missing_indices: List[int],
    ) -> ImputationOutput:
        """
        Impute missing modality summaries based on available ones.
        
        Args:
            available_summaries: [B, N_available, Dim] - Summaries of available modalities
            available_indices: List of modality indices that are available
            missing_indices: List of modality indices that need imputation
            
        Returns:
            ImputationOutput containing imputed summaries and confidence scores
        """
        B, N_avail, D = available_summaries.shape
        device = available_summaries.device
        N_miss = len(missing_indices)
        
        if N_miss == 0:
            # No missing modalities, return empty tensors
            return ImputationOutput(
                imputed_summaries=torch.zeros(B, 0, D, device=device),
                confidence_scores=torch.ones(B, 0, device=device),
                imputed_indices=[],
            )
        
        # Step 1: Add modality positional embeddings to available summaries
        avail_pos = torch.tensor(available_indices, device=device, dtype=torch.long)
        pos_emb = self.modality_embedding(avail_pos)  # [N_avail, D]
        x = available_summaries + pos_emb.unsqueeze(0)  # [B, N_avail, D]
        
        # Step 2: Encode cross-modality context
        context = self.encoder(x)  # [B, N_avail, D]
        
        # Step 3: Generate query embeddings for missing modalities
        miss_pos = torch.tensor(missing_indices, device=device, dtype=torch.long)
        missing_queries = self.modality_embedding(miss_pos)  # [N_miss, D]
        missing_queries = missing_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N_miss, D]
        
        # Step 4: Cross-attention - missing queries attend to encoded context
        # Using scaled dot-product attention
        # Q: missing_queries [B, N_miss, D]
        # K, V: context [B, N_avail, D]
        attn_weights = torch.matmul(
            missing_queries, context.transpose(-2, -1)
        ) / (D ** 0.5)  # [B, N_miss, N_avail]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attended_context = torch.matmul(attn_weights, context)  # [B, N_miss, D]
        
        # Step 5: Combine query embeddings with attended context
        combined = missing_queries + attended_context  # [B, N_miss, D]
        
        # Step 6: Apply imputation head
        imputed = self.impute_head(combined)  # [B, N_miss, D]
        imputed = self.output_ln(imputed)
        
        # Step 7: Compute confidence scores
        confidence = torch.sigmoid(
            self.confidence_head(imputed).squeeze(-1)
        )  # [B, N_miss]
        
        return ImputationOutput(
            imputed_summaries=imputed,
            confidence_scores=confidence,
            imputed_indices=missing_indices,
        )


class ImputationLoss(nn.Module):
    """
    Loss function for training the Evidence Imputation Module.
    
    Uses MSE loss between imputed summaries and teacher (ground truth) summaries,
    weighted by the confidence scores.
    """
    
    def __init__(self, confidence_weight: float = 0.1):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(
        self,
        imputed: torch.Tensor,       # [B, N_miss, D]
        teacher: torch.Tensor,       # [B, N_miss, D]
        confidence: torch.Tensor,    # [B, N_miss]
    ) -> torch.Tensor:
        """
        Compute imputation loss.
        
        The loss encourages:
        1. Imputed summaries to match teacher summaries (MSE)
        2. Model to be confident when imputation is accurate (confidence calibration)
        """
        # MSE loss per sample
        mse_loss = self.mse(imputed, teacher).mean(dim=-1)  # [B, N_miss]
        
        # Reconstruction loss
        recon_loss = mse_loss.mean()
        
        # Confidence calibration: confidence should be high when error is low
        # We use negative correlation as a regularizer
        error_normalized = mse_loss / (mse_loss.max() + 1e-8)
        confidence_loss = (confidence * error_normalized).mean()
        
        total_loss = recon_loss + self.confidence_weight * confidence_loss
        
        return total_loss
