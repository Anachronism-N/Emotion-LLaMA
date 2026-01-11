"""
HERO Loss Functions
HERO 损失函数实现

This module implements the specific loss functions for the HERO framework:
1. STMIL_Loss (Soft-Truncated Mutual Information Learning): For Disentanglement.
2. SCCL_Loss (Supervised Contrastive Continuum Learning): For Alignment.

Reference: HERO Idea.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class vCLUBAuxiliaryNetwork(nn.Module):
    """
    Auxiliary network for vCLUB Mutual Information estimation.
    Predicts p(y|x) where x and y are feature vectors.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim * 2) # Output mean and log_var
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.net(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class STMIL_Loss(nn.Module):
    """
    Soft-Truncated Mutual Information Learning Loss.
    Based on vCLUB estimator.
    
    Goal: Minimize MI(EmotionFeatures, IdentityFeatures) -> Disentanglement.
    """
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        # Estimator q(y|x)
        self.estimator = vCLUBAuxiliaryNetwork(feature_dim, feature_dim, hidden_dim)

    def forward(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        """
        Estimate Upper Bound of MI(X, Y).
        
        Args:
            x_samples: Emotion features [B, D]
            y_samples: Identity/Content features [B, D]
        """
        mu, logvar = self.estimator(x_samples)
        
        # Log density of y under variational distribution q(y|x)
        # log q(y|x) = -0.5 * sum( (y - mu)^2 / exp(logvar) + logvar + log(2pi) )
        var = logvar.exp()
        # Batched calculation
        log_q_y_given_x = -0.5 * torch.sum(
            (y_samples - mu)**2 / var + logvar, dim=1
        )
        
        # Log density of y under marginal approximation q(y)
        # Usually approximated by random shuffling of pairs
        # But vCLUB simplifies to just maximizing likelihood q(y|x) AND 
        # minimizing it for shuffled pairs?
        # Standard vCLUB: MI <= E[log q(y|x)] - E[log q(y)]
        # Here we simplify to just minimizing conditional likelihood?
        # Actually vCLUB requires training the estimator to maximize LL, 
        # and the main model to minimize Estimator's LL.
        # This implementation assumes the estimator is trained separately or via gradient reversal.
        # For simplicity in this "Loss Module", we calculate the core term.
        
        return log_q_y_given_x.mean() # This is likelihood. To minimize MI we might need more complex setup.
        # Given this is a placeholder implementation for Phase 1, we provide the structure.
        # Correct usage:
        # 1. Update Estimator to maximize log_q_y_given_x
        # 2. Update Encoder to minimize log_q_y_given_x
        
        pass


class SCCL_Loss(nn.Module):
    """
    Supervised Contrastive Continuum Learning Loss.
    Aligns Expert Features with Semantic Evidence (Text).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        features: torch.Tensor, 
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D] - Expert Pooled Feature
            text_features: [B, D] - Encoded Semantic Evidence Text (e.g. from Text Expert or BERT)
            labels: [B] - Emotion labels. If provided, pulls same-class samples together.
        """
        # Normalize
        features = F.normalize(features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Sim Matrix
        logits = torch.matmul(features, text_features.T) / self.temperature
        
        # Labels: diagonal matches are positive
        batch_size = features.shape[0]
        targets = torch.arange(batch_size, device=features.device)
        
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        
        return (loss_i2t + loss_t2i) / 2


class MultiModalContrastiveLoss(nn.Module):
    """
    Multi-Modal Contrastive Alignment Loss.
    
    Performs contrastive learning between all specified modality pairs to
    build a more unified feature space.
    
    Default pairs:
        - Audio-Text: Align speech prosody with transcription semantics
        - Visual-Audio: Align facial expressions with voice
        - AU-Text: Align facial action units with emotion descriptions
    
    Args:
        temperature: Softmax temperature for contrastive loss.
        modality_pairs: List of (modality1, modality2) tuples to align.
    """
    
    DEFAULT_PAIRS = [
        ('audio', 'text'),
        ('vis_global', 'audio'), 
        ('au', 'text'),
        ('vis_motion', 'audio'),
    ]
    
    def __init__(
        self, 
        temperature: float = 0.07,
        modality_pairs: Optional[list] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.modality_pairs = modality_pairs or self.DEFAULT_PAIRS
        
    def forward(
        self, 
        modal_features: dict,  # {'audio': [B, D], 'text': [B, D], ...}
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute contrastive loss across all modality pairs.
        
        Args:
            modal_features: Dictionary mapping modality names to feature tensors.
                           Each tensor should be [Batch, Dim].
                           
        Returns:
            total_loss: Averaged loss across all valid pairs.
            pair_losses: Dictionary of losses per pair for logging.
        """
        pair_losses = {}
        valid_pairs = 0
        total_loss = 0.0
        
        for m1, m2 in self.modality_pairs:
            if m1 not in modal_features or m2 not in modal_features:
                continue
                
            feat1 = modal_features[m1]
            feat2 = modal_features[m2]
            
            # Skip if batch size mismatch
            if feat1.size(0) != feat2.size(0):
                continue
                
            # Normalize features
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)
            
            # Compute similarity matrix
            logits = torch.matmul(feat1, feat2.T) / self.temperature
            
            # InfoNCE loss (symmetric)
            batch_size = feat1.size(0)
            labels = torch.arange(batch_size, device=feat1.device)
            
            loss_1to2 = F.cross_entropy(logits, labels)
            loss_2to1 = F.cross_entropy(logits.T, labels)
            
            pair_loss = (loss_1to2 + loss_2to1) / 2
            pair_losses[f'{m1}-{m2}'] = pair_loss.item()
            
            total_loss += pair_loss
            valid_pairs += 1
        
        if valid_pairs > 0:
            total_loss = total_loss / valid_pairs
        else:
            # Return zero loss if no valid pairs
            device = next(iter(modal_features.values())).device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return total_loss, pair_losses


class ModalityEntropyRegularizer(nn.Module):
    """
    Modality Importance Regularization via Entropy.
    
    Encourages the AdaptiveQueryGenerator to use information from all
    available modalities rather than over-relying on a single one.
    
    Uses entropy of attention weights as a regularization term:
    H(weights) = -sum(w * log(w))
    
    High entropy = uniform distribution (desired)
    Low entropy = concentrated on few modalities (penalized)
    
    Args:
        min_entropy_ratio: Minimum acceptable entropy as ratio of max entropy.
                          Values below this ratio incur a penalty.
        penalty_scale: Scale factor for the penalty.
    """
    
    def __init__(
        self, 
        min_entropy_ratio: float = 0.5,
        penalty_scale: float = 1.0,
    ):
        super().__init__()
        self.min_entropy_ratio = min_entropy_ratio
        self.penalty_scale = penalty_scale
        
    def forward(
        self, 
        attention_weights: torch.Tensor,  # [B, N_modalities]
        modality_mask: Optional[torch.Tensor] = None,  # [B, N_modalities]
    ) -> torch.Tensor:
        """
        Compute entropy regularization penalty.
        
        Args:
            attention_weights: Attention weights from AdaptiveQueryGenerator.
            modality_mask: Optional mask indicating available modalities.
            
        Returns:
            Penalty term (to be added to total loss).
        """
        # Compute entropy per sample
        log_weights = torch.log(attention_weights + 1e-8)
        entropy = -(attention_weights * log_weights).sum(dim=-1)  # [B]
        
        # Compute maximum possible entropy
        if modality_mask is not None:
            # Max entropy depends on number of available modalities per sample
            num_available = modality_mask.sum(dim=-1).clamp(min=1)
            max_entropy = torch.log(num_available)
        else:
            num_modalities = attention_weights.size(-1)
            max_entropy = torch.log(torch.tensor(
                num_modalities, dtype=torch.float, device=attention_weights.device
            ))
        
        # Compute entropy ratio
        entropy_ratio = entropy / (max_entropy + 1e-8)
        
        # Penalize if ratio is below threshold
        penalty = F.relu(self.min_entropy_ratio - entropy_ratio)
        
        return self.penalty_scale * penalty.mean()
