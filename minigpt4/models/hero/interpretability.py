"""
HERO Interpretability Module
可解释性模块

This module provides tools for visualizing and interpreting HERO model decisions:
1. Attention Visualization: Visualize modality importance weights
2. CoT Reasoning Logger: Log and parse Chain-of-Thought reasoning outputs
3. Evidence Highlighting: Annotate which modalities contributed to predictions
4. Decision Path Tracker: Track intermediate states during inference

Reference: HERO Extended Optimization Proposals - Section 2.7
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction with interpretability data."""
    sample_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Input metadata
    available_modalities: List[str] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)
    imputed_modalities: List[str] = field(default_factory=list)
    
    # Attention analysis
    modality_weights: Dict[str, float] = field(default_factory=dict)
    dominant_modality: str = ""
    entropy_score: float = 0.0
    
    # Model outputs
    raw_output: str = ""
    parsed_prediction: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # CoT reasoning (if available)
    evidence_analysis: Dict[str, str] = field(default_factory=dict)
    rationale: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class InterpretabilityModule:
    """
    Central module for HERO interpretability and visualization.
    
    Provides methods for:
    - Recording and analyzing model predictions
    - Visualizing attention patterns
    - Logging CoT reasoning chains
    - Exporting interpretability reports
    
    Args:
        save_dir: Directory to save visualizations and logs.
        modality_names: List of modality names in order.
        enable_visualization: Whether to generate matplotlib plots.
    """
    
    DEFAULT_MODALITY_NAMES = ['vis_global', 'vis_motion', 'audio', 'au', 'text', 'synergy']
    
    def __init__(
        self,
        save_dir: str = "./interpretability_output",
        modality_names: Optional[List[str]] = None,
        enable_visualization: bool = True,
    ):
        self.save_dir = save_dir
        self.modality_names = modality_names or self.DEFAULT_MODALITY_NAMES
        self.enable_visualization = enable_visualization
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        
        # Records storage
        self.records: List[PredictionRecord] = []
        
    def record_prediction(
        self,
        sample_id: str,
        modality_weights: torch.Tensor,  # [N_modalities]
        modality_mask: Optional[torch.Tensor] = None,  # [N_modalities]
        raw_output: str = "",
        parsed_prediction: Optional[dict] = None,
        imputed_indices: Optional[List[int]] = None,
    ) -> PredictionRecord:
        """
        Record a prediction with interpretability metadata.
        
        Args:
            sample_id: Unique identifier for this sample.
            modality_weights: Attention weights from AdaptiveQueryGenerator.
            modality_mask: Mask indicating available modalities.
            raw_output: Raw text output from LLM.
            parsed_prediction: Parsed prediction dictionary.
            imputed_indices: Indices of modalities that were imputed.
            
        Returns:
            PredictionRecord with all recorded data.
        """
        weights = modality_weights.detach().cpu().numpy()
        
        # Determine available/missing modalities
        if modality_mask is not None:
            mask = modality_mask.detach().cpu().numpy()
            available = [self.modality_names[i] for i in range(len(mask)) if mask[i] > 0.5]
            missing = [self.modality_names[i] for i in range(len(mask)) if mask[i] <= 0.5]
        else:
            available = self.modality_names.copy()
            missing = []
            
        imputed = []
        if imputed_indices:
            imputed = [self.modality_names[i] for i in imputed_indices]
        
        # Build weight dictionary
        weight_dict = {
            self.modality_names[i]: float(weights[i]) 
            for i in range(min(len(weights), len(self.modality_names)))
        }
        
        # Find dominant modality
        dominant_idx = int(weights.argmax())
        dominant = self.modality_names[dominant_idx] if dominant_idx < len(self.modality_names) else "unknown"
        
        # Compute entropy
        entropy = -float((weights * (weights + 1e-8).clip(min=1e-8).__log__()).sum())
        
        # Parse CoT if available
        evidence_analysis = {}
        rationale = ""
        confidence = 0.0
        
        if parsed_prediction:
            evidence_analysis = parsed_prediction.get('evidence', {})
            rationale = parsed_prediction.get('rationale', '')
            confidence = parsed_prediction.get('confidence', 0.0)
        else:
            # Try to parse from raw output
            parsed = self.parse_cot_output(raw_output)
            if parsed:
                evidence_analysis = parsed.get('evidence', {})
                rationale = parsed.get('rationale', '')
                confidence = parsed.get('confidence', 0.0)
        
        record = PredictionRecord(
            sample_id=sample_id,
            available_modalities=available,
            missing_modalities=missing,
            imputed_modalities=imputed,
            modality_weights=weight_dict,
            dominant_modality=dominant,
            entropy_score=entropy,
            raw_output=raw_output,
            parsed_prediction=parsed_prediction or {},
            confidence=confidence,
            evidence_analysis=evidence_analysis,
            rationale=rationale,
        )
        
        self.records.append(record)
        return record
        
    def parse_cot_output(self, raw_output: str) -> Optional[dict]:
        """
        Parse Chain-of-Thought output from LLM.
        
        Handles both JSON format and structured text format.
        """
        if not raw_output:
            return None
            
        # Try JSON parsing first
        try:
            # Find JSON block
            if '{' in raw_output and '}' in raw_output:
                start = raw_output.find('{')
                end = raw_output.rfind('}') + 1
                json_str = raw_output[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback: Parse structured text
        result = {}
        
        # Extract evidence sections
        if "Evidence Analysis" in raw_output or "证据分析" in raw_output:
            lines = raw_output.split('\n')
            evidence = {}
            for line in lines:
                if ':' in line:
                    for modality in ['Visual', 'Audio', 'Text', 'AU', '视觉', '音频', '文本']:
                        if modality.lower() in line.lower():
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                evidence[modality.lower()] = parts[1].strip()
            result['evidence'] = evidence
            
        # Extract prediction
        if "Prediction" in raw_output or "预测" in raw_output:
            lines = raw_output.split('\n')
            for line in lines:
                if 'emotion' in line.lower() or '情感' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        result['emotion'] = parts[1].strip().strip('"\'')
                        
        return result if result else None
        
    def visualize_attention(
        self,
        record: PredictionRecord,
        save: bool = True,
    ) -> Optional[str]:
        """
        Generate attention weight visualization as a bar chart.
        
        Args:
            record: PredictionRecord to visualize.
            save: Whether to save the plot to disk.
            
        Returns:
            Path to saved image, or None if visualization disabled.
        """
        if not self.enable_visualization:
            return None
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        
        weights = record.modality_weights
        modalities = list(weights.keys())
        values = list(weights.values())
        
        # Create color scheme based on weight values
        colors = ['#3498db' if v < 0.3 else '#2ecc71' if v < 0.5 else '#e74c3c' 
                  for v in values]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(modalities, values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Attention Weight', fontsize=11)
        ax.set_xlabel('Modality', fontsize=11)
        ax.set_title(f'Modality Importance - {record.sample_id}\n'
                    f'Dominant: {record.dominant_modality} | Entropy: {record.entropy_score:.3f}',
                    fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        
        # Mark imputed modalities
        for i, mod in enumerate(modalities):
            if mod in record.imputed_modalities:
                ax.get_xticklabels()[i].set_color('orange')
                ax.get_xticklabels()[i].set_fontweight('bold')
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.save_dir, 'plots', f'attention_{record.sample_id}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        else:
            plt.show()
            return None
            
    def log_cot_reasoning(self, record: PredictionRecord) -> str:
        """
        Log CoT reasoning to a JSON file.
        
        Args:
            record: PredictionRecord to log.
            
        Returns:
            Path to saved log file.
        """
        log_data = {
            'sample_id': record.sample_id,
            'timestamp': record.timestamp,
            'modality_analysis': {
                'weights': record.modality_weights,
                'dominant': record.dominant_modality,
                'entropy': record.entropy_score,
                'available': record.available_modalities,
                'imputed': record.imputed_modalities,
            },
            'reasoning': {
                'evidence': record.evidence_analysis,
                'rationale': record.rationale,
                'confidence': record.confidence,
            },
            'prediction': record.parsed_prediction,
            'raw_output': record.raw_output,
        }
        
        path = os.path.join(self.save_dir, 'logs', f'cot_{record.sample_id}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            
        return path
        
    def export_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        Export a summary report of all recorded predictions.
        
        Returns:
            Path to the exported report.
        """
        if not output_path:
            output_path = os.path.join(self.save_dir, 'summary_report.json')
            
        # Compute aggregate statistics
        num_records = len(self.records)
        
        if num_records == 0:
            summary = {'error': 'No records to summarize'}
        else:
            # Modality usage stats
            modality_usage = {mod: 0.0 for mod in self.modality_names}
            dominant_counts = {mod: 0 for mod in self.modality_names}
            avg_entropy = 0.0
            imputation_rate = 0.0
            
            for record in self.records:
                for mod, weight in record.modality_weights.items():
                    modality_usage[mod] += weight
                if record.dominant_modality in dominant_counts:
                    dominant_counts[record.dominant_modality] += 1
                avg_entropy += record.entropy_score
                if record.imputed_modalities:
                    imputation_rate += 1
                    
            # Normalize
            for mod in modality_usage:
                modality_usage[mod] /= num_records
            avg_entropy /= num_records
            imputation_rate /= num_records
            
            summary = {
                'total_predictions': num_records,
                'average_modality_weights': modality_usage,
                'dominant_modality_counts': dominant_counts,
                'average_entropy': avg_entropy,
                'imputation_rate': imputation_rate,
                'records': [r.to_dict() for r in self.records[-10:]],  # Last 10 records
            }
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Summary report exported to {output_path}")
        return output_path


class AttentionVisualizerHook:
    """
    PyTorch hook for capturing attention weights during forward pass.
    
    Usage:
        hook = AttentionVisualizerHook()
        model.integration_layer.register_forward_hook(hook)
        output = model(input)
        attention_weights = hook.get_last_attention()
    """
    
    def __init__(self):
        self.attention_weights = None
        self.modality_importance = None
        
    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        if hasattr(output, 'attention_weights'):
            self.attention_weights = output.attention_weights.detach()
        if hasattr(output, 'modality_importance'):
            self.modality_importance = output.modality_importance
            
    def get_last_attention(self) -> Optional[torch.Tensor]:
        """Get the last captured attention weights."""
        return self.attention_weights
        
    def get_last_importance(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the last captured modality importance dict."""
        return self.modality_importance
