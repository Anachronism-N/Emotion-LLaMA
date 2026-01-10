#!/usr/bin/env python
"""
HERO Pipeline Test Script
测试 HERO 模块能否正常工作

This script tests the HERO modules without requiring actual data,
using randomly generated tensors to verify the forward pass.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_observation_experts():
    """Test the Observation Expert Layer."""
    print("=" * 60)
    print("Testing Observation Expert Layer")
    print("=" * 60)
    
    from minigpt4.models.hero.observation_experts import (
        ObservationExpertLayer,
        ModalityQFormer,
        SynergyExpert,
    )
    
    batch_size = 2
    seq_len = 10
    
    # Test ModalityQFormer
    print("\n1. Testing ModalityQFormer...")
    qformer = ModalityQFormer(
        input_dim=1024,
        hidden_dim=768,
        num_queries=32,
        num_layers=2,
    )
    
    dummy_input = torch.randn(batch_size, seq_len, 1024)
    output = qformer(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Feature tensor shape: {output.feature_tensor.shape}")
    print(f"   Pooled feature shape: {output.pooled_feature.shape}")
    assert output.feature_tensor.shape == (batch_size, 32, 768)
    assert output.pooled_feature.shape == (batch_size, 768)
    print("   ✓ ModalityQFormer test passed!")
    
    # Test SynergyExpert
    print("\n2. Testing SynergyExpert...")
    synergy = SynergyExpert(
        audio_dim=768,
        visual_dim=768,
        hidden_dim=768,
    )
    
    audio_feat = torch.randn(batch_size, seq_len, 768)
    visual_feat = torch.randn(batch_size, seq_len, 768)
    output = synergy(audio_feat, visual_feat)
    
    print(f"   Audio input shape: {audio_feat.shape}")
    print(f"   Visual input shape: {visual_feat.shape}")
    print(f"   Output feature tensor: {output.feature_tensor.shape}")
    print(f"   Output pooled feature: {output.pooled_feature.shape}")
    assert output.feature_tensor.shape == (batch_size, 1, 768)
    print("   ✓ SynergyExpert test passed!")
    
    # Test full ObservationExpertLayer
    print("\n3. Testing ObservationExpertLayer (full)...")
    expert_layer = ObservationExpertLayer(
        visual_dim=1408,
        video_dim=1024,
        audio_dim=1024,
        au_dim=1024,
        hidden_dim=768,
        num_queries=32,
        llm_hidden_dim=4096,
        include_synergy=True,
    )
    
    visual_feat = torch.randn(batch_size, seq_len, 1408)
    video_feat = torch.randn(batch_size, seq_len, 1024)
    audio_feat = torch.randn(batch_size, seq_len, 1024)
    au_feat = torch.randn(batch_size, seq_len, 1024)
    
    outputs = expert_layer(
        visual_features=visual_feat,
        video_features=video_feat,
        audio_features=audio_feat,
        au_features=au_feat,
    )
    
    print(f"   Number of expert outputs: {len(outputs)}")
    for key, val in outputs.items():
        print(f"   - {key}: feature={val.feature_tensor.shape}, pooled={val.pooled_feature.shape}")
    
    # Test projection to LLM
    projected = expert_layer.project_to_llm(outputs)
    print(f"\n   Projected to LLM dimension:")
    for key, val in projected.items():
        print(f"   - {key}: {val.shape}")
    
    # Test summary vectors
    summaries = expert_layer.get_summary_vectors(outputs)
    print(f"\n   Summary vectors: {summaries.shape}")
    
    print("   ✓ ObservationExpertLayer test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in expert_layer.parameters())
    trainable_params = sum(p.numel() for p in expert_layer.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return True


def test_integration_layer():
    """Test the Evidence Integration Layer."""
    print("\n" + "=" * 60)
    print("Testing Evidence Integration Layer")
    print("=" * 60)
    
    from minigpt4.models.hero.integration_layer import (
        EvidenceIntegrationLayer,
        GlobalQueryGenerator,
        PanoramicGuidedAttention,
        ModalityDropoutTrainer,
    )
    
    batch_size = 2
    num_queries = 32
    hidden_dim = 768
    llm_dim = 4096
    
    # Test GlobalQueryGenerator
    print("\n1. Testing GlobalQueryGenerator...")
    query_gen = GlobalQueryGenerator(
        hidden_dim=hidden_dim,
        num_heads=8,
        max_experts=6,
    )
    
    summary_vectors = torch.randn(batch_size, 5, hidden_dim)
    global_query, expert_attention = query_gen(summary_vectors)
    
    print(f"   Summary vectors shape: {summary_vectors.shape}")
    print(f"   Global query shape: {global_query.shape}")
    print(f"   Expert attention shape: {expert_attention.shape}")
    assert global_query.shape == (batch_size, hidden_dim)
    print("   ✓ GlobalQueryGenerator test passed!")
    
    # Test PanoramicGuidedAttention
    print("\n2. Testing PanoramicGuidedAttention...")
    panoramic = PanoramicGuidedAttention(
        hidden_dim=hidden_dim,
        llm_dim=llm_dim,
        num_heads=8,
        num_output_queries=64,
    )
    
    k_bank = torch.randn(batch_size, 100, hidden_dim)
    integrated, attn_weights, modality_imp = panoramic(
        global_query=global_query,
        k_bank=k_bank,
        modality_boundaries=[32, 64, 96, 100],
    )
    
    print(f"   K-bank shape: {k_bank.shape}")
    print(f"   Integrated context shape: {integrated.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   Modality importance keys: {list(modality_imp.keys())}")
    assert integrated.shape == (batch_size, 64, llm_dim)
    print("   ✓ PanoramicGuidedAttention test passed!")
    
    # Test full EvidenceIntegrationLayer
    print("\n3. Testing EvidenceIntegrationLayer (full)...")
    integration_layer = EvidenceIntegrationLayer(
        hidden_dim=hidden_dim,
        llm_dim=llm_dim,
        num_heads=8,
        num_output_queries=64,
    )
    
    expert_features = {
        'visual': torch.randn(batch_size, num_queries, hidden_dim),
        'video': torch.randn(batch_size, num_queries, hidden_dim),
        'audio': torch.randn(batch_size, num_queries, hidden_dim),
        'au': torch.randn(batch_size, num_queries, hidden_dim),
        'synergy': torch.randn(batch_size, 1, hidden_dim),
    }
    summary_vectors = torch.randn(batch_size, 5, hidden_dim)
    
    output = integration_layer(expert_features, summary_vectors)
    
    print(f"   Integrated context: {output.integrated_context.shape}")
    print(f"   Attention weights: {output.attention_weights.shape}")
    print(f"   Global query: {output.global_query.shape}")
    print(f"   Modality importance: {list(output.modality_importance.keys())}")
    print("   ✓ EvidenceIntegrationLayer test passed!")
    
    # Test ModalityDropoutTrainer
    print("\n4. Testing ModalityDropoutTrainer...")
    trainer = ModalityDropoutTrainer(integration_layer, dropout_prob=0.3)
    output, kl_loss = trainer(expert_features, summary_vectors, training=True)
    
    print(f"   Output integrated context: {output.integrated_context.shape}")
    print(f"   KL divergence loss: {kl_loss.item():.4f}")
    print("   ✓ ModalityDropoutTrainer test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in integration_layer.parameters())
    print(f"\n   Total parameters: {total_params:,}")
    
    return True


def test_full_pipeline():
    """Test the full HERO pipeline with dummy data."""
    print("\n" + "=" * 60)
    print("Testing Full HERO Pipeline")
    print("=" * 60)
    
    from minigpt4.models.hero.observation_experts import ObservationExpertLayer
    from minigpt4.models.hero.integration_layer import EvidenceIntegrationLayer
    
    batch_size = 2
    seq_len = 10
    hidden_dim = 768
    llm_dim = 4096
    num_queries = 32
    
    # Create modules
    print("\n1. Creating HERO modules...")
    observation_experts = ObservationExpertLayer(
        visual_dim=1408,
        video_dim=1024,
        audio_dim=1024,
        au_dim=1024,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        llm_hidden_dim=llm_dim,
        include_synergy=True,
    )
    
    integration_layer = EvidenceIntegrationLayer(
        hidden_dim=hidden_dim,
        llm_dim=llm_dim,
        num_heads=8,
        num_output_queries=64,
    )
    
    # Create dummy inputs
    print("\n2. Creating dummy inputs...")
    visual_feat = torch.randn(batch_size, seq_len, 1408)
    video_feat = torch.randn(batch_size, seq_len, 1024)
    audio_feat = torch.randn(batch_size, seq_len, 1024)
    au_feat = torch.randn(batch_size, seq_len, 1024)
    
    print(f"   Visual features: {visual_feat.shape}")
    print(f"   Video features: {video_feat.shape}")
    print(f"   Audio features: {audio_feat.shape}")
    print(f"   AU features: {au_feat.shape}")
    
    # Forward through observation experts
    print("\n3. Forward through Observation Experts...")
    expert_outputs = observation_experts(
        visual_features=visual_feat,
        video_features=video_feat,
        audio_features=audio_feat,
        au_features=au_feat,
    )
    
    summary_vectors = observation_experts.get_summary_vectors(expert_outputs)
    print(f"   Summary vectors: {summary_vectors.shape}")
    
    # Get feature tensors for integration
    expert_features = {}
    for key, output in expert_outputs.items():
        expert_features[key] = output.feature_tensor
        print(f"   Expert '{key}': {output.feature_tensor.shape}")
    
    # Forward through integration layer
    print("\n4. Forward through Integration Layer...")
    integration_output = integration_layer(
        expert_features=expert_features,
        summary_vectors=summary_vectors,
    )
    
    print(f"   Integrated context: {integration_output.integrated_context.shape}")
    print(f"   Global query: {integration_output.global_query.shape}")
    print(f"   Attention weights: {integration_output.attention_weights.shape}")
    
    # Show modality importance
    print("\n5. Modality Importance Scores:")
    total_attn = sum(integration_output.modality_importance.values())
    for key, val in integration_output.modality_importance.items():
        percentage = (val / total_attn * 100).mean().item()
        print(f"   - {key}: {percentage:.1f}%")
    
    # Memory usage estimate
    print("\n6. Memory Usage Estimate:")
    total_params = sum(p.numel() for p in observation_experts.parameters())
    total_params += sum(p.numel() for p in integration_layer.parameters())
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"   Total parameters: {total_params:,}")
    print(f"   Estimated memory (float32): {memory_mb:.1f} MB")
    
    print("\n   ✓ Full HERO pipeline test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# HERO Module Tests")
    print("#" * 60)
    
    try:
        test_observation_experts()
        test_integration_layer()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
