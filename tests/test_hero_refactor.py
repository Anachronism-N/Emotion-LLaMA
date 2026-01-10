
import torch
import sys
import unittest

from minigpt4.models.hero.observation_experts import ObservationExpertLayer, ModalityQFormer, SynergyExpert
from minigpt4.models.hero.integration_layer import EvidenceIntegrationLayer, PanoramicGuidedAttention, AudioGuidedQueryGenerator, ModalityDropoutTrainer
from minigpt4.models.hero.hero_model import HEROModel

class TestHEROModules(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 768
        self.llm_dim = 4096
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def test_observation_experts(self):
        print("\n" + "=" * 60)
        print("Testing Refactored Observation Expert Layer")
        print("=" * 60)
        
        # 1. Inputs
        vis_global = torch.randn(self.batch_size, 32, 1408).to(self.device) # CLIP patches
        vis_motion = torch.randn(self.batch_size, self.seq_len, 1024).to(self.device) # VideoMAE
        audio = torch.randn(self.batch_size, self.seq_len, 1024).to(self.device) # HuBERT
        au = torch.randn(self.batch_size, self.seq_len, 1024).to(self.device) # AU
        text = torch.randn(self.batch_size, self.seq_len, 768).to(self.device) # ASR
        
        # 2. Layer
        layer = ObservationExpertLayer(
            visual_dim=1408,
            video_dim=1024,
            audio_dim=1024,
            au_dim=1024,
            text_dim=768,
            hidden_dim=self.hidden_dim,
            llm_hidden_dim=self.llm_dim,
            include_synergy=True
        ).to(self.device)
        
        # 3. Forward
        outputs = layer(vis_global, vis_motion, audio, au, text)
        
        # 4. Verify outputs
        expected_keys = ['vis_global', 'vis_motion', 'audio', 'au', 'text', 'synergy']
        for k in expected_keys:
            self.assertIn(k, outputs)
            out = outputs[k]
            print(f"Expert '{k}': Tensor={out.feature_tensor.shape}, Pooled={out.pooled_feature.shape}")
            self.assertEqual(out.feature_tensor.shape[0], self.batch_size)
            self.assertEqual(out.feature_tensor.shape[2], self.hidden_dim)
            # Verify Text Evidence
            print(f"  Evidence Sample: {out.semantic_evidence[0]}")
            self.assertIsInstance(out.semantic_evidence, list)
            
        print("   ✓ ObservationExpertLayer test passed!")
        return outputs

    def test_integration_layer(self):
        print("\n" + "=" * 60)
        print("Testing Refactored Evidence Integration Layer (Audio-Guided)")
        print("=" * 60)
        
        # Create dummy inputs
        expert_features = {
            'vis_global': torch.randn(self.batch_size, 32, self.hidden_dim).to(self.device),
            'vis_motion': torch.randn(self.batch_size, 32, self.hidden_dim).to(self.device),
            'audio': torch.randn(self.batch_size, 32, self.hidden_dim).to(self.device),
            'au': torch.randn(self.batch_size, 32, self.hidden_dim).to(self.device),
            'text': torch.randn(self.batch_size, 32, self.hidden_dim).to(self.device),
            'synergy': torch.randn(self.batch_size, 1, self.hidden_dim).to(self.device),
        }
        summary_vectors = torch.randn(self.batch_size, 6, self.hidden_dim).to(self.device)
        
        layer = EvidenceIntegrationLayer(
            hidden_dim=self.hidden_dim,
            llm_dim=self.llm_dim,
            max_experts=6
        ).to(self.device)
        
        output = layer(expert_features, summary_vectors)
        
        print(f"Integrated Context: {output.integrated_context.shape}")
        print(f"Global Query: {output.global_query.shape}")
        
        self.assertEqual(output.integrated_context.shape, (self.batch_size, 64, self.llm_dim))
        
        # Check if Audio-Guided Query Generator is working
        self.assertIsInstance(layer.query_generator, AudioGuidedQueryGenerator)
        print("   ✓ Used AudioGuidedQueryGenerator")
        
        print("   ✓ EvidenceIntegrationLayer test passed!")

    def test_loss_functions(self):
        print("\n" + "=" * 60)
        print("Testing HERO Loss Functions")
        print("=" * 60)
        from minigpt4.models.hero.hero_loss import STMIL_Loss, SCCL_Loss
        
        # STMIL
        stmil = STMIL_Loss(feature_dim=self.hidden_dim).to(self.device)
        x = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
        y = torch.randn(self.batch_size, self.hidden_dim).to(self.device) # Content
        loss_mi = stmil(x, y)
        print(f"STMIL Loss: {loss_mi.item()}")
        
        # SCCL
        sccl = SCCL_Loss()
        feats = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True).to(self.device)
        text_feats = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True).to(self.device)
        loss_cl = sccl(feats, text_feats)
        loss_cl.backward() # Check grad
        print(f"SCCL Loss: {loss_cl.item()}")
        
        print("   ✓ Loss Functions test passed!")

def run_all_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHEROModules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    run_all_tests()
