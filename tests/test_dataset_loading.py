
import unittest
import torch
import numpy as np
import os
import json
import shutil
from minigpt4.datasets.datasets.hero_dataset import HERODataset

class TestHERODataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.feature_dir = os.path.join(self.test_dir, "features")
        os.makedirs(self.feature_dir, exist_ok=True)
        self.ann_path = os.path.join(self.test_dir, "labels.json")
        
        # Create Dummy Data
        self.video_id = "test_vid_001"
        self.dummy_features = {
            'video_mae': np.random.randn(16, 768).astype(np.float32), # Suppose Base dims
            'clip': np.random.randn(32, 1408).astype(np.float32),
            'hubert': np.random.randn(50, 1024).astype(np.float32),
            'bert': np.random.randn(20, 768).astype(np.float32),
            'au': np.random.randn(50, 1024).astype(np.float32)
        }
        np.save(os.path.join(self.feature_dir, f"{self.video_id}.npy"), self.dummy_features)
        
        # Create Annotation
        ann = [{
            "video_id": self.video_id,
            "emotion": "happy",
            "text": "I am happy"
        }]
        with open(self.ann_path, 'w') as f:
            json.dump(ann, f)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_loading(self):
        dataset = HERODataset(
            vis_processor=lambda x: x,
            text_processor=lambda x: x,
            vis_root=self.feature_dir,
            ann_path=self.ann_path
        )
        
        sample = dataset[0]
        print("Initial keys:", sample.keys())
        
        # Verify Keys
        self.assertIn("video_features", sample)
        self.assertIn("image", sample)
        
        # Verify Shapes
        # Standardize expected shape: [5, 64, 1408]
        feat = sample["video_features"]
        print("Video Features Shape:", feat.shape)
        self.assertEqual(feat.shape, (5, 64, 1024))
        
        # Verify non-zero content (approx check)
        # Index 4 is CLIP (1408 dim)
        clip_slice = feat[4]
        self.assertNotEqual(clip_slice.abs().sum(), 0)
        
        print("HERODataset verification passed!")

if __name__ == '__main__':
    unittest.main()
