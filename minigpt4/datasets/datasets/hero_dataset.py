
import os
import json
import random
import torch
import numpy as np
import warnings
from torch.utils.data import Dataset
from PIL import Image

class HERODataset(Dataset):
    """
    Dataset loader for HERO Framework (Feature-Only Mode).
    Reads .npy dictionary files containing pre-extracted features.
    
    Structure:
    - features/video_id.npy: {'video_mae', 'clip', 'hubert', 'bert', 'au'}
    - labels.json: List of {'video_id', 'emotion', 'text'}
    """
    
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,  # Path to 'features/' folder
        ann_path,  # Path to 'labels.json'
        modality_dropout_prob=0.0,
    ):
        self.feature_root = vis_root # Reusing naming convention but it is feature root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.modality_dropout_prob = modality_dropout_prob
        
        print(f"Loading annotations from {ann_path}")
        with open(ann_path, 'r') as f:
            self.annotation = json.load(f)
            
        print(f"Loaded {len(self.annotation)} samples.")
        
        # Instruction Templates
        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            "Identify the displayed emotion in the video.",
            "Analyze the multimodal evidence and predict the emotion.",
        ]
        
        self.emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']
        self.emo2idx = {emo: ii for ii, emo in enumerate(self.emos)}

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann['video_id']
        feature_path = os.path.join(self.feature_root, f"{video_id}.npy")
        
        if not os.path.exists(feature_path):
            print(f"Warning: Feature file missing for {video_id}")
            # Return a valid dummy or skip? Dataset __getitem__ cannot easily skip.
            # We fail loud or return zeros. Let's return zeros for stability but log.
            return self.__getitem__((index + 1) % len(self))
            
        # Load Features
        try:
            feats_dict = np.load(feature_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            return self.__getitem__((index + 1) % len(self))

        # Extract & Process Features
        # 1. Visual Global (CLIP) -> Used as 'image' input to model
        # The model encode_img_hero takes 'image' tensor directly or patches?
        # HEROModel.encode_img_hero expects "image_patches" from visual_encoder(image).
        # We are passing pre-extracted CLIP features.
        # So we need to Bypass visual encoder or pass features as "image".
        # However, minigpt_base prepares "image" using vis_processor usually.
        # If we are Feature-Only, we might not have the raw image.
        # But `HEROModel` uses `self.visual_encoder(image)`.
        # IF WE ONLY HAVE FEATURES, we must modify HEROModel to accept pre-extracted visual features.
        # OR we pass a dummy image and use the CLIP features from the dictionary.
        # Let's assume we pass the CLIP features as the "image" input, but `HEROModel` needs to handle it.
        # Actually `HEROModel.encode_img_hero` expects `image` tensor.
        # Let's look at `extract_features.py`: it extracts `clip` features.
        # I will pass these `clip` features in `video_features` or a separate key?
        # `HEROModel` uses `image_patches`.
        # SOLUTION: I will pack `clip` features into `video_features`? No, visual global is distinct.
        # I will Put CLIP features into `image` key, but `HEROModel` forward expects a 3D pixel tensor [B, C, H, W].
        # If I pass features [B, N, D], standard ViT encoder will fail.
        # Modifications needed in HEROModel: Check if 'image' is features or pixels.
        
        # For now, let's stack them in `video_features` and handle logic in Model.
        # video_features order: [Motion, Audio, AU, Text]
        # Wait, if I don't give an image, `encode_img_hero` calls `self.visual_encoder(image)`.
        # I must provide a dummy image if I want to use the pipeline without code changes, OR change the code.
        # Given "Feature-Only", I should change `HEROModel` to accept `visual_features` directly.
        # But `encode_img_hero` takes `image, video_features`.
        # I'll create `dummy_image` (zeros) and pass `clip` features as a new modality in `video_features`.
        
        # Current Order in HEROModel.encode_img_hero:
        # 0: VideoMAE
        # 1: Audio
        # 2: AU
        # 3: Text
        # Visual Global comes from `image` arg.
        
        # I'll stack CLIP features as index 4? Or handle it.
        # Let's stack: [Motion, Audio, AU, Text, CLIP]
        # And update HEROModel to look for 5th modality if `image` is dummy.
        
        # Extract Tensors
        def get_tensor(key):
            if key in feats_dict and feats_dict[key] is not None:
                t = torch.tensor(feats_dict[key])
                if len(t.shape) == 3: t = t.mean(dim=0) # [1, Seq, Dim] -> [Seq, Dim]
                if len(t.shape) == 1: t = t.unsqueeze(0)
                return t.float()
            return torch.zeros(1, 768) # Dummy

        # CLIP (32 patches)
        clip_feats = get_tensor('clip') # [N, 1408]
        # Motion (Seq)
        motion_feats = get_tensor('video_mae') # [T, 1024]
        # Audio
        audio_feats = get_tensor('hubert') # [T, 1024]
        # Text
        text_feats = get_tensor('bert') # [T, 768]
        # AU
        au_feats = get_tensor('au') if 'au' in feats_dict else torch.zeros(1, 1024)
        
        # Pad/Truncate dimensions to match what model expects?
        # Model experts have projections (`ModalityQFormer` input_proj). 
        # So we just pass raw dims.
        
        # Construct `video_features` [N_Modalities, Max_Seq, Max_Dim]???
        # No, `torch.stack` requires same tensor shape.
        # Modalities have different Dim and different SeqLen.
        # We cannot simply `stack` them if dims differ.
        # `mer2024.py` did: `video_features = torch.cat(feature_tensors, dim=0)`
        # If dim=0 is Modality Index, then they must match in other dims.
        # But in `mer2024.py`, `FaceMAE_feats` and `VideoMAE_feats` came from `.npy`.
        # If they had different sizes, `cat` would fail unless `dim=0` means Concat in Time?
        # `mer2024.py`: `feature_tensors` tuple. `cat(dim=0)`.
        # If `FaceMAE` is [1, T, D] and `VideoMAE` is [1, T, D], then `cat` -> [2, T, D].
        # It relies on them having same shape.
        
        # PROBLEM: My features have different Dims! (CLIP 1408, VideoMAE 1024, BERT 768).
        # I cannot stack them into one tensor [M, T, D].
        # I must return a Dictionary or a list.
        # `HEROModel.forward` expects `video_features` tensor?
        # Let's check `hero_model.py`:
        # `video_features[:, 0, :, :]` -> Slices indices. Assumes stacked tensor.
        # This implies all modalities MUST have same shape in the input tensor.
        # Workaround: Pad all to max Dim (1408)?
        
        max_dim = 1024
        max_len = 512 # Truncate/Pad seq length for uniform batching
        
        def pad_tensor(t, target_len=64, target_dim=1024):
            # t: [Seq, Dim]
            # Pad dim
            curr_dim = t.shape[-1]
            if curr_dim < target_dim:
                t = torch.cat([t, torch.zeros(t.shape[0], target_dim - curr_dim)], dim=1)
            elif curr_dim > target_dim:
                t = t[:, :target_dim]
                
            # Pad/cut seq
            curr_len = t.shape[0]
            if curr_len < target_len:
                t = torch.cat([t, torch.zeros(target_len - curr_len, target_dim)], dim=0)
            elif curr_len > target_len:
                # Uniform sample? Or truncate. Uniform is better for video.
                indices = torch.linspace(0, curr_len-1, target_len).long()
                t = t[indices]
            return t

        # Pad all to compatible shape [1, 64, 1024]
        # Stack them: [5, 64, 1024]
        
        # Order: 0:Motion, 1:Audio, 2:AU, 3:Text, 4:CLIP(Global)
        feat_list = []
        feat_list.append(pad_tensor(motion_feats))
        feat_list.append(pad_tensor(audio_feats))
        feat_list.append(pad_tensor(au_feats))
        feat_list.append(pad_tensor(text_feats))
        feat_list.append(pad_tensor(clip_feats))
        
        video_features = torch.stack(feat_list, dim=0) # [5, 64, 1024]
        
        # Modality Mask
        modality_mask = torch.ones(5)
        # Random dropout
        video_features, modality_mask = self.apply_modality_dropout(video_features, modality_mask)
        
        # Instruction & Label
        emotion_label = ann['emotion']
        emotion_idx = self.emo2idx.get(emotion_label, 0)
        
        instruction = random.choice(self.emotion_instruction_pool)
        instruction = f"<video><VideoHere></video> {instruction}"
        
        # Dummy Image (Black) since we use pre-extracted features
        image = torch.zeros(3, 224, 224) 

        return {
            "image": image,
            "video_features": video_features,
            "modality_mask": modality_mask,
            "instruction_input": instruction,
            "answer": str(emotion_label),
            "emotion": emotion_idx,
            "image_id": video_id
        }

    def apply_modality_dropout(self, video_features, modality_mask):
        if self.modality_dropout_prob <= 0:
            return video_features, modality_mask
            
        params_retain = 1.0 - self.modality_dropout_prob
        # Only drop if training? Dataset doesn't know 'training' state easily.
        # Usually handled in collator or set prob=0 for eval.
        
        # Mask random modalities
        mask = torch.bernoulli(torch.full(modality_mask.shape, params_retain))
        if mask.sum() == 0: mask[0] = 1.0 # Keep at least one
        
        return video_features * mask[:, None, None], mask
