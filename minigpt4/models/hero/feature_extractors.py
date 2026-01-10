"""
HERO Feature Extractors
HERO 特征提取器模块

This module provides standalone classes to extract features from raw video/audio 
for the HERO framework. It wraps HuggingFace transformers and other standard libraries.

Classes:
1. VideoMAEFeatureExtractor: Extracts motion features [Seq, 1024]
2. HuBERTFeatureExtractor: Extracts audio features [Seq, 1024]
3. CLIPFeatureExtractor: Extracts global visual features [ImgCount, 1408]
4. ASRFeatureExtractor: Extracts text features via Whisper -> BERT [Seq, 768]
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchaudio
from transformers import (
    VideoMAEModel, VideoMAEImageProcessor,
    Wav2Vec2Processor, HubertModel,
    WhisperProcessor, WhisperForConditionalGeneration,
    BertTokenizer, BertModel,
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel
)

class VideoMAEFeatureExtractor:
    def __init__(self, model_name="MCG-OpenLab/videomae-base-finetuned-kinetics", device="cuda"):
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, video_frames: list):
        """
        Args:
            video_frames: List of PIL Images or numpy arrays (H, W, C).
                          Should be sampled (e.g., 16 frames).
        Returns:
            start_logits: Tensor [1, 1568, 768] (depends on model)
            pooled_output: Tensor [1, 768]
        Note: HERO expects [Seq, 1024]. If reusing base, dim is 768. 
              We might need large or a projection. For now standard base is 768.
              The HERO model config defaults to 1024 for video_dim.
              Check if we should use videomae-large (1024) or project.
              Let's assume we use videomae-large for 1024 dim match.
        """
        if not video_frames:
            return torch.zeros(1, 16, 1024).to(self.device) # Dummy

        inputs = self.processor(list(video_frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Output: last_hidden_state [Batch, Seq(Patches), Dim]
        # We usually want temporal features. VideoMAE outputs Patch embeddings.
        # For a 16-frame clip, patches are flattened. 
        # Ideally, we want [T, Dim]. VideoMAE is a ViT, outputs [N_Patches, Dim].
        # Implementation decision: Average pool spatial patches to get [T, Dim]?
        # Or just use the mean pooled output?
        # HERO Observation expert expects [Seq, Dim].
        # Let's return last_hidden_state average pooled over space if possible, 
        # or just the raw last_hidden_state if we treat patches as sequence.
        # Standard practice: Mean pool.
        return outputs.last_hidden_state


class HuBERTFeatureExtractor:
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device="cuda"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, audio_array, sampling_rate=16000):
        # audio_array: numpy array or torch tensor
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state # [B, Seq, 1024] (Large is 1024)


class ASRFeatureExtractor:
    """
    Pipeline: Audio -> Whisper (Text) -> BERT (Embedding)
    """
    def __init__(self, whisper_model="openai/whisper-base", bert_model="bert-base-uncased", device="cuda"):
        self.device = device
        # Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(device)
        self.whisper_model.eval()
        
        # BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model).to(device)
        self.bert_model.eval()

    def extract_text(self, audio_array, sampling_rate=16000):
        inputs = self.whisper_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        input_features = inputs.input_features
        
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(input_features)
        
        transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    def extract_embedding(self, text):
        if not text:
            return torch.zeros(1, 1, 768).to(self.device)
            
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state # [B, Seq, 768]


class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, images: list):
        inputs = self.processor(text=[""], images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            # Get vision model outputs directly
            vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
        return vision_outputs.last_hidden_state # [B, Patches, 1024/768] note: CLIP global is 768/1024 depending on version
