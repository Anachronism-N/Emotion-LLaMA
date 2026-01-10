"""
HERO Feature Extraction Script
HERO 特征提取脚本

This script processes raw video files to extract multimodal features:
1. Visual Motion (VideoMAE)
2. Visual Global (CLIP)
3. Audio (HuBERT)
4. Text (Whisper -> BERT)

Usage:
    python scripts/extract_features.py --video_path video.mp4 --output_dir features/ --device cuda
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Warning: moviepy not found. Audio extraction might fail if not using pre-extracted audio.")
    VideoFileClip = None

from minigpt4.models.hero.feature_extractors import (
    VideoMAEFeatureExtractor,
    HuBERTFeatureExtractor,
    ASRFeatureExtractor,
    CLIPFeatureExtractor
)

def extract_frames(video_path, num_frames=16):
    """Uniformly sample frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    
    # Pad if failed to get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        
    return frames

def extract_audio(video_path, output_audio_path="temp.wav"):
    """Extract audio from video using moviepy."""
    if VideoFileClip is None:
        raise ImportError("moviepy is required for audio extraction.")
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            return None
        video.audio.write_audiofile(output_audio_path, logger=None)
        return output_audio_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--output_dir", required=True, help="Directory to save features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cleanup", action="store_true", help="Delete temp files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.video_path))[0]
    save_path = os.path.join(args.output_dir, f"{basename}.npy")
    
    if os.path.exists(save_path):
        print(f"Features already exist at {save_path}, skipping.")
        return

    print(f"Processing {args.video_path}...")
    
    # Initialize Extractors
    print("Loading models...")
    videomae_extractor = VideoMAEFeatureExtractor(device=args.device)
    hubert_extractor = HuBERTFeatureExtractor(device=args.device)
    asr_extractor = ASRFeatureExtractor(device=args.device)
    clip_extractor = CLIPFeatureExtractor(device=args.device)
    
    features = {}
    
    # 1. Visual Extraction
    print("Extracting Visual Features...")
    frames = extract_frames(args.video_path, num_frames=16)
    
    # Motion (VideoMAE)
    motion_feats = videomae_extractor.extract(frames)
    features['video_mae'] = motion_feats.cpu().numpy()
    
    # Global (CLIP) - Use middle frame or sparse sample
    # Let's take 4 keyframes from the 16
    keyy_indices = [0, 5, 10, 15]
    keyframes = [frames[i] for i in keyy_indices]
    global_feats = clip_extractor.extract(keyframes)
    features['clip'] = global_feats.cpu().numpy()
    
    # 2. Audio Extraction
    print("Extracting Audio/Text Features...")
    audio_path = f"temp_{basename}.wav"
    extracted_audio_path = extract_audio(args.video_path, audio_path)
    
    if extracted_audio_path:
        import torchaudio
        waveform, sr = torchaudio.load(extracted_audio_path)
        # Resample to 16k for HuBERT/Whisper
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # HuBERT
        # Squeeze channel
        if waveform.shape[0] > 1:
            waveform_mono = waveform.mean(dim=0)
        else:
            waveform_mono = waveform.squeeze(0)
            
        audio_feats = hubert_extractor.extract(waveform_mono)
        features['hubert'] = audio_feats.cpu().numpy()
        
        # ASR
        text = asr_extractor.extract_text(waveform_mono)
        print(f"Transcript: {text}")
        text_emb = asr_extractor.extract_embedding(text)
        features['bert'] = text_emb.cpu().numpy()
        features['transcript'] = text
        
        if args.cleanup:
            os.remove(extracted_audio_path)
    else:
        print("No audio found, padding with zeros.")
        features['hubert'] = np.zeros((1, 1, 1024))
        features['bert'] = np.zeros((1, 1, 768))
        features['transcript'] = ""

    # Save
    np.save(save_path, features)
    print(f"Saved features to {save_path}")

if __name__ == "__main__":
    main()
