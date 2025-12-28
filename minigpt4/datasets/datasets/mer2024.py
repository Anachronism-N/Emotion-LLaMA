import glob
import os
import json
import pickle
import random
import time
import itertools
import warnings
import pandas as pd
import json
from copy import deepcopy

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

    

class MER2024Dataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_path,
        au_feature_dir="openface_au_23_UTT",
        modality_dropout_prob=0.0,
    ):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            # "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            # "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            # "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            # "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
        ]

        self.task_pool = [
           "emotion",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.au_feature_dir = au_feature_dir
        self.include_au = bool(au_feature_dir)
        self._warned_missing_au = False
        self._warned_au_dim = False
        self.modality_dropout_prob = modality_dropout_prob
        self.tmp = [x.strip().split(' ') for x in open(ann_path)]
        print(('video number:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        self.emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(self.emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(self.emos): self.emo2idx[ii] = emo

        # # MER2024 transcription
        self.character_lines = pd.read_csv('/home/czb/big_space/datasets/Emotion/MER2024/transcription_all_new.csv')

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path)
        else:
            video_path = os.path.join(self.vis_root, video_name + ".avi")
            image = self.extract_frame(video_path)

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)

        # image_file = '{}.jpg'.format(video_name)
        # image_path = os.path.join(self.vis_root, image_file)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)

        FaceMAE_feats, VideoMAE_feats, Audio_feats, AU_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        if AU_feats is not None:
            if len(AU_feats.shape) == 1:
                AU_feats = AU_feats.unsqueeze(0)
            feature_tensors = (FaceMAE_feats, VideoMAE_feats, Audio_feats, AU_feats)
        else:
            feature_tensors = (FaceMAE_feats, VideoMAE_feats, Audio_feats)
        video_features = torch.cat(feature_tensors, dim=0)
        video_features, modality_mask = self.apply_modality_dropout(video_features)

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            caption = t[2] # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        
        emotion = self.emo2idx[t[2]]
        sentence = self.character_lines.loc[self.character_lines['name'] == video_name, 'sentence_en'].values[0] # MER2024

        character_line = "The person in video says: {}. ".format(sentence)
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))
        # print(instruction)
        
        return {
            "image": image,
            "video_features": video_features,
            "modality_mask": modality_mask,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)
        # Read the first frame
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Release the video capture object
        video_capture.release()
        return frame_rgb

    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_340_23_UTT', video_name + '.npy') # MER2024
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeVideo_399_23_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'HL_23_UTT', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        # AU feature (optional)
        AU_feats = None
        if self.include_au:
            au_feats_path = os.path.join(self.file_path, self.au_feature_dir, video_name + '.npy')
            if os.path.exists(au_feats_path):
                AU_feats = torch.tensor(np.load(au_feats_path))
                AU_feats = self._align_au_dim(AU_feats, Audio_feats)
            else:
                AU_feats = torch.zeros_like(Audio_feats)
                if not self._warned_missing_au:
                    warnings.warn(
                        f"AU feature file missing at {au_feats_path}; using zeros as fallback."
                    )
                    self._warned_missing_au = True

        return FaceMAE_feats, VideoMAE_feats, Audio_feats, AU_feats

    def _align_au_dim(self, au_feats, audio_feats):
        au_dim = au_feats.shape[-1]
        audio_dim = audio_feats.shape[-1]
        if au_dim == audio_dim:
            return au_feats
        if not self._warned_au_dim:
            warnings.warn(
                f"AU feature dim {au_dim} does not match audio dim {audio_dim}; "
                "padding/truncating to match."
            )
            self._warned_au_dim = True
        if au_dim > audio_dim:
            return au_feats[..., :audio_dim]
        pad_size = audio_dim - au_dim
        return F.pad(au_feats, (0, pad_size))

    def apply_modality_dropout(self, video_features):
        num_modalities = video_features.size(0)
        modality_mask = torch.ones(num_modalities, dtype=video_features.dtype)
        if self.modality_dropout_prob <= 0:
            return video_features, modality_mask
        dropped = []
        for idx in range(num_modalities):
            if random.random() < self.modality_dropout_prob:
                dropped.append(idx)
        if len(dropped) == num_modalities:
            dropped.pop(random.randrange(num_modalities))
        if dropped:
            modality_mask[dropped] = 0
        video_features = video_features * modality_mask[:, None]
        return video_features, modality_mask
