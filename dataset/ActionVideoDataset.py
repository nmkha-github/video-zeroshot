import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import os
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.VideoHelper.VideoHelper import VideoHelper


class ActionVideoDataset(Dataset):
    def __init__(self, video_folder, annotation_excel_file):
        self.video_folder = video_folder
        self.annotations = pd.read_excel(annotation_excel_file).filter(
            items=["video", "action", "danger"]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_info = self.annotations.iloc[idx]
        video_index = video_info["video"]
        action_label = video_info["action"]
        danger_label = video_info["danger"]

        video_path = os.path.join(
            self.video_folder, f"Video_{video_index}_{action_label}.mp4"
        )

        return video_index, resize_frames, action_label, danger_label
