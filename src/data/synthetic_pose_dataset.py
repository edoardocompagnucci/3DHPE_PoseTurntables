import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class SyntheticPoseDataset(Dataset):
    def __init__(self, data_root, split_txt, transform=None):
        self.data_root = data_root
        self.split_txt = split_txt
        self.transform = transform

        with open(split_txt) as f:
            self.data_ids = [line.strip() for line in f if line.strip()]

        self.paths = {
            "joints_2d": os.path.join(data_root, "annotations", "joints_2d"),
            "joints_3d": os.path.join(data_root, "annotations", "joints_3d"),
            "rot_mats": os.path.join(data_root, "annotations", "rot_mats"),
            "K": os.path.join(data_root, "annotations", "K"),
            "R": os.path.join(data_root, "annotations", "R"),
            "t": os.path.join(data_root, "annotations", "t"),
            "rgb": os.path.join(data_root, "raw", "rgb")
        }

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]

        rgb_path = os.path.join(self.paths["rgb"], f"{data_id}.png")
        rgb = np.array(Image.open(rgb_path)) if os.path.exists(rgb_path) else None
        
        sample = {
            "joints_2d": torch.tensor(
                np.load(os.path.join(self.paths["joints_2d"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "joints_3d": torch.tensor(
                np.load(os.path.join(self.paths["joints_3d"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "rot_mats": torch.tensor(
                np.load(os.path.join(self.paths["rot_mats"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "K": torch.tensor(
                np.load(os.path.join(self.paths["K"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "R": torch.tensor(
                np.load(os.path.join(self.paths["R"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "t": torch.tensor(
                np.load(os.path.join(self.paths["t"], f"{data_id}.npy")),
                dtype=torch.float32
            ),
            "rgb": torch.tensor(rgb)
        }
        

        if self.transform:
            sample = self.transform(sample)
            
        return sample