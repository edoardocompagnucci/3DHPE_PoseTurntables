import torch
from torch.utils.data import Dataset
import os
import numpy as np

class SyntheticPoseDataset(Dataset):
    def __init__(self, data_root, split_txt, transform=None):
        self.data_root = data_root
        self.split_text = split_txt
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
            "img": os.path.join(data_root, "raw", "rgb")
        }

    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]

        sample = {
            "joints_2d": np.load(os.path.join(self.paths["joints_2d"], f"{data_id}.npy")),
            "joints_3d": np.load(os.path.join(self.paths["joints_3d"], f"{data_id}.npy")),
            "rot_mats": np.load(os.path.join(self.paths["joints_3d"], f"{data_id}.npy")),
            "K": np.load(os.path.join(self.paths["K"], f"{data_id}.npy")),
            "R": np.load(os.path.join(self.paths["R"], f"{data_id}.npy")),
            "t": np.load(os.path.join(self.paths["t"], f"{data_id}.npy")),
            "img": np.load(os.path.join(self.paths["img"], f"{data_id}.png"))
            
        }

        if self.transform:
            sample = self.transform(sample)
            
        return idx
    
txt = r"C:\Users\Drako\Desktop\Projects\3dhpe_a\data\splits\train.txt"
data_root = r"C:\Users\Drako\Desktop\Projects\3dhpe_a\data"
d = SyntheticPoseDataset(data_root=data_root, split_txt=txt)

