import os, json, numpy as np
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils

json_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..",  "data", "meta", "joints_mapping.json"))

with open(json_path, "r") as f:
    mapping = json.load(f)

SMPL_TO_MPII = np.array(
    [-1 if m is None else m for m in mapping["smpl2mpii"]],
    dtype=np.int16
)


class SyntheticPoseDataset(Dataset):

    def __init__(self, data_root, split_txt, transform=None):
        self.root = data_root
        self.transform = transform
        with open(split_txt) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

        j = lambda *p: os.path.join(data_root, *p)
        self.paths = dict(
            joints_2d=j("annotations", "joints_2d"),
            joints_3d=j("annotations", "joints_3d"),
            rot_mats=j("annotations", "rot_mats"),
            K=j("annotations", "K"),
            R=j("annotations", "R"),
            t=j("annotations", "t"),
            rgb=j("raw", "rgb")
        )

    def __len__(self): return len(self.ids)

    @staticmethod
    def mpii_to_smpl(kp2d):
        out = np.zeros((24, 2), dtype=kp2d.dtype)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d[mpii_idx]
        return out


    def __getitem__(self, idx):
        did = self.ids[idx]
        load = lambda key: np.load(os.path.join(self.paths[key], f"{did}.npy"))

        kp2d_mpii = load("joints_2d")
        kp2d_smpl = self.mpii_to_smpl(kp2d_mpii)
        
        rot_mats = load("rot_mats")
        rot_mats_tensor = torch.tensor(rot_mats, dtype=torch.float32)
        rot_6d = rotation_utils.rot_matrix_to_6d(rot_mats_tensor)

        sample = {
            "joints_2d_mpii": torch.tensor(kp2d_mpii, dtype=torch.float32),
            "joints_2d":      torch.tensor(kp2d_smpl, dtype=torch.float32),
            "joints_3d":      torch.tensor(load("joints_3d"), dtype=torch.float32),
            "rot_mats":       rot_mats_tensor,
            "rot_6d":         rot_6d,
            "K":              torch.tensor(load("K"), dtype=torch.float32),
            "R":              torch.tensor(load("R"), dtype=torch.float32),
            "t":              torch.tensor(load("t"), dtype=torch.float32)
        }

        rgb_path = os.path.join(self.paths["rgb"], f"{did}.png")
        if os.path.exists(rgb_path):
            sample["rgb"] = torch.tensor(np.array(Image.open(rgb_path)))

        root = sample["joints_3d"][0].clone()
        sample["joints_3d_centered"] = sample["joints_3d"] - root
        sample["joints_3d_world"]    = sample["joints_3d"].clone()

        if self.transform:
            sample = self.transform(sample)

        return sample
