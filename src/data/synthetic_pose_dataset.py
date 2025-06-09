import os
import json
import numpy as np
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils
from src.utils.camera_augmentation import CameraViewpointAugmenter

json_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "meta", "joints_mapping.json"))
with open(json_path, "r") as f:
    mapping = json.load(f)

SMPL_TO_MPII = np.array([-1 if m is None else m for m in mapping["smpl2mpii"]], dtype=np.int16)


class SyntheticPoseDataset(Dataset):
    def __init__(self, data_root, split_txt, transform=None, 
                 augment_2d=False,
                 noise_std=0.02,
                 confidence_noise=0.005,
                 max_shift=0.005,
                 camera_aug_rotation_deg=10.0,
                 camera_aug_translation_m=0.05):
        
        self.root = data_root
        self.transform = transform
        self.augment_2d = augment_2d
        self.noise_std = noise_std
        self.confidence_noise = confidence_noise
        self.max_shift = max_shift
 
        self.unreliable_joints = [10, 11, 15, 22, 23]

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

        self.camera_augmenter = CameraViewpointAugmenter(
            max_rotation_deg=camera_aug_rotation_deg,
            max_translation_m=camera_aug_translation_m
        )
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def mpii_to_smpl(kp2d):
        out = np.zeros((24, 2), dtype=kp2d.dtype)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d[mpii_idx]
        return out

    def augment_2d_keypoints_pixel_space(self, joints_2d_pixel):

        if not self.augment_2d:
            return joints_2d_pixel

        augmented = joints_2d_pixel.clone()

        joints_np = augmented.cpu().numpy()

        angle = np.random.uniform(-30.0, +30.0)
        scale = np.random.uniform(0.7, 1.3)
        tx    = np.random.uniform(-0.1, +0.1) * 512.0 
        ty    = np.random.uniform(-0.1, +0.1) * 512.0

        alpha = np.deg2rad(angle)

        M = np.array([
            [scale * np.cos(alpha), -scale * np.sin(alpha), tx],
            [scale * np.sin(alpha),  scale * np.cos(alpha), ty]
        ], dtype=np.float32)   # (2×3)

        joints_homo = np.concatenate([
            joints_np, 
            np.ones((joints_np.shape[0], 1), dtype=np.float32)
        ], axis=1)

        aug_np = (M @ joints_homo.T).T

        augmented = torch.from_numpy(aug_np).float()

        if self.noise_std > 0:
            pixel_noise = torch.randn_like(augmented) * (self.noise_std * 512.0)
            augmented = augmented + pixel_noise

        if self.confidence_noise > 0:
            for joint_idx in self.unreliable_joints:
                if joint_idx < augmented.shape[0] and torch.rand(1) < 0.2:
                    extra_noise = torch.randn(2) * (self.confidence_noise * 512.0)
                    augmented[joint_idx] += extra_noise

        if self.max_shift > 0:
            if torch.rand(1) < 0.15:
                global_shift = (torch.rand(2) - 0.5) * 2 * (self.max_shift * 512.0)
                augmented += global_shift
        augmented = torch.clamp(augmented, 0.0, 512.0)

        return augmented
    
    def __getitem__(self, idx):
        did = self.ids[idx]
        load = lambda key: np.load(os.path.join(self.paths[key], f"{did}.npy"))

        kp2d_mpii = load("joints_2d")  # Original stored 2D keypoints
        joints_2d_mpii = torch.tensor(kp2d_mpii, dtype=torch.float32)
        
        joints_3d = torch.tensor(load("joints_3d"), dtype=torch.float32)
        K = torch.tensor(load("K"), dtype=torch.float32)
        R = torch.tensor(load("R"), dtype=torch.float32)
        t = torch.tensor(load("t"), dtype=torch.float32)

        if self.augment_2d:
            # 50% chance to use camera augmentation, 50% chance to keep original
            if np.random.random() < 0.5:
                augmented_2d_mpii = self.camera_augmenter.augment_viewpoint(
                    joints_3d, K, R, t
                )
                if augmented_2d_mpii is not None:
                    joints_2d_mpii = augmented_2d_mpii
            #joints_2d_mpii = self.augment_2d_keypoints_pixel_space(joints_2d_mpii)
            # Otherwise keep original stored 2D keypoints (joints_2d_mpii unchanged)
        
        kp2d_smpl = torch.tensor(self.mpii_to_smpl(joints_2d_mpii.numpy()), dtype=torch.float32)

        rot_mats = load("rot_mats")
        rot_mats_tensor = torch.tensor(rot_mats, dtype=torch.float32)
        rot_6d = rotation_utils.rot_matrix_to_6d(rot_mats_tensor)

        sample = {
            "joints_2d_mpii": joints_2d_mpii,
            "joints_2d": kp2d_smpl,
            "joints_3d": joints_3d,
            "rot_mats": rot_mats_tensor,
            "rot_6d": rot_6d,
            "K": K,
            "R": R,
            "t": t
        }

        rgb_path = os.path.join(self.paths["rgb"], f"{did}.png")
        if os.path.exists(rgb_path):
            sample["rgb"] = torch.tensor(np.array(Image.open(rgb_path)))

        root = sample["joints_3d"][0].clone()
        sample["joints_3d_centered"] = sample["joints_3d"] - root
        sample["joints_3d_world"] = sample["joints_3d"].clone()

        if self.transform:
            sample = self.transform(sample)

        return sample