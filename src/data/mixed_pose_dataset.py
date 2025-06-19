import os
import sys
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

# Load joint mapping
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils

# Load joint mapping
json_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "meta", "joints_mapping.json"))
with open(json_path, "r") as f:
    mapping = json.load(f)

SMPL_TO_MPII = np.array([-1 if m is None else m for m in mapping["smpl2mpii"]], dtype=np.int16)


class SyntheticPoseDataset:
    """Simple adapter for synthetic pose data"""
    _ZERO_EPS = 1e-3
    _MAX_ZEROS = 6
    
    def __init__(self, data_root: str, split_txt: str):
        self.root = data_root
        
        # Load frame IDs
        with open(split_txt) as f:
            all_ids = [ln.strip() for ln in f if ln.strip()]
        
        # Set up paths
        self.paths = {
            'joints_2d': os.path.join(data_root, "annotations", "joints_2d"),
            'joints_3d': os.path.join(data_root, "annotations", "joints_3d"),
            'rot_mats': os.path.join(data_root, "annotations", "rot_mats"),
        }
        
        # Filter out samples with too many zero keypoints
        self.ids = []
        for did in all_ids:
            kp2d_mpii = np.load(os.path.join(self.paths['joints_2d'], f"{did}.npy"))
            num_zeros = np.sum(np.abs(kp2d_mpii) < self._ZERO_EPS, axis=1).sum()
            if num_zeros <= self._MAX_ZEROS:
                self.ids.append(did)
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def mpii_to_smpl(kp2d):
        """Convert MPII to SMPL keypoint format"""
        # Ensure input is numpy array
        kp2d = np.array(kp2d, dtype=np.float32)
        out = np.zeros((24, 2), dtype=np.float32)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0 and mpii_idx < len(kp2d):
                out[smpl_idx] = kp2d[mpii_idx]
        return out
    
    def __getitem__(self, idx):
        did = self.ids[idx]
        
        # Load data
        kp2d_mpii = np.load(os.path.join(self.paths['joints_2d'], f"{did}.npy"))
        joints_3d = np.load(os.path.join(self.paths['joints_3d'], f"{did}.npy"))
        rot_mats = np.load(os.path.join(self.paths['rot_mats'], f"{did}.npy"))
        
        # Convert to SMPL format
        joints_2d = self.mpii_to_smpl(kp2d_mpii)
        
        # Center 3D joints
        root = joints_3d[0].copy()
        joints_3d_centered = joints_3d - root
        
        # Convert rotation matrices to 6D
        rot_mats_tensor = torch.tensor(rot_mats, dtype=torch.float32)
        rot_6d = rotation_utils.rot_matrix_to_6d(rot_mats_tensor)
        
        return {
            "joints_2d": torch.tensor(joints_2d, dtype=torch.float32),
            "joints_3d_centered": torch.tensor(joints_3d_centered, dtype=torch.float32),
            "joints_3d_world": torch.tensor(joints_3d, dtype=torch.float32),
            "rot_mats": rot_mats_tensor,
            "rot_6d": rot_6d,
            "data_type": "synthetic"
        }


class RealPoseDataset:
    """Simple adapter for real pose data (3DPW processed)"""
    
    def __init__(self, data_root: str, split: str):
        self.data_root = data_root
        self.split = split
        
        # Load 3DPW processed data
        detections_dir = os.path.join(data_root, "3DPW_processed", "detections")
        self.samples = []
        
        if not os.path.exists(detections_dir):
            raise FileNotFoundError(f"3DPW processed detections not found: {detections_dir}")
        
        # Load all pickle files for the specified split
        for filename in os.listdir(detections_dir):
            if not filename.endswith('_detections.pkl'):
                continue
                
            file_split = filename.split('_')[0]
            if split and file_split != split:
                continue
                
            filepath = os.path.join(detections_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    seq_data = pickle.load(f)
                
                # Extract valid samples
                for frame_idx, frame_data in seq_data['detections'].items():
                    for actor_idx, actor_data in frame_data.items():
                        if (actor_data['matched'] and 
                            actor_data['keypoints'] is not None and
                            actor_data['joints_3d_centered'] is not None):
                            
                            self.samples.append(actor_data)
                            
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} real pose samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def mpii_to_smpl(kp2d):
        """Convert MPII to SMPL keypoint format"""
        # Ensure input is numpy array
        kp2d = np.array(kp2d, dtype=np.float32)
        out = np.zeros((24, 2), dtype=np.float32)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0 and mpii_idx < len(kp2d):
                out[smpl_idx] = kp2d[mpii_idx]
        return out
    
    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # Convert MPII to SMPL format
        joints_2d_mpii = np.array(data['keypoints'], dtype=np.float32)
        joints_2d_smpl = self.mpii_to_smpl(joints_2d_mpii)
        
        # Get data and ensure they are numpy arrays
        joints_3d_centered = np.array(data['joints_3d_centered'], dtype=np.float32)
        rot_mats = np.array(data['rot_mats'], dtype=np.float32)
        rot_6d = np.array(data['rot_6d'], dtype=np.float32)
        
        # For real data, we don't have world coordinates, so we use centered as world
        joints_3d_world = joints_3d_centered
        
        return {
            "joints_2d": torch.tensor(joints_2d_smpl, dtype=torch.float32),
            "joints_3d_centered": torch.tensor(joints_3d_centered, dtype=torch.float32),
            "joints_3d_world": torch.tensor(joints_3d_world, dtype=torch.float32),
            "rot_mats": torch.tensor(rot_mats, dtype=torch.float32),
            "rot_6d": torch.tensor(rot_6d, dtype=torch.float32),
            "data_type": "real"
        }


class MixedPoseDataset(Dataset):
    """Mixed dataset combining synthetic and real pose data with flexible ratios"""
    
    def __init__(self, 
                 data_root: str,
                 synthetic_split_txt: Optional[str] = None,
                 real_split: Optional[str] = None,
                 synthetic_ratio: float = 1.0,
                 real_ratio: float = 1.0,
                 transform=None):
        """
        Args:
            data_root: Root directory containing both synthetic and real data
            synthetic_split_txt: Path to synthetic data split file
            real_split: Split name for real data ('train', 'validation', 'test')
            synthetic_ratio: Ratio of synthetic samples to include (0.0 to 1.0)
            real_ratio: Ratio of real samples to include (0.0 to 1.0)
            transform: Transform to apply to samples
        """
        self.transform = transform
        self.indices = []
        
        # Load synthetic dataset if requested
        if synthetic_split_txt and synthetic_ratio > 0:
            synthetic_dataset = SyntheticPoseDataset(data_root, synthetic_split_txt)
            num_synthetic = int(len(synthetic_dataset) * synthetic_ratio)
            
            # Add synthetic indices
            synthetic_indices = np.random.choice(len(synthetic_dataset), num_synthetic, replace=False)
            for idx in synthetic_indices:
                self.indices.append(('synthetic', idx, synthetic_dataset))
            
            print(f"Added {num_synthetic} synthetic samples (ratio: {synthetic_ratio:.2f})")
        
        # Load real dataset if requested
        if real_split and real_ratio > 0:
            real_dataset = RealPoseDataset(data_root, real_split)
            num_real = int(len(real_dataset) * real_ratio)
            
            # Add real indices
            real_indices = np.random.choice(len(real_dataset), num_real, replace=False)
            for idx in real_indices:
                self.indices.append(('real', idx, real_dataset))
            
            print(f"Added {num_real} real samples (ratio: {real_ratio:.2f})")
        
        # Shuffle all indices
        np.random.shuffle(self.indices)
        
        print(f"Total mixed samples: {len(self)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        dataset_type, actual_idx, dataset = self.indices[idx]
        
        # Get sample from appropriate dataset
        sample = dataset[actual_idx]
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_mixed_dataset(data_root: str,
                        split: str = 'train',
                        synthetic_ratio: float = 1.0,
                        real_ratio: float = 1.0,
                        transform=None):
    """
    Create a mixed pose dataset with specified ratios
    
    Args:
        data_root: Root data directory
        split: Data split ('train', 'val', 'test')
        synthetic_ratio: Ratio of synthetic samples (0.0 to 1.0)
        real_ratio: Ratio of real samples (0.0 to 1.0)
        transform: Transform to apply
    
    Returns:
        MixedPoseDataset instance
    """
    synthetic_split_txt = None
    real_split = None
    
    if synthetic_ratio > 0:
        synthetic_split_txt = os.path.join(data_root, "splits", f"{split}.txt")
    
    if real_ratio > 0:
        real_split = {'train': 'train', 'val': 'validation', 'test': 'test'}.get(split, split)
    
    return MixedPoseDataset(
        data_root=data_root,
        synthetic_split_txt=synthetic_split_txt,
        real_split=real_split,
        synthetic_ratio=synthetic_ratio,
        real_ratio=real_ratio,
        transform=transform
    )