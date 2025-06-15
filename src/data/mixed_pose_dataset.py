import os
import sys
import json
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils
from src.utils.camera_augmentation import CameraViewpointAugmenter

# Load joint mapping
json_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "meta", "joints_mapping.json"))
with open(json_path, "r") as f:
    mapping = json.load(f)

SMPL_TO_MPII = np.array([-1 if m is None else m for m in mapping["smpl2mpii"]], dtype=np.int16)


class BasePoseDataset(ABC):
    """Abstract base class for pose datasets"""
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        pass
    
    @abstractmethod
    def get_data_type(self) -> str:
        """Return 'synthetic' or 'real'"""
        pass


class SyntheticPoseAdapter(BasePoseDataset):
    """Adapter for synthetic pose data"""
    
    def __init__(self, data_root: str, split_txt: str, transform=None, 
                 augment_2d: bool = False, **aug_kwargs):
        self.root = data_root
        self.transform = transform
        self.augment_2d = augment_2d
        
        # Load frame IDs
        with open(split_txt) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]
        
        # Set up paths
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
        
        # Camera augmentation
        self.camera_augmenter = CameraViewpointAugmenter(
            max_rotation_deg=aug_kwargs.get('camera_aug_rotation_deg', 10.0),
            max_translation_m=aug_kwargs.get('camera_aug_translation_m', 0.05)
        ) if augment_2d else None
        
        # 2D augmentation parameters
        self.noise_std = aug_kwargs.get('noise_std', 0.02)
        self.confidence_noise = aug_kwargs.get('confidence_noise', 0.005)
        self.max_shift = aug_kwargs.get('max_shift', 0.005)
        self.unreliable_joints = [10, 11, 15, 22, 23]
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def get_data_type(self) -> str:
        return 'synthetic'
    
    @staticmethod
    def mpii_to_smpl(kp2d):
        """Convert MPII to SMPL keypoint format"""
        out = np.zeros((24, 2), dtype=kp2d.dtype)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d[mpii_idx]
        return out
    
    def augment_2d_keypoints_pixel_space(self, joints_2d_pixel):
        """Apply 2D keypoint augmentation in pixel space"""
        if not self.augment_2d:
            return joints_2d_pixel

        augmented = joints_2d_pixel.clone()
        joints_np = augmented.cpu().numpy()

        # Geometric augmentation
        angle = np.random.uniform(-30.0, +30.0)
        scale = np.random.uniform(0.7, 1.3)
        tx = np.random.uniform(-0.1, +0.1) * 512.0 
        ty = np.random.uniform(-0.1, +0.1) * 512.0

        alpha = np.deg2rad(angle)
        M = np.array([
            [scale * np.cos(alpha), -scale * np.sin(alpha), tx],
            [scale * np.sin(alpha),  scale * np.cos(alpha), ty]
        ], dtype=np.float32)

        joints_homo = np.concatenate([
            joints_np, 
            np.ones((joints_np.shape[0], 1), dtype=np.float32)
        ], axis=1)

        aug_np = (M @ joints_homo.T).T
        augmented = torch.from_numpy(aug_np).float()

        # Add noise
        if self.noise_std > 0:
            pixel_noise = torch.randn_like(augmented) * (self.noise_std * 512.0)
            augmented = augmented + pixel_noise

        # Confidence noise for unreliable joints
        if self.confidence_noise > 0:
            for joint_idx in self.unreliable_joints:
                if joint_idx < augmented.shape[0] and torch.rand(1) < 0.2:
                    extra_noise = torch.randn(2) * (self.confidence_noise * 512.0)
                    augmented[joint_idx] += extra_noise

        # Global shift
        if self.max_shift > 0 and torch.rand(1) < 0.15:
            global_shift = (torch.rand(2) - 0.5) * 2 * (self.max_shift * 512.0)
            augmented += global_shift
        
        augmented = torch.clamp(augmented, 0.0, 512.0)
        return augmented
    
    def __getitem__(self, idx: int) -> Dict:
        did = self.ids[idx]
        load = lambda key: np.load(os.path.join(self.paths[key], f"{did}.npy"))

        # Load 2D keypoints (MPII format)
        kp2d_mpii = load("joints_2d")
        joints_2d_mpii = torch.tensor(kp2d_mpii, dtype=torch.float32)
        
        # Load 3D data
        joints_3d = torch.tensor(load("joints_3d"), dtype=torch.float32)
        K = torch.tensor(load("K"), dtype=torch.float32)
        R = torch.tensor(load("R"), dtype=torch.float32)
        t = torch.tensor(load("t"), dtype=torch.float32)

        # Apply camera augmentation if enabled
        if self.augment_2d and self.camera_augmenter and np.random.random() < 0.5:
            augmented_2d_mpii = self.camera_augmenter.augment_viewpoint(joints_3d, K, R, t)
            if augmented_2d_mpii is not None:
                joints_2d_mpii = augmented_2d_mpii
        
        # Convert MPII to SMPL format
        kp2d_smpl = torch.tensor(self.mpii_to_smpl(joints_2d_mpii.numpy()), dtype=torch.float32)

        # Load rotation data
        rot_mats = load("rot_mats")
        rot_mats_tensor = torch.tensor(rot_mats, dtype=torch.float32)
        rot_6d = rotation_utils.rot_matrix_to_6d(rot_mats_tensor)

        # Create sample
        sample = {
            "joints_2d_mpii": joints_2d_mpii,
            "joints_2d": kp2d_smpl,
            "joints_3d": joints_3d,
            "rot_mats": rot_mats_tensor,
            "rot_6d": rot_6d,
            "K": K,
            "R": R,
            "t": t,
            "data_type": "synthetic",
            "frame_id": did,
            "sequence_name": "synthetic_data",
            "frame_idx": -1,
            "actor_idx": 0
        }

        # RGB data is not used in training, so we skip it for consistent batching

        # Root-center 3D joints
        root = sample["joints_3d"][0].clone()
        sample["joints_3d_centered"] = sample["joints_3d"] - root
        sample["joints_3d_world"] = sample["joints_3d"].clone()

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample


class RealPoseAdapter(BasePoseDataset):
    """Adapter for real pose data (3DPW processed)"""
    
    def __init__(self, data_root: str, split: str = None, transform=None,
                 augment_2d: bool = False, **aug_kwargs):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.augment_2d = augment_2d
        
        # 2D augmentation parameters
        self.noise_std = aug_kwargs.get('noise_std', 0.01)  # Less noise for real data
        self.confidence_noise = aug_kwargs.get('confidence_noise', 0.003)
        self.max_shift = aug_kwargs.get('max_shift', 0.003)
        
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
                
                # Extract valid samples (matched detections with complete data)
                for frame_idx, frame_data in seq_data['detections'].items():
                    for actor_idx, actor_data in frame_data.items():
                        if (actor_data['matched'] and 
                            actor_data['keypoints'] is not None and
                            actor_data['joints_3d_centered'] is not None):
                            
                            self.samples.append({
                                'sequence_name': seq_data['sequence_name'],
                                'split': seq_data['split'], 
                                'frame_idx': frame_idx,
                                'actor_idx': actor_idx,
                                'data': actor_data
                            })
                            
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} real pose samples from {split} split")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_data_type(self) -> str:
        return 'real'
    
    @staticmethod
    def mpii_to_smpl(kp2d):
        """Convert MPII to SMPL keypoint format"""
        out = np.zeros((24, 2), dtype=kp2d.dtype)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d[mpii_idx]
        return out
    
    def augment_2d_keypoints_light(self, joints_2d):
        """Light augmentation for real 2D keypoints"""
        if not self.augment_2d:
            return joints_2d
            
        augmented = joints_2d.clone()
        
        # Light geometric transformation
        if torch.rand(1) < 0.3:  # 30% chance
            angle = np.random.uniform(-5.0, 5.0)  # Smaller rotation
            scale = np.random.uniform(0.95, 1.05)  # Smaller scale change
            
            alpha = np.deg2rad(angle)
            center = torch.mean(augmented, dim=0)
            
            # Center, transform, then uncenter
            centered = augmented - center
            cos_a, sin_a = np.cos(alpha), np.sin(alpha)
            rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            
            transformed = (centered @ rot_matrix.T) * scale
            augmented = transformed + center
        
        # Light noise
        if self.noise_std > 0:
            noise = torch.randn_like(augmented) * (self.noise_std * 512.0)
            augmented = augmented + noise
        
        # Small global shift
        if self.max_shift > 0 and torch.rand(1) < 0.1:
            shift = (torch.rand(2) - 0.5) * 2 * (self.max_shift * 512.0)
            augmented = augmented + shift
        
        augmented = torch.clamp(augmented, 0.0, 512.0)
        return augmented
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        data = sample_info['data']
        
        # Convert data to tensors
        joints_2d_mpii = torch.tensor(data['keypoints'], dtype=torch.float32)  # (16, 2)
        joints_3d_centered = torch.tensor(data['joints_3d_centered'], dtype=torch.float32)  # (24, 3)
        rot_mats = torch.tensor(data['rot_mats'], dtype=torch.float32)  # (24, 3, 3)
        rot_6d = torch.tensor(data['rot_6d'], dtype=torch.float32)  # (24, 6)
        K = torch.tensor(data['K'], dtype=torch.float32)
        R = torch.tensor(data['R'], dtype=torch.float32)
        t = torch.tensor(data['t'], dtype=torch.float32)
        
        # Apply light augmentation to 2D keypoints
        joints_2d_mpii = self.augment_2d_keypoints_light(joints_2d_mpii)
        
        # Convert MPII to SMPL format
        joints_2d_smpl = torch.tensor(self.mpii_to_smpl(joints_2d_mpii.numpy()), dtype=torch.float32)
        
        # Create sample - FIXED: Added missing frame_id key
        sample = {
            "joints_2d_mpii": joints_2d_mpii,
            "joints_2d": joints_2d_smpl,
            "joints_3d_centered": joints_3d_centered,
            "rot_mats": rot_mats,
            "rot_6d": rot_6d,
            "K": K,
            "R": R,
            "t": t,
            "data_type": "real",
            "frame_id": f"{sample_info['sequence_name']}_{sample_info['frame_idx']}_{sample_info['actor_idx']}",  # FIXED: Added frame_id
            "sequence_name": sample_info['sequence_name'],
            "frame_idx": sample_info['frame_idx'],
            "actor_idx": sample_info['actor_idx']
        }
        
        # For consistency with synthetic data, also provide joints_3d and joints_3d_world
        # (real data is already root-centered, so we set world = centered)
        sample["joints_3d"] = joints_3d_centered.clone()
        sample["joints_3d_world"] = joints_3d_centered.clone()
        
        # RGB data is not used in training and not available for real data
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class MixedPoseDataset(Dataset):
    """Mixed dataset combining synthetic and real pose data"""
    
    def __init__(self, 
                 data_root: str,
                 synthetic_split_txt: str = None,
                 real_split: str = None,
                 synthetic_ratio: float = 0.7,
                 transform=None,
                 augment_2d: bool = False,
                 **aug_kwargs):
        """
        Args:
            data_root: Root directory containing both synthetic and real data
            synthetic_split_txt: Path to synthetic data split file
            real_split: Split name for real data ('train', 'validation', 'test')
            synthetic_ratio: Ratio of synthetic samples (0.0 to 1.0)
            transform: Transform to apply to samples
            augment_2d: Whether to apply 2D augmentations
            **aug_kwargs: Additional augmentation parameters
        """
        self.synthetic_ratio = synthetic_ratio
        self.transform = transform
        
        # Initialize datasets
        self.datasets = {}
        
        # Add synthetic dataset if requested
        if synthetic_split_txt and synthetic_ratio > 0:
            self.datasets['synthetic'] = SyntheticPoseAdapter(
                data_root=data_root,
                split_txt=synthetic_split_txt,
                transform=None,  # Apply transform at mixed level
                augment_2d=augment_2d,
                **aug_kwargs
            )
        
        # Add real dataset if requested  
        if real_split and synthetic_ratio < 1.0:
            self.datasets['real'] = RealPoseAdapter(
                data_root=data_root,
                split=real_split,
                transform=None,  # Apply transform at mixed level
                augment_2d=augment_2d,
                **aug_kwargs
            )
        
        if not self.datasets:
            raise ValueError("At least one dataset (synthetic or real) must be specified")
        
        # Calculate total length and mixing strategy
        self._setup_mixing_strategy()
        
        print(f"Mixed dataset initialized:")
        for name, dataset in self.datasets.items():
            print(f"  {name}: {len(dataset)} samples")
        print(f"  Total mixed samples: {len(self)}")
        print(f"  Synthetic ratio: {synthetic_ratio:.2f}")
    
    def _setup_mixing_strategy(self):
        """Setup the mixing strategy for the datasets"""
        total_synthetic = len(self.datasets.get('synthetic', []))
        total_real = len(self.datasets.get('real', []))
        
        if 'synthetic' in self.datasets and 'real' in self.datasets:
            # Both datasets available - mix according to ratio
            target_synthetic = int(max(total_synthetic, total_real) * self.synthetic_ratio)
            target_real = int(max(total_synthetic, total_real) * (1 - self.synthetic_ratio))
            
            # Create sampling indices with repetition if needed
            if total_synthetic > 0:
                synthetic_indices = np.random.choice(total_synthetic, target_synthetic, replace=True)
                self.synthetic_samples = [(idx, 'synthetic') for idx in synthetic_indices]
            else:
                self.synthetic_samples = []
                
            if total_real > 0:
                real_indices = np.random.choice(total_real, target_real, replace=True)
                self.real_samples = [(idx, 'real') for idx in real_indices]
            else:
                self.real_samples = []
            
            # Combine and shuffle
            self.mixed_indices = self.synthetic_samples + self.real_samples
            random.shuffle(self.mixed_indices)
            
        elif 'synthetic' in self.datasets:
            # Only synthetic data
            self.mixed_indices = [(idx, 'synthetic') for idx in range(total_synthetic)]
        else:
            # Only real data
            self.mixed_indices = [(idx, 'real') for idx in range(total_real)]
    
    def __len__(self) -> int:
        return len(self.mixed_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        # Get the actual dataset index and type
        actual_idx, dataset_type = self.mixed_indices[idx]
        
        # Get sample from appropriate dataset
        sample = self.datasets[dataset_type][actual_idx]
        
        # Add mixing metadata
        sample['mixed_idx'] = idx
        sample['source_dataset'] = dataset_type
        sample['source_idx'] = actual_idx
        
        # Apply transform at mixed level
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples by dataset type"""
        distribution = defaultdict(int)
        for _, dataset_type in self.mixed_indices:
            distribution[dataset_type] += 1
        return dict(distribution)
    
    def reshuffle(self):
        """Reshuffle the mixing order (call between epochs)"""
        self._setup_mixing_strategy()


# Convenience function for easy dataset creation
def create_mixed_dataset(data_root: str,
                        mode: str = 'mixed',
                        split: str = 'train',
                        synthetic_ratio: float = 0.7,
                        transform=None,
                        augment_2d: bool = True,
                        **aug_kwargs) -> Union[MixedPoseDataset, SyntheticPoseAdapter, RealPoseAdapter]:
    """
    Create a pose dataset with specified configuration
    
    Args:
        data_root: Root data directory
        mode: 'mixed', 'synthetic', or 'real'
        split: Data split ('train', 'val', 'test')
        synthetic_ratio: Ratio of synthetic samples when mode='mixed'
        transform: Transform to apply
        augment_2d: Whether to apply 2D augmentations
        **aug_kwargs: Additional augmentation parameters
    
    Returns:
        Appropriate dataset instance
    """
    if mode == 'synthetic':
        split_txt = os.path.join(data_root, "splits", f"{split}.txt")
        return SyntheticPoseAdapter(
            data_root=data_root,
            split_txt=split_txt,
            transform=transform,
            augment_2d=augment_2d,
            **aug_kwargs
        )
    
    elif mode == 'real':
        # Map split names
        real_split = {'train': 'train', 'val': 'validation', 'test': 'test'}.get(split, split)
        return RealPoseAdapter(
            data_root=data_root,
            split=real_split,
            transform=transform,
            augment_2d=augment_2d,
            **aug_kwargs
        )
    
    elif mode == 'mixed':
        split_txt = os.path.join(data_root, "splits", f"{split}.txt")
        real_split = {'train': 'train', 'val': 'validation', 'test': 'test'}.get(split, split)
        
        return MixedPoseDataset(
            data_root=data_root,
            synthetic_split_txt=split_txt,
            real_split=real_split,
            synthetic_ratio=synthetic_ratio,
            transform=transform,
            augment_2d=augment_2d,
            **aug_kwargs
        )
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'mixed', 'synthetic', 'real'")