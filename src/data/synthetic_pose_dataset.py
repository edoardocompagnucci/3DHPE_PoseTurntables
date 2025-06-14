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

# FIXED: Consistent joint definitions
SMPL_EXTREMITY_JOINTS = [7, 8, 10, 11, 18, 19, 20, 21, 22, 23]  # ankles, feet, wrists, hands
SMPL_CORE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17]  # pelvis, hips, knees, spine, shoulders, elbows

# FIXED: Map SMPL extremities to MPII indices for corruption
def get_mpii_extremity_indices():
    """Get MPII indices that correspond to SMPL extremity joints"""
    mpii_extremities = []
    for smpl_idx in SMPL_EXTREMITY_JOINTS:
        mpii_idx = SMPL_TO_MPII[smpl_idx] if smpl_idx < len(SMPL_TO_MPII) else -1
        if mpii_idx >= 0:  # Valid mapping exists
            mpii_extremities.append(mpii_idx)
    return mpii_extremities

MPII_EXTREMITY_JOINTS = get_mpii_extremity_indices()


class SyntheticPoseDataset(Dataset):
    def __init__(self, data_root, split_txt, transform=None, 
                 augment_2d=False,
                 noise_std=0.02,
                 confidence_noise=0.005,
                 max_shift=0.005,
                 camera_aug_rotation_deg=8.0,
                 camera_aug_translation_m=0.02,
                 break_domain_discrimination=True,
                 corruption_schedule='progressive'):  # NEW: corruption scheduling
        
        self.root = data_root
        self.transform = transform
        self.augment_2d = augment_2d
        self.noise_std = noise_std
        self.confidence_noise = confidence_noise
        self.max_shift = max_shift
        self.break_domain_discrimination = break_domain_discrimination
        self.corruption_schedule = corruption_schedule
        
        # Use consistent joint definitions
        self.extremity_joints_smpl = SMPL_EXTREMITY_JOINTS
        self.core_joints_smpl = SMPL_CORE_JOINTS
        self.extremity_joints_mpii = MPII_EXTREMITY_JOINTS

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
        
        print(f"📊 SyntheticPoseDataset initialized:")
        print(f"   Samples: {len(self.ids)}")
        print(f"   Domain breaking: {break_domain_discrimination}")
        print(f"   MPII extremities: {self.extremity_joints_mpii}")
        
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def mpii_to_smpl(kp2d):
        out = np.zeros((24, 2), dtype=kp2d.dtype)
        for smpl_idx, mpii_idx in enumerate(SMPL_TO_MPII):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d[mpii_idx]
        return out
    
    def generate_synthetic_confidence(self, joints_2d_mpii, corrupted_joints, base_confidence=0.9):
        """
        FIXED: Generate realistic confidence scores that don't leak domain info
        """
        num_joints = joints_2d_mpii.shape[0]
        
        # FIXED: More realistic confidence distribution
        # Real data typically has confidence 0.4-0.9, so we match this range
        if len(corrupted_joints) > 0:
            # When corrupted, generate confidence similar to real data
            confidence = torch.rand(num_joints) * 0.5 + 0.4  # Range 0.4-0.9
        else:
            # Clean synthetic data can have higher confidence but not perfect
            confidence = torch.rand(num_joints) * 0.2 + 0.8  # Range 0.8-1.0
        
        # FIXED: Reduce confidence for actually corrupted joints
        for joint_idx in corrupted_joints:
            if joint_idx < num_joints:
                # Corrupted joints get lower confidence
                confidence[joint_idx] = torch.rand(()).item() * 0.4 + 0.3  # Range 0.3-0.7
        
        # Add slight noise to make it realistic
        confidence += torch.randn(num_joints) * 0.05
        confidence = torch.clamp(confidence, 0.3, 1.0)
        
        return confidence
        
    def get_corruption_params(self, epoch=None):
        """Get corruption parameters based on epoch and schedule"""
        if not self.break_domain_discrimination:
            return 0.0, 0.0  # No corruption
            
        if epoch is None:
            corruption_prob = 0.7
            corruption_strength = 1.0
        else:
            if self.corruption_schedule == 'progressive':
                # Start high, gradually reduce
                corruption_prob = max(0.4, 0.8 - epoch * 0.008)
                corruption_strength = max(0.5, 1.2 - epoch * 0.01)
            elif self.corruption_schedule == 'constant':
                corruption_prob = 0.6
                corruption_strength = 0.8
            else:  # adaptive
                # High corruption early, then stabilize
                if epoch < 20:
                    corruption_prob = 0.8
                    corruption_strength = 1.0
                else:
                    corruption_prob = 0.5
                    corruption_strength = 0.7
                    
        return corruption_prob, corruption_strength
        
    def apply_extremity_corruption(self, joints_2d_mpii, epoch=None):
        """
        FIXED: Improved extremity corruption with better mapping and strategies
        """
        if not self.break_domain_discrimination:
            return joints_2d_mpii, []
        
        corrupted_joints = []
        joints_2d = joints_2d_mpii.clone()
        
        corruption_prob, corruption_strength = self.get_corruption_params(epoch)
        
        # Apply corruption with current probability
        if torch.rand(()).item() < corruption_prob:
            # FIXED: Select extremities based on proper MPII mapping
            available_extremities = [idx for idx in self.extremity_joints_mpii if idx < joints_2d.shape[0]]
            
            if len(available_extremities) > 0:
                # Corrupt 50-80% of available extremities
                num_to_corrupt = max(1, int(len(available_extremities) * (0.5 + torch.rand(()).item() * 0.3)))
                joints_to_corrupt = torch.randperm(len(available_extremities))[:num_to_corrupt]
                
                for idx in joints_to_corrupt:
                    joint_idx = available_extremities[idx]
                    corrupted_joints.append(joint_idx)
                    
                    # FIXED: Multiple corruption strategies with proper scaling
                    corruption_type = torch.rand(()).item()
                    
                    if corruption_type < 0.4:
                        # Strategy 1: Systematic bias (mimicking detection drift)
                        bias_direction = torch.randn(2) * 0.1 * corruption_strength
                        # Add position-dependent bias
                        if joints_2d[joint_idx, 0] > 0:  # Right side joints
                            bias_direction[0] += 0.05 * corruption_strength
                        else:  # Left side joints  
                            bias_direction[0] -= 0.05 * corruption_strength
                        joints_2d[joint_idx] += bias_direction
                        
                    elif corruption_type < 0.7:
                        # Strategy 2: High noise (detection uncertainty)
                        noise_level = (0.05 + torch.rand(()).item() * 0.1) * corruption_strength
                        joints_2d[joint_idx] += torch.randn(2) * noise_level
                        
                    else:
                        # Strategy 3: Detection failure (complete miss)
                        if torch.rand(()).item() < 0.2 * corruption_strength:
                            # Occasionally completely wrong position
                            joints_2d[joint_idx] = torch.rand(2) * 2.0 - 1.0  # Random position in [-1,1]
                        else:
                            # Or copy from nearby joint with large offset
                            if len(available_extremities) > 1:
                                other_joint = available_extremities[torch.randint(0, len(available_extremities), (1,)).item()]
                                if other_joint != joint_idx and other_joint < joints_2d.shape[0]:
                                    joints_2d[joint_idx] = joints_2d[other_joint] + torch.randn(2) * 0.15 * corruption_strength
            
            # FIXED: Add correlated errors between connected extremities
            self._add_correlated_errors(joints_2d, corrupted_joints, corruption_strength)
        
        # Keep joints in reasonable range
        joints_2d = torch.clamp(joints_2d, -1.5, 1.5)
        
        return joints_2d, corrupted_joints
    
    def _add_correlated_errors(self, joints_2d, corrupted_joints, corruption_strength):
        """Add correlated errors between connected joints"""
        # This would need the actual MPII joint structure
        # For now, add some general correlation between nearby joints
        if len(corrupted_joints) > 0:
            # If multiple joints corrupted, make them slightly correlated
            if len(corrupted_joints) >= 2:
                for i in range(1, len(corrupted_joints)):
                    if corrupted_joints[i] < joints_2d.shape[0] and corrupted_joints[0] < joints_2d.shape[0]:
                        # Add small correlation
                        correlation = torch.randn(2) * 0.03 * corruption_strength
                        joints_2d[corrupted_joints[i]] += correlation

    def __getitem__(self, idx):
        did = self.ids[idx]
        load = lambda key: np.load(os.path.join(self.paths[key], f"{did}.npy"))

        kp2d_mpii = load("joints_2d")  # Original stored 2D keypoints
        joints_2d_mpii = torch.tensor(kp2d_mpii, dtype=torch.float32)
        
        joints_3d = torch.tensor(load("joints_3d"), dtype=torch.float32)
        K = torch.tensor(load("K"), dtype=torch.float32)
        R = torch.tensor(load("R"), dtype=torch.float32)
        t = torch.tensor(load("t"), dtype=torch.float32)

        # Store original clean keypoints for debugging
        joints_2d_mpii_clean = joints_2d_mpii.clone()
        
        if self.augment_2d:
            # First apply camera augmentation
            if np.random.random() < 0.5:
                augmented_2d_mpii = self.camera_augmenter.augment_viewpoint(
                    joints_3d, K, R, t
                )
                if augmented_2d_mpii is not None:
                    joints_2d_mpii = augmented_2d_mpii
            
            # FIXED: Apply extremity-specific corruption with epoch info
            # Note: We don't have epoch info here, so we'll use None
            joints_2d_mpii, corrupted_joints = self.apply_extremity_corruption(joints_2d_mpii)
            
            # Generate confidence scores based on corruption
            confidence_scores = self.generate_synthetic_confidence(joints_2d_mpii, corrupted_joints)
        else:
            # Validation: minimal corruption but still some variation to avoid perfect domain signals
            if self.break_domain_discrimination and torch.rand(()).item() < 0.1:
                joints_2d_mpii, corrupted_joints = self.apply_extremity_corruption(joints_2d_mpii)
                confidence_scores = self.generate_synthetic_confidence(joints_2d_mpii, corrupted_joints)
            else:
                # High but not perfect confidence
                confidence_scores = torch.rand(16) * 0.15 + 0.85  # Range 0.85-1.0
        
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
            "t": t,
            "sample_id": did,
            "avg_confidence": confidence_scores.mean().item(),
            "confidence_scores": confidence_scores,
            "sequence_name": "synthetic",
            "frame_idx": -1,
            "actor_idx": -1,
        }

        root = sample["joints_3d"][0].clone()
        sample["joints_3d_centered"] = sample["joints_3d"] - root
        sample["joints_3d_world"] = sample["joints_3d"].clone()

        if self.transform:
            sample = self.transform(sample)

        return sample