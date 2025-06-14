"""
FIXED: 3DPW Dataset Loader for Domain Mixing Training
"""

import os
import pickle
import json
import numpy as np
import sys
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils

# FIXED: Consistent joint definitions
SMPL_EXTREMITY_JOINTS = [7, 8, 10, 11, 18, 19, 20, 21, 22, 23]  # ankles, feet, wrists, hands
SMPL_CORE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17]  # pelvis, hips, knees, spine, shoulders, elbows


class ThreeDPWDataset(Dataset):
  
    def __init__(self, data_root, split="train", transform=None, min_confidence=0.3):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.min_confidence = min_confidence

        self.detections_dir = os.path.join(data_root, "3DPW_processed", "detections")

        mapping_path = os.path.join(data_root, "meta", "joints_mapping.json")
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        self.smpl_to_mpii = np.array([-1 if m is None else m for m in mapping["smpl2mpii"]], dtype=np.int16)

        self.samples = self._load_samples()
        
        print(f"✅ Loaded {len(self.samples)} valid 3DPW samples from {split} split")
        
    def _load_samples(self):
        samples = []
        
        if not os.path.exists(self.detections_dir):
            print(f"❌ Detections directory not found: {self.detections_dir}")
            return samples
        
        detection_files = [f for f in os.listdir(self.detections_dir) 
                          if f.startswith(f"{self.split}_") and f.endswith("_detections.pkl")]
        
        print(f"📂 Found {len(detection_files)} detection files for {self.split} split")
        
        for det_file in detection_files:
            det_path = os.path.join(self.detections_dir, det_file)
            
            try:
                with open(det_path, 'rb') as f:
                    seq_data = pickle.load(f)
                
                seq_name = seq_data['sequence_name']

                for frame_idx, frame_data in seq_data['detections'].items():
                    for actor_idx, actor_data in frame_data.items():

                        if (actor_data['matched'] and 
                            actor_data['keypoints'] is not None and
                            actor_data['joints_3d_centered'] is not None and
                            actor_data['rot_6d'] is not None):

                            scores = np.array(actor_data['scores'])
                            if np.mean(scores) >= self.min_confidence:
                                sample_id = f"{seq_name}_frame{frame_idx:05d}_actor{actor_idx}"
                                samples.append({
                                    'sample_id': sample_id,
                                    'sequence_name': seq_name,
                                    'frame_idx': frame_idx,
                                    'actor_idx': actor_idx,
                                    'detection_file': det_path,
                                    'avg_confidence': np.mean(scores),
                                    'confidence_scores': scores  # Store per-joint scores
                                })
            
            except Exception as e:
                print(f"⚠️  Error loading {det_file}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def mpii_to_smpl(self, kp2d_mpii):
        out = np.zeros((24, 2), dtype=kp2d_mpii.dtype)
        for smpl_idx, mpii_idx in enumerate(self.smpl_to_mpii):
            if mpii_idx >= 0:
                out[smpl_idx] = kp2d_mpii[mpii_idx]
        return out
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        with open(sample_info['detection_file'], 'rb') as f:
            seq_data = pickle.load(f)

        frame_idx = sample_info['frame_idx']
        actor_idx = sample_info['actor_idx']
        actor_data = seq_data['detections'][frame_idx][actor_idx]

        kp2d_mpii = np.array(actor_data['keypoints'], dtype=np.float32)
        joints_2d_mpii = torch.tensor(kp2d_mpii, dtype=torch.float32)

        kp2d_smpl = self.mpii_to_smpl(kp2d_mpii)
        joints_2d = torch.tensor(kp2d_smpl, dtype=torch.float32)

        joints_3d_centered = np.array(actor_data['joints_3d_centered'], dtype=np.float32)
        joints_3d_centered = torch.tensor(joints_3d_centered, dtype=torch.float32)

        rotations_6d = np.array(actor_data['rot_6d'], dtype=np.float32)  
        rot_6d = torch.tensor(rotations_6d, dtype=torch.float32)

        rot_matrices = rotation_utils.rot_6d_to_matrix(rot_6d.reshape(1, 24, 6)).squeeze(0)
        
        K = np.array(actor_data['K'], dtype=np.float32)  
        R = np.array(actor_data['R'], dtype=np.float32)  
        t = np.array(actor_data['t'], dtype=np.float32)  
        
        K = torch.tensor(K, dtype=torch.float32)
        R = torch.tensor(R, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        
        # Get confidence scores
        confidence_scores = torch.tensor(actor_data['scores'], dtype=torch.float32)
        
        sample = {
            "joints_2d_mpii": joints_2d_mpii,
            "joints_2d": joints_2d,               
            "joints_3d_centered": joints_3d_centered,  
            "rot_6d": rot_6d,                 
            "rot_mats": rot_matrices, 
            "K": K,                               
            "R": R,                               
            "t": t,
            "sample_id": sample_info['sample_id'],
            "avg_confidence": sample_info['avg_confidence'],
            "confidence_scores": confidence_scores,  # Per-joint confidence
            "sequence_name": sample_info['sequence_name'],
            "frame_idx": sample_info['frame_idx'],
            "actor_idx": sample_info['actor_idx']
        }
        
        sample["joints_3d"] = joints_3d_centered.clone()  
        sample["joints_3d_world"] = joints_3d_centered.clone()  
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_dataset_stats(self):
        """Get statistics about the loaded dataset"""
        if not self.samples:
            return {}
        
        sequences = set(s['sequence_name'] for s in self.samples)
        confidences = [s['avg_confidence'] for s in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'unique_sequences': len(sequences),
            'sequence_names': sorted(sequences),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'split': self.split
        }


class JointLevelMixedDataset(Dataset):
    """
    ENHANCED: Dataset that mixes at joint level AND includes pure real samples
    """
    
    def __init__(self, synthetic_dataset, threedpw_dataset, 
                 joint_mix_prob=0.3, pure_real_prob=0.2, seed=42):
        """
        Args:
            synthetic_dataset: SyntheticPoseDataset instance
            threedpw_dataset: ThreeDPWDataset instance  
            joint_mix_prob: Probability of mixing joints within a sample
            pure_real_prob: Probability of including pure real samples
            seed: Random seed
        """
        self.synthetic_dataset = synthetic_dataset
        self.threedpw_dataset = threedpw_dataset
        self.joint_mix_prob = joint_mix_prob
        self.pure_real_prob = pure_real_prob
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create paired samples for mixing
        self.num_synthetic = len(synthetic_dataset)
        self.num_real = len(threedpw_dataset)
        
        # For each synthetic sample, assign a random real sample for potential mixing
        if self.num_real > 0:
            self.real_pairs = np.random.randint(0, self.num_real, size=self.num_synthetic)
        else:
            self.real_pairs = []
        
        print(f"📊 ENHANCED Joint-Level Mixed Dataset:")
        print(f"   Synthetic samples: {self.num_synthetic}")
        print(f"   Real samples: {self.num_real}")
        print(f"   Joint mix probability: {joint_mix_prob:.1%}")
        print(f"   Pure real probability: {pure_real_prob:.1%}")
    
    def __len__(self):
        return self.num_synthetic
    
    def __getitem__(self, idx):
        # ENHANCED: Three-way decision for better domain exposure
        choice = torch.rand(()).item()
        
        if choice < self.pure_real_prob and self.num_real > 0:
            # Option 1: Pure real sample (for domain discriminator training)
            real_idx = self.real_pairs[idx]
            sample = self.threedpw_dataset[real_idx]
            sample['dataset_type'] = '3dpw'
            return sample
            
        elif choice < self.pure_real_prob + self.joint_mix_prob and self.num_real > 0:
            # Option 2: Joint-level mixed sample (for domain breaking)
            syn_sample = self.synthetic_dataset[idx]
            real_idx = self.real_pairs[idx]
            real_sample = self.threedpw_dataset[real_idx]
            mixed_sample = self._mix_joints(syn_sample, real_sample)
            mixed_sample['dataset_type'] = 'joint_mixed'
            return mixed_sample
            
        else:
            # Option 3: Pure synthetic sample
            sample = self.synthetic_dataset[idx]
            sample['dataset_type'] = 'synthetic'
            return sample
    
    def _mix_joints(self, syn_sample, real_sample):
        """FIXED: Mix synthetic and real joints with consistent indices"""
        mixed_sample = syn_sample.copy()
        
        # Randomly decide mixing strategy
        strategy = torch.rand(()).item()
        
        if strategy < 0.33:
            # Strategy 1: Real core + Synthetic extremities
            # Replace core joints with real data
            for joint_idx in SMPL_CORE_JOINTS:
                if joint_idx < mixed_sample['joints_2d'].shape[0]:
                    mixed_sample['joints_2d'][joint_idx] = real_sample['joints_2d'][joint_idx]
            
            # FIXED: Blend confidence scores properly
            mixed_confidence = syn_sample['confidence_scores'].clone()
            # For mixed data, create intermediate confidence
            mixed_sample['avg_confidence'] = (syn_sample['avg_confidence'] + real_sample['avg_confidence']) / 2
            
        elif strategy < 0.66:
            # Strategy 2: Synthetic core + Real extremities
            # Replace extremity joints with real data
            for joint_idx in SMPL_EXTREMITY_JOINTS:
                if joint_idx < mixed_sample['joints_2d'].shape[0]:
                    mixed_sample['joints_2d'][joint_idx] = real_sample['joints_2d'][joint_idx]
                    
        else:
            # Strategy 3: Random mixing (50% of all joints)
            num_joints = mixed_sample['joints_2d'].shape[0]
            joints_to_replace = torch.randperm(num_joints)[:num_joints//2]
            for joint_idx in joints_to_replace:
                if joint_idx < real_sample['joints_2d'].shape[0]:
                    mixed_sample['joints_2d'][joint_idx] = real_sample['joints_2d'][joint_idx]
        
        # Update average confidence for mixed samples
        mixed_sample['avg_confidence'] = (syn_sample['avg_confidence'] + real_sample['avg_confidence']) / 2
        
        # FIXED: Update confidence scores to reflect mixing
        # This prevents the model from using confidence as a domain signal
        mixed_confidence = (syn_sample['confidence_scores'] + real_sample['confidence_scores']) / 2
        mixed_sample['confidence_scores'] = mixed_confidence
        
        return mixed_sample


class MixedPoseDataset(Dataset):
    """
    FIXED: Mixed dataset with proper domain labeling
    """
    
    def __init__(self, synthetic_dataset, threedpw_dataset, 
                 real_data_ratio=0.2, seed=42):
        """
        Args:
            synthetic_dataset: SyntheticPoseDataset instance
            threedpw_dataset: ThreeDPWDataset instance  
            real_data_ratio: Fraction of samples that should be real (3DPW) data
            seed: Random seed for reproducible mixing
        """
        self.synthetic_dataset = synthetic_dataset
        self.threedpw_dataset = threedpw_dataset
        self.real_data_ratio = real_data_ratio
        
        np.random.seed(seed)
        
        total_synthetic = len(synthetic_dataset)
        total_real = len(threedpw_dataset)
        
        if total_real == 0:
            print("⚠️  No 3DPW samples available, using only synthetic data")
            self.real_data_ratio = 0.0
        
        if self.real_data_ratio > 0:
            target_real_samples = int(total_synthetic * self.real_data_ratio / (1 - self.real_data_ratio))
            target_real_samples = min(target_real_samples, total_real)
        else:
            target_real_samples = 0
        
        self.synthetic_indices = list(range(total_synthetic))
        
        if target_real_samples > 0:
            self.real_indices = np.random.choice(
                total_real, 
                size=target_real_samples, 
                replace=target_real_samples > total_real
            ).tolist()
        else:
            self.real_indices = []
        
        self.mixed_samples = []
        
        for idx in self.synthetic_indices:
            self.mixed_samples.append(('synthetic', idx))
        
        for idx in self.real_indices:
            self.mixed_samples.append(('real', idx))
        
        np.random.shuffle(self.mixed_samples)
        
        actual_real_ratio = len(self.real_indices) / len(self.mixed_samples) if self.mixed_samples else 0
        
        print(f"📊 Mixed Dataset Statistics:")
        print(f"   Total samples: {len(self.mixed_samples)}")
        print(f"   Synthetic samples: {len(self.synthetic_indices)} ({(1-actual_real_ratio)*100:.1f}%)")
        print(f"   Real (3DPW) samples: {len(self.real_indices)} ({actual_real_ratio*100:.1f}%)")
        print(f"   Target real ratio: {real_data_ratio:.1%}, Actual: {actual_real_ratio:.1%}")
    
    def __len__(self):
        return len(self.mixed_samples)
    
    def __getitem__(self, idx):
        """Get sample from either synthetic or real dataset"""
        dataset_type, sample_idx = self.mixed_samples[idx]
        
        if dataset_type == 'synthetic':
            sample = self.synthetic_dataset[sample_idx]
            sample['dataset_type'] = 'synthetic'
        else:  
            sample = self.threedpw_dataset[sample_idx]
            sample['dataset_type'] = '3dpw'
        
        return sample
    
    def get_mixing_stats(self):
        """Get statistics about the current mixing"""
        synthetic_count = sum(1 for dt, _ in self.mixed_samples if dt == 'synthetic')
        real_count = len(self.mixed_samples) - synthetic_count
        
        return {
            'total_samples': len(self.mixed_samples),
            'synthetic_samples': synthetic_count,
            'real_samples': real_count,
            'real_ratio': real_count / len(self.mixed_samples) if self.mixed_samples else 0,
            'synthetic_ratio': synthetic_count / len(self.mixed_samples) if self.mixed_samples else 0
        }


def create_domain_mixing_datasets(data_root, real_data_ratio=0.2, transform=None, 
                                 min_confidence=0.3, seed=42, use_joint_mixing=True):
    """
    FIXED: Create mixed training and validation datasets with proper setup
    """
    
    from src.data.synthetic_pose_dataset import SyntheticPoseDataset
    
    train_split_txt = os.path.join(data_root, "splits", "train.txt")
    val_split_txt = os.path.join(data_root, "splits", "val.txt")
    
    synthetic_train = SyntheticPoseDataset(
        data_root=data_root,
        split_txt=train_split_txt,
        transform=transform,
        augment_2d=True,
        break_domain_discrimination=True,  # Enable extremity corruption
        corruption_schedule='progressive'
    )
    
    synthetic_val = SyntheticPoseDataset(
        data_root=data_root,
        split_txt=val_split_txt,
        transform=transform,
        augment_2d=False,
        break_domain_discrimination=True  # Light corruption even in validation
    )
    
    threedpw_train = ThreeDPWDataset(
        data_root=data_root,
        split="train",
        transform=transform,
        min_confidence=min_confidence
    )
    
    threedpw_val = ThreeDPWDataset(
        data_root=data_root,
        split="validation",  
        transform=transform,
        min_confidence=min_confidence
    )
    
    if use_joint_mixing:
        # Use ENHANCED joint-level mixing for training
        mixed_train = JointLevelMixedDataset(
            synthetic_dataset=synthetic_train,
            threedpw_dataset=threedpw_train,
            joint_mix_prob=0.3,
            pure_real_prob=0.2,  # Include 20% pure real samples for domain discriminator
            seed=seed
        )
    else:
        # Original sample-level mixing
        mixed_train = MixedPoseDataset(
            synthetic_dataset=synthetic_train,
            threedpw_dataset=threedpw_train,
            real_data_ratio=real_data_ratio,
            seed=seed
        )
    
    # Validation uses sample-level mixing
    mixed_val = MixedPoseDataset(
        synthetic_dataset=synthetic_val,
        threedpw_dataset=threedpw_val,
        real_data_ratio=real_data_ratio,
        seed=seed + 1  
    )
    
    print(f"\n✅ Created domain mixing datasets")
    if use_joint_mixing:
        print(f"   Training: Joint-level mixing enabled")
    else:
        print(f"   Training: Sample-level mixing with {real_data_ratio:.1%} real data")
    
    return mixed_train, mixed_val


if __name__ == "__main__":
    """Test the datasets"""
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        # Test with joint mixing
        train_dataset, val_dataset = create_domain_mixing_datasets(
            data_root=data_root,
            real_data_ratio=0.586,
            use_joint_mixing=True
        )
        
        print("\n📋 Testing joint-mixed training sample:")
        sample = train_dataset[0]
        print(f"   Dataset type: {sample['dataset_type']}")
        print(f"   Confidence: {sample['avg_confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")