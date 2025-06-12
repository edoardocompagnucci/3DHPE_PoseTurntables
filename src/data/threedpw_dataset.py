"""
3DPW Dataset Loader for Domain Mixing Training
Create this file as: src/data/threedpw_dataset.py
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
                                    'avg_confidence': np.mean(scores)
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


class MixedPoseDataset(Dataset):
    """
    Mixed dataset combining synthetic and 3DPW data for domain mixing training
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
                                 min_confidence=0.3, seed=42):
    """
    Convenience function to create mixed training and validation datasets
    
    Args:
        data_root: Root directory containing both synthetic and 3DPW data
        real_data_ratio: Fraction of real data in mixed dataset
        transform: Transform to apply to all samples
        min_confidence: Minimum confidence for 3DPW samples
        seed: Random seed for reproducible mixing
    
    Returns:
        train_dataset, val_dataset (both MixedPoseDataset instances)
    """
    
    
    from src.data.synthetic_pose_dataset import SyntheticPoseDataset
    
    
    train_split_txt = os.path.join(data_root, "splits", "train.txt")
    val_split_txt = os.path.join(data_root, "splits", "val.txt")
    
    synthetic_train = SyntheticPoseDataset(
        data_root=data_root,
        split_txt=train_split_txt,
        transform=transform,
        augment_2d=True  
    )
    
    synthetic_val = SyntheticPoseDataset(
        data_root=data_root,
        split_txt=val_split_txt,
        transform=transform,
        augment_2d=False
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
    
    
    mixed_train = MixedPoseDataset(
        synthetic_dataset=synthetic_train,
        threedpw_dataset=threedpw_train,
        real_data_ratio=real_data_ratio,
        seed=seed
    )
    
    mixed_val = MixedPoseDataset(
        synthetic_dataset=synthetic_val,
        threedpw_dataset=threedpw_val,
        real_data_ratio=real_data_ratio,
        seed=seed + 1  
    )
    
    print(f"\n✅ Created domain mixing datasets with {real_data_ratio:.1%} real data")
    
    return mixed_train, mixed_val


if __name__ == "__main__":
    """Test the 3DPW dataset loader"""
    
    
    data_root = os.path.join(PROJECT_ROOT, "data")
    
    try:
        dataset = ThreeDPWDataset(
            data_root=data_root,
            split="train",
            min_confidence=0.3
        )
        
        if len(dataset) > 0:
            
            sample = dataset[0]
            
            print(f"\n📋 Sample format verification:")
            print(f"   joints_2d_mpii shape: {sample['joints_2d_mpii'].shape}")
            print(f"   joints_2d shape: {sample['joints_2d'].shape}")
            print(f"   joints_3d_centered shape: {sample['joints_3d_centered'].shape}")
            print(f"   rot_6d shape: {sample['rot_6d'].shape}")
            print(f"   K shape: {sample['K'].shape}")
            print(f"   R shape: {sample['R'].shape}")
            print(f"   t shape: {sample['t'].shape}")
            print(f"   Sample ID: {sample['sample_id']}")
            print(f"   Avg confidence: {sample['avg_confidence']:.3f}")
            
            
            stats = dataset.get_dataset_stats()
            print(f"\n📊 Dataset statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        else:
            print("⚠️  No samples found in dataset")
    
    except Exception as e:
        print(f"❌ Error testing 3DPW dataset: {e}")
        print("Make sure you have run the preprocessing script first!")

