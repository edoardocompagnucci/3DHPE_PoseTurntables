import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import synthetic_pose_dataset as sd

def rot_matrix_to_6d(rot_matrices):
    return torch.cat([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], dim=-1)

def rot_6d_to_matrix(rot_6d):
    batch_shape = rot_6d.shape[:-1]
    rot_6d_flat = rot_6d.reshape(-1, 6)
    
    a1 = rot_6d_flat[:, :3]
    a2 = rot_6d_flat[:, 3:]
    
    b1 = a1 / torch.norm(a1, dim=1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=1)
    
    rot_matrix = torch.stack([b1, b2, b3], dim=-1)
    return rot_matrix.reshape(*batch_shape, 3, 3)

def test_conversion():
    data_root = os.path.join(PROJECT_ROOT, "data")
    split_txt = os.path.join(data_root, "splits", "val.txt")
    
    dataset = sd.SyntheticPoseDataset(data_root=data_root, split_txt=split_txt)
    sample = dataset[0]
    
    rot_mats = sample["rot_mats"]
    print(f"Original shape: {rot_mats.shape}")
    
    rot_6d = rot_matrix_to_6d(rot_mats)
    print(f"6D shape: {rot_6d.shape}")
    
    rot_recon = rot_6d_to_matrix(rot_6d)
    print(f"Reconstructed shape: {rot_recon.shape}")
    
    error = torch.norm(rot_mats - rot_recon)
    print(f"Reconstruction error: {error.item()}")
    
    max_error = torch.max(torch.norm(rot_mats - rot_recon, dim=(1,2)))
    print(f"Max single matrix error: {max_error.item()}")
    
    success = error < 0.01
    print(f"Success: {success}")
    
    return success

if __name__ == "__main__":
    success = test_conversion()
    print(f"{'PASSED' if success else 'FAILED'}")