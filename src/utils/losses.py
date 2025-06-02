import torch
import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import rotation_utils

_bone_data_loaded = False
_edges = None
_reference_bone_lengths = None

def _load_bone_data():
    global _bone_data_loaded, _edges, _reference_bone_lengths
    
    if _bone_data_loaded:
        return

    joint_mapping_path = os.path.join(PROJECT_ROOT, "data", "meta", "joints_mapping.json")
    with open(joint_mapping_path) as f:
        bone_graph = json.load(f)
    _edges = bone_graph["edges"]

    bone_lengths_path = os.path.join(PROJECT_ROOT, "data", "meta", "bone_lengths.npy")
    reference_lengths = np.load(bone_lengths_path)
    _reference_bone_lengths = torch.tensor(reference_lengths, dtype=torch.float32)
    
    _bone_data_loaded = True
    print(f"✅ Loaded bone data: {len(_edges)} bones")

def mpjpe_loss(predicted, target):

    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))

    return joint_error.mean()


def geodesic_loss(pred_6d, target_6d):

    if pred_6d.dim() == 2:
        batch_size = pred_6d.shape[0]
        num_joints = pred_6d.shape[1] // 6
        pred_6d = pred_6d.reshape(batch_size, num_joints, 6)
        target_6d = target_6d.reshape(batch_size, num_joints, 6)
    
    pred_rot = rotation_utils.rot_6d_to_matrix(pred_6d) 
    target_rot = rotation_utils.rot_6d_to_matrix(target_6d)
    
    relative_rot = torch.matmul(pred_rot.transpose(-1, -2), target_rot)

    trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(dim=-1)
    
    trace_clamped = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)

    geodesic_dist = torch.acos(torch.abs(trace_clamped))
    
    return geodesic_dist.mean()


def rotation_mse_loss(pred_6d, target_6d):
    return torch.nn.functional.mse_loss(pred_6d, target_6d)


def bone_length_loss(predicted_poses):
    _load_bone_data()
    
    batch_size = predicted_poses.shape[0]
    device = predicted_poses.device
    reference_lengths = _reference_bone_lengths.to(device)

    predicted_lengths = []
    for parent, child in _edges:
        bone_vector = predicted_poses[:, child] - predicted_poses[:, parent]
        bone_length = torch.norm(bone_vector, dim=1)
        predicted_lengths.append(bone_length)
    
    predicted_lengths = torch.stack(predicted_lengths, dim=1)
    target_lengths = reference_lengths.unsqueeze(0).expand(batch_size, -1)

    normalized_pred = predicted_lengths / (target_lengths + 1e-6)
    loss = torch.mean((normalized_pred - 1.0)**2)
    
    return loss


def combined_pose_loss(pred_dict, target_dict, pos_weight=1.0, rot_weight=0.1, use_geodesic=True):

    pos_loss = mpjpe_loss(pred_dict['positions'], target_dict['positions'])
    
    if use_geodesic:
        rot_loss = geodesic_loss(pred_dict['rotations'], target_dict['rotations'])
    else:
        rot_loss = rotation_mse_loss(pred_dict['rotations'], target_dict['rotations'])

    total_loss = pos_weight * pos_loss + rot_weight * rot_loss
    
    return {
        'total': total_loss,
        'position': pos_loss,
        'rotation': rot_loss
    }


def combined_pose_bone_loss(pred_dict, target_dict, pos_weight=1.0, rot_weight=0.1, bone_weight=0.05, use_geodesic=True):
    pos_loss = mpjpe_loss(pred_dict['positions'], target_dict['positions'])
    
    if use_geodesic:
        rot_loss = geodesic_loss(pred_dict['rotations'], target_dict['rotations'])
    else:
        rot_loss = rotation_mse_loss(pred_dict['rotations'], target_dict['rotations'])
    
    batch_size = pred_dict['positions'].shape[0]
    pred_positions_3d = pred_dict['positions'].reshape(batch_size, 24, 3)
    bone_loss = bone_length_loss(pred_positions_3d)
    
    total_loss = pos_weight * pos_loss + rot_weight * rot_loss + bone_weight * bone_loss
    
    return {
        'total': total_loss,
        'position': pos_loss,
        'rotation': rot_loss,
        'bone': bone_loss
    }