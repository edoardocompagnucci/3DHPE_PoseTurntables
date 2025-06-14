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

# FIXED: Consistent joint definitions across all files
SMPL_EXTREMITY_JOINTS = [7, 8, 10, 11, 18, 19, 20, 21, 22, 23]  # ankles, feet, wrists, hands
SMPL_CORE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17]  # pelvis, hips, knees, spine, shoulders, elbows


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


def extremity_weighted_mpjpe_loss(predicted, target, extremity_weight=2.0):
    """
    MPJPE loss with higher weight for extremity joints
    """
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))  # (B, J)
    
    # Create weight vector
    weights = torch.ones(num_joints, device=joint_error.device)
    for joint_idx in SMPL_EXTREMITY_JOINTS:
        if joint_idx < num_joints:
            weights[joint_idx] = extremity_weight
    
    # Normalize weights to maintain overall scale
    weights = weights / weights.mean()
    
    # Apply weights
    weighted_error = joint_error * weights.unsqueeze(0)
    
    return weighted_error.mean()


def adaptive_extremity_loss(predicted, target, epoch=0, max_weight=3.0):
    """
    Adaptive extremity weighting with improved scheduling
    """
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))  # (B, J)
    
    # FIXED: Smoother progressive weight schedule
    if epoch < 10:
        extremity_weight = 1.0  # Start equal
    elif epoch < 30:
        extremity_weight = 1.0 + (epoch - 10) / 20 * 1.0  # Gradually increase to 2.0
    elif epoch < 60:
        extremity_weight = 2.0 + (epoch - 30) / 30 * 1.0  # Continue to 3.0
    else:
        extremity_weight = min(max_weight, 3.0)  # Cap at max
    
    # Create weight vector
    weights = torch.ones(num_joints, device=joint_error.device)
    for joint_idx in SMPL_EXTREMITY_JOINTS:
        if joint_idx < num_joints:
            weights[joint_idx] = extremity_weight
    
    # FIXED: Error-adaptive weighting with better normalization
    with torch.no_grad():
        mean_errors = joint_error.mean(dim=0)  # Average error per joint
        error_scale = torch.clamp(mean_errors / (mean_errors.mean() + 1e-6), 0.5, 2.0)
        weights = weights * error_scale
    
    # Normalize to maintain scale
    weights = weights / weights.mean()
    
    # Apply weights
    weighted_error = joint_error * weights.unsqueeze(0)
    
    return weighted_error.mean(), extremity_weight


def confidence_aware_loss(predicted, target, confidence_scores=None, joint_mapping=None):
    """
    Weight loss by detection confidence with proper joint mapping
    """
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))  # (B, J)
    
    if confidence_scores is not None:
        # FIXED: Proper confidence handling
        if confidence_scores.shape[1] == num_joints:
            # Direct SMPL confidence
            confidence_weights = confidence_scores
        else:
            # MPII confidence - map to SMPL or use average
            avg_confidence = confidence_scores.mean(dim=1, keepdim=True)  # (B, 1)
            confidence_weights = avg_confidence.expand(-1, num_joints)  # (B, J)
        
        # Normalize confidence to prevent extreme weighting
        confidence_weights = torch.clamp(confidence_weights, 0.3, 1.0)
        
        weighted_error = joint_error * confidence_weights
    else:
        weighted_error = joint_error
    
    return weighted_error.mean()


def mpjpe_loss(predicted, target):
    """Original MPJPE loss for compatibility"""
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))

    return joint_error.mean()


def geodesic_loss(pred_6d, target_6d):
    if pred_6d.dim() == 2:
        batch_size  = pred_6d.shape[0]
        num_joints  = pred_6d.shape[1] // 6
        pred_6d     = pred_6d.reshape(batch_size, num_joints, 6)
        target_6d   = target_6d.reshape(batch_size, num_joints, 6)

    pred_rot   = rotation_utils.rot_6d_to_matrix(pred_6d)   # (B, J, 3,3)
    target_rot = rotation_utils.rot_6d_to_matrix(target_6d) # (B, J, 3,3)

    relative_rot = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(dim=-1)  # (B, J)

    trace_clamped = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)

    geodesic_dist = torch.acos(trace_clamped)

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
    """Original combined loss for compatibility"""
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


def combined_pose_bone_loss(pred_dict, target_dict, pos_weight=1.0, rot_weight=0.1, 
                           bone_weight=0.05, use_geodesic=True):
    """Original combined loss with bone for compatibility"""
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


def domain_breaking_loss(pred_dict, target_dict, epoch=0, confidence_scores=None,
                        pos_weight=1.0, rot_weight=0.1, bone_weight=0.05, 
                        use_extremity_weighting=True, use_confidence=True):
    """
    FIXED: Improved domain-breaking loss function
    """
    
    # Position loss with extremity weighting
    if use_extremity_weighting:
        pos_loss, extremity_weight = adaptive_extremity_loss(
            pred_dict['positions'], 
            target_dict['positions'], 
            epoch=epoch
        )
    else:
        pos_loss = mpjpe_loss(pred_dict['positions'], target_dict['positions'])
        extremity_weight = 1.0
    
    # FIXED: Apply confidence weighting properly
    if use_confidence and confidence_scores is not None:
        confidence_loss = confidence_aware_loss(
            pred_dict['positions'], 
            target_dict['positions'],
            confidence_scores
        )
        # Blend confidence and extremity weighting
        pos_loss = 0.7 * pos_loss + 0.3 * confidence_loss
    
    # Rotation loss
    rot_loss = geodesic_loss(pred_dict['rotations'], target_dict['rotations'])
    
    # Bone loss
    batch_size = pred_dict['positions'].shape[0]
    pred_positions_3d = pred_dict['positions'].reshape(batch_size, 24, 3)
    bone_loss = bone_length_loss(pred_positions_3d)
    
    # Total loss
    total_loss = pos_weight * pos_loss + rot_weight * rot_loss + bone_weight * bone_loss
    
    return {
        'total': total_loss,
        'position': pos_loss,
        'rotation': rot_loss,
        'bone': bone_loss,
        'extremity_weight': extremity_weight
    }