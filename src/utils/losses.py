import torch
from utils import rotation_utils

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

    geodesic_dist = torch.acos(torch.abs(trace_clamped))  # (B, J)
    
    return geodesic_dist.mean()


def rotation_mse_loss(pred_6d, target_6d):
    return torch.nn.functional.mse_loss(pred_6d, target_6d)


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
