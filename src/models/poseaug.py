import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PoseAug(nn.Module):
    def __init__(
        self,
        num_joints=24,
        rotation_range=0.2,
        bone_scale_range=0.1,
        camera_jitter=0.1,
        img_size=512
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.rotation_range = rotation_range
        self.bone_scale_range = bone_scale_range
        self.camera_jitter = camera_jitter
        self.img_size = img_size
        
        # SMPL kinematic chain
        self.parents = [
            -1,  # 0: pelvis (root)
            0,   # 1: left_hip (parent: pelvis)
            0,   # 2: right_hip (parent: pelvis)
            0,   # 3: spine1 (parent: pelvis)
            1,   # 4: left_knee (parent: left_hip)
            2,   # 5: right_knee (parent: right_hip)
            3,   # 6: spine2 (parent: spine1)
            4,   # 7: left_ankle (parent: left_knee)
            5,   # 8: right_ankle (parent: right_knee)
            6,   # 9: spine3 (parent: spine2)
            7,   # 10: left_foot (parent: left_ankle)
            8,   # 11: right_foot (parent: right_ankle)
            9,   # 12: neck (parent: spine3)
            9,   # 13: left_collar (parent: spine3)
            9,   # 14: right_collar (parent: spine3)
            12,  # 15: head (parent: neck)
            13,  # 16: left_shoulder (parent: left_collar)
            14,  # 17: right_shoulder (parent: right_collar)
            16,  # 18: left_elbow (parent: left_shoulder)
            17,  # 19: right_elbow (parent: right_shoulder)
            18,  # 20: left_wrist (parent: left_elbow)
            19,  # 21: right_wrist (parent: right_elbow)
            20,  # 22: left_hand (parent: left_wrist)
            21,  # 23: right_hand (parent: right_wrist)
        ]
    
    def rodrigues_rotation_matrix(self, axis, angle):
        """Generate rotation matrix from axis-angle representation using Rodrigues formula"""
        batch_size = axis.shape[0]
        device = axis.device
        
        # Normalize axis
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
        
        cos_angle = torch.cos(angle).unsqueeze(-1)  # (B, 1)
        sin_angle = torch.sin(angle).unsqueeze(-1)  # (B, 1)
        
        # Cross product matrix [axis]_x
        zeros = torch.zeros(batch_size, device=device)
        K = torch.stack([
            torch.stack([zeros, -axis[:, 2], axis[:, 1]], dim=1),
            torch.stack([axis[:, 2], zeros, -axis[:, 0]], dim=1),
            torch.stack([-axis[:, 1], axis[:, 0], zeros], dim=1)
        ], dim=1)  # (B, 3, 3)
        
        # Identity matrix
        I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Rodrigues formula: R = I + sin(θ)[axis]_x + (1-cos(θ))[axis]_x^2
        axis_outer = torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(1))  # (B, 3, 3)
        R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * (axis_outer - I)
        
        return R

    def forward(self, poses_3d, rot_mats, camera_params=None):
        batch_size = poses_3d.shape[0]
        device = poses_3d.device
        
        poses_aug = poses_3d.clone()
        rot_mats_aug = rot_mats.clone()
        
        # Augment pose with kinematic constraints
        for i in range(1, self.num_joints):
            parent = self.parents[i]
            if parent >= 0:
                # Get current bone vector
                bone = poses_aug[:, i] - poses_aug[:, parent]
                bone_length = torch.norm(bone, dim=1, keepdim=True)
                
                # Generate random rotation
                random_axis = torch.randn(batch_size, 3, device=device)
                random_axis = random_axis / (torch.norm(random_axis, dim=1, keepdim=True) + 1e-8)
                angle = (torch.rand(batch_size, device=device) * 2 - 1) * self.rotation_range
                
                # Create rotation matrix
                R_aug = self.rodrigues_rotation_matrix(random_axis, angle)
                
                # Apply rotation to bone
                rotated_bone = torch.bmm(R_aug, bone.unsqueeze(-1)).squeeze(-1)
                
                # Apply bone length scaling
                scale = 1.0 + (torch.rand(batch_size, 1, device=device) * 2 - 1) * self.bone_scale_range
                new_bone = rotated_bone * scale * bone_length / (torch.norm(rotated_bone, dim=1, keepdim=True) + 1e-8)
                
                # Update joint position
                poses_aug[:, i] = poses_aug[:, parent] + new_bone
                
                # Update rotation matrix for this joint
                rot_mats_aug[:, i] = torch.bmm(rot_mats_aug[:, i], R_aug)

        # Camera parameter augmentation and 2D projection
        poses_2d_aug = None
        camera_params_aug = None
        
        if camera_params is not None:
            K, R, t = camera_params['K'], camera_params['R'], camera_params['t']
            
            # Augment camera parameters
            K_aug = K.clone()
            R_aug = R.clone()
            t_aug = t.clone()
            
            # Focal length perturbation
            focal_scale = 1.0 + (torch.rand(batch_size, 2, device=device) * 2 - 1) * self.camera_jitter * 0.1
            K_aug[:, 0, 0] = K[:, 0, 0] * focal_scale[:, 0]  # fx
            K_aug[:, 1, 1] = K[:, 1, 1] * focal_scale[:, 1]  # fy
            
            # Camera pose perturbation
            yaw = (torch.rand(batch_size, device=device) * 2 - 1) * self.camera_jitter * 0.1
            pitch = (torch.rand(batch_size, device=device) * 2 - 1) * self.camera_jitter * 0.05
            
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
            
            # Yaw rotation (around Y-axis)
            R_yaw = torch.zeros(batch_size, 3, 3, device=device)
            R_yaw[:, 0, 0] = cos_yaw
            R_yaw[:, 0, 2] = sin_yaw
            R_yaw[:, 1, 1] = 1.0
            R_yaw[:, 2, 0] = -sin_yaw
            R_yaw[:, 2, 2] = cos_yaw
            
            # Pitch rotation (around X-axis)
            R_pitch = torch.zeros(batch_size, 3, 3, device=device)
            R_pitch[:, 0, 0] = 1.0
            R_pitch[:, 1, 1] = cos_pitch
            R_pitch[:, 1, 2] = -sin_pitch
            R_pitch[:, 2, 1] = sin_pitch
            R_pitch[:, 2, 2] = cos_pitch
            
            R_delta = torch.bmm(R_yaw, R_pitch)
            R_aug = torch.bmm(R, R_delta)
            
            # Translation perturbation
            t_jitter = torch.randn(batch_size, 3, device=device) * self.camera_jitter * 0.05
            t_aug = t + t_jitter
            
            # Project to 2D and normalize
            poses_2d_aug = self._project_and_normalize(poses_aug, {'K': K_aug, 'R': R_aug, 't': t_aug})
            camera_params_aug = {'K': K_aug, 'R': R_aug, 't': t_aug}
        
        return {
            'poses_3d_aug': poses_aug,
            'poses_2d_aug': poses_2d_aug,
            'rot_mats_aug': rot_mats_aug,
            'camera_params_aug': camera_params_aug
        }
    
    def _project_and_normalize(self, poses_3d, camera_params):
        """Project 3D poses to 2D and normalize to [-1, 1] range"""
        K, R, t = camera_params['K'], camera_params['R'], camera_params['t']
        batch_size = poses_3d.shape[0]
        
        # Transform to camera coordinates
        poses_3d_flat = poses_3d.reshape(batch_size, -1, 3)
        poses_cam = torch.bmm(R, poses_3d_flat.transpose(1, 2)) + t.unsqueeze(-1)
        poses_cam = poses_cam.transpose(1, 2)  # (B, N, 3)
        
        # Project to 2D
        u = poses_cam[:, :, 0] / poses_cam[:, :, 2].clamp(min=1e-5)
        v = poses_cam[:, :, 1] / poses_cam[:, :, 2].clamp(min=1e-5)
        
        u = K[:, 0, 0].unsqueeze(1) * u + K[:, 0, 2].unsqueeze(1)
        v = K[:, 1, 1].unsqueeze(1) * v + K[:, 1, 2].unsqueeze(1)
        
        poses_2d = torch.stack([u, v], dim=2)  # (B, N, 2)
        
        # Normalize to [-1, 1] range (same as NormalizerJoints2d)
        poses_2d_norm = (poses_2d / self.img_size) * 2 - 1
        
        return poses_2d_norm

    def rot_matrix_to_6d(self, rot_matrices):
        """Convert rotation matrices to 6D representation"""
        return torch.cat([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], dim=-1)