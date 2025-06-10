import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiablePoseAug(nn.Module):
    """
    Differentiable, learnable PoseAug module based on CameraViewpointAugmenter.
    Makes camera viewpoint augmentation learnable with differentiable operations.
    """
    def __init__(self, 
                 initial_max_rotation_deg=15.0, 
                 initial_max_translation_m=0.1,
                 augmentation_prob=0.5):
        super(DifferentiablePoseAug, self).__init__()
        
        
        
        self.log_max_rotation = nn.Parameter(
            torch.log(torch.tensor(initial_max_rotation_deg * np.pi / 180.0))  
        )
        self.log_max_translation = nn.Parameter(
            torch.log(torch.tensor(initial_max_translation_m))
        )
        
        
        self.rotation_bias = nn.Parameter(torch.zeros(3))
        self.translation_bias = nn.Parameter(torch.zeros(3))
        
        
        self.rotation_scale = nn.Parameter(torch.ones(3))
        self.translation_scale = nn.Parameter(torch.ones(3))
        
        self.augmentation_prob = augmentation_prob
        
        
        smpl2mpii = [6, 3, 2, None, 4, 1, None, 5, 0, 7, None, None, 8, None, None, 9, 13, 12, 14, 11, 15, 10, None, None]
        self.mpii_to_smpl_indices = []
        for mpii_idx in range(16):  
            for smpl_idx, mpii_mapped in enumerate(smpl2mpii):
                if mpii_mapped == mpii_idx:
                    self.mpii_to_smpl_indices.append(smpl_idx)
                    break
        
        
        self.register_buffer('mpii_to_smpl_tensor', torch.tensor(self.mpii_to_smpl_indices))
        
    def forward(self, joints_2d_mpii, joints_3d, K, R, t, training=True):
        """
        Apply differentiable camera viewpoint augmentation.
        
        Args:
            joints_2d_mpii: (B, 16, 2) - Current MPII 2D keypoints
            joints_3d: (B, 24, 3) - SMPL 3D joints in world coordinates  
            K: (B, 3, 3) - Camera intrinsic matrix
            R: (B, 3, 3) - Camera rotation matrix
            t: (B, 3) - Camera translation vector
            training: bool - Whether in training mode
            
        Returns:
            augmented_joints_2d_mpii: (B, 16, 2) - Augmented MPII 2D keypoints
        """
        if not training:
            return joints_2d_mpii
            
        batch_size = joints_2d_mpii.shape[0]
        device = joints_2d_mpii.device
        
        
        augment_mask = torch.rand(batch_size, device=device) < self.augmentation_prob
        
        if not augment_mask.any():
            return joints_2d_mpii
        
        
        max_rotation = torch.exp(self.log_max_rotation)
        max_translation = torch.exp(self.log_max_translation)
        
        
        augmented_R = R.clone()
        augmented_t = t.clone()
        
        
        augment_indices = torch.where(augment_mask)[0]
        
        if len(augment_indices) > 0:
            
            augmented_R[augment_indices] = self._perturb_rotation_batch(
                R[augment_indices], max_rotation
            )
            
            
            augmented_t[augment_indices] = self._perturb_translation_batch(
                t[augment_indices], max_translation
            )
        
        
        augmented_2d_smpl = self._reproject_joints_batch(joints_3d, K, augmented_R, augmented_t)
        
        
        augmented_2d_mpii = self._extract_mpii_from_smpl_batch(augmented_2d_smpl)
        
        
        result = joints_2d_mpii.clone()
        result[augment_mask] = augmented_2d_mpii[augment_mask]
        
        return result
    
    def _perturb_rotation_batch(self, R_batch, max_rotation):
        """
        Apply differentiable rotation perturbation to a batch of rotation matrices.
        Based on your original _perturb_rotation logic.
        """
        batch_size = R_batch.shape[0]
        device = R_batch.device
        
        
        angles = (torch.rand(batch_size, 3, device=device) * 2 - 1) * max_rotation
        angles = angles * self.rotation_scale + self.rotation_bias
        
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        
        Rx = torch.zeros(batch_size, 3, 3, device=device)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos_angles[:, 0]
        Rx[:, 1, 2] = -sin_angles[:, 0]
        Rx[:, 2, 1] = sin_angles[:, 0]
        Rx[:, 2, 2] = cos_angles[:, 0]
        
        
        Ry = torch.zeros(batch_size, 3, 3, device=device)
        Ry[:, 0, 0] = cos_angles[:, 1]
        Ry[:, 0, 2] = sin_angles[:, 1]
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -sin_angles[:, 1]
        Ry[:, 2, 2] = cos_angles[:, 1]
        
        
        Rz = torch.zeros(batch_size, 3, 3, device=device)
        Rz[:, 0, 0] = cos_angles[:, 2]
        Rz[:, 0, 1] = -sin_angles[:, 2]
        Rz[:, 1, 0] = sin_angles[:, 2]
        Rz[:, 1, 1] = cos_angles[:, 2]
        Rz[:, 2, 2] = 1
        
        
        perturbation = torch.bmm(torch.bmm(Rz, Ry), Rx)
        
        
        perturbed_R = torch.bmm(perturbation, R_batch)
        
        return perturbed_R
    
    def _perturb_translation_batch(self, t_batch, max_translation):
        """
        Apply differentiable translation perturbation to a batch of translation vectors.
        Based on your original _perturb_translation logic.
        """
        batch_size = t_batch.shape[0]
        device = t_batch.device
        
        
        noise = (torch.rand(batch_size, 3, device=device) * 2 - 1) * max_translation
        noise = noise * self.translation_scale + self.translation_bias
        
        perturbed_t = t_batch + noise
        
        return perturbed_t
    
    def _reproject_joints_batch(self, joints_3d_world, K, R, t):
        """
        Differentiable batch reprojection of 3D joints to 2D.
        Based on your original _reproject_joints logic.
        
        Args:
            joints_3d_world: (B, 24, 3)
            K: (B, 3, 3)
            R: (B, 3, 3) 
            t: (B, 3)
            
        Returns:
            proj2d: (B, 24, 2)
        """
        batch_size = joints_3d_world.shape[0]
        
        
        
        joints_3d_transposed = joints_3d_world.transpose(-1, -2)
        
        
        cam_pts = torch.bmm(R, joints_3d_transposed)
        
        
        cam_pts = cam_pts + t.unsqueeze(-1)
        
        
        proj = torch.bmm(K, cam_pts)
        
        
        proj_transposed = proj.transpose(-1, -2)  
        
        
        depth = proj_transposed[:, :, 2:3] + 1e-8  
        proj2d = proj_transposed[:, :, :2] / depth  
        
        
        proj2d = torch.clamp(proj2d, 0.0, 512.0)
        
        return proj2d
    
    def _extract_mpii_from_smpl_batch(self, smpl_2d_batch):
        """
        Extract MPII joints from SMPL joints for a batch.
        Based on your original _extract_mpii_from_smpl logic.
        
        Args:
            smpl_2d_batch: (B, 24, 2)
            
        Returns:
            mpii_2d_batch: (B, 16, 2)
        """
        
        
        mpii_2d_batch = smpl_2d_batch[:, self.mpii_to_smpl_tensor]  
        
        return mpii_2d_batch
    
    def get_augmentation_stats(self):
        """
        Get current augmentation parameter values for monitoring.
        """
        max_rotation_deg = torch.exp(self.log_max_rotation) * 180.0 / np.pi
        max_translation_m = torch.exp(self.log_max_translation)
        
        return {
            'max_rotation_deg': max_rotation_deg.item(),
            'max_translation_m': max_translation_m.item(),
            'rotation_bias': self.rotation_bias.detach().cpu().numpy(),
            'translation_bias': self.translation_bias.detach().cpu().numpy(),
            'rotation_scale': self.rotation_scale.detach().cpu().numpy(),
            'translation_scale': self.translation_scale.detach().cpu().numpy(),
        }


class PoseAugWrapper(nn.Module):
    """
    Wrapper to integrate DifferentiablePoseAug with existing dataset format.
    Handles the conversion between dataset format and PoseAug format.
    Also handles coordinate normalization conversion.
    """
    def __init__(self, poseaug_module, img_size=512):
        super(PoseAugWrapper, self).__init__()
        self.poseaug = poseaug_module
        self.img_size = img_size
        
        
        smpl2mpii = [6, 3, 2, None, 4, 1, None, 5, 0, 7, None, None, 8, None, None, 9, 13, 12, 14, 11, 15, 10, None, None]
        self.smpl_to_mpii_mapping = smpl2mpii
    
    def normalized_to_pixel(self, joints_normalized):
        """Convert from normalized [-1, 1] to pixel [0, img_size] coordinates"""
        return (joints_normalized + 1) * self.img_size / 2
    
    def pixel_to_normalized(self, joints_pixel):
        """Convert from pixel [0, img_size] to normalized [-1, 1] coordinates"""
        return (joints_pixel / self.img_size) * 2 - 1
    
    def mpii_to_smpl_batch(self, joints_2d_mpii_batch):
        """Convert batch of MPII keypoints to SMPL format"""
        batch_size = joints_2d_mpii_batch.shape[0]
        joints_2d_smpl_batch = torch.zeros(batch_size, 24, 2, 
                                         dtype=joints_2d_mpii_batch.dtype,
                                         device=joints_2d_mpii_batch.device)
        
        
        for smpl_idx, mpii_idx in enumerate(self.smpl_to_mpii_mapping):
            if mpii_idx is not None:
                joints_2d_smpl_batch[:, smpl_idx] = joints_2d_mpii_batch[:, mpii_idx]
                
        return joints_2d_smpl_batch
    
    def forward(self, joints_2d_smpl_batch, joints_3d_batch, K_batch, R_batch, t_batch, training=True):
        """
        Apply PoseAug to a batch of samples in the format expected by the training loop.
        
        Args:
            joints_2d_smpl_batch: (B, 24, 2) - SMPL format 2D keypoints in NORMALIZED [-1,1] coordinates
            joints_3d_batch: (B, 24, 3) - 3D joints
            K_batch: (B, 3, 3) - Camera intrinsics
            R_batch: (B, 3, 3) - Camera rotation
            t_batch: (B, 3) - Camera translation
            training: bool
            
        Returns:
            augmented_joints_2d_smpl: (B, 24, 2) - Augmented SMPL format 2D keypoints in NORMALIZED [-1,1] coordinates
        """
        if not training:
            return joints_2d_smpl_batch
        
        batch_size = joints_2d_smpl_batch.shape[0]
        device = joints_2d_smpl_batch.device
        
        
        joints_2d_smpl_pixel = self.normalized_to_pixel(joints_2d_smpl_batch)
        
        
        joints_2d_mpii_pixel = torch.zeros(batch_size, 16, 2, device=device)
        for mpii_idx, smpl_idx in enumerate(self.poseaug.mpii_to_smpl_indices):
            joints_2d_mpii_pixel[:, mpii_idx] = joints_2d_smpl_pixel[:, smpl_idx]
        
        
        augmented_joints_2d_mpii_pixel = self.poseaug(
            joints_2d_mpii_pixel, joints_3d_batch, K_batch, R_batch, t_batch, training
        )
        
        
        augmented_joints_2d_smpl_pixel = self.mpii_to_smpl_batch(augmented_joints_2d_mpii_pixel)
        
        
        augmented_joints_2d_smpl_normalized = self.pixel_to_normalized(augmented_joints_2d_smpl_pixel)
        
        return augmented_joints_2d_smpl_normalized