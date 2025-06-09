import torch
import numpy as np

class CameraViewpointAugmenter:
    def __init__(self, max_rotation_deg=15.0, max_translation_m=0.1):
        self.max_rotation_deg = max_rotation_deg
        self.max_translation_m = max_translation_m
        
        # Create MPII→SMPL reverse mapping from the smpl2mpii mapping
        # smpl2mpii = [6, 3, 2, null, 4, 1, null, 5, 0, 7, null, null, 8, null, null, 9, 13, 12, 14, 11, 15, 10, null, null]
        smpl2mpii = [6, 3, 2, None, 4, 1, None, 5, 0, 7, None, None, 8, None, None, 9, 13, 12, 14, 11, 15, 10, None, None]
        
        # Create reverse mapping: mpii_to_smpl_indices[mpii_idx] = smpl_idx
        self.mpii_to_smpl_indices = []
        for mpii_idx in range(16):  # MPII has 16 joints
            for smpl_idx, mpii_mapped in enumerate(smpl2mpii):
                if mpii_mapped == mpii_idx:
                    self.mpii_to_smpl_indices.append(smpl_idx)
                    break
        
        # Result should be: [8, 5, 2, 1, 4, 7, 0, 9, 12, 15, 21, 19, 17, 16, 18, 20]
        # This means: MPII joint 0 comes from SMPL joint 8, MPII joint 1 from SMPL joint 5, etc.
    
    def augment_viewpoint(self, joints_3d_world, K, R, t):
        max_attempts = 10
        
        for attempt in range(max_attempts):
            perturbed_R = self._perturb_rotation(R)
            perturbed_t = self._perturb_translation(t)
            
            augmented_2d_full = self._reproject_joints(joints_3d_world, K, perturbed_R, perturbed_t)
            
            if augmented_2d_full is not None:
                # Extract MPII joints from SMPL according to proper mapping
                augmented_2d_mpii = self._extract_mpii_from_smpl(augmented_2d_full)
                return augmented_2d_mpii
        
        # Fallback: reproject with original camera and extract MPII subset
        original_2d_full = self._reproject_joints(joints_3d_world, K, R, t)
        if original_2d_full is not None:
            return self._extract_mpii_from_smpl(original_2d_full)
        return None
    
    def _extract_mpii_from_smpl(self, smpl_2d):
        """Extract 16 MPII joints from 24 SMPL joints according to mapping"""
        mpii_2d = torch.zeros((16, 2), dtype=smpl_2d.dtype)
        for mpii_idx, smpl_idx in enumerate(self.mpii_to_smpl_indices):
            mpii_2d[mpii_idx] = smpl_2d[smpl_idx]
        return mpii_2d
    
    def _perturb_rotation(self, R):
        if isinstance(R, torch.Tensor):
            R_np = R.cpu().numpy()
        else:
            R_np = R
            
        angles = np.random.uniform(
            -np.deg2rad(self.max_rotation_deg), 
            np.deg2rad(self.max_rotation_deg), 
            3
        )
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        perturbation = Rz @ Ry @ Rx
        perturbed_R_np = perturbation @ R_np
        
        return torch.tensor(perturbed_R_np, dtype=torch.float32)
    
    def _perturb_translation(self, t):
        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy()
        else:
            t_np = t
            
        noise = np.random.uniform(-self.max_translation_m, self.max_translation_m, 3)
        perturbed_t_np = t_np + noise
        
        return torch.tensor(perturbed_t_np, dtype=torch.float32)
    
    def _reproject_joints(self, joints_3d_world, K, R, t):
        if isinstance(joints_3d_world, torch.Tensor):
            j3d = joints_3d_world.cpu().numpy()
        else:
            j3d = joints_3d_world
            
        if isinstance(K, torch.Tensor):
            K_np = K.cpu().numpy()
        else:
            K_np = K
            
        if isinstance(R, torch.Tensor):
            R_np = R.cpu().numpy()
        else:
            R_np = R
            
        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy()
        else:
            t_np = t
        
        cam_pts = (R_np @ j3d.T) + t_np.reshape(3, 1)
        
        # Check for invalid projections (behind camera)
        if np.any(cam_pts[2, :] <= 0.1):
            return None
            
        proj = (K_np @ cam_pts).T
        proj2d = proj[:, :2] / proj[:, 2:3]
        
        # Check if all joints are within image bounds (with margin)
        margin = 50
        if (np.any(proj2d < -margin) or 
            np.any(proj2d > 512 + margin)):
            return None
        
        # Only apply soft clipping to keep joints in bounds
        proj2d = np.clip(proj2d, 0, 512)
        
        return torch.tensor(proj2d, dtype=torch.float32)