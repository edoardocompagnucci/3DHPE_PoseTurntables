# 3D Human Pose Estimation - PoseTurntables (Project A)

Implementation of a 2D-to-3D pose lifter for the PoseTurntables project.

## Model Architecture

The current implementation uses an MLP-based lifter with the following features:
- 4-layer fully connected network (1024→1024→512→256 neurons)
- BatchNorm and Dropout (0.25) applied after each hidden layer
- ReLU activations
- Input: 2D keypoints (24 joints × 2 coordinates)
- Output: 3D joint positions (24 joints × 3 coordinates)

## Performance

On the validation set, the model achieves:
- MPJPE: 42.6 mm

## Training Process

The model is trained with:
- Adam optimizer (lr=5e-4, weight_decay=1e-4)
- ReduceLROnPlateau scheduler (factor=0.7, patience=3)
- Early stopping (patience=15)
- Input normalization to range [-1, 1]
- Gradient clipping (max_norm=1.0)

## Training Process

For real-world inference, the system uses:
- MMPose RTMPose-MPII for 2D keypoint detection from RGB images
- 2D keypoints are mapped from MPII format (16 joints) to SMPL format (24 joints)
- The trained MLP lifter converts 2D keypoints to 3D joint positions
- End-to-end pipeline from RGB image to 3D pose estimation

## Dataset

- 11 000 synthetic samples in total (generated using Side Effects Houdini)
- 3D joint positions in world space (24 joints × 3 coordinates)
- 2D keypoints following MPII skeleton (16 joints)
- Camera intrinsics K and extrinsics R, t (not used yet)
- 80/20 train/validation split
- 2D keypoints mapped from MPII style (16 joints) to SMPL format (24 joints) for training and inference