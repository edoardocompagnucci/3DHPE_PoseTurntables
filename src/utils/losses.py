import torch

def mpjpe_loss(predicted, target):
    """
    Mean Per Joint Position Error (MPJPE) 
    """

    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1] // 3

    pred = predicted.reshape(batch_size, num_joints, 3)
    targ = target.reshape(batch_size, num_joints, 3)

    # Calculate Euclidean distance for each joint -> sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)  
    joint_error = torch.sqrt(torch.sum((pred - targ) ** 2, dim=2))

    return joint_error.mean()