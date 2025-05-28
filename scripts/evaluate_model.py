# scripts/evaluate_model.py
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lifter import MLPLifter
from src.data.synthetic_pose_dataset import SyntheticPoseDataset
from src.utils.transforms import NormalizerJoints2d
from src.utils.losses import mpjpe_loss

def evaluate_model(model_path, data_root, split_txt, num_joints=16, img_size=512):
    """Comprehensive evaluation of a model checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MLPLifter(num_joints=num_joints).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Load validation dataset
    normalizer = NormalizerJoints2d(img_size=img_size)
    dataset = SyntheticPoseDataset(
        data_root=data_root,
        split_txt=split_txt,
        transform=normalizer
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Per-joint error tracking
    joint_errors = np.zeros(num_joints)
    total_samples = 0
    
    # Overall MPJPE
    total_mpjpe = 0.0
    
    # Evaluate
    with torch.no_grad():
        for batch in loader:
            inputs = batch["joints_2d"].to(device)
            target = batch["joints_3d_centered"].to(device)
            
            # Forward pass
            preds = model(inputs)
            
            # Calculate overall MPJPE
            mpjpe = mpjpe_loss(preds, target).item()
            total_mpjpe += mpjpe * inputs.size(0)
            
            # Calculate per-joint error
            batch_size = inputs.size(0)
            preds_3d = preds.reshape(batch_size, num_joints, 3)
            target_3d = target.reshape(batch_size, num_joints, 3)
            
            # Euclidean distance per joint
            joint_dists = torch.sqrt(torch.sum((preds_3d - target_3d) ** 2, dim=2))
            
            # Add to running totals
            joint_errors += joint_dists.sum(dim=0).cpu().numpy()
            total_samples += batch_size
    
    # Calculate final metrics
    avg_mpjpe = total_mpjpe / total_samples
    joint_errors = joint_errors / total_samples
    
    # Load joint names if available
    joint_names = [f"Joint {i}" for i in range(num_joints)]
    try:
        bone_graph_path = os.path.join(data_root, "meta", "bone_graph.json")
        with open(bone_graph_path, 'r') as f:
            bone_graph = json.load(f)
            joint_names = bone_graph.get("order", joint_names)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Results directory
    results_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(results_dir, exist_ok=True)
    
    # Print results
    print(f"Overall MPJPE: {avg_mpjpe*1000:.2f} mm")
    print("\nPer-joint MPJPE:")
    for i, (name, error) in enumerate(zip(joint_names, joint_errors)):
        print(f"{i:2d} {name:15s}: {error*1000:.2f} mm")
    
    # Plot per-joint error
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_joints), joint_errors * 1000)
    plt.xticks(range(num_joints), joint_names, rotation=90)
    plt.ylabel("MPJPE (mm)")
    plt.title(f"Per-Joint MPJPE - Overall: {avg_mpjpe*1000:.2f} mm")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_joint_error.png"))
    plt.show()
    
    return avg_mpjpe, joint_errors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate 3D pose estimation model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"), 
                        help="Path to data directory")
    parser.add_argument("--split", default="val", choices=["train", "val"], 
                        help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    split_txt = os.path.join(args.data_root, "splits", f"{args.split}.txt")
    
    evaluate_model(args.model, args.data_root, split_txt)
    