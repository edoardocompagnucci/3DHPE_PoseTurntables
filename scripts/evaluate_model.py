import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_PATH = r"checkpoints\mlp_lifter_rotation_20250529_232201\final_model.pth"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
BATCH_SIZE = 64
sys.path.insert(0, PROJECT_ROOT)

from src.models.rot_head import MLPLifterRotationHead
from src.data.synthetic_pose_dataset import SyntheticPoseDataset
from src.utils.transforms import NormalizerJoints2d
from src.utils.losses import mpjpe_loss, geodesic_loss
from src.utils import rotation_utils

def rotation_mae_degrees(pred_rot_6d, target_rot_6d):
    if pred_rot_6d.dim() == 2:
        batch_size = pred_rot_6d.shape[0]
        num_joints = pred_rot_6d.shape[1] // 6
        pred_rot_6d = pred_rot_6d.reshape(batch_size, num_joints, 6)
        target_rot_6d = target_rot_6d.reshape(batch_size, num_joints, 6)

    pred_matrices = rotation_utils.rot_6d_to_matrix(pred_rot_6d)
    target_matrices = rotation_utils.rot_6d_to_matrix(target_rot_6d)
    
    rel_rot = torch.matmul(pred_matrices.transpose(-1, -2), target_matrices)

    trace = torch.diagonal(rel_rot, dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    angles_rad = torch.acos(torch.abs(cos_angle))
    angles_deg = angles_rad * 180.0 / np.pi
    
    return angles_deg

def evaluate_pose_and_rotation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📊 Combined Position + Rotation Evaluation")
    print("=" * 50)
    
    model = MLPLifterRotationHead(num_joints=24).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✅ Loaded model: {os.path.basename(MODEL_PATH)}")
    if 'best_val_mpjpe' in checkpoint:
        training_mpjpe = checkpoint['best_val_mpjpe'] * 1000
        print(f"📝 Training best MPJPE: {training_mpjpe:.1f} mm")
    
    # Load validation dataset
    split_txt = os.path.join(DATA_ROOT, "splits", "val.txt")
    normalizer = NormalizerJoints2d(img_size=512)
    dataset = SyntheticPoseDataset(data_root=DATA_ROOT, split_txt=split_txt, transform=normalizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"📂 Validation samples: {len(dataset)}")
    
    # Joint names for reporting
    joint_mapping_path = os.path.join(DATA_ROOT, "meta", "joints_mapping.json")
    with open(joint_mapping_path, 'r') as f:
        joint_mapping = json.load(f)
    joint_names = joint_mapping["smpl_names"]
    
    print(f"✅ Loaded {len(joint_names)} joint names from mapping file")
    
    # Collect errors
    position_errors = []  # MPJPE per sample
    rotation_errors = []  # MAE per sample  
    per_joint_position_errors = [[] for _ in range(24)]
    per_joint_rotation_errors = [[] for _ in range(24)]
    
    print("🔄 Evaluating...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            
            # Forward pass
            outputs = model(inputs)
            pred_pos = outputs['positions']  # (B, 72)
            pred_rot = outputs['rotations']  # (B, 144)
            
            # Position errors (MPJPE)
            batch_size = inputs.size(0)
            pred_pos_3d = pred_pos.reshape(batch_size, 24, 3)
            target_pos_3d = target_pos.reshape(batch_size, 24, 3)
            
            # Per-sample MPJPE
            sample_mpjpe = torch.norm(pred_pos_3d - target_pos_3d, dim=2).mean(dim=1)  # (B,)
            position_errors.extend(sample_mpjpe.cpu().numpy())
            
            # Per-joint position errors
            joint_pos_errors = torch.norm(pred_pos_3d - target_pos_3d, dim=2)  # (B, J)
            for j in range(24):
                per_joint_position_errors[j].extend(joint_pos_errors[:, j].cpu().numpy())
            
            # Rotation errors (MAE in degrees)
            pred_rot_6d = pred_rot.reshape(batch_size, 24, 6)
            target_rot_6d = target_rot.reshape(batch_size, 24, 6)
            
            # Per-sample rotation MAE
            sample_rot_mae = rotation_mae_degrees(pred_rot_6d, target_rot_6d)  # (B, J)
            sample_mean_mae = sample_rot_mae.mean(dim=1)  # (B,)
            rotation_errors.extend(sample_mean_mae.cpu().numpy())
            
            # Per-joint rotation errors
            for j in range(24):
                per_joint_rotation_errors[j].extend(sample_rot_mae[:, j].cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} samples...")
    
    # Compute overall statistics
    overall_mpjpe = np.mean(position_errors) * 1000  # Convert to mm
    overall_rot_mae = np.mean(rotation_errors)  # Already in degrees
    
    per_joint_mpjpe = [np.mean(errors) * 1000 for errors in per_joint_position_errors]
    per_joint_rot_mae = [np.mean(errors) for errors in per_joint_rotation_errors]
    
    # Print results
    print(f"\n📈 Overall Results:")
    print(f"   Position MPJPE: {overall_mpjpe:.1f} mm ± {np.std(position_errors)*1000:.1f} mm")
    print(f"   Rotation MAE:   {overall_rot_mae:.1f}° ± {np.std(rotation_errors):.1f}°")
    print(f"   Samples evaluated: {len(position_errors)}")
    
    # Performance assessment
    pos_excellent = overall_mpjpe < 50
    pos_good = overall_mpjpe < 60
    rot_excellent = overall_rot_mae < 15
    rot_good = overall_rot_mae < 25
    
    print(f"\n🎯 Performance Assessment:")
    if pos_excellent:
        print(f"   ✅ Position: EXCELLENT ({overall_mpjpe:.1f} mm)")
    elif pos_good:
        print(f"   ✅ Position: GOOD ({overall_mpjpe:.1f} mm)")
    else:
        print(f"   ⚠️  Position: NEEDS WORK ({overall_mpjpe:.1f} mm)")
        
    if rot_excellent:
        print(f"   ✅ Rotation: EXCELLENT ({overall_rot_mae:.1f}°)")
    elif rot_good:
        print(f"   ✅ Rotation: GOOD ({overall_rot_mae:.1f}°)")
    else:
        print(f"   ⚠️  Rotation: NEEDS WORK ({overall_rot_mae:.1f}°)")
    
    # Detailed per-joint analysis (top 5 worst for each)
    print(f"\n📊 Worst Performing Joints:")
    
    # Position errors
    pos_worst = sorted(zip(joint_names, per_joint_mpjpe), key=lambda x: x[1], reverse=True)
    print(f"   Position (MPJPE):")
    for i, (name, error) in enumerate(pos_worst[:5]):
        print(f"     {i+1}. {name:15s}: {error:5.1f} mm")
    
    # Rotation errors  
    rot_worst = sorted(zip(joint_names, per_joint_rot_mae), key=lambda x: x[1], reverse=True)
    print(f"   Rotation (MAE):")
    for i, (name, error) in enumerate(rot_worst[:5]):
        print(f"     {i+1}. {name:15s}: {error:5.1f}°")
    
    # Create visualizations
    results_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(results_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Position error histogram
    ax1.hist(np.array(position_errors) * 1000, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(overall_mpjpe, color='red', linestyle='--', label=f'Mean: {overall_mpjpe:.1f}mm')
    ax1.set_xlabel('Position Error (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Position Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2.hist(rotation_errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(overall_rot_mae, color='red', linestyle='--', label=f'Mean: {overall_rot_mae:.1f}°')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Rotation Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-joint position errors
    joint_indices = range(24)
    ax3.bar(joint_indices, per_joint_mpjpe, alpha=0.7, color='blue')
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('MPJPE (mm)')
    ax3.set_title('Per-Joint Position Errors')
    ax3.set_xticks(joint_indices[::4])
    ax3.grid(True, alpha=0.3)
    
    # Per-joint rotation errors
    ax4.bar(joint_indices, per_joint_rot_mae, alpha=0.7, color='orange')
    ax4.set_xlabel('Joint Index')
    ax4.set_ylabel('MAE (degrees)')
    ax4.set_title('Per-Joint Rotation Errors')
    ax4.set_xticks(joint_indices[::4])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    plot_path = os.path.join(results_dir, "combined_pose_rotation_evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plots: {plot_path}")
    
    # Save detailed report
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Combined Position + Rotation Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
        f.write(f"Samples evaluated: {len(position_errors)}\n\n")
        f.write(f"Overall Results:\n")
        f.write(f"  Position MPJPE: {overall_mpjpe:.1f} ± {np.std(position_errors)*1000:.1f} mm\n")
        f.write(f"  Rotation MAE:   {overall_rot_mae:.1f} ± {np.std(rotation_errors):.1f}°\n\n")
        f.write(f"Per-Joint Position Errors (MPJPE in mm):\n")
        for name, error in zip(joint_names, per_joint_mpjpe):
            f.write(f"  {name:15s}: {error:5.1f}\n")
        f.write(f"\nPer-Joint Rotation Errors (MAE in degrees):\n")
        for name, error in zip(joint_names, per_joint_rot_mae):
            f.write(f"  {name:15s}: {error:5.1f}\n")
    
    print(f"💾 Saved report: {report_path}")
    
    return {
        'position_mpjpe': overall_mpjpe,
        'rotation_mae': overall_rot_mae,
        'per_joint_position': per_joint_mpjpe,
        'per_joint_rotation': per_joint_rot_mae
    }

if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print(f"💡 Please update MODEL_PATH in the script configuration")
        print(f"💡 Available checkpoints:")
        checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        if os.path.exists(checkpoints_dir):
            for item in os.listdir(checkpoints_dir):
                item_path = os.path.join(checkpoints_dir, item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                    for file in os.listdir(item_path):
                        if file.endswith('.pth'):
                            print(f"      🔗 {file}")
        exit(1)
    
    results = evaluate_pose_and_rotation()
    
    print(f"\n🎉 Evaluation complete!")
    print(f"📊 Position MPJPE: {results['position_mpjpe']:.1f} mm")
    print(f"📊 Rotation MAE: {results['rotation_mae']:.1f}°")

