#!/usr/bin/env python3
"""
Evaluation script for models trained on mixed datasets
"""

import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.rot_head import MLPLifterRotationHead
from src.data.mixed_pose_dataset import create_mixed_dataset
from src.utils.transforms import NormalizerJoints2d
from src.utils.losses import mpjpe_loss, geodesic_loss
from src.utils import rotation_utils


def rotation_mae_degrees(pred_rot_6d, target_rot_6d):
    """Calculate rotation MAE in degrees"""
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


def evaluate_model(model_path, data_root, eval_mode='mixed', eval_split='val', batch_size=64):
    """
    Evaluate a model on the specified dataset
    
    Args:
        model_path: Path to the model checkpoint
        data_root: Root directory of the data
        eval_mode: 'mixed', 'synthetic', or 'real'
        eval_split: 'train', 'val', or 'test'
        batch_size: Batch size for evaluation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📊 MIXED DATASET MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Evaluation mode: {eval_mode}")
    print(f"Evaluation split: {eval_split}")
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = MLPLifterRotationHead(num_joints=24).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Print training info if available
    if 'hyperparameters' in checkpoint:
        hp = checkpoint['hyperparameters']
        print(f"\nTraining Info:")
        print(f"  Dataset mode: {hp.get('dataset_mode', 'unknown')}")
        print(f"  Synthetic ratio: {hp.get('synthetic_ratio', 'N/A')}")
        print(f"  Training MPJPE: {checkpoint.get('best_val_mpjpe', 0)*1000:.1f} mm")
        print(f"  Epochs trained: {hp.get('num_epochs_trained', 'unknown')}")
    
    if 'dataset_info' in checkpoint:
        di = checkpoint['dataset_info']
        print(f"  Training samples: {di.get('train_samples', 'unknown')}")
        if di.get('train_distribution'):
            print(f"  Training distribution: {di['train_distribution']}")
    
    # Create evaluation dataset
    print(f"\n📂 Loading evaluation dataset...")
    try:
        eval_dataset = create_mixed_dataset(
            data_root=data_root,
            mode=eval_mode,
            split=eval_split,
            synthetic_ratio=0.5,  # Default ratio for mixed evaluation
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False  # No augmentation during evaluation
        )
        print(f"✅ Evaluation dataset loaded: {len(eval_dataset)} samples")
        
        if hasattr(eval_dataset, 'get_dataset_distribution'):
            distribution = eval_dataset.get_dataset_distribution()
            print(f"   Distribution: {distribution}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load evaluation dataset: {e}")
    
    # Create DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Load joint names for reporting
    joint_mapping_path = os.path.join(data_root, "meta", "joints_mapping.json")
    with open(joint_mapping_path, 'r') as f:
        joint_mapping = json.load(f)
    joint_names = joint_mapping["smpl_names"]
    
    # Evaluation metrics storage
    all_position_errors = []
    all_rotation_errors = []
    per_joint_position_errors = [[] for _ in range(24)]
    per_joint_rotation_errors = [[] for _ in range(24)]
    
    # Track performance by data type
    errors_by_type = {'synthetic': [], 'real': []}
    rot_errors_by_type = {'synthetic': [], 'real': []}
    
    print(f"\n🔄 Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            
            # Forward pass
            pred_pos, pred_rot = model(inputs)
            
            # Calculate position errors (MPJPE)
            batch_size = inputs.size(0)
            pred_pos_3d = pred_pos.reshape(batch_size, 24, 3)
            target_pos_3d = target_pos.reshape(batch_size, 24, 3)
            
            # Per-sample MPJPE
            sample_mpjpe = torch.norm(pred_pos_3d - target_pos_3d, dim=2).mean(dim=1)
            all_position_errors.extend(sample_mpjpe.cpu().numpy())
            
            # Per-joint position errors
            joint_pos_errors = torch.norm(pred_pos_3d - target_pos_3d, dim=2)
            for j in range(24):
                per_joint_position_errors[j].extend(joint_pos_errors[:, j].cpu().numpy())
            
            # Calculate rotation errors
            pred_rot_6d = pred_rot.reshape(batch_size, 24, 6)
            target_rot_6d = target_rot.reshape(batch_size, 24, 6)
            
            sample_rot_mae = rotation_mae_degrees(pred_rot_6d, target_rot_6d)
            sample_mean_mae = sample_rot_mae.mean(dim=1)
            all_rotation_errors.extend(sample_mean_mae.cpu().numpy())
            
            # Per-joint rotation errors
            for j in range(24):
                per_joint_rotation_errors[j].extend(sample_rot_mae[:, j].cpu().numpy())
            
            # Track errors by data type if available
            if 'data_type' in batch:
                for i, dtype in enumerate(batch['data_type']):
                    pos_error = sample_mpjpe[i].cpu().numpy()
                    rot_error = sample_mean_mae[i].cpu().numpy()
                    
                    if dtype in errors_by_type:
                        errors_by_type[dtype].append(pos_error)
                        rot_errors_by_type[dtype].append(rot_error)
            
            if (batch_idx + 1) % 20 == 0:
                processed = (batch_idx + 1) * batch_size
                print(f"  Processed {processed} samples...")
    
    # Calculate overall statistics
    overall_mpjpe = np.mean(all_position_errors) * 1000  # Convert to mm
    overall_rot_mae = np.mean(all_rotation_errors)  # Already in degrees
    
    per_joint_mpjpe = [np.mean(errors) * 1000 for errors in per_joint_position_errors]
    per_joint_rot_mae = [np.mean(errors) for errors in per_joint_rotation_errors]
    
    # Print results
    print(f"\n📈 Overall Results:")
    print(f"   Position MPJPE: {overall_mpjpe:.1f} mm ± {np.std(all_position_errors)*1000:.1f} mm")
    print(f"   Rotation MAE:   {overall_rot_mae:.1f}° ± {np.std(all_rotation_errors):.1f}°")
    print(f"   Samples evaluated: {len(all_position_errors)}")
    
    # Results by data type
    if any(len(errors) > 0 for errors in errors_by_type.values()):
        print(f"\n📊 Results by Data Type:")
        for dtype in ['synthetic', 'real']:
            if len(errors_by_type[dtype]) > 0:
                pos_mean = np.mean(errors_by_type[dtype]) * 1000
                rot_mean = np.mean(rot_errors_by_type[dtype])
                count = len(errors_by_type[dtype])
                print(f"   {dtype.capitalize():>10}: "
                      f"MPJPE {pos_mean:5.1f}mm, "
                      f"Rot {rot_mean:4.1f}°, "
                      f"({count} samples)")
    
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
    
    # Worst performing joints
    print(f"\n📊 Worst Performing Joints:")
    pos_worst = sorted(zip(joint_names, per_joint_mpjpe), key=lambda x: x[1], reverse=True)
    print(f"   Position (MPJPE):")
    for i, (name, error) in enumerate(pos_worst[:5]):
        print(f"     {i+1}. {name:15s}: {error:5.1f} mm")
    
    rot_worst = sorted(zip(joint_names, per_joint_rot_mae), key=lambda x: x[1], reverse=True)
    print(f"   Rotation (MAE):")
    for i, (name, error) in enumerate(rot_worst[:5]):
        print(f"     {i+1}. {name:15s}: {error:5.1f}°")
    
    # Create visualizations
    results_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Position error histogram
    axes[0, 0].hist(np.array(all_position_errors) * 1000, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(overall_mpjpe, color='red', linestyle='--', label=f'Mean: {overall_mpjpe:.1f}mm')
    axes[0, 0].set_xlabel('Position Error (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Position Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation error histogram
    axes[0, 1].hist(all_rotation_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(overall_rot_mae, color='red', linestyle='--', label=f'Mean: {overall_rot_mae:.1f}°')
    axes[0, 1].set_xlabel('Rotation Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Rotation Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-joint position errors
    joint_indices = range(24)
    axes[0, 2].bar(joint_indices, per_joint_mpjpe, alpha=0.7, color='blue')
    axes[0, 2].set_xlabel('Joint Index')
    axes[0, 2].set_ylabel('MPJPE (mm)')
    axes[0, 2].set_title('Per-Joint Position Errors')
    axes[0, 2].set_xticks(joint_indices[::4])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Per-joint rotation errors
    axes[1, 0].bar(joint_indices, per_joint_rot_mae, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Joint Index')
    axes[1, 0].set_ylabel('MAE (degrees)')
    axes[1, 0].set_title('Per-Joint Rotation Errors')
    axes[1, 0].set_xticks(joint_indices[::4])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Data type comparison (if available)
    if any(len(errors) > 0 for errors in errors_by_type.values()):
        types = [t for t in ['synthetic', 'real'] if len(errors_by_type[t]) > 0]
        pos_means = [np.mean(errors_by_type[t]) * 1000 for t in types]
        rot_means = [np.mean(rot_errors_by_type[t]) for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, pos_means, width, label='Position MPJPE (mm)', alpha=0.7)
        axes[1, 1].set_xlabel('Data Type')
        axes[1, 1].set_ylabel('Position Error (mm)')
        axes[1, 1].set_title('Performance by Data Type - Position')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(types)
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].bar(x, rot_means, width, label='Rotation MAE (°)', alpha=0.7, color='orange')
        axes[1, 2].set_xlabel('Data Type')
        axes[1, 2].set_ylabel('Rotation Error (degrees)')
        axes[1, 2].set_title('Performance by Data Type - Rotation')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(types)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Data type info\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 2].text(0.5, 0.5, 'Data type info\nnot available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    
    # Save results
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(results_dir, f"evaluation_{model_name}_{eval_mode}_{eval_split}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plots: {plot_path}")
    
    # Save detailed report
    report_path = os.path.join(results_dir, f"evaluation_report_{model_name}_{eval_mode}_{eval_split}.txt")
    with open(report_path, 'w') as f:
        f.write("Mixed Dataset Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation mode: {eval_mode}\n")
        f.write(f"Evaluation split: {eval_split}\n")
        f.write(f"Samples evaluated: {len(all_position_errors)}\n\n")
        
        f.write(f"Overall Results:\n")
        f.write(f"  Position MPJPE: {overall_mpjpe:.1f} ± {np.std(all_position_errors)*1000:.1f} mm\n")
        f.write(f"  Rotation MAE:   {overall_rot_mae:.1f} ± {np.std(all_rotation_errors):.1f}°\n\n")
        
        # Results by data type
        if any(len(errors) > 0 for errors in errors_by_type.values()):
            f.write(f"Results by Data Type:\n")
            for dtype in ['synthetic', 'real']:
                if len(errors_by_type[dtype]) > 0:
                    pos_mean = np.mean(errors_by_type[dtype]) * 1000
                    rot_mean = np.mean(rot_errors_by_type[dtype])
                    count = len(errors_by_type[dtype])
                    f.write(f"  {dtype.capitalize()}: MPJPE {pos_mean:.1f}mm, Rot {rot_mean:.1f}°, ({count} samples)\n")
            f.write("\n")
        
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
        'per_joint_rotation': per_joint_rot_mae,
        'errors_by_type': errors_by_type,
        'rot_errors_by_type': rot_errors_by_type
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate mixed dataset model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--mode', type=str, choices=['mixed', 'synthetic', 'real'], 
                       default='mixed', help='Evaluation dataset mode')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='val', help='Evaluation split')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.model):
        args.model = os.path.join(PROJECT_ROOT, args.model)
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(PROJECT_ROOT, args.data_root)
    
    try:
        results = evaluate_model(
            model_path=args.model,
            data_root=args.data_root,
            eval_mode=args.mode,
            eval_split=args.split,
            batch_size=args.batch_size
        )
        
        print(f"\n🎉 Evaluation complete!")
        print(f"📊 Position MPJPE: {results['position_mpjpe']:.1f} mm")
        print(f"📊 Rotation MAE: {results['rotation_mae']:.1f}°")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())