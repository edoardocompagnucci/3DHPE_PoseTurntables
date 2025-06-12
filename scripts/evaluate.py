import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(PROJECT_ROOT))

def evaluate_model(checkpoint_path, data_root, dataset_type='mixed', real_data_ratio=0.2, 
                   batch_size=64, device=None, min_confidence=0.3):

    from src.models.rot_head import MLPLifterRotationHead
    from src.utils.transforms import NormalizerJoints2d
    from src.data.synthetic_pose_dataset import SyntheticPoseDataset
    from src.data.threedpw_dataset import ThreeDPWDataset, MixedPoseDataset

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 Evaluating on {device}")
    print(f"📂 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"📊 Dataset type: {dataset_type}")
    
    model = MLPLifterRotationHead(num_joints=24, dropout=0.1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✅ Model loaded successfully")

    normalizer = NormalizerJoints2d(img_size=512)
    val_split_txt = os.path.join(data_root, "splits", "val.txt")

    if dataset_type == 'synthetic':
        val_dataset = SyntheticPoseDataset(
            data_root=data_root,
            split_txt=val_split_txt,
            transform=normalizer,
            augment_2d=False
        )
        print(f"📊 Synthetic validation samples: {len(val_dataset)}")
        
    elif dataset_type == '3dpw':
        val_dataset = ThreeDPWDataset(
            data_root=data_root,
            split="validation",
            transform=normalizer,
            min_confidence=min_confidence
        )
        print(f"📊 3DPW validation samples: {len(val_dataset)}")
        
    elif dataset_type == 'mixed':
        synthetic_val = SyntheticPoseDataset(
            data_root=data_root,
            split_txt=val_split_txt,
            transform=normalizer,
            augment_2d=False
        )
        threedpw_val = ThreeDPWDataset(
            data_root=data_root,
            split="validation",
            transform=normalizer,
            min_confidence=min_confidence
        )
        val_dataset = MixedPoseDataset(
            synthetic_dataset=synthetic_val,
            threedpw_dataset=threedpw_val,
            real_data_ratio=real_data_ratio,
            seed=42
        )
        print(f"📊 Mixed validation samples: {len(val_dataset)}")
        
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'synthetic', '3dpw', or 'mixed'")
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluation
    print(f"🔄 Running evaluation...")
    
    total_mpjpe = 0.0
    total_samples = 0
    synthetic_mpjpe = 0.0
    real_mpjpe = 0.0
    synthetic_count = 0
    real_count = 0
    
    per_joint_errors = np.zeros(24)
    per_joint_counts = np.zeros(24)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            
            # Forward pass
            pos3d, _ = model(inputs)
            
            # Reshape predictions
            batch_size = inputs.size(0)
            pred_pos_3d = pos3d.reshape(batch_size, 24, 3)
            target_pos_3d = target_pos.reshape(batch_size, 24, 3)
            
            # Compute per-sample MPJPE
            joint_errors = torch.norm(pred_pos_3d - target_pos_3d, dim=2)  # (B, J)
            sample_mpjpe = joint_errors.mean(dim=1)  # (B,)
            
            # Accumulate per-joint errors
            per_joint_errors += joint_errors.sum(dim=0).cpu().numpy()
            per_joint_counts += batch_size
            
            # Overall MPJPE
            total_mpjpe += sample_mpjpe.sum().item()
            total_samples += batch_size
            
            # Separate synthetic vs real if mixed dataset
            if dataset_type == 'mixed':
                dataset_types = batch["dataset_type"]
                for i, dt in enumerate(dataset_types):
                    if dt == 'synthetic':
                        synthetic_mpjpe += sample_mpjpe[i].item()
                        synthetic_count += 1
                    else:
                        real_mpjpe += sample_mpjpe[i].item()
                        real_count += 1
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                processed = (batch_idx + 1) * batch_size
                print(f"  Processed {processed}/{len(val_dataset)} samples...")
    
    # Calculate final metrics
    overall_mpjpe_mm = (total_mpjpe / total_samples) * 1000.0
    per_joint_mpjpe_mm = (per_joint_errors / per_joint_counts) * 1000.0
    
    results = {
        'overall_mpjpe_mm': overall_mpjpe_mm,
        'total_samples': total_samples,
        'dataset_type': dataset_type,
        'per_joint_mpjpe_mm': per_joint_mpjpe_mm.tolist(),
    }
    
    # Add dataset-specific metrics
    if dataset_type == 'mixed':
        if synthetic_count > 0:
            results['synthetic_mpjpe_mm'] = (synthetic_mpjpe / synthetic_count) * 1000.0
            results['synthetic_samples'] = synthetic_count
        if real_count > 0:
            results['real_mpjpe_mm'] = (real_mpjpe / real_count) * 1000.0
            results['real_samples'] = real_count
        results['real_data_ratio'] = real_data_ratio
    
    # Print results
    print(f"\n📈 Evaluation Results:")
    print(f"   Overall MPJPE: {overall_mpjpe_mm:.1f} mm")
    print(f"   Total samples: {total_samples}")
    
    if dataset_type == 'mixed':
        if synthetic_count > 0:
            print(f"   Synthetic MPJPE: {results['synthetic_mpjpe_mm']:.1f} mm ({synthetic_count} samples)")
        if real_count > 0:
            print(f"   Real MPJPE: {results['real_mpjpe_mm']:.1f} mm ({real_count} samples)")
    
    # Performance assessment
    if overall_mpjpe_mm < 50:
        print(f"   ✅ EXCELLENT performance ({overall_mpjpe_mm:.1f} mm)")
    elif overall_mpjpe_mm < 70:
        print(f"   ✅ GOOD performance ({overall_mpjpe_mm:.1f} mm)")
    elif overall_mpjpe_mm < 100:
        print(f"   ⚠️  FAIR performance ({overall_mpjpe_mm:.1f} mm)")
    else:
        print(f"   ❌ NEEDS IMPROVEMENT ({overall_mpjpe_mm:.1f} mm)")
    
    return results


# Example usage
if __name__ == "__main__":
    results = evaluate_model(
        checkpoint_path=r"checkpoints\mlp_lifter_domain_mixing_20250611_211849\final_model.pth",
        data_root="data/",
        dataset_type='3dpw',  # 'synthetic', '3dpw', or 'mixed'
    )

    print(f"MPJPE: {results['overall_mpjpe_mm']:.1f} mm")