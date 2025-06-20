import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data.mixed_pose_dataset import create_mixed_dataset
from models.rot_head import MLPLifterResidualHead
from utils.losses import combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d


def main():
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Training hyperparameters
    BATCH_SIZE = 64  
    DROPOUT_RATE = 0.25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 350
    NUM_JOINTS = 24
    IMG_SIZE = 512
    
    # Noise augmentation parameter
    KEYPOINT_NOISE_STD = 0.04  # Add 1% noise to 2D keypoints

    # Loss weights (you might need to adjust pos_weight since MSE has different scale)
    LOSS_POS_WEIGHT = 60.0  # Increased because MSE has smaller values than MPJPE
    LOSS_ROT_WEIGHT = 0.45
    LOSS_BONE_WEIGHT = 0.2

    # SYNTHETIC ONLY TRAINING
    SYNTHETIC_RATIO = 1.0  # Use all synthetic data
    REAL_RATIO = 0.0       # No real data
    
    # Early stopping
    early_stopping_patience = 50
    early_stopping_counter = 0

    # Checkpoint saving
    CHECKPOINT_EVERY_N_EPOCHS = 10  # Save checkpoint every 10 epochs

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training mode: SYNTHETIC TRAIN / REAL VALIDATION")
    print(f"  This setup shows the domain gap between synthetic and real data")
    print(f"  Train on: 100% synthetic data")
    print(f"  Validate on: 100% real data")
    print(f"Keypoint noise std: {KEYPOINT_NOISE_STD}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    # Create datasets
    print("Creating datasets...")
    
    # Training dataset - synthetic only
    train_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        split='train',
        synthetic_ratio=SYNTHETIC_RATIO,
        real_ratio=REAL_RATIO,
        transform=normalizer
    )
    
    # Validation dataset - use REAL data to see domain gap
    val_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        split='val',
        synthetic_ratio=0.0,  # No synthetic for validation
        real_ratio=1.0,       # All real data for validation
        transform=normalizer
    )
    
    print(f"Train dataset: {len(train_dataset)} samples (synthetic only)")
    print(f"Val dataset: {len(val_dataset)} samples (real data)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,   
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )

    # Model setup
    model = MLPLifterResidualHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Use MPJPE for scheduler (not MSE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=7, verbose=True
    )

    # Experiment setup
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"synthetic_only_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # Training loop
    best_val_mpjpe = float("inf")
    best_state = None
    train_mpjpe_history, val_mpjpe_history = [], []
    train_pos_losses, val_pos_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Adjust bone loss weight
        if epoch == 100:
            LOSS_BONE_WEIGHT = 0.05

        # Training phase
        model.train()
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        running_train_mpjpe = 0.0
        
        epoch_data_types = {'synthetic': 0, 'real': 0}
        
        for batch in train_loader:
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            
            # Add Gaussian noise to 2D keypoints during training
            if model.training:
                noise = torch.randn_like(inputs) * KEYPOINT_NOISE_STD
                inputs = inputs + noise

            if 'data_type' in batch:
                for dtype in batch['data_type']:
                    epoch_data_types[dtype] = epoch_data_types.get(dtype, 0) + 1

            pos3d, rot6d = model(inputs)

            pred_dict = {
                'positions': pos3d,
                'rotations': rot6d
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot
            }
            
            loss_dict = combined_pose_bone_loss(
                pred_dict, target_dict, 
                pos_weight=LOSS_POS_WEIGHT, 
                rot_weight=LOSS_ROT_WEIGHT,
                bone_weight=LOSS_BONE_WEIGHT
            )
            
            loss = loss_dict['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()
            running_train_mpjpe += loss_dict['mpjpe'].item()  # Track MPJPE

        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        avg_train_mpjpe = running_train_mpjpe / len(train_loader)
        
        train_pos_losses.append(avg_train_pos)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)
        train_mpjpe_history.append(avg_train_mpjpe)

        # Validation phase
        model.eval()
        running_val_pos = 0.0
        running_val_rot = 0.0
        running_val_bone = 0.0
        running_val_mpjpe = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target_pos = batch["joints_3d_centered"].to(device)
                target_rot = batch["rot_6d"].to(device)
                
                # No noise during validation
                
                pos3d, rot6d = model(inputs)
                
                pred_dict = {
                    'positions': pos3d,
                    'rotations': rot6d
                }
                target_dict = {
                    'positions': target_pos.flatten(1),
                    'rotations': target_rot
                }
                
                loss_dict = combined_pose_bone_loss(
                    pred_dict, target_dict, 
                    pos_weight=LOSS_POS_WEIGHT, 
                    rot_weight=LOSS_ROT_WEIGHT,
                    bone_weight=LOSS_BONE_WEIGHT
                )
                
                running_val_pos += loss_dict['position'].item()
                running_val_rot += loss_dict['rotation'].item()
                running_val_bone += loss_dict['bone'].item()
                running_val_mpjpe += loss_dict['mpjpe'].item()  # Track MPJPE

        avg_val_pos = running_val_pos / len(val_loader)
        avg_val_rot = running_val_rot / len(val_loader)
        avg_val_bone = running_val_bone / len(val_loader)
        avg_val_mpjpe = running_val_mpjpe / len(val_loader)
        
        val_pos_losses.append(avg_val_pos)
        val_rot_losses.append(avg_val_rot)
        val_bone_losses.append(avg_val_bone)
        val_mpjpe_history.append(avg_val_mpjpe)

        # Logging - always show MPJPE in mm
        train_mpjpe_mm = avg_train_mpjpe * 1000.0
        val_mpjpe_mm = avg_val_mpjpe * 1000.0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} [SYNTHETIC TRAIN / REAL VAL]")
        print(f"  Train - MPJPE: {train_mpjpe_mm:.1f}mm, MSE: {avg_train_pos:.4f}, Rot: {avg_train_rot:.4f}, Bone: {avg_train_bone:.4f}")
        print(f"  Val   - MPJPE: {val_mpjpe_mm:.1f}mm, MSE: {avg_val_pos:.4f}, Rot: {avg_val_rot:.4f}, Bone: {avg_val_bone:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        total_samples = sum(epoch_data_types.values())
        if total_samples > 0:
            synth_pct = epoch_data_types.get('synthetic', 0) / total_samples * 100
            real_pct = epoch_data_types.get('real', 0) / total_samples * 100
            print(f"  Data: Synthetic {synth_pct:.1f}%, Real {real_pct:.1f}%")

        # Use validation MPJPE for scheduler and best model selection
        scheduler.step(avg_val_mpjpe)

        # Save best model based on MPJPE
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            best_state = model.state_dict().copy()
            early_stopping_counter = 0
            print(f"✓ New best validation MPJPE: {val_mpjpe_mm:.1f} mm")
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        # Save checkpoint every N epochs
        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_mpjpe": avg_train_mpjpe,
                "val_mpjpe": avg_val_mpjpe,
                "best_val_mpjpe": best_val_mpjpe,
                "synthetic_ratio": SYNTHETIC_RATIO,
                "real_ratio": REAL_RATIO,
                "train_history": {
                    "mpjpe": train_mpjpe_history,
                    "pos_loss": train_pos_losses,
                    "rot_loss": train_rot_losses,
                    "bone_loss": train_bone_losses
                },
                "val_history": {
                    "mpjpe": val_mpjpe_history,
                    "pos_loss": val_pos_losses,
                    "rot_loss": val_rot_losses,
                    "bone_loss": val_bone_losses
                },
                "hyperparameters": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "dropout_rate": DROPOUT_RATE,
                    "pos_weight": LOSS_POS_WEIGHT,
                    "rot_weight": LOSS_ROT_WEIGHT,
                    "bone_weight": LOSS_BONE_WEIGHT,
                    "synthetic_ratio": SYNTHETIC_RATIO,
                    "real_ratio": REAL_RATIO,
                    "keypoint_noise_std": KEYPOINT_NOISE_STD
                }
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    # Finalization
    model.load_state_dict(best_state)
    print(f"\nTraining complete! Best Real Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # Create learning curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MPJPE curve (primary metric)
    train_mpjpe_mm = [mpjpe * 1000.0 for mpjpe in train_mpjpe_history]
    val_mpjpe_mm = [mpjpe * 1000.0 for mpjpe in val_mpjpe_history]
    
    axes[0, 0].plot(train_mpjpe_mm, label="Train MPJPE (Synthetic)", alpha=0.8)
    axes[0, 0].plot(val_mpjpe_mm, label="Val MPJPE (Real)", alpha=0.8)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MPJPE (mm)")
    axes[0, 0].set_title("MPJPE - Domain Gap Visualization")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Rotation loss
    axes[0, 1].plot(train_rot_losses, label="Train Rot Loss", alpha=0.8)
    axes[0, 1].plot(val_rot_losses, label="Val Rot Loss", alpha=0.8)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Geodesic Loss")
    axes[0, 1].set_title("Rotation Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bone loss
    axes[1, 0].plot(train_bone_losses, label="Train Bone Loss", alpha=0.8)
    axes[1, 0].plot(val_bone_losses, label="Val Bone Loss", alpha=0.8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Bone Length Loss")
    axes[1, 0].set_title("Bone Consistency Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # MSE loss (position) - use the 4th subplot
    axes[1, 1].plot(train_pos_losses, label="Train MSE", alpha=0.8)
    axes[1, 1].plot(val_pos_losses, label="Val MSE", alpha=0.8)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].set_title("Position MSE Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(experiment_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curves → {curve_path}")

    # Save final model
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "synthetic_ratio": SYNTHETIC_RATIO,
        "real_ratio": REAL_RATIO,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": LOSS_BONE_WEIGHT,
            "synthetic_ratio": SYNTHETIC_RATIO,
            "real_ratio": REAL_RATIO,
            "num_epochs_trained": epoch,
            "keypoint_noise_std": KEYPOINT_NOISE_STD
        }
    }, final_path)
    print(f"Final model → {final_path}")


if __name__ == "__main__":
    main()