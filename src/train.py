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


def get_curriculum_ratio(epoch, start_epoch=10, increment=0.1, increment_every=10, max_ratio=1.0):
    """
    Calculate synthetic data ratio based on curriculum learning schedule.
    
    Args:
        epoch: Current epoch number
        start_epoch: Epoch to start introducing synthetic data (default: 10)
        increment: How much to increase synthetic ratio each time (default: 0.1 = 10%)
        increment_every: Increase ratio every N epochs (default: 10)
        max_ratio: Maximum synthetic ratio (default: 1.0 = 100%)
    
    Returns:
        Synthetic ratio for current epoch
    """
    if epoch < start_epoch:
        return 0.0  # No synthetic data before start_epoch
    
    # Calculate how many increments have passed
    epochs_since_start = epoch - start_epoch
    num_increments = epochs_since_start // increment_every + 1
    
    # Calculate current ratio
    current_ratio = num_increments * increment
    
    # Cap at maximum ratio
    return min(current_ratio, max_ratio)


def evaluate_on_real_validation(model, data_root, device, batch_size=64):
    """Evaluate model on real 3DPW validation data and return MPJPE"""
    print("📊 Evaluating on real 3DPW validation set...")
    
    # Create real-only validation dataset
    real_val_dataset = create_mixed_dataset(
        data_root=data_root,
        split='val',
        synthetic_ratio=0.0,  # No synthetic data
        real_ratio=1.0,       # All real data
        transform=NormalizerJoints2d(img_size=512)
    )
    
    if len(real_val_dataset) == 0:
        print("⚠️  No real validation samples found!")
        return None
        
    print(f"   Loaded {len(real_val_dataset)} real validation samples")
    
    real_val_loader = DataLoader(
        real_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    model.eval()
    total_mpjpe = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in real_val_loader:
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            
            pred_pos, _ = model(inputs)
            
            batch_size = inputs.size(0)
            pred_pos_3d = pred_pos.reshape(batch_size, 24, 3)
            target_pos_3d = target_pos.reshape(batch_size, 24, 3)
            
            sample_mpjpe = torch.norm(pred_pos_3d - target_pos_3d, dim=2).mean(dim=1)
            
            total_mpjpe += sample_mpjpe.sum().item()
            total_samples += batch_size
    
    avg_mpjpe_mm = (total_mpjpe / total_samples) * 1000.0 if total_samples > 0 else 0.0
    
    print(f"   Real validation MPJPE: {avg_mpjpe_mm:.1f} mm ({total_samples} samples)")
    
    model.train()
    return avg_mpjpe_mm


def main():
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Training hyperparameters
    BATCH_SIZE = 64  
    DROPOUT_RATE = 0.25
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 350
    NUM_JOINTS = 24
    IMG_SIZE = 512
    
    # Noise augmentation parameter
    KEYPOINT_NOISE_STD = 0.01  # Add 1% noise to 2D keypoints

    # Loss weights (you might need to adjust pos_weight since MSE has different scale)
    LOSS_POS_WEIGHT = 100.0  # Increased because MSE has smaller values than MPJPE
    LOSS_ROT_WEIGHT = 0.2
    LOSS_BONE_WEIGHT = 0.2

    # Curriculum learning configuration
    CURRICULUM_START_EPOCH = 10      # When to start introducing synthetic data
    CURRICULUM_INCREMENT = 0.05       # How much to increase synthetic ratio (10%)
    CURRICULUM_INCREMENT_EVERY = 5  # Increase every N epochs
    CURRICULUM_MAX_RATIO = 1.0       # Maximum synthetic ratio
    
    # Real data ratio (stays constant)
    REAL_RATIO = 1
    
    # Early stopping
    early_stopping_patience = 50
    early_stopping_counter = 0

    # Evaluation
    EVAL_EVERY_N_EPOCHS = 20
    CHECKPOINT_EVERY_N_EPOCHS = 10  # Save checkpoint every 10 epochs

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Curriculum learning: Start at epoch {CURRICULUM_START_EPOCH}, +{CURRICULUM_INCREMENT*100}% every {CURRICULUM_INCREMENT_EVERY} epochs")
    print(f"Real ratio: {REAL_RATIO}")
    print(f"Keypoint noise std: {KEYPOINT_NOISE_STD}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    # Create initial datasets (will be recreated during training for curriculum)
    print("Creating initial datasets...")
    
    # Start with 0% synthetic data
    initial_synthetic_ratio = 0.0
    
    train_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        split='train',
        synthetic_ratio=initial_synthetic_ratio,
        real_ratio=REAL_RATIO,
        transform=normalizer
    )
    
    val_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        split='val',
        synthetic_ratio=0.0,
        real_ratio=1.0,
        transform=normalizer
    )
    
    print(f"Initial train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create data loaders (will be recreated when dataset changes)
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
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Experiment setup
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"mixed_pose_{datetime.now():%Y%m%d_%H%M%S}_curriculum"
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
    real_val_mpjpe_history = []
    real_val_epochs = []
    
    # Track current synthetic ratio to avoid unnecessary dataset recreation
    current_synthetic_ratio = initial_synthetic_ratio

    for epoch in range(1, NUM_EPOCHS + 1):
        # Calculate synthetic ratio for this epoch
        epoch_synthetic_ratio = get_curriculum_ratio(
            epoch, 
            start_epoch=CURRICULUM_START_EPOCH,
            increment=CURRICULUM_INCREMENT,
            increment_every=CURRICULUM_INCREMENT_EVERY,
            max_ratio=CURRICULUM_MAX_RATIO
        )
        
        # Recreate dataset if ratio has changed
        if epoch_synthetic_ratio != current_synthetic_ratio:
            current_synthetic_ratio = epoch_synthetic_ratio
            print(f"\n📚 Curriculum update: Synthetic ratio now {current_synthetic_ratio*100:.0f}%")
            
            # Recreate training dataset with new ratio
            train_dataset = create_mixed_dataset(
                data_root=DATA_ROOT,
                split='train',
                synthetic_ratio=current_synthetic_ratio,
                real_ratio=REAL_RATIO,
                transform=normalizer
            )
            
            # Recreate training dataloader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                num_workers=4, 
                pin_memory=True,
                drop_last=True
            )
            
            print(f"   Recreated training dataset: {len(train_dataset)} samples")
        
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

        # Real validation evaluation
        real_val_mpjpe = None
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            real_val_mpjpe = evaluate_on_real_validation(model, DATA_ROOT, device, BATCH_SIZE)
            if real_val_mpjpe is not None:
                real_val_mpjpe_history.append(real_val_mpjpe)
                real_val_epochs.append(epoch)

        # Logging - always show MPJPE in mm
        train_mpjpe_mm = avg_train_mpjpe * 1000.0
        val_mpjpe_mm = avg_val_mpjpe * 1000.0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} [Synthetic: {current_synthetic_ratio*100:.0f}%]")
        print(f"  Train - MPJPE: {train_mpjpe_mm:.1f}mm, MSE: {avg_train_pos:.4f}, Rot: {avg_train_rot:.4f}, Bone: {avg_train_bone:.4f}")
        print(f"  Val   - MPJPE: {val_mpjpe_mm:.1f}mm, MSE: {avg_val_pos:.4f}, Rot: {avg_val_rot:.4f}, Bone: {avg_val_bone:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if real_val_mpjpe is not None:
            print(f"  Real Val MPJPE: {real_val_mpjpe:.1f}mm 🎯")
        
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
                "real_val_mpjpe": real_val_mpjpe,
                "curriculum_synthetic_ratio": current_synthetic_ratio,
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
                    "curriculum_start": CURRICULUM_START_EPOCH,
                    "curriculum_increment": CURRICULUM_INCREMENT,
                    "curriculum_every": CURRICULUM_INCREMENT_EVERY,
                    "real_ratio": REAL_RATIO,
                    "keypoint_noise_std": KEYPOINT_NOISE_STD
                }
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    # Finalization
    model.load_state_dict(best_state)
    print(f"\nTraining complete! Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # Final evaluation
    final_real_mpjpe = evaluate_on_real_validation(model, DATA_ROOT, device, BATCH_SIZE)
    if final_real_mpjpe is not None:
        print(f"🎯 Final Real Validation MPJPE: {final_real_mpjpe:.1f} mm")

    # Create learning curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MPJPE curve (primary metric)
    train_mpjpe_mm = [mpjpe * 1000.0 for mpjpe in train_mpjpe_history]
    val_mpjpe_mm = [mpjpe * 1000.0 for mpjpe in val_mpjpe_history]
    
    axes[0, 0].plot(train_mpjpe_mm, label="Train MPJPE", alpha=0.8)
    axes[0, 0].plot(val_mpjpe_mm, label="Val MPJPE", alpha=0.8)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MPJPE (mm)")
    axes[0, 0].set_title("MPJPE - Primary Metric")
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

    # Real validation MPJPE
    if real_val_mpjpe_history:
        axes[1, 1].plot(real_val_epochs, real_val_mpjpe_history, 'ro-', label="Real Val MPJPE", alpha=0.8)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MPJPE (mm)")
        axes[1, 1].set_title("Real 3DPW Validation Performance")
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
        "final_real_val_mpjpe": final_real_mpjpe,
        "final_synthetic_ratio": current_synthetic_ratio,
        "real_val_history": {
            "epochs": real_val_epochs,
            "mpjpe_values": real_val_mpjpe_history
        },
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": LOSS_BONE_WEIGHT,
            "curriculum_start": CURRICULUM_START_EPOCH,
            "curriculum_increment": CURRICULUM_INCREMENT,
            "curriculum_every": CURRICULUM_INCREMENT_EVERY,
            "real_ratio": REAL_RATIO,
            "num_epochs_trained": epoch,
            "keypoint_noise_std": KEYPOINT_NOISE_STD
        }
    }, final_path)
    print(f"Final model → {final_path}")


if __name__ == "__main__":
    main()