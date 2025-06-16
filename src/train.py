import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import the new mixed dataset
from data.mixed_pose_dataset import create_mixed_dataset
from models.rot_head import MLPLifterRotationHead
from utils.losses import mpjpe_loss, combined_pose_loss, combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d


def evaluate_on_real_validation(model, data_root, device, batch_size=64):
    """
    Evaluate model on real 3DPW validation data and return MPJPE
    
    Args:
        model: The trained model
        data_root: Path to data root
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        float: Mean MPJPE in millimeters
    """
    print("📊 Evaluating on real 3DPW validation set...")
    
    # Create real validation dataset
    try:
        real_val_dataset = create_mixed_dataset(
            data_root=data_root,
            mode='real',
            split='val',  # This maps to 'validation' split internally
            transform=NormalizerJoints2d(img_size=512),
            augment_2d=False  # No augmentation during evaluation
        )
        
        if len(real_val_dataset) == 0:
            print("⚠️  No real validation samples found!")
            return None
            
        print(f"   Loaded {len(real_val_dataset)} real validation samples")
        
    except Exception as e:
        print(f"❌ Failed to load real validation dataset: {e}")
        return None
    
    # Create DataLoader
    real_val_loader = DataLoader(
        real_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # Evaluation
    model.eval()
    total_mpjpe = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(real_val_loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            
            # Forward pass
            pred_pos, _ = model(inputs)  # We only need positions for MPJPE
            
            # Calculate MPJPE
            batch_size = inputs.size(0)
            pred_pos_3d = pred_pos.reshape(batch_size, 24, 3)
            target_pos_3d = target_pos.reshape(batch_size, 24, 3)
            
            # Per-sample MPJPE
            sample_mpjpe = torch.norm(pred_pos_3d - target_pos_3d, dim=2).mean(dim=1)
            
            total_mpjpe += sample_mpjpe.sum().item()
            total_samples += batch_size
    
    # Calculate average MPJPE in mm
    avg_mpjpe_mm = (total_mpjpe / total_samples) * 1000.0 if total_samples > 0 else 0.0
    
    print(f"   Real validation MPJPE: {avg_mpjpe_mm:.1f} mm ({total_samples} samples)")
    
    model.train()  # Switch back to training mode
    return avg_mpjpe_mm


def main():
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Training hyperparameters
    BATCH_SIZE = 64  
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 350
    NUM_JOINTS = 24
    IMG_SIZE = 512

    # Loss weights
    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.1
    LOSS_BONE_WEIGHT = 0.15

    # Dataset configuration
    DATASET_MODE = 'mixed'  # 'mixed', 'synthetic', 'real'
    SYNTHETIC_RATIO = 0.7   # Only used when mode='mixed'
    
    # Augmentation parameters
    AUGMENT_2D = False
    CAMERA_AUG_ROTATION_DEG = 0.0
    CAMERA_AUG_TRANSLATION_M = 0.00
    NOISE_STD = 0.00
    CONFIDENCE_NOISE = 0.00
    MAX_SHIFT = 0.00

    # Early stopping
    early_stopping_patience = 50
    early_stopping_counter = 0

    # Evaluation configuration
    EVAL_EVERY_N_EPOCHS = 10  # Evaluate on real validation every N epochs

    # =============================================================================
    # SETUP
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset mode: {DATASET_MODE}")
    if DATASET_MODE == 'mixed':
        print(f"Synthetic ratio: {SYNTHETIC_RATIO}")

    # Create transform
    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    # =============================================================================
    # DATASET CREATION
    # =============================================================================
    print("Creating datasets...")
    
    # Training dataset
    train_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        mode=DATASET_MODE,
        split='train',
        synthetic_ratio=SYNTHETIC_RATIO,
        transform=normalizer,
        augment_2d=AUGMENT_2D,
        camera_aug_rotation_deg=CAMERA_AUG_ROTATION_DEG,
        camera_aug_translation_m=CAMERA_AUG_TRANSLATION_M,
        noise_std=NOISE_STD,
        confidence_noise=CONFIDENCE_NOISE,
        max_shift=MAX_SHIFT
    )
    
    # Validation dataset (usually less augmentation)
    val_dataset = create_mixed_dataset(
        data_root=DATA_ROOT,
        mode=DATASET_MODE,
        split='val',
        synthetic_ratio=SYNTHETIC_RATIO,
        transform=normalizer,
        augment_2d=False,  # No augmentation for validation
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Print dataset distribution if mixed
    if hasattr(train_dataset, 'get_dataset_distribution'):
        train_dist = train_dataset.get_dataset_distribution()
        val_dist = val_dataset.get_dataset_distribution()
        print(f"Train distribution: {train_dist}")
        print(f"Val distribution: {val_dist}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset,   
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )

    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, verbose=True
    )

    # =============================================================================
    # EXPERIMENT SETUP
    # =============================================================================
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"mixed_pose_{DATASET_MODE}_{datetime.now():%Y%m%d_%H%M%S}"
    if DATASET_MODE == 'mixed':
        exp_name += f"_synth{int(SYNTHETIC_RATIO*100)}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    best_val_mpjpe = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []
    
    # Track real validation performance
    real_val_mpjpe_history = []
    real_val_epochs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Adjust bone loss weight
        if epoch == 100:
            LOSS_BONE_WEIGHT = 0.05
        
        # Reshuffle mixed dataset between epochs
        if hasattr(train_dataset, 'reshuffle'):
            train_dataset.reshuffle()

        # =============================================================================
        # TRAINING PHASE
        # =============================================================================
        model.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        
        # Track data type distribution in current epoch
        epoch_data_types = {'synthetic': 0, 'real': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)

            # Track data types in batch
            if 'data_type' in batch:
                for dtype in batch['data_type']:
                    epoch_data_types[dtype] = epoch_data_types.get(dtype, 0) + 1

            # Forward pass
            pos3d, rot6d = model(inputs)

            # Compute loss
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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            running_loss += loss.item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()

        # Average training losses
        avg_train = running_loss / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)

        # =============================================================================
        # VALIDATION PHASE
        # =============================================================================
        model.eval()
        running_val = 0.0
        running_val_pos = 0.0
        running_val_rot = 0.0
        running_val_bone = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target_pos = batch["joints_3d_centered"].to(device)
                target_rot = batch["rot_6d"].to(device)
                
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
                
                running_val += loss_dict['total'].item()
                running_val_pos += loss_dict['position'].item()
                running_val_rot += loss_dict['rotation'].item()
                running_val_bone += loss_dict['bone'].item()

        # Average validation losses
        avg_val = running_val / len(val_loader)
        avg_val_pos = running_val_pos / len(val_loader)
        avg_val_rot = running_val_rot / len(val_loader)
        avg_val_bone = running_val_bone / len(val_loader)
        
        val_losses.append(avg_val)
        val_rot_losses.append(avg_val_rot)
        val_bone_losses.append(avg_val_bone)

        # =============================================================================
        # REAL VALIDATION EVALUATION (EVERY 10 EPOCHS)
        # =============================================================================
        real_val_mpjpe = None
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            print(f"\n🔍 Running evaluation on real 3DPW validation set (epoch {epoch})...")
            real_val_mpjpe = evaluate_on_real_validation(model, DATA_ROOT, device, BATCH_SIZE)
            if real_val_mpjpe is not None:
                real_val_mpjpe_history.append(real_val_mpjpe)
                real_val_epochs.append(epoch)
                print(f"📈 Real validation MPJPE: {real_val_mpjpe:.1f} mm")

        # =============================================================================
        # LOGGING AND CHECKPOINTING
        # =============================================================================
        train_mm = avg_train_pos * 1000.0
        val_mm = avg_val_pos * 1000.0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {avg_train_rot:.4f}, Bone: {avg_train_bone:.4f}")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {avg_val_rot:.4f}, Bone: {avg_val_bone:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Print real validation MPJPE if available
        if real_val_mpjpe is not None:
            print(f"  Real Val MPJPE: {real_val_mpjpe:.1f}mm 🎯")
        
        # Print data distribution for mixed datasets
        if DATASET_MODE == 'mixed':
            total_samples = sum(epoch_data_types.values())
            if total_samples > 0:
                synth_pct = epoch_data_types.get('synthetic', 0) / total_samples * 100
                real_pct = epoch_data_types.get('real', 0) / total_samples * 100
                print(f"  Data: Synthetic {synth_pct:.1f}%, Real {real_pct:.1f}%")

        # Learning rate scheduling
        scheduler.step(avg_val_pos)

        # Save best model
        if avg_val_pos < best_val_mpjpe:
            best_val_mpjpe = avg_val_pos
            best_state = model.state_dict().copy()
            early_stopping_counter = 0
            print(f"✓ New best validation MPJPE: {val_mm:.1f} mm")
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # =============================================================================
    # FINALIZATION
    # =============================================================================
    # Load best model
    model.load_state_dict(best_state)
    print(f"Training complete! Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # Run final evaluation on real validation set
    print(f"\n🏁 Final evaluation on real 3DPW validation set...")
    final_real_mpjpe = evaluate_on_real_validation(model, DATA_ROOT, device, BATCH_SIZE)
    if final_real_mpjpe is not None:
        print(f"🎯 Final Real Validation MPJPE: {final_real_mpjpe:.1f} mm")

    # Create and save learning curves
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Position loss
    axes[0, 0].plot([l*1000.0 if isinstance(l, (int, float)) else l['position']*1000.0 for l in train_losses], 
                   label="Train MPJPE", alpha=0.8)
    axes[0, 0].plot([l*1000.0 if isinstance(l, (int, float)) else l['position']*1000.0 for l in val_losses], 
                   label="Val MPJPE", alpha=0.8)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MPJPE (mm)")
    axes[0, 0].set_title("Position Loss")
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
    axes[0, 2].plot(train_bone_losses, label="Train Bone Loss", alpha=0.8)
    axes[0, 2].plot(val_bone_losses, label="Val Bone Loss", alpha=0.8)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Bone Length Loss")
    axes[0, 2].set_title("Bone Consistency Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Real validation MPJPE over time
    if real_val_mpjpe_history:
        axes[1, 0].plot(real_val_epochs, real_val_mpjpe_history, 'ro-', label="Real Val MPJPE", alpha=0.8)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MPJPE (mm)")
        axes[1, 0].set_title("Real 3DPW Validation Performance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Show improvement
        if len(real_val_mpjpe_history) > 1:
            improvement = real_val_mpjpe_history[0] - real_val_mpjpe_history[-1]
            axes[1, 0].text(0.05, 0.95, f"Improvement: {improvement:.1f}mm", 
                           transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    else:
        axes[1, 0].text(0.5, 0.5, 'No real validation data', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Real 3DPW Validation Performance")

    # Training progress summary
    axes[1, 1].text(0.1, 0.9, f"Training Summary:", transform=axes[1, 1].transAxes, fontweight='bold')
    axes[1, 1].text(0.1, 0.8, f"Best Val MPJPE: {best_val_mpjpe*1000:.1f} mm", transform=axes[1, 1].transAxes)
    if final_real_mpjpe is not None:
        axes[1, 1].text(0.1, 0.7, f"Final Real MPJPE: {final_real_mpjpe:.1f} mm", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"Dataset Mode: {DATASET_MODE}", transform=axes[1, 1].transAxes)
    if DATASET_MODE == 'mixed':
        axes[1, 1].text(0.1, 0.5, f"Synthetic Ratio: {SYNTHETIC_RATIO:.1f}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f"Epochs Trained: {epoch}", transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    # Performance comparison (if real validation data available)
    if real_val_mpjpe_history:
        performance_data = ['Mixed Val', 'Real Val (Final)']
        performance_values = [best_val_mpjpe*1000, real_val_mpjpe_history[-1]]
        
        bars = axes[1, 2].bar(performance_data, performance_values, alpha=0.7, 
                             color=['blue', 'orange'])
        axes[1, 2].set_ylabel('MPJPE (mm)')
        axes[1, 2].set_title('Final Performance Comparison')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value:.1f}mm', ha='center', va='bottom')
    else:
        axes[1, 2].text(0.5, 0.5, 'No comparison data', ha='center', va='center', 
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Performance Comparison')
    
    plt.tight_layout()
    curve_path = os.path.join(experiment_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curves → {curve_path}")

    # Save final model and metadata
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_rot_loss": avg_val_rot,
        "best_val_bone_loss": avg_val_bone,
        "final_real_val_mpjpe": final_real_mpjpe,
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
            "dataset_mode": DATASET_MODE,
            "synthetic_ratio": SYNTHETIC_RATIO if DATASET_MODE == 'mixed' else None,
            "augment_2d": AUGMENT_2D,
            "num_epochs_trained": epoch,
            "eval_every_n_epochs": EVAL_EVERY_N_EPOCHS
        },
        "dataset_info": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_distribution": train_dataset.get_dataset_distribution() if hasattr(train_dataset, 'get_dataset_distribution') else None,
            "val_distribution": val_dataset.get_dataset_distribution() if hasattr(val_dataset, 'get_dataset_distribution') else None
        }
    }, final_path)
    print(f"Final model → {final_path}")


if __name__ == "__main__":
    main()