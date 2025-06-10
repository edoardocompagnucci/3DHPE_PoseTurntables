import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.rot_head import MLPLifterRotationHead
from utils.losses import mpjpe_loss, combined_pose_loss, combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d
from models.poseaug import DifferentiablePoseAug, PoseAugWrapper
from src.data.threedpw_dataset import create_domain_mixing_datasets

def main():
    SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT= "checkpoints"

    BATCH_SIZE     = 64  
    DROPOUT_RATE = 0.1
    LEARNING_RATE  = 2e-4
    WEIGHT_DECAY   = 1e-4
    NUM_EPOCHS     = 350
    NUM_JOINTS     = 24
    IMG_SIZE = 512

    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.1
    LOSS_BONE_WEIGHT = 0.15

    POSEAUG_LR = 1e-4 
    POSEAUG_INITIAL_ROTATION_DEG = 8.0
    POSEAUG_INITIAL_TRANSLATION_M = 0.02

    early_stopping_patience = 25
    early_stopping_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    train_dataset, val_dataset = create_domain_mixing_datasets(
        data_root=DATA_ROOT,
        real_data_ratio=0.2,
        transform=normalizer,
        min_confidence=0.3
    )
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=False)

    # Create main model
    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)
    
    # Create learnable PoseAug module
    #poseaug_module = DifferentiablePoseAug(
    #    initial_max_rotation_deg=POSEAUG_INITIAL_ROTATION_DEG,
    #    initial_max_translation_m=POSEAUG_INITIAL_TRANSLATION_M,
    #    augmentation_prob=0.5
    #).to(device)
    
    # Wrap PoseAug for easy integration with training loop
    #poseaug_wrapper = PoseAugWrapper(poseaug_module, img_size=IMG_SIZE).to(device)
    
    # Create optimizers - separate for model and PoseAug parameters
    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #poseaug_optimizer = torch.optim.Adam(poseaug_module.parameters(), lr=POSEAUG_LR, weight_decay=WEIGHT_DECAY * 0.1)
    
    # Schedulers
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    #poseaug_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    poseaug_optimizer, mode="min", factor=0.7, patience=3, verbose=False
    #)

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name       = f"mlp_lifter_rotation_poseaug_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")
    
    # Print initial PoseAug parameters
    #initial_stats = poseaug_module.get_augmentation_stats()
    #print(f"Initial PoseAug parameters:")
    #print(f"  Max rotation: {initial_stats['max_rotation_deg']:.1f}°")
    #print(f"  Max translation: {initial_stats['max_translation_m']:.3f}m")

    best_val_mpjpe = float("inf")
    best_state     = None
    best_poseaug_state = None
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == 100:
            LOSS_BONE_WEIGHT = 0.05
            
        # Training phase
        model.train()
        #poseaug_wrapper.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            inputs = batch["joints_2d"].to(device)  # (B, 24, 2)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            
            # Camera parameters for PoseAug
            joints_3d_world = batch["joints_3d_world"].to(device)  # (B, 24, 3)
            K = batch["K"].to(device)  # (B, 3, 3)
            R = batch["R"].to(device)  # (B, 3, 3) 
            t = batch["t"].to(device)  # (B, 3)

            # Apply learnable PoseAug to augment 2D keypoints
            #augmented_inputs = poseaug_wrapper(
            #    inputs, joints_3d_world, K, R, t, training=True
            #)
            augmented_inputs = inputs
            # Forward pass through main model with augmented inputs
            pos3d, rot6d = model(augmented_inputs)

            pred_dict = {
                'positions': pos3d,
                'rotations': rot6d
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot
            }
            
            loss_dict = combined_pose_bone_loss(pred_dict, target_dict, 
                                pos_weight=LOSS_POS_WEIGHT, 
                                rot_weight=LOSS_ROT_WEIGHT,
                                bone_weight=LOSS_BONE_WEIGHT)
            
            loss = loss_dict['total']

            # Backward pass and optimization
            model_optimizer.zero_grad()
            #poseaug_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #torch.nn.utils.clip_grad_norm_(poseaug_module.parameters(), max_norm=0.5)
            
            model_optimizer.step()
            #poseaug_optimizer.step()
            
            running_loss += loss.item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()

        avg_train = running_loss / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)

        # Validation phase
        model.eval()
        #poseaug_wrapper.eval()
        running_val = 0.0
        running_val_pos = 0.0
        running_val_rot = 0.0
        running_val_bone = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target_pos = batch["joints_3d_centered"].to(device)
                target_rot = batch["rot_6d"].to(device)
                
                # No augmentation during validation
                pos3d, rot6d = model(inputs)
                
                pred_dict = {
                    'positions': pos3d,
                    'rotations': rot6d
                }
                target_dict = {
                    'positions': target_pos.flatten(1),
                    'rotations': target_rot
                }
                
                loss_dict = combined_pose_bone_loss(pred_dict, target_dict, 
                                  pos_weight=LOSS_POS_WEIGHT, 
                                  rot_weight=LOSS_ROT_WEIGHT,
                                  bone_weight=LOSS_BONE_WEIGHT)
                
                running_val += loss_dict['total'].item()
                running_val_pos += loss_dict['position'].item()
                running_val_rot += loss_dict['rotation'].item()
                running_val_bone += loss_dict['bone'].item()

        avg_val = running_val / len(val_loader)
        avg_val_pos = running_val_pos / len(val_loader)
        avg_val_rot = running_val_rot / len(val_loader)
        avg_val_bone = running_val_bone / len(val_loader)
        
        val_losses.append(avg_val)
        val_rot_losses.append(avg_val_rot)
        val_bone_losses.append(avg_val_bone)

        train_mm = avg_train_pos * 1000.0
        val_mm   = avg_val_pos * 1000.0
        rot_train = avg_train_rot
        rot_val = avg_val_rot
        bone_train = avg_train_bone
        bone_val = avg_val_bone

        current_model_lr = model_optimizer.param_groups[0]['lr']
        #current_poseaug_lr = poseaug_optimizer.param_groups[0]['lr']
        #poseaug_grad_norm = torch.nn.utils.clip_grad_norm_(poseaug_module.parameters(), max_norm=0.5)

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {rot_train:.4f}, Bone: {bone_train:.4f}")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {rot_val:.4f}, Bone: {bone_val:.4f}")
        #print(f"  LR - Model: {current_model_lr:.6f}, PoseAug: {current_poseaug_lr:.6f}")
        #print(f"  PoseAug grad norm: {poseaug_grad_norm:.6f}")
        
        # Print PoseAug stats every 10 epochs
        #if epoch % 10 == 0:
        #    aug_stats = poseaug_module.get_augmentation_stats()
        #    print(f"  PoseAug - Rotation: {aug_stats['max_rotation_deg']:.1f}°, Translation: {aug_stats['max_translation_m']:.3f}m")

        model_scheduler.step(avg_val_pos)
        #poseaug_scheduler.step(avg_val_pos)

        if avg_val_pos < best_val_mpjpe:
            best_val_mpjpe = avg_val_pos
            best_state     = model.state_dict().copy()
            #best_poseaug_state = poseaug_module.state_dict().copy()
            early_stopping_counter = 0
            print(f"✓ New best validation MPJPE: {val_mm:.1f} mm")
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Load best states
    model.load_state_dict(best_state)
    #poseaug_module.load_state_dict(best_poseaug_state)
    print(f"Training complete! Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # Print final PoseAug parameters
    #final_stats = poseaug_module.get_augmentation_stats()
    #print(f"Final PoseAug parameters:")
    #print(f"  Max rotation: {final_stats['max_rotation_deg']:.1f}°")
    #print(f"  Max translation: {final_stats['max_translation_m']:.3f}m")

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.plot([l*1000.0 for l in [loss['position'] if isinstance(loss, dict) else loss for loss in train_losses]], 
             label="Train MPJPE", alpha=0.8)
    ax1.plot([l*1000.0 for l in [loss['position'] if isinstance(loss, dict) else loss for loss in val_losses]], 
             label="Val MPJPE", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MPJPE (mm)")
    ax1.set_title("Position Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_rot_losses, label="Train Rot Loss", alpha=0.8)
    ax2.plot(val_rot_losses, label="Val Rot Loss", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Geodesic Loss")
    ax2.set_title("Rotation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(train_bone_losses, label="Train Bone Loss", alpha=0.8)
    ax3.plot(val_bone_losses, label="Val Bone Loss", alpha=0.8)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Bone Length Loss")
    ax3.set_title("Bone Consistency Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(experiment_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curves → {curve_path}")

    # Save final model with PoseAug parameters
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        #"poseaug_state": poseaug_module.state_dict(),
        "model_optimizer_state": model_optimizer.state_dict(),
        #"poseaug_optimizer_state": poseaug_optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_rot_loss": avg_val_rot,
        "best_val_bone_loss": avg_val_bone,
        #"final_poseaug_stats": final_stats,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "poseaug_lr": POSEAUG_LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": LOSS_BONE_WEIGHT,
            "poseaug_initial_rotation_deg": POSEAUG_INITIAL_ROTATION_DEG,
            "poseaug_initial_translation_m": POSEAUG_INITIAL_TRANSLATION_M
        }
    }, final_path)
    print(f"Final model with PoseAug → {final_path}")

if __name__ == "__main__":
    main()