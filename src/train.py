import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.rot_head import MLPLifterRotationHead
from utils.losses import mpjpe_loss, combined_pose_loss, combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d
from src.data.threedpw_dataset import create_domain_mixing_datasets

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    BATCH_SIZE = 64  
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 350
    NUM_JOINTS = 24
    IMG_SIZE = 512

    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.1
    LOSS_BONE_WEIGHT = 0.15

    early_stopping_patience = 25
    early_stopping_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    current_real_ratio = 0.2
    train_dataset, val_dataset = create_domain_mixing_datasets(
        data_root=DATA_ROOT,
        real_data_ratio=current_real_ratio,
        transform=normalizer,
        min_confidence=0.3
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=12,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=6)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=6)

    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout=DROPOUT_RATE).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0=50,  # Initial cycle length - matches domain shift epochs
        T_mult=1,  # Keep consistent cycle length 
        eta_min=1e-5  # Minimum learning rate
    )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"mlp_lifter_domain_mixing_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    best_val_mpjpe = float("inf")
    best_state = None
    best_val_mpjpe_per_ratio = {}  # Track best performance per ratio
    best_state_per_ratio = {}  # Track best model state per ratio
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []
    syn_losses, real_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= 50:
            target_real_ratio = 0.2
            LOSS_BONE_WEIGHT = 0.15  # Initial value
        elif epoch <= 100:
            target_real_ratio = 0.3  
            LOSS_BONE_WEIGHT = 0.05  # First reduction
        elif epoch <= 150:
            target_real_ratio = 0.4
            LOSS_BONE_WEIGHT = 0.025  # Second reduction
        else:
            target_real_ratio = 0.5
            LOSS_BONE_WEIGHT = 0.01  # Final value
            
        if target_real_ratio != current_real_ratio:
            current_real_ratio = target_real_ratio
            early_stopping_counter = 0  # Reset counter on domain shift
            print(f"Updating real data ratio to {current_real_ratio:.1%}")
            train_dataset, val_dataset = create_domain_mixing_datasets(
                data_root=DATA_ROOT,
                real_data_ratio=current_real_ratio,
                transform=normalizer,
                min_confidence=0.3
            )

            train_loader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=12,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=6)
            
            val_loader = DataLoader(val_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=12,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=6)
            
        model.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        syn_loss_sum, real_loss_sum = 0.0, 0.0
        syn_count, real_count = 0, 0
        
        for batch in train_loader:
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            dataset_types = batch["dataset_type"]

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
            
            loss = loss_dict['total']

            for i, dataset_type in enumerate(dataset_types):
                sample_loss = loss_dict['total'].item() / len(dataset_types)
                if dataset_type == 'synthetic':
                    syn_loss_sum += sample_loss
                    syn_count += 1
                else:
                    real_loss_sum += sample_loss
                    real_count += 1

            model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            
            running_loss += loss.item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()

        avg_train = running_loss / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        avg_syn_loss = syn_loss_sum / syn_count if syn_count > 0 else 0
        avg_real_loss = real_loss_sum / real_count if real_count > 0 else 0
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)
        syn_losses.append(avg_syn_loss)
        real_losses.append(avg_real_loss)

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
        val_mm = avg_val_pos * 1000.0
        rot_train = avg_train_rot
        rot_val = avg_val_rot
        bone_train = avg_train_bone
        bone_val = avg_val_bone

        current_model_lr = model_optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{NUM_EPOCHS} (Real: {current_real_ratio:.1%})")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {rot_train:.4f}, Bone: {bone_train:.4f}")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {rot_val:.4f}, Bone: {bone_val:.4f}")
        print(f"  Syn/Real - Syn: {avg_syn_loss:.4f}, Real: {avg_real_loss:.4f}")
        print(f"  LR: {current_model_lr:.6f}")

        model_scheduler.step()

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
                
        # Track best model per ratio
        if current_real_ratio not in best_val_mpjpe_per_ratio or avg_val_pos < best_val_mpjpe_per_ratio[current_real_ratio]:
            best_val_mpjpe_per_ratio[current_real_ratio] = avg_val_pos
            best_state_per_ratio[current_real_ratio] = model.state_dict().copy()
            print(f"✓ New best for {current_real_ratio:.1%} real data: {val_mm:.1f} mm")

    # Determine highest ratio that completed training
    final_ratio = max(best_state_per_ratio.keys())  
    model.load_state_dict(best_state_per_ratio[final_ratio])
    print(f"Training complete! Best Val MPJPE overall: {best_val_mpjpe*1000.0:.1f} mm")
    print(f"Using best model from {final_ratio:.1%} real data with MPJPE: {best_val_mpjpe_per_ratio[final_ratio]*1000.0:.1f} mm")

    # Create visualization with per-ratio performance
    real_ratios = sorted(best_val_mpjpe_per_ratio.keys())
    ratio_epochs = []
    ratio_perfs = []

    for ratio in real_ratios:
        ratio_epochs.append(f"{ratio:.0%}")
        ratio_perfs.append(best_val_mpjpe_per_ratio[ratio] * 1000.0)

    fig = plt.figure(figsize=(16, 18))
    
    # Create a 3x2 grid
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)  # Fifth subplot (bottom left)

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

    ax4.plot(syn_losses, label="Synthetic Loss", alpha=0.8)
    ax4.plot(real_losses, label="Real Loss", alpha=0.8)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.set_title("Synthetic vs Real Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add new subplot for per-ratio performance
    ax5.bar(ratio_epochs, ratio_perfs, alpha=0.7, color='green')
    ax5.set_xlabel("Real Data Ratio")
    ax5.set_ylabel("Best MPJPE (mm)")
    ax5.set_title("Best Performance Per Domain Ratio")
    for i, v in enumerate(ratio_perfs):
        ax5.text(i, v + 1, f"{v:.1f}", ha='center')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(experiment_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curves → {curve_path}")

    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_optimizer_state": model_optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_rot_loss": avg_val_rot,
        "best_val_bone_loss": avg_val_bone,
        "best_val_mpjpe_per_ratio": best_val_mpjpe_per_ratio,
        "final_selected_ratio": final_ratio,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": LOSS_BONE_WEIGHT,
            "final_real_ratio": current_real_ratio,
            "scheduler": "CosineAnnealingWarmRestarts"
        }
    }, final_path)
    print(f"Final model → {final_path}")

if __name__ == "__main__":
    main()