import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.rot_head import MLPLifterRotationHead
from models.poseaug import PoseAug
from utils.losses import mpjpe_loss, combined_pose_loss, combined_pose_bone_loss
from utils.transforms import NormalizerJoints2d
from utils import rotation_utils

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    BATCH_SIZE = 34   
    DROPOUT_RATE = 0.25
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 220
    NUM_JOINTS = 24
    IMG_SIZE = 512

    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.1
    LOSS_BONE_WEIGHT = 0.05
    
    POSEAUG_START_EPOCH = 5
    POSEAUG_FREQUENCY = 2
    POSEAUG_WEIGHT = 0.5

    early_stopping_patience = 25
    early_stopping_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    train_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "train.txt"),
        transform=normalizer,
        augment_2d=True,
    )
    val_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "val.txt"),
        transform=normalizer,
        augment_2d=False
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=False)

    model = MLPLifterRotationHead(num_joints=NUM_JOINTS, dropout_rate=DROPOUT_RATE).to(device)
    poseaug = PoseAug(num_joints=NUM_JOINTS, img_size=IMG_SIZE).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=3, verbose=True
    )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name = f"mlp_lifter_poseaug_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")
    print(f"PoseAug will start at epoch {POSEAUG_START_EPOCH}")

    best_val_mpjpe = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    train_rot_losses, val_rot_losses = [], []
    train_bone_losses, val_bone_losses = [], []
    poseaug_applied_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        running_aug_loss = 0.0
        epoch_aug_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            inputs = batch["joints_2d"]
            target_pos = batch["joints_3d_centered"]
            target_rot = batch["rot_6d"]

            outputs = model(inputs)
            
            pred_dict = {
                'positions': outputs['positions'],
                'rotations': outputs['rotations']
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot.flatten(1)
            }
            
            orig_loss_dict = combined_pose_bone_loss(
                pred_dict, target_dict, 
                pos_weight=LOSS_POS_WEIGHT, 
                rot_weight=LOSS_ROT_WEIGHT,
                bone_weight=LOSS_BONE_WEIGHT
            )
            
            total_loss = orig_loss_dict['total']
            
            if epoch >= POSEAUG_START_EPOCH and batch_idx % POSEAUG_FREQUENCY == 0:
                try:
                    camera_params = {
                        'K': batch["K"],
                        'R': batch["R"],
                        't': batch["t"]
                    }
                    
                    aug_data = poseaug(batch["joints_3d_world"], batch["rot_mats"], camera_params)
                    
                    aug_inputs = aug_data['poses_2d_aug']
                    aug_target_3d = aug_data['poses_3d_aug']
                    aug_rot_mats = aug_data['rot_mats_aug']
                    
                    aug_target_pos = aug_target_3d - aug_target_3d[:, 0:1]
                    
                    aug_target_rot_6d = rotation_utils.rot_matrix_to_6d(aug_rot_mats)
                    
                    aug_outputs = model(aug_inputs)
                    
                    aug_pred_dict = {
                        'positions': aug_outputs['positions'],
                        'rotations': aug_outputs['rotations']
                    }
                    aug_target_dict = {
                        'positions': aug_target_pos.flatten(1),
                        'rotations': aug_target_rot_6d.flatten(1)
                    }
                    
                    aug_loss_dict = combined_pose_bone_loss(
                        aug_pred_dict, aug_target_dict,
                        pos_weight=LOSS_POS_WEIGHT,
                        rot_weight=LOSS_ROT_WEIGHT,
                        bone_weight=LOSS_BONE_WEIGHT
                    )
                    
                    total_loss = orig_loss_dict['total'] + POSEAUG_WEIGHT * aug_loss_dict['total']
                    
                    running_aug_loss += aug_loss_dict['total'].item()
                    epoch_aug_count += 1
                    poseaug_applied_count += 1
                    
                except Exception as e:
                    print(f"PoseAug failed for batch {batch_idx}: {e}")
                    pass
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            running_pos_loss += orig_loss_dict['position'].item()
            running_rot_loss += orig_loss_dict['rotation'].item()
            running_bone_loss += orig_loss_dict['bone'].item()

        avg_train = running_loss / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        avg_train_aug = running_aug_loss / max(epoch_aug_count, 1)
        
        train_losses.append(avg_train)
        train_rot_losses.append(avg_train_rot)
        train_bone_losses.append(avg_train_bone)

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
                
                outputs = model(inputs)
                
                pred_dict = {
                    'positions': outputs['positions'],
                    'rotations': outputs['rotations']
                }
                target_dict = {
                    'positions': target_pos.flatten(1),
                    'rotations': target_rot.flatten(1)
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
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train - Pos: {train_mm:.1f}mm, Rot: {avg_train_rot:.4f}, Bone: {avg_train_bone:.4f}")
        if epoch >= POSEAUG_START_EPOCH and epoch_aug_count > 0:
            print(f"  PoseAug - Applied: {epoch_aug_count}, Avg Loss: {avg_train_aug:.4f}")
        print(f"  Val   - Pos: {val_mm:.1f}mm, Rot: {avg_val_rot:.4f}, Bone: {avg_val_bone:.4f}")
        print(f"  LR: {current_lr:.6f}")

        scheduler.step(avg_val_pos)

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

    model.load_state_dict(best_state)
    print(f"Training complete! Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")
    print(f"Total PoseAug applications: {poseaug_applied_count}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.plot([l*1000.0 for l in train_losses], 
             label="Train Total", alpha=0.8)
    ax1.plot([l*1000.0 for l in val_losses], 
             label="Val Total", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss * 1000")
    ax1.set_title("Total Loss (with PoseAug)")
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

    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_rot_loss": avg_val_rot,
        "best_val_bone_loss": avg_val_bone,
        "poseaug_applied_count": poseaug_applied_count,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "pos_weight": LOSS_POS_WEIGHT,
            "rot_weight": LOSS_ROT_WEIGHT,
            "bone_weight": LOSS_BONE_WEIGHT,
            "poseaug_used": True,
            "poseaug_start_epoch": POSEAUG_START_EPOCH,
            "poseaug_frequency": POSEAUG_FREQUENCY,
            "poseaug_weight": POSEAUG_WEIGHT,
            "poseaug_params": {
                "rotation_range": poseaug.rotation_range,
                "bone_scale_range": poseaug.bone_scale_range,
                "camera_jitter": poseaug.camera_jitter
            }
        }
    }, final_path)
    print(f"Final model → {final_path}")

if __name__ == "__main__":
    main()