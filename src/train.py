import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.lifter import MLPLifter, MLPLifter_v2
from utils.losses import mpjpe_loss
from utils.transforms import NormalizerJoints2d

def main():
    SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT= "checkpoints"
    
    # Improved hyperparameters
    BATCH_SIZE     = 64   
    DROPOUT_RATE = 0.25
    LEARNING_RATE  = 5e-4
    WEIGHT_DECAY   = 1e-4
    NUM_EPOCHS     = 180
    NUM_JOINTS     = 24
    IMG_SIZE = 512

    early_stopping_patience = 15  # Increased patience
    early_stopping_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalizer = NormalizerJoints2d(img_size=IMG_SIZE)

    train_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "train.txt"),
        transform=normalizer
    )
    val_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "val.txt"),
        transform=normalizer
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Initialize model with dropout
    model = MLPLifter_v2(num_joints=NUM_JOINTS, dropout_rate=DROPOUT_RATE).to(device)
    
    # Add weight decay to optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # More aggressive scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=3, verbose=True
    )

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name       = f"mlp_lifter_regularized_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    best_val_mpjpe = float("inf")
    best_state     = None
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training phase
        model.train()  # Important: set to training mode for dropout
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["joints_2d"].to(device)
            target = batch["joints_3d_centered"].to(device)

            preds = model(inputs)
            loss  = mpjpe_loss(preds, target)

            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation phase
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target = batch["joints_3d_centered"].to(device)
                preds  = model(inputs)
                running_val += mpjpe_loss(preds, target).item()

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)

        train_mm = avg_train * 1000.0
        val_mm   = avg_val   * 1000.0

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
            f"Train MPJPE: {train_mm:.1f} mm,  "
            f"Val MPJPE: {val_mm:.1f} mm,  "
            f"Lr: {current_lr:.6f}")

        scheduler.step(avg_val)

        # Save best model
        if avg_val < best_val_mpjpe:
            best_val_mpjpe = avg_val
            best_state     = model.state_dict().copy()
            early_stopping_counter = 0
            print(f"✓ New best validation MPJPE: {val_mm:.1f} mm")
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Load best model
    model.load_state_dict(best_state)
    print(f"Training complete! Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # Plot learning curves
    plt.figure(figsize=(10,6))
    plt.plot([l*1000.0 for l in train_losses], label="Train MPJPE", alpha=0.8)
    plt.plot([l*1000.0 for l in val_losses],   label="Val MPJPE", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MPJPE (mm)")
    plt.title("Learning Curve with Regularization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    curve_path = os.path.join(experiment_dir, "learning_curve.png")
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curve → {curve_path}")

    # Save final model
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE
        }
    }, final_path)
    print(f"Final model → {final_path}")

if __name__ == "__main__":
    main()