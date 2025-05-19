import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.synthetic_pose_dataset import SyntheticPoseDataset
from models.lifter import MLPLifter
from utils.losses import mpjpe_loss

def main():
    # -------------------------------------------------------------------------
    # 1) Config
    # -------------------------------------------------------------------------
    SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT= "checkpoints"
    BATCH_SIZE     = 64
    LEARNING_RATE  = 1e-3
    NUM_EPOCHS     = 50
    NUM_JOINTS     = 16 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2) Datasets & Loaders
    # -------------------------------------------------------------------------
    train_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "train.txt")
    )
    val_dataset = SyntheticPoseDataset(
        data_root=DATA_ROOT,
        split_txt=os.path.join(DATA_ROOT, "splits", "val.txt")
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # -------------------------------------------------------------------------
    # 3) Model, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    model     = MLPLifter(num_joints=NUM_JOINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # -------------------------------------------------------------------------
    # 4) Checkpoint dir
    # -------------------------------------------------------------------------
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    exp_name       = f"mlp_lifter_{datetime.now():%Y%m%d_%H%M%S}"
    experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # -------------------------------------------------------------------------
    # 5) Training loop
    # -------------------------------------------------------------------------
    best_val_mpjpe = float("inf")
    best_state     = None
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ——— Train —————————————————————————————————————————
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["joints_2d"].to(device)
            target = batch["joints_3d"].to(device)

            preds = model(inputs)
            loss  = mpjpe_loss(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ——— Validate ——————————————————————————————————————
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["joints_2d"].to(device)
                target = batch["joints_3d"].to(device)
                preds  = model(inputs)
                running_val += mpjpe_loss(preds, target).item()

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)

        # Convert to millimeters for logging
        train_mm = avg_train * 1000.0
        val_mm   = avg_val   * 1000.0

        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
              f"Train MPJPE: {train_mm:.1f} mm,  "
              f"Val MPJPE: {val_mm:.1f} mm")

        # Scheduler step on meters (raw loss)
        scheduler.step(avg_val)

        # Save best
        if avg_val < best_val_mpjpe:
            best_val_mpjpe = avg_val
            best_state     = model.state_dict().copy()
            ckpt_name = f"best_epoch{epoch:02d}_mpjpe{val_mm:.1f}mm.pth"
            ckpt_path = os.path.join(experiment_dir, ckpt_name)
            torch.save({
                "epoch": epoch,
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "train_loss_m": avg_train,
                "val_loss_m":   avg_val,
            }, ckpt_path)
            print(f"→ New best model saved: {ckpt_name}")

    # Load best model for final save
    model.load_state_dict(best_state)
    print(f"Training complete!  Best Val MPJPE: {best_val_mpjpe*1000.0:.1f} mm")

    # -------------------------------------------------------------------------
    # 6) Plot learning curves (in mm)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot([l*1000.0 for l in train_losses], label="Train")
    plt.plot([l*1000.0 for l in val_losses],   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MPJPE (mm)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(experiment_dir, "learning_curve.png")
    plt.savefig(curve_path)
    print(f"Learning curve → {curve_path}")

    # -------------------------------------------------------------------------
    # 7) Final checkpoint
    # -------------------------------------------------------------------------
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch":      NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_m": best_val_mpjpe,
    }, final_path)
    print(f"Final model → {final_path}")

if __name__ == "__main__":
    main()
